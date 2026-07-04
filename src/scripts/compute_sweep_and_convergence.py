"""SNR sweep + convergence fraction in one pass (shared per-pulse metrics where possible)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataset_utils import PulseGridConfig, build_frog_dataloaders
from evaluate_cnn import per_pulse_amb_l1_and_sim_cnn_at_snr
from frog_reconstruction_model import MODEL_REGISTRY, extract_pulse_prediction
from pcgpa_reconstruct import (
    _pcgpa_subsample_indices,
    per_pulse_amb_l1_and_sim_pcgpa_at_snr,
)
from trace_noise import add_trace_noise_awgn

SNR_SWEEP_DB = np.arange(-10.0, 31.0, 5.0)
DEFAULT_OUT = _SRC / "checkpoints/benchmark/multires_2k_snr0_sweep_convergence.npz"
DEFAULT_CKPT_2K = _SRC / "checkpoints/benchmark/multires_2k_noisy_trace_lam15_snr0.pt"
DEFAULT_CKPT_60K = _SRC / "checkpoints/large_60k_multires_50ep.pt"


def _load_multires(path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    n = int(ckpt.get("N", 64))
    model_name = ckpt.get("model_name", "multires")
    if "train_config" in ckpt:
        model_name = ckpt["train_config"].get("model_name", model_name)
    model = MODEL_REGISTRY[model_name](out_dim=2 * n)
    model(torch.zeros(1, 1, n, n))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def _mean_std(arr: np.ndarray) -> tuple[float, float]:
    a = np.asarray(arr, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


def _cnn_sweep_and_convergence(
    model: torch.nn.Module,
    test_loader,
    snr_sweep_db: np.ndarray,
    sub_idx: np.ndarray,
    threshold: float,
    *,
    label: str,
) -> dict[str, np.ndarray]:
    l1_m, l1_s, sim_m, sim_s, conv = [], [], [], [], []
    for snr_db in snr_sweep_db:
        print(f"  {label} @ {float(snr_db):.1f} dB …", flush=True)
        l1_per, sim_per = per_pulse_amb_l1_and_sim_cnn_at_snr(
            model, test_loader, float(snr_db)
        )
        m, s = _mean_std(l1_per)
        l1_m.append(m)
        l1_s.append(s)
        m, s = _mean_std(sim_per)
        sim_m.append(m)
        sim_s.append(s)
        conv.append(float(np.mean(sim_per[sub_idx] < float(threshold))))
    return {
        "l1_m": np.asarray(l1_m, dtype=np.float64),
        "l1_s": np.asarray(l1_s, dtype=np.float64),
        "sim_m": np.asarray(sim_m, dtype=np.float64),
        "sim_s": np.asarray(sim_s, dtype=np.float64),
        "conv_frac": np.asarray(conv, dtype=np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--n-subsample", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--ckpt-2k", default=str(DEFAULT_CKPT_2K))
    parser.add_argument("--ckpt-60k", default=str(DEFAULT_CKPT_60K))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--pcgpa-maxiter", type=int, default=200)
    parser.add_argument("--pcgpa-n-restarts", type=int, default=3)
    parser.add_argument("--skip-pcgpa", action="store_true")
    parser.add_argument("--skip-60k", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"Complete: {out} (use --force to recompute)", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frog_device = torch.device("cpu")
    print(f"device: {device}  (FROG on {frog_device})", flush=True)

    grid = PulseGridConfig(n=64)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    bundle = build_frog_dataloaders(
        n_train=1,
        n_val=1,
        n_test=int(args.n_test),
        batch_size=32,
        seed=int(args.seed),
        device=frog_device,
        grid=grid,
    )
    test_loader = bundle.test_loader
    I_test = bundle.test_loader.dataset.tensors[0].cpu().numpy()
    E_test = bundle.test_loader.dataset.tensors[1].cpu().numpy()
    w_vec = bundle.w_vec
    del bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()

    n_test = int(args.n_test)
    n_sub = min(int(args.n_subsample), n_test)
    sub_idx = _pcgpa_subsample_indices(n_test, n_sub, int(args.seed))
    sub_idx_sorted = np.sort(sub_idx)
    print(
        f"n_test={n_test}  n_subsample={n_sub}  threshold={float(args.threshold):.2f}",
        flush=True,
    )

    t0 = time.time()
    results: dict[str, np.ndarray] = {}

    ckpt_2k = Path(args.ckpt_2k)
    print(f"\nMultires 2K + trace from {ckpt_2k.name} …", flush=True)
    inner = _load_multires(ckpt_2k, device)

    class _TraceWrapper(torch.nn.Module):
        def __init__(self, net: torch.nn.Module) -> None:
            super().__init__()
            self.net = net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return extract_pulse_prediction(self.net(x))

    model_2k = _TraceWrapper(inner).to(device)
    r2k = _cnn_sweep_and_convergence(
        model_2k,
        test_loader,
        SNR_SWEEP_DB,
        sub_idx_sorted,
        float(args.threshold),
        label="2K+trace",
    )
    for k, v in r2k.items():
        results[f"trace_{k}"] = v
    del model_2k, inner
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not args.skip_60k:
        ckpt_60k = Path(args.ckpt_60k)
        print(f"\nMultires 60K from {ckpt_60k.name} …", flush=True)
        model_60k = _load_multires(ckpt_60k, device)
        r60 = _cnn_sweep_and_convergence(
            model_60k,
            test_loader,
            SNR_SWEEP_DB,
            sub_idx_sorted,
            float(args.threshold),
            label="60K",
        )
        for k, v in r60.items():
            results[f"m60k_{k}"] = v
        del model_60k
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not args.skip_pcgpa:
        print(
            f"\nPCGPA on {n_sub} pulses (maxiter={args.pcgpa_maxiter}, "
            f"restarts={args.pcgpa_n_restarts}) …",
            flush=True,
        )
        pcgpa_l1_m, pcgpa_l1_s = [], []
        pcgpa_sim_m, pcgpa_sim_s = [], []
        pcgpa_conv = []
        pcgpa_kw = dict(
            add_noise_fn=add_trace_noise_awgn,
            dt=grid.dt,
            sigma_omega=grid.resolved_sigma_omega,
            maxiter=int(args.pcgpa_maxiter),
            n_restarts=int(args.pcgpa_n_restarts),
            seed=int(args.seed),
            omega_axis=w_vec,
        )
        for snr_db in SNR_SWEEP_DB:
            print(f"  PCGPA @ {float(snr_db):.1f} dB …", flush=True)
            l1_per, sim_per = per_pulse_amb_l1_and_sim_pcgpa_at_snr(
                I_test,
                E_test,
                float(snr_db),
                n_subsample=n_sub,
                show_progress=True,
                **pcgpa_kw,
            )
            m, s = _mean_std(l1_per)
            pcgpa_l1_m.append(m)
            pcgpa_l1_s.append(s)
            m, s = _mean_std(sim_per)
            pcgpa_sim_m.append(m)
            pcgpa_sim_s.append(s)
            pcgpa_conv.append(float(np.mean(sim_per < float(args.threshold))))
        results["pcgpa_l1_m"] = np.asarray(pcgpa_l1_m, dtype=np.float64)
        results["pcgpa_l1_s"] = np.asarray(pcgpa_l1_s, dtype=np.float64)
        results["pcgpa_sim_m"] = np.asarray(pcgpa_sim_m, dtype=np.float64)
        results["pcgpa_sim_s"] = np.asarray(pcgpa_sim_s, dtype=np.float64)
        results["pcgpa_conv_frac"] = np.asarray(pcgpa_conv, dtype=np.float64)
        results["pcgpa_n_subsample"] = np.asarray(n_sub, dtype=np.int32)

    meta = dict(
        snr_sweep_db=SNR_SWEEP_DB,
        seed=int(args.seed),
        n_test=n_test,
        n_subsample=n_sub,
        sub_indices=sub_idx_sorted,
        convergence_threshold=float(args.threshold),
        ckpt_2k=str(ckpt_2k),
        ckpt_60k=str(args.ckpt_60k),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **meta, **results)
    print(f"\nComplete: {out}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
