"""Compute fraction of test pulses with best-ambiguity SIMILARITY_ERROR < threshold."""

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
from evaluate_cnn import convergence_fraction_cnn_sweep
from frog_reconstruction_model import MODEL_REGISTRY, extract_pulse_prediction
from pcgpa_reconstruct import per_pulse_similarity_amb_pcgpa_at_snr
from trace_noise import add_trace_noise_awgn

SNR_SWEEP_DB = np.arange(-10.0, 31.0, 5.0)
DEFAULT_OUT = _SRC / "checkpoints/benchmark/multires_sim_convergence_frac_lt_0p1.npz"


def _load_multires_on_device(path: Path, device: torch.device) -> torch.nn.Module:
    """Load Multires checkpoint; LazyLinear init on CPU avoids GPU bad-allocation after heavy FROG build."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    n = int(ckpt.get("N", 64))
    model_name = ckpt.get("model_name", "multires")
    if "train_config" in ckpt:
        model_name = ckpt["train_config"].get("model_name", model_name)
    model = MODEL_REGISTRY[model_name](out_dim=2 * n)
    model(torch.zeros(1, 1, n, n))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def _save_npz(out: Path, **arrays) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(f"  checkpoint saved: {out}", flush=True)


def _load_partial(out: Path) -> dict[str, np.ndarray]:
    if not out.exists():
        return {}
    z = np.load(out)
    return {k: z[k] for k in z.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--pcgpa-maxiter", type=int, default=200)
    parser.add_argument("--pcgpa-n-restarts", type=int, default=3)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cnn-only", action="store_true")
    parser.add_argument("--pcgpa-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.output)
    partial = {} if args.force else _load_partial(out)
    if out.exists() and not args.force:
        have_cnn = "multires_2k_trace" in partial and "multires_60k" in partial
        have_pcgpa = "pcgpa" in partial and np.all(np.isfinite(partial["pcgpa"]))
        if have_cnn and have_pcgpa:
            print(f"Complete: {out} (use --force to recompute)", flush=True)
            return
        print(f"Resuming from partial checkpoint: {out}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frog_device = torch.device("cpu")
    print(f"device: {device}  (FROG trace generation on {frog_device})", flush=True)
    grid = PulseGridConfig(n=64)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    # Only the test split is needed; avoid generating 2k train FROG traces (RAM/GPU pressure).
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
    w_vec = bundle.w_vec
    I_test = bundle.test_loader.dataset.tensors[0].cpu().numpy()
    E_test = bundle.test_loader.dataset.tensors[1].cpu().numpy()
    del bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()

    meta = dict(
        snr_sweep_db=SNR_SWEEP_DB,
        threshold=float(args.threshold),
        n_test=int(args.n_test),
        seed=int(args.seed),
        pcgpa_maxiter=int(args.pcgpa_maxiter),
        pcgpa_n_restarts=int(args.pcgpa_n_restarts),
    )

    t0 = time.time()
    results: dict[str, np.ndarray] = {
        k: partial[k]
        for k in ("multires_2k_trace", "multires_60k", "pcgpa")
        if k in partial
    }

    run_cnn = not args.pcgpa_only and (
        args.force
        or "multires_2k_trace" not in results
        or "multires_60k" not in results
    )
    run_pcgpa = not args.cnn_only and (
        args.force
        or "pcgpa" not in results
        or not np.all(np.isfinite(results.get("pcgpa", np.array([np.nan]))))
    )

    if run_cnn:
        opt_ckpt = _SRC / "checkpoints/benchmark/multires_2k_noisy_trace_lambda_opt.pt"
        print(f"\nMultires 2K + trace from {opt_ckpt.name} …", flush=True)
        inner = _load_multires_on_device(opt_ckpt, device)

        class _TraceWrapper(torch.nn.Module):
            def __init__(self, net: torch.nn.Module) -> None:
                super().__init__()
                self.net = net

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return extract_pulse_prediction(self.net(x))

        model_trace = _TraceWrapper(inner).to(device)
        model_trace.eval()
        results["multires_2k_trace"] = convergence_fraction_cnn_sweep(
            model_trace,
            test_loader,
            SNR_SWEEP_DB,
            threshold=float(args.threshold),
        )
        del model_trace, inner
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  2K+trace done in {time.time() - t0:.0f}s", flush=True)

        ckpt_60k = _SRC / "checkpoints/large_60k_multires_50ep.pt"
        print(f"\nMultires 60K from {ckpt_60k.name} …", flush=True)
        model_60k = _load_multires_on_device(ckpt_60k, device)
        t1 = time.time()
        results["multires_60k"] = convergence_fraction_cnn_sweep(
            model_60k,
            test_loader,
            SNR_SWEEP_DB,
            threshold=float(args.threshold),
        )
        del model_60k
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  60K done in {time.time() - t1:.0f}s", flush=True)

        pcgpa_placeholder = results.get(
            "pcgpa", np.full(len(SNR_SWEEP_DB), np.nan, dtype=np.float64)
        )
        _save_npz(
            out,
            **meta,
            multires_2k_trace=results["multires_2k_trace"],
            multires_60k=results["multires_60k"],
            pcgpa=pcgpa_placeholder,
        )

    if run_pcgpa:
        if "multires_2k_trace" not in results or "multires_60k" not in results:
            raise RuntimeError("PCGPA requires CNN results in checkpoint (run without --pcgpa-only first).")

        print(
            f"\nPCGPA (maxiter={args.pcgpa_maxiter}, restarts={args.pcgpa_n_restarts}, "
            f"n={int(args.n_test)}) …",
            flush=True,
        )
        t2 = time.time()
        pcgpa_fracs = results.get(
            "pcgpa", np.full(len(SNR_SWEEP_DB), np.nan, dtype=np.float64)
        ).copy()
        for i, snr_db in enumerate(SNR_SWEEP_DB):
            if np.isfinite(pcgpa_fracs[i]) and not args.force:
                print(f"  skip PCGPA @ {float(snr_db):.1f} dB (cached)", flush=True)
                continue
            print(f"  PCGPA convergence @ {float(snr_db):.1f} dB …", flush=True)
            per = per_pulse_similarity_amb_pcgpa_at_snr(
                I_test,
                E_test,
                float(snr_db),
                add_noise_fn=add_trace_noise_awgn,
                dt=grid.dt,
                sigma_omega=grid.resolved_sigma_omega,
                maxiter=int(args.pcgpa_maxiter),
                n_subsample=None,
                seed=int(args.seed),
                n_restarts=int(args.pcgpa_n_restarts),
                show_progress=True,
                omega_axis=w_vec,
            )
            pcgpa_fracs[i] = float(np.mean(per < float(args.threshold)))
            results["pcgpa"] = pcgpa_fracs
            _save_npz(
                out,
                **meta,
                multires_2k_trace=results["multires_2k_trace"],
                multires_60k=results["multires_60k"],
                pcgpa=pcgpa_fracs,
            )
        print(f"  PCGPA done in {time.time() - t2:.0f}s", flush=True)

    print(f"\nComplete: {out}  (total {time.time() - t0:.0f}s)", flush=True)
    for name, fracs in results.items():
        print(f"  {name}: {fracs}", flush=True)


if __name__ == "__main__":
    main()
