"""Multires 60K baseline (λ=0) on notebook modified-C1 + FROGNetPadded traces."""

from __future__ import annotations

import argparse
import importlib.util
import json
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

from data_generation import stochastic_pulse_config_notebook_modified_c1
from dataset_utils import (
    build_notebook_c1_padded_frog,
    build_stochastic_padded_frog_dataloaders,
    notebook_c1_spectral_fft_params,
)
from evaluate_cnn import run_cnn_snr_sweep, save_cnn_sweep
from frog_reconstruction_model import extract_pulse_prediction
from train import build_model

_STOCH_SCRIPT = _SRC / "scripts" / "run_multires_2k_stochastic_noisy_trace_lambda.py"
_spec = importlib.util.spec_from_file_location("_stoch_lam", _STOCH_SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

subset_loader = _mod.subset_loader
train_multires_noisy_trace_loss_early_stop = _mod.train_multires_noisy_trace_loss_early_stop
calibrate_trace_scale = _mod.calibrate_trace_scale
SNR_SWEEP_DB = _mod.SNR_SWEEP_DB

DEFAULT_CKPT = _SRC / "checkpoints/benchmark/stochastic_c1_padded_multires_60k_baseline.pt"
DEFAULT_META = (
    _SRC / "checkpoints/benchmark/stochastic_c1_padded_multires_60k_baseline_meta.json"
)
DEFAULT_SWEEP = (
    _SRC / "checkpoints/benchmark/stochastic_c1_padded_multires_60k_baseline_sweep.npz"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=60000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--train-snr-min", type=float, default=0.0)
    parser.add_argument("--train-snr-max", type=float, default=30.0)
    parser.add_argument("--val-snr-db", type=float, default=15.0)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--meta-json", default=str(DEFAULT_META))
    parser.add_argument("--sweep-output", default=str(DEFAULT_SWEEP))
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    meta_path = Path(args.meta_json)
    sweep_path = Path(args.sweep_output)

    if ckpt_path.exists() and sweep_path.exists() and not args.force:
        print(
            f"Exists: {ckpt_path} and {sweep_path} (use --force to recompute)",
            flush=True,
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    n = 64
    train_snr_range = (float(args.train_snr_min), float(args.train_snr_max))

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    grid = stochastic_pulse_config_notebook_modified_c1(n=n)
    spectral = notebook_c1_spectral_fft_params(
        n=n,
        t_total_fs=grid.t_total_fs,
        n_spikes=grid.n_spikes,
    )
    print(
        f"notebook C1 padded 60K: T={grid.t_total_fs} fs, N_spikes={grid.n_spikes}, "
        f"N_FFT={spectral['n_fft']}, λ=0 (pulse L1 only)",
        flush=True,
    )

    t_all0 = time.perf_counter()
    t_data0 = time.perf_counter()
    bundle = build_stochastic_padded_frog_dataloaders(
        n_train=int(args.n_train),
        n_val=max(int(args.n_val), 64),
        n_test=max(int(args.n_test), 64),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
        grid=grid,
        canonicalize_mode="tstar",
    )
    data_gen_sec = time.perf_counter() - t_data0
    print(f"data generation + FROG traces: {data_gen_sec:.1f}s", flush=True)

    val_loader = subset_loader(bundle.val_loader, int(args.n_val))
    test_loader = subset_loader(bundle.test_loader, int(args.n_test))
    frog = build_notebook_c1_padded_frog(device, n=n, spectral=spectral)

    train_sec = None
    if ckpt_path.exists() and not args.force:
        print(f"Loading checkpoint: {ckpt_path}", flush=True)
        meta = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(n, device, model_name="multires")
        model.load_state_dict(meta["model_state_dict"])
        train_out = {
            "best_epoch": int(meta.get("best_epoch", -1)),
            "stopped_epoch": int(meta.get("stopped_epoch", -1)),
            "best_val_l1": float(meta.get("best_val_l1", float("nan"))),
            "trace_scale": float(meta.get("trace_scale", float("nan"))),
            "history": None,
        }
    else:
        print(
            f"train={int(args.n_train)} val={int(args.n_val)} test={int(args.n_test)}; "
            f"SNR [{train_snr_range[0]:.0f},{train_snr_range[1]:.0f}] dB; "
            f"val @{args.val_snr_db:.0f} dB",
            flush=True,
        )
        model = build_model(n, device, model_name="multires")
        trace_scale = calibrate_trace_scale(
            build_model(n, device, model_name="multires"),
            frog,
            bundle.train_loader,
            device=device,
        )
        t_tr0 = time.perf_counter()
        train_out = train_multires_noisy_trace_loss_early_stop(
            model,
            bundle.train_loader,
            val_loader,
            frog,
            lam=0.0,
            trace_scale=trace_scale,
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            lr=float(args.lr),
            train_snr_db_range=train_snr_range,
            val_snr_db=float(args.val_snr_db),
        )
        train_sec = time.perf_counter() - t_tr0
        hist = train_out["history"]
        payload = {
            "model_state_dict": model.state_dict(),
            "lam": 0.0,
            "best_epoch": train_out["best_epoch"],
            "stopped_epoch": train_out["stopped_epoch"],
            "best_val_l1": train_out["best_val_l1"],
            "trace_scale": train_out["trace_scale"],
            "train_losses": hist.train_losses,
            "val_l1_pulses": hist.val_l1_pulses,
            "train_snr_db_range": list(train_snr_range),
            "val_snr_db": float(args.val_snr_db),
            "n_train": int(args.n_train),
            "n_val": int(args.n_val),
            "n_test": int(args.n_test),
            "pulse_kind": "stochastic_c1_padded_tstar",
            "n_fft": int(spectral["n_fft"]),
            "t_total_fs": float(grid.t_total_fs),
            "n_spikes": int(grid.n_spikes),
            "wall_time_sec": train_sec,
            "data_generation_sec": data_gen_sec,
        }
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, ckpt_path)
        meta_path.write_text(
            json.dumps(
                {
                    "best_epoch": train_out["best_epoch"],
                    "stopped_epoch": train_out["stopped_epoch"],
                    "best_val_l1": train_out["best_val_l1"],
                    "lam": 0.0,
                    "n_train": int(args.n_train),
                    "pulse_kind": "stochastic_c1_padded_tstar",
                    "n_fft": int(spectral["n_fft"]),
                    "wall_time_sec": train_sec,
                    "data_generation_sec": data_gen_sec,
                    "device": str(device),
                    "cuda_name": (
                        torch.cuda.get_device_name(0) if device.type == "cuda" else None
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"Saved checkpoint: {ckpt_path}  "
            f"(train wall={train_sec:.1f}s, best_ep={train_out['best_epoch']})",
            flush=True,
        )

    sweep_sec = None
    if not args.skip_sweep:
        if sweep_path.exists() and not args.force:
            print(f"Sweep exists: {sweep_path}", flush=True)
        else:
            print(f"\nSNR sweep on test set (n={int(args.n_test)}) …", flush=True)
            model.eval()

            class _PulseModel(torch.nn.Module):
                def __init__(self, net: torch.nn.Module) -> None:
                    super().__init__()
                    self.net = net

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return extract_pulse_prediction(self.net(x))

            t_sw0 = time.perf_counter()
            sweep = run_cnn_snr_sweep(
                _PulseModel(model),
                test_loader,
                SNR_SWEEP_DB,
                experiment_name="stochastic_c1_padded_multires_60k_baseline",
                verbose=True,
            )
            save_cnn_sweep(sweep_path, sweep)
            sweep_sec = time.perf_counter() - t_sw0
            print(f"Saved sweep: {sweep_path}  (wall={sweep_sec:.1f}s)", flush=True)

    total_sec = time.perf_counter() - t_all0
    print(
        f"Done in {total_sec:.0f}s "
        f"(data={data_gen_sec:.0f}s"
        + (f", train={train_sec:.0f}s" if train_sec is not None else "")
        + (f", sweep={sweep_sec:.0f}s" if sweep_sec is not None else "")
        + ")",
        flush=True,
    )


if __name__ == "__main__":
    main()
