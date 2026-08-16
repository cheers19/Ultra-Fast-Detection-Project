"""Multires 2K λ-search on notebook modified-C1 pulses + FROGNetPadded traces.

Uses ``stochastic_pulses_generator_NB.ipynb`` protocol:
  T=53 fs, N_spikes=300, canonicalize_field_tstar, padded FFT (N_FFT≈128).
Does not overwrite legacy Experiment A artifacts.
"""

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

DEFAULT_OUTPUT = "checkpoints/benchmark/stochastic_c1_padded_multires_2k_noisy_trace_lambda.npz"
DEFAULT_META = "checkpoints/benchmark/stochastic_c1_padded_multires_2k_noisy_trace_lambda_meta.json"
DEFAULT_OPT_SWEEP = (
    "checkpoints/benchmark/stochastic_c1_padded_multires_2k_noisy_trace_lambda_opt_sweep.npz"
)
DEFAULT_BASE_SWEEP = (
    "checkpoints/benchmark/stochastic_c1_padded_multires_2k_noisy_trace_lambda_baseline_sweep.npz"
)
DEFAULT_CKPT_DIR = "checkpoints/benchmark/stochastic_c1_padded_multires_2k_noisy_trace_lambda"
DEFAULT_OPT_CKPT = (
    "checkpoints/benchmark/stochastic_c1_padded_multires_2k_noisy_trace_lambda_opt.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--train-snr-min", type=float, default=0.0)
    parser.add_argument("--train-snr-max", type=float, default=30.0)
    parser.add_argument("--val-snr-db", type=float, default=15.0)
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=3.0)
    parser.add_argument("--lambda-steps", type=int, default=5)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--meta-json", default=DEFAULT_META)
    parser.add_argument("--sweep-output", default=DEFAULT_OPT_SWEEP)
    parser.add_argument("--baseline-sweep-output", default=DEFAULT_BASE_SWEEP)
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--opt-checkpoint", default=DEFAULT_OPT_CKPT)
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        device = torch.device("cpu")
        print("device: cpu", flush=True)

    n = 64
    train_snr_range = (float(args.train_snr_min), float(args.train_snr_max))
    lambda_grid = np.linspace(
        float(args.lambda_min), float(args.lambda_max), int(args.lambda_steps)
    )
    n_lam = len(lambda_grid)

    out_npz = _SRC / args.output
    out_meta = _SRC / args.meta_json
    sweep_out = _SRC / args.sweep_output
    baseline_sweep_out = _SRC / args.baseline_sweep_output
    ckpt_dir = _SRC / args.ckpt_dir
    opt_ckpt = _SRC / args.opt_checkpoint

    if out_npz.exists() and not args.force and not args.resume:
        d = np.load(out_npz)
        print(f"Results exist: {out_npz}", flush=True)
        print(
            f"  lambda_opt={float(d['lambda_opt']):.4f}  "
            f"best_val={float(d['best_val_at_opt']):.5f}",
            flush=True,
        )
        return

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    grid = stochastic_pulse_config_notebook_modified_c1(n=n)
    spectral = notebook_c1_spectral_fft_params(
        n=n,
        t_total_fs=grid.t_total_fs,
        n_spikes=grid.n_spikes,
    )
    print(
        f"notebook C1 padded: T={grid.t_total_fs} fs, N_spikes={grid.n_spikes}, "
        f"sigma_spike={grid.coherence_time_fs:.4f} fs, "
        f"sigma_t_c={grid.t_center_std_fs:.4f} fs, "
        f"N_FFT={spectral['n_fft']}, dE_actual={spectral['de_new_actual']:.6f} eV",
        flush=True,
    )

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

    cal_model = build_model(n, device, model_name="multires")
    trace_scale = calibrate_trace_scale(
        cal_model, frog, bundle.train_loader, device=device
    )
    del cal_model
    print(f"trace_scale = {trace_scale:.4f}", flush=True)
    print(
        f"train={int(args.n_train)} / val={int(args.n_val)} / test={int(args.n_test)}; "
        f"SNR train [{train_snr_range[0]:.0f},{train_snr_range[1]:.0f}] dB; "
        f"val @{args.val_snr_db:.0f} dB",
        flush=True,
    )

    best_val_l1 = np.full(n_lam, np.nan)
    best_epochs = np.full(n_lam, -1, dtype=np.int32)
    stopped_epochs = np.full(n_lam, -1, dtype=np.int32)
    wall_times_sec = np.full(n_lam, np.nan)
    run_log: list[dict] = []
    t0 = time.perf_counter()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for li, lam in enumerate(lambda_grid):
        ckpt_path = ckpt_dir / f"lam_{float(lam):.4f}.pt"
        if args.resume and ckpt_path.exists() and not args.force:
            meta = torch.load(ckpt_path, map_location=device, weights_only=False)
            best_val_l1[li] = float(meta["best_val_l1"])
            best_epochs[li] = int(meta["best_epoch"])
            stopped_epochs[li] = int(meta["stopped_epoch"])
            wall_times_sec[li] = float(meta.get("wall_time_sec", np.nan))
            run_log.append(meta.get("log_entry", meta))
            print(
                f"lam={lam:.4f} (cached)  best_ep={best_epochs[li]}  "
                f"best_val={best_val_l1[li]:.5f}  "
                f"wall={wall_times_sec[li]:.1f}s",
                flush=True,
            )
            continue

        print(f"\n--- lam = {lam:.4f} ({li + 1}/{n_lam}) ---", flush=True)
        t_lam0 = time.perf_counter()
        model = build_model(n, device, model_name="multires")
        result = train_multires_noisy_trace_loss_early_stop(
            model,
            bundle.train_loader,
            val_loader,
            frog,
            lam=float(lam),
            trace_scale=trace_scale,
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            lr=float(args.lr),
            train_snr_db_range=train_snr_range,
            val_snr_db=float(args.val_snr_db),
        )
        wall_lam = time.perf_counter() - t_lam0
        best_val_l1[li] = result["best_val_l1"]
        best_epochs[li] = result["best_epoch"]
        stopped_epochs[li] = result["stopped_epoch"]
        wall_times_sec[li] = wall_lam

        log_entry = {
            "lam": float(lam),
            "best_epoch": result["best_epoch"],
            "stopped_epoch": result["stopped_epoch"],
            "best_val_l1": result["best_val_l1"],
            "wall_time_sec": wall_lam,
            "trace_scale": trace_scale,
            "train_losses": result["history"].train_losses,
            "val_l1_pulses": result["history"].val_l1_pulses,
            "train_snr_db_range": list(train_snr_range),
            "val_snr_db": float(args.val_snr_db),
            "n_val": int(args.n_val),
            "n_train": int(args.n_train),
            "n_test": int(args.n_test),
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "early_stop": result["stopped_epoch"] < int(args.max_epochs),
            "pulse_kind": "stochastic_c1_padded_tstar",
            "n_fft": int(spectral["n_fft"]),
            "t_total_fs": float(grid.t_total_fs),
            "n_spikes": int(grid.n_spikes),
            "coherence_time_fs": float(grid.coherence_time_fs),
            "t_center_std_fs": float(grid.t_center_std_fs),
        }
        run_log.append(log_entry)
        torch.save(
            {"model_state_dict": model.state_dict(), **log_entry, "log_entry": log_entry},
            ckpt_path,
        )
        print(
            f"  lam={lam:.4f}  best_epoch={result['best_epoch']}  "
            f"best_val_L1={result['best_val_l1']:.5f}  wall={wall_lam:.1f}s",
            flush=True,
        )

    train_wall_sec = time.perf_counter() - t0
    print(f"\nAll lambda checkpoints saved under {ckpt_dir}", flush=True)
    print(f"λ-grid wall time: {train_wall_sec:.1f}s", flush=True)

    run_log = []
    best_val_l1 = np.full(n_lam, np.nan)
    best_epochs = np.full(n_lam, -1, dtype=np.int32)
    stopped_epochs = np.full(n_lam, -1, dtype=np.int32)
    wall_times_sec = np.full(n_lam, np.nan)
    for li, lam in enumerate(lambda_grid):
        ckpt_path = ckpt_dir / f"lam_{float(lam):.4f}.pt"
        meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        entry = meta.get(
            "log_entry", {k: v for k, v in meta.items() if k != "model_state_dict"}
        )
        run_log.append(entry)
        best_val_l1[li] = float(meta["best_val_l1"])
        best_epochs[li] = int(meta["best_epoch"])
        stopped_epochs[li] = int(meta["stopped_epoch"])
        wall_times_sec[li] = float(meta.get("wall_time_sec", np.nan))

    opt_idx = int(np.nanargmin(best_val_l1))
    baseline_idx = int(np.argmin(np.abs(lambda_grid - 0.0)))
    lambda_opt = float(lambda_grid[opt_idx])
    lambda_baseline = float(lambda_grid[baseline_idx])
    print(
        f"\nlambda* = {lambda_opt:.4f}  (best val L1 = {best_val_l1[opt_idx]:.5f})",
        flush=True,
    )
    print(
        f"baseline λ = {lambda_baseline:.4f}  "
        f"(best val L1 = {best_val_l1[baseline_idx]:.5f})",
        flush=True,
    )

    timing_summary = {
        "data_generation_sec": data_gen_sec,
        "lambda_grid_train_sec": float(np.nansum(wall_times_sec)),
        "per_lambda_wall_time_sec": {
            f"{float(lam):.4f}": float(wall_times_sec[i])
            for i, lam in enumerate(lambda_grid)
        },
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(
        json.dumps({"runs": run_log, "timing": timing_summary}, indent=2),
        encoding="utf-8",
    )

    np.savez(
        out_npz,
        lambda_grid=lambda_grid,
        best_val_l1=best_val_l1,
        best_epochs=best_epochs,
        stopped_epochs=stopped_epochs,
        wall_times_sec=wall_times_sec,
        lambda_opt=lambda_opt,
        lambda_opt_idx=opt_idx,
        lambda_baseline=lambda_baseline,
        lambda_baseline_idx=baseline_idx,
        best_val_at_opt=float(best_val_l1[opt_idx]),
        best_val_at_baseline=float(best_val_l1[baseline_idx]),
        trace_scale=trace_scale,
        n_train=int(args.n_train),
        n_val=int(args.n_val),
        n_test=int(args.n_test),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        train_snr_min=float(train_snr_range[0]),
        train_snr_max=float(train_snr_range[1]),
        val_snr_db=float(args.val_snr_db),
        seed=int(args.seed),
        pulse_kind="stochastic_c1_padded_tstar",
        n_fft=int(spectral["n_fft"]),
        t_total_fs=float(grid.t_total_fs),
        n_spikes=int(grid.n_spikes),
        data_generation_sec=data_gen_sec,
        lambda_grid_train_sec=float(np.nansum(wall_times_sec)),
    )

    opt_src = ckpt_dir / f"lam_{lambda_opt:.4f}.pt"
    if opt_src.exists():
        opt_data = torch.load(opt_src, map_location=device, weights_only=False)
        opt_data["lambda_opt"] = lambda_opt
        torch.save(opt_data, opt_ckpt)

    if not args.skip_sweep:
        class _PulseModel(torch.nn.Module):
            def __init__(self, net: torch.nn.Module) -> None:
                super().__init__()
                self.net = net

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return extract_pulse_prediction(self.net(x))

        t_sw0 = time.perf_counter()
        print(f"\nSNR sweep (λ*={lambda_opt:.4f}) …", flush=True)
        model = build_model(n, device, model_name="multires")
        model.load_state_dict(
            torch.load(opt_ckpt, map_location=device, weights_only=False)[
                "model_state_dict"
            ]
        )
        model.eval()
        sweep = run_cnn_snr_sweep(
            _PulseModel(model),
            test_loader,
            SNR_SWEEP_DB,
            experiment_name="stochastic_c1_padded_multires_2k_trace_lambda_opt",
            verbose=True,
        )
        save_cnn_sweep(sweep_out, sweep)

        baseline_src = ckpt_dir / f"lam_{lambda_baseline:.4f}.pt"
        print(
            f"\nSNR sweep (baseline λ={lambda_baseline:.4f}, no trace loss) …",
            flush=True,
        )
        model = build_model(n, device, model_name="multires")
        model.load_state_dict(
            torch.load(baseline_src, map_location=device, weights_only=False)[
                "model_state_dict"
            ]
        )
        model.eval()
        sweep_b = run_cnn_snr_sweep(
            _PulseModel(model),
            test_loader,
            SNR_SWEEP_DB,
            experiment_name="stochastic_c1_padded_multires_2k_baseline",
            verbose=True,
        )
        save_cnn_sweep(baseline_sweep_out, sweep_b)
        sweep_sec = time.perf_counter() - t_sw0
        timing_summary["snr_sweeps_sec"] = sweep_sec
        out_meta.write_text(
            json.dumps({"runs": run_log, "timing": timing_summary}, indent=2),
            encoding="utf-8",
        )
        print(f"SNR sweeps wall time: {sweep_sec:.1f}s", flush=True)

    total_sec = time.perf_counter() - t0 + data_gen_sec
    print(f"Done. Total wall (data+train+sweeps approx): {total_sec:.1f}s", flush=True)


if __name__ == "__main__":
    main()
