"""Multires 2K physics (trace λ-search) for current stochastic NB C1 params."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from data_generation import StochasticPulseConfig
from dataset_utils import (
    build_notebook_c1_padded_frog,
    build_stochastic_padded_frog_dataloaders,
    notebook_c1_spectral_fft_params,
)
from evaluate_cnn import run_cnn_snr_sweep, save_cnn_sweep
from frog_reconstruction_model import extract_pulse_prediction
from spectral_grid import compute_sigma_t_center
from train import build_model

_STOCH = SRC / "scripts" / "run_multires_2k_stochastic_noisy_trace_lambda.py"
_spec = importlib.util.spec_from_file_location("_stoch_lam", _STOCH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

subset_loader = _mod.subset_loader
train_multires_noisy_trace_loss_early_stop = _mod.train_multires_noisy_trace_loss_early_stop
calibrate_trace_scale = _mod.calibrate_trace_scale
SNR_SWEEP_DB = _mod.SNR_SWEEP_DB

# Current free params from stochastic_pulses_generator_NB.ipynb
T_TOTAL = 53.0
N_POINTS = 64
N_SPIKES = 400
COHERENCE_TIME_FS = 0.025 * T_TOTAL
PULSE_TEMPORAL_FRACTION = 0.34
FRACTION_FROM_NYQUIST = 0.52
CANON = "tstar"

N_TRAIN = 2048
N_VAL = 200
N_TEST = 512
BATCH = 64
SEED = 0
LR = 1e-3
MAX_EPOCHS = 200
PATIENCE = 25
TRAIN_SNR = (0.0, 30.0)
VAL_SNR_DB = 15.0
LAMBDA_GRID = np.linspace(0.0, 3.0, 5)

OUT_DIR = SRC / "checkpoints" / "benchmark" / "exp_a_prime_diagnostics"
TAG = "updated_c1_nb_multires_2k_physics"
CKPT_DIR = OUT_DIR / f"{TAG}_lam_ckpts"
META_JSON = OUT_DIR / f"{TAG}_meta.json"
NPZ = OUT_DIR / f"{TAG}.npz"
OPT_CKPT = OUT_DIR / f"{TAG}_opt.pt"
OPT_SWEEP = OUT_DIR / f"{TAG}_opt_snr_sweep.npz"
BASE_SWEEP = OUT_DIR / f"{TAG}_baseline_snr_sweep.npz"


class _PulseModel(torch.nn.Module):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return extract_pulse_prediction(self.net(x))


def main(*, force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if NPZ.exists() and META_JSON.exists() and OPT_SWEEP.exists() and not force:
        d = np.load(NPZ)
        print(
            f"[cache] {TAG}: lambda*={float(d['lambda_opt']):.4f} "
            f"best_val={float(d['best_val_at_opt']):.4f}"
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    sigma_tc = float(
        compute_sigma_t_center(T_TOTAL, N_SPIKES, PULSE_TEMPORAL_FRACTION)
    )
    grid = StochasticPulseConfig(
        n=N_POINTS,
        t_total_fs=T_TOTAL,
        n_spikes=N_SPIKES,
        coherence_time_fs=COHERENCE_TIME_FS,
        t_center_std_fs=sigma_tc,
        delta_energy_ev_range=6.5 * 0.0001,
    )
    spectral = notebook_c1_spectral_fft_params(
        n=N_POINTS,
        t_total_fs=T_TOTAL,
        n_spikes=N_SPIKES,
        pulse_temporal_fraction=PULSE_TEMPORAL_FRACTION,
        fraction_from_nyquist=FRACTION_FROM_NYQUIST,
    )
    print(
        f"C1: T={T_TOTAL} spikes={N_SPIKES} σ_spike={COHERENCE_TIME_FS:.4f} "
        f"f_pulse={PULSE_TEMPORAL_FRACTION} σ_tc={sigma_tc:.4f} canon={CANON}",
        flush=True,
    )
    print(
        f"Spectral: κ={FRACTION_FROM_NYQUIST} N_FFT={spectral['n_fft']} "
        f"dE={spectral['de_new_actual']:.6f} eV",
        flush=True,
    )
    print(f"λ grid: {LAMBDA_GRID}", flush=True)

    bundle = build_stochastic_padded_frog_dataloaders(
        n_train=N_TRAIN,
        n_val=max(N_VAL, 64),
        n_test=max(N_TEST, 64),
        batch_size=BATCH,
        seed=SEED,
        device=device,
        grid=grid,
        canonicalize_mode=CANON,
        spectral=spectral,
        pulse_temporal_fraction=PULSE_TEMPORAL_FRACTION,
        fraction_from_nyquist=FRACTION_FROM_NYQUIST,
    )
    val_loader = subset_loader(bundle.val_loader, N_VAL)
    test_loader = subset_loader(bundle.test_loader, N_TEST)
    frog = build_notebook_c1_padded_frog(device, n=N_POINTS, spectral=spectral)

    cal_model = build_model(N_POINTS, device, model_name="multires")
    trace_scale = calibrate_trace_scale(
        cal_model, frog, bundle.train_loader, device=device
    )
    del cal_model
    print(f"trace_scale = {trace_scale:.4f}", flush=True)

    n_lam = len(LAMBDA_GRID)
    best_val_l1 = np.full(n_lam, np.nan)
    best_epochs = np.full(n_lam, -1, dtype=np.int32)
    stopped_epochs = np.full(n_lam, -1, dtype=np.int32)
    wall_times_sec = np.full(n_lam, np.nan)
    run_log: list[dict] = []

    for li, lam in enumerate(LAMBDA_GRID):
        ckpt_path = CKPT_DIR / f"lam_{float(lam):.4f}.pt"
        if ckpt_path.exists() and not force:
            meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            best_val_l1[li] = float(meta["best_val_l1"])
            best_epochs[li] = int(meta["best_epoch"])
            stopped_epochs[li] = int(meta["stopped_epoch"])
            wall_times_sec[li] = float(meta.get("wall_time_sec", np.nan))
            run_log.append(meta.get("log_entry", meta))
            print(
                f"[cache] lam={lam:.4f}  best_val={best_val_l1[li]:.5f}  "
                f"ep={best_epochs[li]}",
                flush=True,
            )
            continue

        print(f"\n--- lam = {lam:.4f} ({li + 1}/{n_lam}) ---", flush=True)
        t_lam0 = time.perf_counter()
        model = build_model(N_POINTS, device, model_name="multires")
        result = train_multires_noisy_trace_loss_early_stop(
            model,
            bundle.train_loader,
            val_loader,
            frog,
            lam=float(lam),
            trace_scale=trace_scale,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            lr=LR,
            train_snr_db_range=TRAIN_SNR,
            val_snr_db=VAL_SNR_DB,
        )
        wall_lam = time.perf_counter() - t_lam0
        best_val_l1[li] = float(result["best_val_l1"])
        best_epochs[li] = int(result["best_epoch"])
        stopped_epochs[li] = int(result["stopped_epoch"])
        wall_times_sec[li] = wall_lam
        log_entry = {
            "lam": float(lam),
            "best_epoch": int(result["best_epoch"]),
            "stopped_epoch": int(result["stopped_epoch"]),
            "best_val_l1": float(result["best_val_l1"]),
            "wall_time_sec": wall_lam,
            "trace_scale": float(trace_scale),
            "train_losses": list(result["history"].train_losses),
            "val_l1_pulses": list(result["history"].val_l1_pulses),
            "train_snr_db_range": list(TRAIN_SNR),
            "val_snr_db": VAL_SNR_DB,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_test": N_TEST,
            "n_spikes": N_SPIKES,
            "coherence_time_fs": COHERENCE_TIME_FS,
            "pulse_temporal_fraction": PULSE_TEMPORAL_FRACTION,
            "fraction_from_nyquist": FRACTION_FROM_NYQUIST,
            "t_center_std_fs": sigma_tc,
            "n_fft": int(spectral["n_fft"]),
            "de_new_actual": float(spectral["de_new_actual"]),
            "canonicalize_mode": CANON,
        }
        run_log.append(log_entry)
        torch.save(
            {"model_state_dict": model.state_dict(), **log_entry, "log_entry": log_entry},
            ckpt_path,
        )
        print(
            f"  lam={lam:.4f}  best_ep={result['best_epoch']}  "
            f"best_val={result['best_val_l1']:.5f}  wall={wall_lam:.1f}s",
            flush=True,
        )

    # Reload log consistently
    run_log = []
    for li, lam in enumerate(LAMBDA_GRID):
        meta = torch.load(
            CKPT_DIR / f"lam_{float(lam):.4f}.pt",
            map_location="cpu",
            weights_only=False,
        )
        entry = meta.get(
            "log_entry", {k: v for k, v in meta.items() if k != "model_state_dict"}
        )
        run_log.append(entry)
        best_val_l1[li] = float(meta["best_val_l1"])
        best_epochs[li] = int(meta["best_epoch"])
        stopped_epochs[li] = int(meta["stopped_epoch"])
        wall_times_sec[li] = float(meta.get("wall_time_sec", np.nan))

    opt_idx = int(np.nanargmin(best_val_l1))
    base_idx = int(np.argmin(np.abs(LAMBDA_GRID - 0.0)))
    lambda_opt = float(LAMBDA_GRID[opt_idx])
    lambda_base = float(LAMBDA_GRID[base_idx])
    print(
        f"\nlambda* = {lambda_opt:.4f}  (best val L1 = {best_val_l1[opt_idx]:.5f})",
        flush=True,
    )
    print(
        f"baseline λ = {lambda_base:.4f}  (best val L1 = {best_val_l1[base_idx]:.5f})",
        flush=True,
    )

    np.savez(
        NPZ,
        lambda_grid=LAMBDA_GRID,
        best_val_l1=best_val_l1,
        best_epochs=best_epochs,
        stopped_epochs=stopped_epochs,
        wall_times_sec=wall_times_sec,
        lambda_opt=lambda_opt,
        lambda_opt_idx=opt_idx,
        lambda_baseline=lambda_base,
        best_val_at_opt=float(best_val_l1[opt_idx]),
        best_val_at_baseline=float(best_val_l1[base_idx]),
        trace_scale=float(trace_scale),
        n_fft=int(spectral["n_fft"]),
        n_spikes=N_SPIKES,
        fraction_from_nyquist=FRACTION_FROM_NYQUIST,
        pulse_temporal_fraction=PULSE_TEMPORAL_FRACTION,
    )
    META_JSON.write_text(
        json.dumps(
            {
                "runs": run_log,
                "lambda_opt": lambda_opt,
                "lambda_baseline": lambda_base,
                "pulse_params": {
                    "t_total_fs": T_TOTAL,
                    "n_spikes": N_SPIKES,
                    "coherence_time_fs": COHERENCE_TIME_FS,
                    "pulse_temporal_fraction": PULSE_TEMPORAL_FRACTION,
                    "fraction_from_nyquist": FRACTION_FROM_NYQUIST,
                    "t_center_std_fs": sigma_tc,
                    "n_fft": int(spectral["n_fft"]),
                    "de_new_actual": float(spectral["de_new_actual"]),
                    "canonicalize_mode": CANON,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    opt_src = CKPT_DIR / f"lam_{lambda_opt:.4f}.pt"
    opt_data = torch.load(opt_src, map_location=device, weights_only=False)
    opt_data["lambda_opt"] = lambda_opt
    torch.save(opt_data, OPT_CKPT)

    # SNR sweeps: λ* and λ=0
    print(f"\nSNR sweep (λ*={lambda_opt:.4f}) …", flush=True)
    model = build_model(N_POINTS, device, model_name="multires")
    model.load_state_dict(opt_data["model_state_dict"])
    model.eval()
    sweep = run_cnn_snr_sweep(
        _PulseModel(model),
        test_loader,
        SNR_SWEEP_DB,
        experiment_name=f"{TAG}_opt",
        verbose=True,
    )
    save_cnn_sweep(OPT_SWEEP, sweep)

    print(f"\nSNR sweep (baseline λ={lambda_base:.4f}) …", flush=True)
    base_data = torch.load(
        CKPT_DIR / f"lam_{lambda_base:.4f}.pt",
        map_location=device,
        weights_only=False,
    )
    model = build_model(N_POINTS, device, model_name="multires")
    model.load_state_dict(base_data["model_state_dict"])
    model.eval()
    sweep_b = run_cnn_snr_sweep(
        _PulseModel(model),
        test_loader,
        SNR_SWEEP_DB,
        experiment_name=f"{TAG}_baseline",
        verbose=True,
    )
    save_cnn_sweep(BASE_SWEEP, sweep_b)
    print(f"Done. Artifacts under {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
