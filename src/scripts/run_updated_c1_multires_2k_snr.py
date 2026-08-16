"""Multires 2K (λ=0) + SNR sweep for current stochastic_pulses_generator_NB C1 params."""
from __future__ import annotations

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
    build_stochastic_padded_frog_dataloaders,
    notebook_c1_spectral_fft_params,
)
from evaluate_cnn import run_cnn_snr_sweep, save_cnn_sweep
from frog_reconstruction_model import TraceToPulseMultires, extract_pulse_prediction
from spectral_grid import compute_sigma_t_center
from train import train_trace_to_pulse_early_stopping

# Current free params from stochastic_pulses_generator_NB.ipynb (code cell)
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
SNR_SWEEP_DB = np.arange(-10.0, 31.0, 5.0)

OUT_DIR = SRC / "checkpoints" / "benchmark" / "exp_a_prime_diagnostics"
NAME = "updated_c1_nb_multires_2k_lam0"
CKPT = OUT_DIR / f"{NAME}.pt"
JSON_PATH = OUT_DIR / f"{NAME}.json"
SWEEP_PATH = OUT_DIR / f"{NAME}_snr_sweep.npz"


def main(*, force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if JSON_PATH.exists() and CKPT.exists() and SWEEP_PATH.exists() and not force:
        print(f"[cache] {NAME} already complete: {JSON_PATH}")
        d = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        print(
            f"  best_val={d['best_val_l1']:.4f} ep={d['best_epoch']} "
            f"wall={d['wall_sec']:.1f}s"
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

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

    torch.manual_seed(SEED)
    t0 = time.perf_counter()
    bundle = build_stochastic_padded_frog_dataloaders(
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_test=N_TEST,
        batch_size=BATCH,
        seed=SEED,
        device=device,
        grid=grid,
        canonicalize_mode=CANON,
        spectral=spectral,
        pulse_temporal_fraction=PULSE_TEMPORAL_FRACTION,
        fraction_from_nyquist=FRACTION_FROM_NYQUIST,
    )

    model = TraceToPulseMultires(
        out_dim=2 * N_POINTS, filters_per_branch=(8, 16, 32)
    ).to(device)
    model(torch.zeros(1, 1, N_POINTS, N_POINTS, device=device))

    # λ=0: pulse L1 only (standard Multires 2K baseline protocol)
    result = train_trace_to_pulse_early_stopping(
        model,
        bundle.train_loader,
        bundle.val_loader,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        lr=LR,
        train_snr_db_range=TRAIN_SNR,
        val_snr_db=VAL_SNR_DB,
        verbose=True,
    )

    class _PulseModel(torch.nn.Module):
        def __init__(self, net: torch.nn.Module) -> None:
            super().__init__()
            self.net = net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return extract_pulse_prediction(self.net(x))

    wrap = _PulseModel(model)
    print("\nSNR sweep …", flush=True)
    sweep = run_cnn_snr_sweep(
        wrap,
        bundle.test_loader,
        SNR_SWEEP_DB,
        experiment_name=NAME,
        verbose=True,
    )
    save_cnn_sweep(SWEEP_PATH, sweep)
    wall = time.perf_counter() - t0

    out = {
        "name": NAME,
        "best_epoch": int(result.best_epoch),
        "stopped_epoch": int(result.stopped_epoch),
        "best_val_l1": float(result.best_val_l1),
        "train_losses": list(result.history.train_losses),
        "val_l1_pulses": list(result.history.val_l1_pulses),
        "wall_sec": float(wall),
        "meta": {
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_test": N_TEST,
            "lambda": 0.0,
            "t_total_fs": T_TOTAL,
            "n_spikes": N_SPIKES,
            "coherence_time_fs": COHERENCE_TIME_FS,
            "pulse_temporal_fraction": PULSE_TEMPORAL_FRACTION,
            "t_center_std_fs": sigma_tc,
            "fraction_from_nyquist": FRACTION_FROM_NYQUIST,
            "n_fft": int(spectral["n_fft"]),
            "de_new_actual": float(spectral["de_new_actual"]),
            "canonicalize_mode": CANON,
            "train_snr": list(TRAIN_SNR),
            "val_snr_db": VAL_SNR_DB,
            "seed": SEED,
            "device": str(device),
            "sweep_path": str(SWEEP_PATH),
        },
    }
    torch.save({"model_state_dict": model.state_dict(), "result": out}, CKPT)
    JSON_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"done: valL1={out['best_val_l1']:.4f} ep={out['best_epoch']} "
        f"wall={wall:.1f}s\n  {JSON_PATH}\n  {SWEEP_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force=force)
