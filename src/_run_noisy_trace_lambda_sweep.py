"""Train Model A with noisy trace loss; sweep λ in {1..5}."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import data_c_amb_loss_diagnostics as diag

OUT = Path("checkpoints/benchmark/data_c_amb_loss_diagnostics")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAM_GRID = [1.0, 2.0, 3.0, 4.0, 5.0]
TAG_NEW = "model_a_raw_pulse_loss_noisy_trace"

print("device", DEVICE)
res = diag.run_model_a_noisy_trace_lambda_sweep(
    LAM_GRID,
    OUT,
    force=False,
    n_train=2048,
    n_val=200,
    n_test=512,
    seed=0,
    max_epochs=200,
    patience=25,
    device=DEVICE,
    verbose=True,
    winner_tag=TAG_NEW,
)

loader = diag.build_data_c_test_loader(
    n_train=2048, n_val=200, n_test=512, seed=0, device=DEVICE
)
diag.run_and_save_test_snr_sweep(
    OUT / f"{TAG_NEW}_model.pt",
    OUT / f"{TAG_NEW}_test_snr_sweep.npz",
    test_loader=loader,
    snr_sweep_db=np.arange(-10.0, 31.0, 5.0),
    device=DEVICE,
    experiment_name="data_c_model_a_noisy_trace",
    verbose=True,
)
print("DONE", res["best_lam"], res["best_score"])
