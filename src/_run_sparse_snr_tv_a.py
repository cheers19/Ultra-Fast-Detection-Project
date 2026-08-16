"""Train Model A sparse train+val SNR {0,15,30}: λ=3 and λ=0; then test SNR sweeps."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_LOG = Path(__file__).resolve().parent / "_sparse_snr_tv_a_log.txt"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_log_f = open(_LOG, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_f)
sys.stderr = _Tee(sys.__stderr__, _log_f)

import data_c_amb_loss_diagnostics as diag

OUT = Path("checkpoints/benchmark/data_c_amb_loss_diagnostics")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPARSE_SNR = [0.0, 15.0, 30.0]
SNR_SWEEP = np.arange(-10.0, 31.0, 5.0)

print("device", DEVICE)

for tag, lam in [
    ("model_a_raw_sparse_snr_tv_lam3", 3.0),
    ("model_a_raw_sparse_snr_tv_lam0", 0.0),
]:
    hist_path = OUT / f"{tag}_history.npz"
    if hist_path.exists():
        print("skip train", tag)
    else:
        print(f"Training {tag} λ={lam} ...", flush=True)
        res = diag.train_data_c_amb_diagnostics(
            pulse_loss_mode="raw",
            lam=lam,
            n_train=2048,
            n_val=200,
            n_test=512,
            seed=0,
            max_epochs=200,
            patience=25,
            device=DEVICE,
            verbose=True,
            train_snr_db_values=SPARSE_SNR,
            val_snr_db_values=SPARSE_SNR,
            val_snr_db_range=(0.0, 30.0),
            trace_loss_ref="clean",
        )
        diag.save_run_artifacts(res, OUT, tag)
        print("saved", tag, "best_epoch", res["best_epoch"], "best", res["best_score"])

loader = diag.build_data_c_test_loader(
    n_train=2048, n_val=200, n_test=512, seed=0, device=DEVICE
)
for tag in ("model_a_raw_sparse_snr_tv_lam3", "model_a_raw_sparse_snr_tv_lam0"):
    sweep = OUT / f"{tag}_test_snr_sweep.npz"
    if sweep.exists():
        print("skip sweep", tag)
        continue
    print("SNR sweep", tag, flush=True)
    diag.run_and_save_test_snr_sweep(
        OUT / f"{tag}_model.pt",
        sweep,
        test_loader=loader,
        snr_sweep_db=SNR_SWEEP,
        device=DEVICE,
        experiment_name=f"data_c_{tag}",
        verbose=True,
    )
print("DONE")
