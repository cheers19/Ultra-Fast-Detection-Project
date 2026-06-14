"""Train Multires 6K with SNR curriculum and run benchmark sweep."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataset_utils import PulseGridConfig, build_frog_dataloaders
from frognet import FROGNet
from model_comparison_benchmark import (
    load_benchmark_sweep,
    run_multires_benchmark_sweep,
    save_benchmark_sweep,
    train_multires_curriculum_early_stop,
)
from train import load_checkpoint
from trace_noise import add_trace_noise_awgn as add_trace_noise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
N = 64
N_VAL = 512
N_TEST = 512
BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 15
VAL_SNR_DB = 15.0
SNR_SWEEP_DB = np.arange(-10.0, 31.0, 5.0)
BENCH_DIR = Path("checkpoints/benchmark")
CKPT = BENCH_DIR / "multires_6k_curriculum_es.pt"
SWEEP = BENCH_DIR / "sweep_multires_6k_curriculum_es.npz"


def main() -> None:
    print("DEVICE:", DEVICE)
    if CKPT.is_file():
        model, meta = load_checkpoint(CKPT, DEVICE)
        print(f"Loaded curriculum model: {CKPT} (best epoch {meta.get('best_epoch', '?')})")
    else:
        print("Training Multires 6K with SNR curriculum …")
        model, res = train_multires_curriculum_early_stop(
            n_train=6_000,
            checkpoint_path=CKPT,
            experiment_name="multires_6k_curriculum_es",
            device=DEVICE,
            seed=SEED,
            n_val=N_VAL,
            n_test=N_TEST,
            batch_size=BATCH_SIZE,
            lr=LR,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            val_snr_db=VAL_SNR_DB,
            n=N,
            add_noise_fn=add_trace_noise,
            verbose=True,
        )
        print(f"Done: best epoch {res.best_epoch}, val L1={res.best_val_l1:.5f}")

    if SWEEP.is_file():
        print(f"Loaded sweep: {SWEEP}")
        return

    bundle = build_frog_dataloaders(
        n_train=2048,
        n_val=N_VAL,
        n_test=N_TEST,
        batch_size=BATCH_SIZE,
        seed=SEED,
        device=DEVICE,
        grid=PulseGridConfig(n=N),
    )
    frog = FROGNet(num_delay_steps=N).to(DEVICE)
    sweep_cur = run_multires_benchmark_sweep(
        model,
        bundle.test_loader,
        SNR_SWEEP_DB,
        label="Multires 6K curriculum",
        frog=frog,
        device=DEVICE,
        add_noise_fn=add_trace_noise,
    )
    save_benchmark_sweep(SWEEP, sweep_cur)
    print(f"Saved sweep: {SWEEP}")


if __name__ == "__main__":
    main()
