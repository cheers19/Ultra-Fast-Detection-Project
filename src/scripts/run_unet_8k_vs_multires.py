"""Train regular U-Net on 8K pulses and compare to Multires 6K baseline."""

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
    train_unet_early_stop,
)
from train import load_checkpoint
from trace_noise import add_trace_noise_awgn as add_trace_noise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
N = 64
N_TRAIN = 8_000
N_VAL = 512
N_TEST = 512
BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 15
VAL_SNR_DB = 15.0
SNR_SWEEP_DB = np.arange(-10.0, 31.0, 5.0)
BENCH_DIR = Path("checkpoints/benchmark")
UNET_CKPT = BENCH_DIR / "unet_8k_es.pt"
UNET_SWEEP = BENCH_DIR / "sweep_unet_8k_es.npz"
MULTIRES_SWEEP = BENCH_DIR / "sweep_multires_6k_es.npz"


def main() -> None:
    print("DEVICE:", DEVICE)
    if UNET_CKPT.is_file():
        model, meta = load_checkpoint(UNET_CKPT, DEVICE)
        print(f"Loaded U-Net 8K: {UNET_CKPT} (best epoch {meta.get('best_epoch', '?')})")
    else:
        print(f"Training U-Net 8K (n_train={N_TRAIN}) …")
        model, res = train_unet_early_stop(
            n_train=N_TRAIN,
            checkpoint_path=UNET_CKPT,
            experiment_name="unet_8k_es",
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

    if not UNET_SWEEP.is_file():
        sweep_device = torch.device("cpu")
        sweep_batch = 8
        print(f"U-Net sweep on CPU (batch={sweep_batch})")
        bundle = build_frog_dataloaders(
            n_train=2048,
            n_val=N_VAL,
            n_test=N_TEST,
            batch_size=sweep_batch,
            seed=SEED,
            device=sweep_device,
            grid=PulseGridConfig(n=N),
        )
        frog = FROGNet(num_delay_steps=N).to(sweep_device)
        model = model.to(sweep_device)
        sweep = run_multires_benchmark_sweep(
            model,
            bundle.test_loader,
            SNR_SWEEP_DB,
            label="U-Net 8K",
            frog=frog,
            device=sweep_device,
            add_noise_fn=add_trace_noise,
        )
        save_benchmark_sweep(UNET_SWEEP, sweep)
        print(f"Saved {UNET_SWEEP}")
    else:
        print(f"Loaded sweep: {UNET_SWEEP}")

    if not MULTIRES_SWEEP.is_file():
        raise FileNotFoundError(f"Missing {MULTIRES_SWEEP}")

    sw_u = load_benchmark_sweep(UNET_SWEEP)
    sw_m = load_benchmark_sweep(MULTIRES_SWEEP)
    sw_m.pulse_l1.label = "Multires 6K"
    sw_u.pulse_l1.label = "U-Net 8K"

    print("\n=== U-Net 8K vs Multires 6K ===")
    print(f"{'SNR':>5} | {'L1 UNet':>8} | {'L1 Mult':>8} | {'Sim UNet':>9} | {'Sim Mult':>9}")
    for i, snr in enumerate(SNR_SWEEP_DB):
        print(
            f"{snr:5.0f} | {sw_u.pulse_l1.mean[i]:8.4f} | {sw_m.pulse_l1.mean[i]:8.4f} | "
            f"{sw_u.similarity_error.mean[i]:9.5f} | {sw_m.similarity_error.mean[i]:9.5f}"
        )


if __name__ == "__main__":
    main()
