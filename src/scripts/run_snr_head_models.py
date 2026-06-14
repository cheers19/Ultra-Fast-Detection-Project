"""Train Multires / U-Net 6K with auxiliary SNR head and run benchmark sweeps."""

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
    DEFAULT_SNR_HEAD_LOSS_WEIGHT,
    load_benchmark_sweep,
    run_multires_benchmark_sweep,
    save_benchmark_sweep,
    train_multires_snr_early_stop,
    train_unet_snr_early_stop,
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

SPECS = (
    {
        "name": "multires_6k_snr",
        "train_fn": train_multires_snr_early_stop,
        "ckpt": BENCH_DIR / "multires_6k_snr_es.pt",
        "sweep": BENCH_DIR / "sweep_multires_6k_snr_es.npz",
        "sweep_label": "Multires 6K + SNR head",
    },
    {
        "name": "unet_6k_snr",
        "train_fn": train_unet_snr_early_stop,
        "ckpt": BENCH_DIR / "unet_6k_snr_es.pt",
        "sweep": BENCH_DIR / "sweep_unet_6k_snr_es.npz",
        "sweep_label": "U-Net 6K + SNR head",
    },
)


def _train_if_needed(spec: dict) -> torch.nn.Module:
    ckpt: Path = spec["ckpt"]
    if ckpt.is_file():
        model, meta = load_checkpoint(ckpt, DEVICE)
        print(
            f"Loaded {spec['name']}: {ckpt} "
            f"(best epoch {meta.get('best_epoch', '?')})"
        )
        return model
    print(f"Training {spec['name']} …")
    model, res = spec["train_fn"](
        n_train=6_000,
        checkpoint_path=ckpt,
        experiment_name=f"{spec['name']}_es",
        device=DEVICE,
        seed=SEED,
        n_val=N_VAL,
        n_test=N_TEST,
        batch_size=BATCH_SIZE,
        lr=LR,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        val_snr_db=VAL_SNR_DB,
        snr_loss_weight=DEFAULT_SNR_HEAD_LOSS_WEIGHT,
        n=N,
        add_noise_fn=add_trace_noise,
        verbose=True,
    )
    print(f"  best epoch {res.best_epoch}, val L1={res.best_val_l1:.5f}")
    return model


def _sweep_if_needed(spec: dict, model: torch.nn.Module) -> None:
    sweep_path: Path = spec["sweep"]
    if sweep_path.is_file():
        print(f"Loaded sweep: {sweep_path}")
        return
    sweep_device = DEVICE
    sweep_batch = BATCH_SIZE
    if spec["name"].startswith("unet"):
        sweep_device = torch.device("cpu")
        sweep_batch = 8
        print(f"Sweep {spec['name']} on CPU (batch={sweep_batch}) to avoid GPU OOM")
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
    if sweep_device != DEVICE:
        model = model.to(sweep_device)
    sweep = run_multires_benchmark_sweep(
        model,
        bundle.test_loader,
        SNR_SWEEP_DB,
        label=spec["sweep_label"],
        frog=frog,
        device=sweep_device,
        add_noise_fn=add_trace_noise,
    )
    save_benchmark_sweep(sweep_path, sweep)
    print(f"Saved sweep: {sweep_path}")
    if sweep_device != DEVICE:
        model.to(DEVICE)


def _compare(spec: dict, baseline_sweep_path: Path, baseline_label: str) -> None:
    new_sw = load_benchmark_sweep(spec["sweep"])
    base_sw = load_benchmark_sweep(baseline_sweep_path)
    print(f"\n=== {spec['name']} vs baseline ({baseline_label}) ===")
    print(f"{'SNR':>5} | {'L1 base':>8} | {'L1 new':>8} | {'dL1':>8} | {'Sim base':>9} | {'Sim new':>9} | {'dSim':>8}")
    for i, snr in enumerate(new_sw.snr_sweep_db):
        dl = new_sw.pulse_l1.mean[i] - base_sw.pulse_l1.mean[i]
        ds = new_sw.similarity_error.mean[i] - base_sw.similarity_error.mean[i]
        print(
            f"{snr:5.0f} | {base_sw.pulse_l1.mean[i]:8.4f} | {new_sw.pulse_l1.mean[i]:8.4f} | "
            f"{dl:+8.4f} | {base_sw.similarity_error.mean[i]:9.5f} | "
            f"{new_sw.similarity_error.mean[i]:9.5f} | {ds:+8.5f}"
        )


def main() -> None:
    print("DEVICE:", DEVICE)
    print(f"SNR head loss weight: {DEFAULT_SNR_HEAD_LOSS_WEIGHT}")
    for spec in SPECS:
        model = _train_if_needed(spec)
        _sweep_if_needed(spec, model)

    _compare(
        SPECS[0],
        BENCH_DIR / "sweep_multires_6k_es.npz",
        "Multires 6K",
    )
    _compare(
        SPECS[1],
        BENCH_DIR / "sweep_unet_6k_es.npz",
        "U-Net 6K",
    )


if __name__ == "__main__":
    main()
