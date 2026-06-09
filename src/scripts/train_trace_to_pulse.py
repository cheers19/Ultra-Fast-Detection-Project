#!/usr/bin/env python3
"""Train TraceToPulseCNN on simulated FROG data (CLI for long runs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataset_utils import PulseGridConfig, build_frog_dataloaders
from train import TrainConfig, build_model, save_checkpoint, train_trace_to_pulse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TraceToPulseCNN on simulated SHG-FROG traces.")
    p.add_argument("--n-train", type=int, default=60_000)
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--n-test", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=64, help="time samples / trace width")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/large_60k.pt",
        help="output checkpoint path",
    )
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "cnn_large"],
        help="architecture: cnn (baseline) or cnn_large (wider/deeper)",
    )
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_name = args.experiment_name or Path(args.checkpoint).stem
    config = TrainConfig(
        n=args.n,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        experiment_name=exp_name,
        model_name=args.model,
        device=args.device,
    )
    device = config.resolve_device()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print(f"DEVICE: {device}")
    print(f"Training {exp_name}: model={args.model}, n_train={config.n_train}, epochs={config.epochs}")

    bundle = build_frog_dataloaders(
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        batch_size=config.batch_size,
        seed=config.seed,
        device=device,
        grid=PulseGridConfig(n=config.n),
    )
    model = build_model(config.n, device, model_name=config.model_name)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    history = train_trace_to_pulse(
        model,
        bundle.train_loader,
        bundle.val_loader,
        epochs=config.epochs,
        lr=config.lr,
        train_snr_db_range=config.train_snr_db_range,
        val_snr_db=config.val_snr_db,
    )
    ckpt = save_checkpoint(config.checkpoint_path, model, config, history)
    print("Saved:", ckpt)


if __name__ == "__main__":
    main()
