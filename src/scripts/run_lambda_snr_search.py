"""λ vs. trace SNR: select on val, report on test. Used by reconstruction_snr_experiments.ipynb."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataset_utils import PulseGridConfig, build_frog_dataloaders
from frognet import FROGNet
from pulse_metrics import (
    best_l1_ambiguity,
    packed_batch_to_complex,
    pulse_packed_l1_loss_torch,
    unpack_packed_field,
)
from trace_noise import add_trace_noise_awgn
from train import load_checkpoint


def train_with_physics_loss(
    model: nn.Module,
    loader: DataLoader,
    frog: FROGNet,
    *,
    device: torch.device,
    snr_db: float,
    lam: float,
    epochs: int,
    lr: float,
) -> nn.Module:
    trace_crit = nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for I_clean, E_true in loader:
            I_clean = I_clean.to(device)
            E_true = E_true.to(device)
            I_noisy = add_trace_noise_awgn(I_clean, float(snr_db))
            optimizer.zero_grad(set_to_none=True)
            E_pred = model(I_noisy.unsqueeze(1))
            loss = pulse_packed_l1_loss_torch(E_pred, E_true)
            if float(lam) > 0.0:
                loss = loss + float(lam) * trace_crit(frog(E_pred), I_noisy)
            loss.backward()
            optimizer.step()
    return model


def mean_best_l1(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    snr_db: float,
) -> float:
    model.eval()
    per: list[float] = []
    with torch.no_grad():
        for I_clean, E_true in loader:
            I_clean = I_clean.to(device)
            I_noisy = add_trace_noise_awgn(I_clean, float(snr_db))
            E_pred = model(I_noisy.unsqueeze(1))
            rec = packed_batch_to_complex(E_pred.cpu())
            for i in range(rec.shape[0]):
                e_true = unpack_packed_field(E_true[i].cpu().numpy())
                per.append(float(best_l1_ambiguity(rec[i], e_true)))
    return float(np.mean(per))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-n", type=int, default=512)
    parser.add_argument("--test-n", type=int, default=512)
    parser.add_argument("--train-n", type=int, default=512)
    parser.add_argument("--search-epochs", type=int, default=5)
    parser.add_argument("--finetune-lr", type=float, default=5e-5)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/large_60k_multires_50ep.pt",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/lambda_opt_vs_snr_val512.npz",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(args.seed)
    n = 64
    t_total = 250.0
    val_n = int(args.val_n)
    test_n = int(args.test_n)

    snr_grid = np.arange(-10.0, 31.0, 5.0)
    lambda_grid = np.linspace(0.0, 0.2, 10)

    grid = PulseGridConfig(n=n, t_total=t_total)
    bundle = build_frog_dataloaders(
        n_train=max(int(args.train_n), 64),
        n_val=val_n,
        n_test=test_n,
        batch_size=64,
        seed=seed,
        device=device,
        grid=grid,
    )
    train_loader = DataLoader(
        Subset(bundle.train_loader.dataset, range(int(args.train_n))),
        batch_size=64,
        shuffle=True,
    )
    val_loader = bundle.val_loader
    test_loader = bundle.test_loader

    frog = FROGNet(num_delay_steps=n).to(device)
    frog.eval()

    ckpt = _SRC / args.checkpoint
    base_model, _ = load_checkpoint(ckpt, device)

    n_snr = len(snr_grid)
    n_lam = len(lambda_grid)
    l1_val_grid = np.full((n_snr, n_lam), np.nan, dtype=np.float64)
    lambda_opt = np.full(n_snr, np.nan, dtype=np.float64)
    l1_val_at_opt = np.full(n_snr, np.nan, dtype=np.float64)
    l1_test_at_opt = np.full(n_snr, np.nan, dtype=np.float64)

    t0 = time.time()
    for si, snr_db in enumerate(snr_grid):
        best_val = float("inf")
        best_model = None
        best_li = 0

        for li, lam in enumerate(lambda_grid):
            if float(lam) == 0.0:
                model = base_model
            else:
                model, _ = load_checkpoint(ckpt, device)
                train_with_physics_loss(
                    model,
                    train_loader,
                    frog,
                    device=device,
                    snr_db=float(snr_db),
                    lam=float(lam),
                    epochs=int(args.search_epochs),
                    lr=float(args.finetune_lr),
                )
            val_l1 = mean_best_l1(
                model, val_loader, device=device, snr_db=float(snr_db)
            )
            l1_val_grid[si, li] = val_l1
            print(
                f"  SNR {float(snr_db):+.0f} dB  lam={float(lam):.4f}  "
                f"val L1={val_l1:.5f}",
                flush=True,
            )
            if val_l1 < best_val:
                best_val = val_l1
                best_li = li
                best_model = model

        lambda_opt[si] = float(lambda_grid[best_li])
        l1_val_at_opt[si] = float(best_val)
        l1_test_at_opt[si] = mean_best_l1(
            best_model, test_loader, device=device, snr_db=float(snr_db)
        )
        print(
            f"SNR {float(snr_db):+.0f} dB -> lam_opt={lambda_opt[si]:.4f}  "
            f"val L1={l1_val_at_opt[si]:.5f}  test L1={l1_test_at_opt[si]:.5f}  "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    out = _SRC / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        snr_grid=snr_grid,
        lambda_grid=lambda_grid,
        l1_val_grid=l1_val_grid,
        lambda_opt=lambda_opt,
        l1_val_at_opt=l1_val_at_opt,
        l1_test_at_opt=l1_test_at_opt,
        val_n=val_n,
        test_n=test_n,
        selection="val",
        search_epochs=int(args.search_epochs),
        train_n=int(args.train_n),
    )
    print(f"Saved {out}  total {time.time() - t0:.0f}s", flush=True)
    print("lambda_opt:", lambda_opt, flush=True)


if __name__ == "__main__":
    main()
