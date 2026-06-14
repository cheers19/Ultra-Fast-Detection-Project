"""Train Multires 2K on clean traces; search trace-L1 loss weight λ; eval @ 15 dB test."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataset_utils import PulseGridConfig, build_frog_dataloaders
from frog_reconstruction_model import extract_pulse_prediction
from frognet import FROGNet
from pulse_metrics import (
    best_l1_ambiguity,
    packed_batch_to_complex,
    pulse_packed_l1_loss_torch,
    unpack_packed_field,
)
from trace_noise import add_trace_noise_awgn
from train import EarlyStopTrainResult, TrainHistory, build_model


@dataclass
class CleanTraceLossTrainResult:
    history: TrainHistory
    best_epoch: int
    best_val_l1: float
    stopped_epoch: int
    lam: float


def train_multires_clean_trace_loss_early_stop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    frog: FROGNet,
    *,
    lam: float,
    max_epochs: int,
    patience: int,
    lr: float,
    val_snr_db: float = 15.0,
    verbose: bool = True,
) -> CleanTraceLossTrainResult:
    """Train on noise-free traces; loss = L1(pulse) + λ·L1(FROG(E_pred), I_clean)."""
    pulse_crit = pulse_packed_l1_loss_torch
    trace_crit = nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = TrainHistory()
    device = next(model.parameters()).device

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    stopped_epoch = 0

    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for I_clean, E_true in train_loader:
            I_clean = I_clean.to(device)
            E_true = E_true.to(device)
            optimizer.zero_grad(set_to_none=True)
            E_pred = extract_pulse_prediction(model(I_clean.unsqueeze(1)))
            loss = pulse_crit(E_pred, E_true)
            if float(lam) > 0.0:
                loss = loss + float(lam) * trace_crit(frog(E_pred), I_clean)
            loss.backward()
            optimizer.step()
            b = I_clean.shape[0]
            running += loss.item() * b
            n_seen += b
        history.train_losses.append(running / max(n_seen, 1))

        model.eval()
        vsum, vcount = 0.0, 0
        with torch.no_grad():
            for I_clean, E_true in val_loader:
                I_clean = I_clean.to(device)
                E_true = E_true.to(device)
                I_noisy = add_trace_noise_awgn(I_clean, float(val_snr_db))
                E_pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
                vloss = pulse_crit(E_pred, E_true)
                b = I_clean.shape[0]
                vsum += vloss.item() * b
                vcount += b
        val_l1 = vsum / max(vcount, 1)
        history.val_l1_pulses.append(val_l1)

        if val_l1 < best_val:
            best_val = val_l1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(
                f"  lam={float(lam):.4f}  epoch {epoch + 1:03d}/{max_epochs}  "
                f"train_loss={history.train_losses[-1]:.5f}  "
                f"val_L1@{val_snr_db:.1f}dB={val_l1:.5f}",
                flush=True,
            )

        if epochs_no_improve >= patience:
            stopped_epoch = epoch + 1
            break
    else:
        stopped_epoch = max_epochs

    if best_state is not None:
        model.load_state_dict(best_state)

    return CleanTraceLossTrainResult(
        history=history,
        best_epoch=best_epoch,
        best_val_l1=best_val,
        stopped_epoch=stopped_epoch,
        lam=float(lam),
    )


def mean_best_l1_at_snr(
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
            E_true = E_true.to(device)
            I_noisy = add_trace_noise_awgn(I_clean, float(snr_db))
            E_pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
            rec = packed_batch_to_complex(E_pred.cpu())
            for i in range(rec.shape[0]):
                e_true = unpack_packed_field(E_true[i].cpu().numpy())
                per.append(float(best_l1_ambiguity(rec[i], e_true)))
    return float(np.mean(per))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    lam: float,
    result: CleanTraceLossTrainResult,
    eval_snr_db: float,
    test_l1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "lam": float(lam),
            "best_epoch": result.best_epoch,
            "best_val_l1": result.best_val_l1,
            "stopped_epoch": result.stopped_epoch,
            "eval_snr_db": float(eval_snr_db),
            "test_l1": float(test_l1),
        },
        path,
    )


def plot_lambda_vs_test_l1(
    lambda_grid: np.ndarray,
    test_l1_grid: np.ndarray,
    lambda_opt: float,
    *,
    eval_snr_db: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lambda_grid, test_l1_grid, "o-", lw=2, ms=6, label="test pulse L1 (best ambiguity)")
    ax.axvline(lambda_opt, color="C3", ls="--", lw=1.5, label=f"λ* = {lambda_opt:.4f}")
    idx = int(np.nanargmin(test_l1_grid))
    ax.scatter([lambda_grid[idx]], [test_l1_grid[idx]], s=120, c="C3", zorder=5, edgecolors="k")
    ax.set_xlabel("trace L1 loss weight λ")
    ax.set_ylabel("mean best-ambiguity pulse L1")
    ax.set_title(
        f"Multires 2K (clean train) — test error @ {eval_snr_db:.0f} dB vs. λ"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-val", type=int, default=512)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--eval-snr-db", type=float, default=15.0)
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=0.2)
    parser.add_argument("--lambda-steps", type=int, default=11)
    parser.add_argument(
        "--output",
        default="checkpoints/benchmark/multires_2k_clean_trace_lambda.npz",
    )
    parser.add_argument(
        "--plot",
        default="figures/multires_2k_clean_trace_lambda_15db.png",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="checkpoints/benchmark/multires_2k_clean_trace_lambda",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip λ values whose checkpoint already exists in --ckpt-dir",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 64
    eval_snr = float(args.eval_snr_db)
    lambda_grid = np.linspace(
        float(args.lambda_min), float(args.lambda_max), int(args.lambda_steps)
    )

    out_npz = _SRC / args.output
    out_plot = _SRC / args.plot
    ckpt_dir = _SRC / args.ckpt_dir

    if out_npz.exists() and not args.force:
        print(f"Cache exists: {out_npz}  (use --force to recompute)", flush=True)
        data = np.load(out_npz)
        plot_lambda_vs_test_l1(
            data["lambda_grid"],
            data["test_l1_grid"],
            float(data["lambda_opt"]),
            eval_snr_db=eval_snr,
            out_path=out_plot,
        )
        print(f"λ* = {float(data['lambda_opt']):.4f}  test L1 = {float(data['test_l1_at_opt']):.5f}")
        return

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    bundle = build_frog_dataloaders(
        n_train=int(args.n_train),
        n_val=int(args.n_val),
        n_test=int(args.n_test),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
        grid=PulseGridConfig(n=n),
    )
    frog = FROGNet(num_delay_steps=n).to(device)

    n_lam = len(lambda_grid)
    val_l1_grid = np.full(n_lam, np.nan, dtype=np.float64)
    test_l1_grid = np.full(n_lam, np.nan, dtype=np.float64)
    best_epochs = np.full(n_lam, -1, dtype=np.int32)

    t0 = time.time()
    for li, lam in enumerate(lambda_grid):
        ckpt_path = ckpt_dir / f"lam_{float(lam):.4f}.pt"
        if args.resume and ckpt_path.exists() and not args.force:
            meta = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = build_model(n, device, model_name="multires")
            model.load_state_dict(meta["model_state_dict"])
            val_l1 = mean_best_l1_at_snr(
                model, bundle.val_loader, device=device, snr_db=eval_snr
            )
            test_l1 = mean_best_l1_at_snr(
                model, bundle.test_loader, device=device, snr_db=eval_snr
            )
            val_l1_grid[li] = val_l1
            test_l1_grid[li] = test_l1
            best_epochs[li] = int(meta.get("best_epoch", -1))
            print(
                f"λ={float(lam):.4f}  (cached)  val L1={val_l1:.5f}  "
                f"test L1={test_l1:.5f}",
                flush=True,
            )
            continue

        print(f"\n=== λ = {float(lam):.4f} ({li + 1}/{n_lam}) ===", flush=True)
        model = build_model(n, device, model_name="multires")
        result = train_multires_clean_trace_loss_early_stop(
            model,
            bundle.train_loader,
            bundle.val_loader,
            frog,
            lam=float(lam),
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            lr=float(args.lr),
            val_snr_db=eval_snr,
        )
        val_l1 = mean_best_l1_at_snr(
            model, bundle.val_loader, device=device, snr_db=eval_snr
        )
        test_l1 = mean_best_l1_at_snr(
            model, bundle.test_loader, device=device, snr_db=eval_snr
        )
        val_l1_grid[li] = val_l1
        test_l1_grid[li] = test_l1
        best_epochs[li] = result.best_epoch
        save_checkpoint(
            ckpt_path,
            model,
            lam=float(lam),
            result=result,
            eval_snr_db=eval_snr,
            test_l1=test_l1,
        )
        print(
            f"λ={float(lam):.4f}  best_epoch={result.best_epoch}  "
            f"val L1={val_l1:.5f}  test L1={test_l1:.5f}  "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    opt_idx = int(np.nanargmin(test_l1_grid))
    lambda_opt = float(lambda_grid[opt_idx])
    l1_test_at_opt = float(test_l1_grid[opt_idx])
    l1_val_at_opt = float(val_l1_grid[opt_idx])

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        lambda_grid=lambda_grid,
        val_l1_grid=val_l1_grid,
        test_l1_grid=test_l1_grid,
        best_epochs=best_epochs,
        lambda_opt=lambda_opt,
        l1_test_at_opt=l1_test_at_opt,
        l1_val_at_opt=l1_val_at_opt,
        eval_snr_db=eval_snr,
        n_train=int(args.n_train),
        seed=int(args.seed),
    )
    plot_lambda_vs_test_l1(
        lambda_grid,
        test_l1_grid,
        lambda_opt,
        eval_snr_db=eval_snr,
        out_path=out_plot,
    )

    print(
        f"\nDone. λ* = {lambda_opt:.4f}  test L1 @ {eval_snr:.0f} dB = {l1_test_at_opt:.5f}  "
        f"(val L1 = {l1_val_at_opt:.5f})",
        flush=True,
    )
    print(f"Saved: {out_npz}", flush=True)
    print(f"Plot:  {out_plot}", flush=True)


if __name__ == "__main__":
    main()
