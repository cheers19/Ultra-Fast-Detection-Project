"""Multires 2K clean-train λ search @ 15/20/25 dB; 30 eval samples; scaled trace L1 loss."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

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
from train import TrainHistory, build_model


@dataclass
class CleanTraceLossTrainResult:
    history: TrainHistory
    best_epoch: int
    best_val_l1: float
    stopped_epoch: int
    lam: float
    trace_scale: float


def trace_l1_sum_batch_torch(i_pred: torch.Tensor, i_ref: torch.Tensor) -> torch.Tensor:
    """Mean over batch of sum |ΔI| over trace pixels (matches benchmark trace_l1 convention)."""
    return (i_pred - i_ref).abs().flatten(1).sum(dim=-1).mean()


def calibrate_trace_scale(
    model: torch.nn.Module,
    frog: FROGNet,
    loader: DataLoader,
    *,
    device: torch.device,
    n_batches: int = 8,
) -> float:
    """Median trace_l1_sum / pulse_l1_sum on clean batches (random init) for comparable loss terms."""
    model.eval()
    ratios: list[float] = []
    with torch.no_grad():
        for bi, (I_clean, E_true) in enumerate(loader):
            if bi >= n_batches:
                break
            I_clean = I_clean.to(device)
            E_true = E_true.to(device)
            E_pred = extract_pulse_prediction(model(I_clean.unsqueeze(1)))
            p = float(pulse_packed_l1_loss_torch(E_pred, E_true).item())
            t = float(trace_l1_sum_batch_torch(frog(E_pred), I_clean).item())
            if p > 1e-8:
                ratios.append(t / p)
    if not ratios:
        return float(64 * 64 / (2 * 64))  # n²/(2n) fallback
    return float(np.median(ratios))


def train_multires_clean_trace_loss_early_stop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    frog: FROGNet,
    *,
    lam: float,
    trace_scale: float,
    max_epochs: int,
    patience: int,
    lr: float,
    val_snr_db: float,
    verbose: bool = True,
) -> CleanTraceLossTrainResult:
    """Clean input; loss = L1_pulse + λ·(L1_trace_sum / trace_scale)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = TrainHistory()
    device = next(model.parameters()).device
    scale = max(float(trace_scale), 1e-8)

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
            pulse_l1 = pulse_packed_l1_loss_torch(E_pred, E_true)
            loss = pulse_l1
            if float(lam) > 0.0:
                trace_l1 = trace_l1_sum_batch_torch(frog(E_pred), I_clean)
                loss = loss + float(lam) * (trace_l1 / scale)
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
                vloss = pulse_packed_l1_loss_torch(E_pred, E_true)
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
                f"val_L1@{val_snr_db:.0f}dB={val_l1:.5f}",
                flush=True,
            )

        if epochs_no_improve >= patience:
            stopped_epoch = epoch + 1
            if verbose:
                print(
                    f"  early stop: no val improvement for {patience} epochs; "
                    f"stopped at epoch {stopped_epoch}, best epoch {best_epoch}",
                    flush=True,
                )
            break
    else:
        stopped_epoch = max_epochs
        if verbose:
            print(
                f"  reached max_epochs={max_epochs}; best epoch {best_epoch}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return CleanTraceLossTrainResult(
        history=history,
        best_epoch=best_epoch,
        best_val_l1=best_val,
        stopped_epoch=stopped_epoch,
        lam=float(lam),
        trace_scale=scale,
    )


def mean_best_l1_at_snr(
    model: torch.nn.Module,
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


def subset_loader(base: DataLoader, n: int) -> DataLoader:
    n = min(int(n), len(base.dataset))
    return DataLoader(
        Subset(base.dataset, range(n)),
        batch_size=min(int(base.batch_size), n),
        shuffle=False,
    )


def plot_lambda_curves(
    snr_grid: np.ndarray,
    lambda_grid: np.ndarray,
    test_l1_grid: np.ndarray,
    lambda_opt: np.ndarray,
    *,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(snr_grid), figsize=(4.5 * len(snr_grid), 4), sharey=True)
    if len(snr_grid) == 1:
        axes = [axes]
    for ax, si, snr in zip(axes, range(len(snr_grid)), snr_grid):
        y = test_l1_grid[si]
        ax.plot(lambda_grid, y, "o-", lw=2, ms=6)
        lo = int(np.nanargmin(y))
        ax.axvline(lambda_opt[si], color="C3", ls="--", lw=1.5)
        ax.scatter([lambda_grid[lo]], [y[lo]], s=100, c="C3", zorder=5, edgecolors="k")
        ax.set_xlabel("λ")
        ax.set_title(f"test @ {snr:.0f} dB (n=30)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mean best-ambiguity pulse L1")
    fig.suptitle("Multires 2K clean train — scaled trace L1 loss")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--snr-list", type=str, default="25,20,15")
    parser.add_argument("--lambda-steps", type=int, default=8)
    parser.add_argument(
        "--output",
        default="checkpoints/benchmark/multires_2k_clean_trace_lambda_snr_grid.npz",
    )
    parser.add_argument(
        "--meta-json",
        default="checkpoints/benchmark/multires_2k_clean_trace_lambda_snr_grid_meta.json",
    )
    parser.add_argument(
        "--plot",
        default="figures/multires_2k_clean_trace_lambda_snr_grid.png",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="checkpoints/benchmark/multires_2k_clean_trace_lambda_snr_grid",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        device = torch.device("cpu")
        print("device: cpu (CUDA not available — training will be much slower)", flush=True)
    n = 64
    snr_grid = np.array([float(x) for x in args.snr_list.split(",")], dtype=np.float64)
    lambda_grid = np.linspace(0.0, 1.0, int(args.lambda_steps))
    n_eval = int(args.n_eval)

    out_npz = _SRC / args.output
    out_meta = _SRC / args.meta_json
    out_plot = _SRC / args.plot
    ckpt_dir = _SRC / args.ckpt_dir

    if out_npz.exists() and not args.force and not args.resume:
        d = np.load(out_npz)
        plot_lambda_curves(
            d["snr_grid"], d["lambda_grid"], d["test_l1_grid"], d["lambda_opt"], out_path=out_plot
        )
        for si, snr in enumerate(d["snr_grid"]):
            print(
                f"SNR {snr:.0f} dB: lam*={float(d['lambda_opt'][si]):.4f}  "
                f"test L1={float(d['l1_test_at_opt'][si]):.5f}",
                flush=True,
            )
        return

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    bundle = build_frog_dataloaders(
        n_train=int(args.n_train),
        n_val=max(n_eval, 64),
        n_test=max(n_eval, 64),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
        grid=PulseGridConfig(n=n),
    )
    val_loader = subset_loader(bundle.val_loader, n_eval)
    test_loader = subset_loader(bundle.test_loader, n_eval)
    frog = FROGNet(num_delay_steps=n).to(device)

    cal_model = build_model(n, device, model_name="multires")
    trace_scale = calibrate_trace_scale(
        cal_model, frog, bundle.train_loader, device=device
    )
    del cal_model
    print(f"trace_scale (median trace_sum/pulse_sum on clean batches) = {trace_scale:.4f}", flush=True)

    n_snr = len(snr_grid)
    n_lam = len(lambda_grid)
    val_l1_grid = np.full((n_snr, n_lam), np.nan)
    test_l1_grid = np.full((n_snr, n_lam), np.nan)
    best_epochs = np.full((n_snr, n_lam), -1, dtype=np.int32)
    stopped_epochs = np.full((n_snr, n_lam), -1, dtype=np.int32)

    run_log: list[dict] = []
    t0 = time.time()

    for si, snr_db in enumerate(snr_grid):
        print(f"\n======== SNR {snr_db:.0f} dB ({si + 1}/{n_snr}) ========", flush=True)
        snr_ckpt_dir = ckpt_dir / f"snr_{snr_db:.0f}db"
        snr_ckpt_dir.mkdir(parents=True, exist_ok=True)

        for li, lam in enumerate(lambda_grid):
            ckpt_path = snr_ckpt_dir / f"lam_{float(lam):.4f}.pt"
            if args.resume and ckpt_path.exists() and not args.force:
                meta = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = build_model(n, device, model_name="multires")
                model.load_state_dict(meta["model_state_dict"])
                val_l1 = mean_best_l1_at_snr(model, val_loader, device=device, snr_db=snr_db)
                test_l1 = mean_best_l1_at_snr(model, test_loader, device=device, snr_db=snr_db)
                val_l1_grid[si, li] = val_l1
                test_l1_grid[si, li] = test_l1
                best_epochs[si, li] = int(meta["best_epoch"])
                stopped_epochs[si, li] = int(meta["stopped_epoch"])
                run_log.append(meta.get("log_entry", {}))
                print(
                    f"  lam={lam:.4f} (cached)  best_ep={best_epochs[si, li]}  "
                    f"stop_ep={stopped_epochs[si, li]}  test L1={test_l1:.5f}",
                    flush=True,
                )
                continue

            print(f"\n--- lam = {lam:.4f} ({li + 1}/{n_lam}) ---", flush=True)
            model = build_model(n, device, model_name="multires")
            result = train_multires_clean_trace_loss_early_stop(
                model,
                bundle.train_loader,
                val_loader,
                frog,
                lam=float(lam),
                trace_scale=trace_scale,
                max_epochs=int(args.max_epochs),
                patience=int(args.patience),
                lr=float(args.lr),
                val_snr_db=float(snr_db),
            )
            val_l1 = mean_best_l1_at_snr(model, val_loader, device=device, snr_db=snr_db)
            test_l1 = mean_best_l1_at_snr(model, test_loader, device=device, snr_db=snr_db)
            val_l1_grid[si, li] = val_l1
            test_l1_grid[si, li] = test_l1
            best_epochs[si, li] = result.best_epoch
            stopped_epochs[si, li] = result.stopped_epoch

            log_entry = {
                "snr_db": float(snr_db),
                "lam": float(lam),
                "best_epoch": result.best_epoch,
                "stopped_epoch": result.stopped_epoch,
                "best_val_l1": result.best_val_l1,
                "val_l1_eval": val_l1,
                "test_l1_eval": test_l1,
                "trace_scale": trace_scale,
                "early_stop": result.stopped_epoch < int(args.max_epochs),
            }
            run_log.append(log_entry)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    **log_entry,
                    "log_entry": log_entry,
                },
                ckpt_path,
            )
            print(
                f"  lam={lam:.4f}  best_epoch={result.best_epoch}  "
                f"stopped_epoch={result.stopped_epoch}  "
                f"val L1={val_l1:.5f}  test L1={test_l1:.5f}",
                flush=True,
            )

    lambda_opt = np.array(
        [float(lambda_grid[int(np.nanargmin(test_l1_grid[si]))]) for si in range(n_snr)]
    )
    l1_test_at_opt = np.array(
        [float(np.nanmin(test_l1_grid[si])) for si in range(n_snr)]
    )
    l1_val_at_opt = np.array(
        [
            float(val_l1_grid[si, int(np.nanargmin(test_l1_grid[si]))])
            for si in range(n_snr)
        ]
    )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        snr_grid=snr_grid,
        lambda_grid=lambda_grid,
        val_l1_grid=val_l1_grid,
        test_l1_grid=test_l1_grid,
        best_epochs=best_epochs,
        stopped_epochs=stopped_epochs,
        lambda_opt=lambda_opt,
        l1_test_at_opt=l1_test_at_opt,
        l1_val_at_opt=l1_val_at_opt,
        trace_scale=trace_scale,
        n_eval=n_eval,
        n_train=int(args.n_train),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        seed=int(args.seed),
    )
    out_meta.write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    plot_lambda_curves(
        snr_grid, lambda_grid, test_l1_grid, lambda_opt, out_path=out_plot
    )

    print(f"\nDone in {time.time() - t0:.0f}s", flush=True)
    for si, snr in enumerate(snr_grid):
        print(
            f"SNR {snr:.0f} dB: lam*={lambda_opt[si]:.4f}  test L1={l1_test_at_opt[si]:.5f}  "
            f"(best/stop epochs per lam in meta JSON)",
            flush=True,
        )
    print(f"Saved: {out_npz}", flush=True)
    print(f"Meta:  {out_meta}", flush=True)
    print(f"Plot:  {out_plot}", flush=True)


if __name__ == "__main__":
    main()
