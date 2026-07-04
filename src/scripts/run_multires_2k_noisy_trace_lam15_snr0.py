"""Train Multires 2K with noisy trace L1 loss: λ=1.5 fixed, train SNR uniform 0–30 dB."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

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
from pulse_metrics import pulse_packed_l1_loss_torch
from train import build_model

sys.path.insert(0, str(_SRC / "scripts"))
from run_multires_2k_noisy_trace_lambda import (
    calibrate_trace_scale,
    subset_loader,
    train_multires_noisy_trace_loss_early_stop,
)

DEFAULT_CKPT = _SRC / "checkpoints/benchmark/multires_2k_noisy_trace_lam15_snr0.pt"
DEFAULT_META = _SRC / "checkpoints/benchmark/multires_2k_noisy_trace_lam15_snr0_meta.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--train-snr-min", type=float, default=0.0)
    parser.add_argument("--train-snr-max", type=float, default=30.0)
    parser.add_argument("--val-snr-db", type=float, default=15.0)
    parser.add_argument("--lam", type=float, default=1.5)
    parser.add_argument("--output", default=str(DEFAULT_CKPT))
    parser.add_argument("--meta-json", default=str(DEFAULT_META))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_ckpt = _SRC / args.output
    out_meta = _SRC / args.meta_json
    if out_ckpt.exists() and not args.force:
        print(f"Checkpoint exists: {out_ckpt} (use --force to retrain)", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    lam = float(args.lam)
    train_snr_range = (float(args.train_snr_min), float(args.train_snr_max))

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    n = 64
    bundle = build_frog_dataloaders(
        n_train=int(args.n_train),
        n_val=max(int(args.n_val), 64),
        n_test=max(int(args.n_test), 64),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
        grid=PulseGridConfig(n=n),
    )
    val_loader = subset_loader(bundle.val_loader, int(args.n_val))
    frog = FROGNet(num_delay_steps=n).to(device)

    cal_model = build_model(n, device, model_name="multires")
    trace_scale = calibrate_trace_scale(cal_model, frog, bundle.train_loader, device=device)
    del cal_model
    print(f"λ={lam:.4f}  trace_scale={trace_scale:.4f}", flush=True)
    print(
        f"train SNR in [{train_snr_range[0]:.0f}, {train_snr_range[1]:.0f}] dB; "
        f"val @ {args.val_snr_db:.0f} dB (n={int(args.n_val)})",
        flush=True,
    )

    t0 = time.time()
    model = build_model(n, device, model_name="multires")
    result = train_multires_noisy_trace_loss_early_stop(
        model,
        bundle.train_loader,
        val_loader,
        frog,
        lam=lam,
        trace_scale=trace_scale,
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        lr=float(args.lr),
        train_snr_db_range=train_snr_range,
        val_snr_db=float(args.val_snr_db),
    )

    meta = {
        "lam": lam,
        "best_epoch": result.best_epoch,
        "stopped_epoch": result.stopped_epoch,
        "best_val_l1": result.best_val_l1,
        "trace_scale": trace_scale,
        "train_losses": result.history.train_losses,
        "val_l1_pulses": result.history.val_l1_pulses,
        "train_snr_db_range": list(train_snr_range),
        "val_snr_db": float(args.val_snr_db),
        "n_val": int(args.n_val),
        "n_train": int(args.n_train),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "early_stop": result.stopped_epoch < int(args.max_epochs),
    }

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **meta}, out_ckpt)
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"\nDone in {time.time() - t0:.0f}s — best_epoch={result.best_epoch}  "
        f"best_val_L1={result.best_val_l1:.5f}",
        flush=True,
    )
    print(f"Saved: {out_ckpt}", flush=True)
    print(f"Meta:  {out_meta}", flush=True)


if __name__ == "__main__":
    main()
