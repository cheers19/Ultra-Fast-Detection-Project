"""Multires 2K on stochastic Data C — train with trace L1 loss only (no pulse term, no λ)."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from data_generation import stochastic_pulse_config_data_c
from dataset_utils import build_stochastic_frog_dataloaders
from evaluate_cnn import run_cnn_snr_sweep, save_cnn_sweep
from frog_reconstruction_model import extract_pulse_prediction
from frognet import FROGNet
from pulse_metrics import pulse_packed_l1_loss_torch
from trace_noise import add_trace_noise_awgn
from train import TrainHistory, build_model

_LAM_SCRIPT = _SRC / "scripts" / "run_multires_2k_stochastic_data_c_noisy_trace_lambda.py"
_spec = importlib.util.spec_from_file_location("_data_c_lam", _LAM_SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

subset_loader = _mod.subset_loader
trace_l1_sum_batch_torch = _mod.trace_l1_sum_batch_torch
calibrate_trace_scale = _mod.calibrate_trace_scale
SNR_SWEEP_DB = _mod.SNR_SWEEP_DB

DEFAULT_CKPT = _SRC / "checkpoints/benchmark/stochastic_data_c_multires_2k_trace_only.pt"
DEFAULT_META = _SRC / "checkpoints/benchmark/stochastic_data_c_multires_2k_trace_only_meta.json"
DEFAULT_SWEEP = _SRC / "checkpoints/benchmark/stochastic_data_c_multires_2k_trace_only_sweep.npz"
DEFAULT_CKPT_VALTRACE = _SRC / "checkpoints/benchmark/stochastic_data_c_multires_2k_trace_only_valtrace.pt"
DEFAULT_META_VALTRACE = _SRC / "checkpoints/benchmark/stochastic_data_c_multires_2k_trace_only_valtrace_meta.json"
DEFAULT_SWEEP_VALTRACE = _SRC / "checkpoints/benchmark/stochastic_data_c_multires_2k_trace_only_valtrace_sweep.npz"


def _val_trace_loss(
    model: torch.nn.Module,
    val_loader,
    frog: FROGNet,
    *,
    val_snr_db: float,
    scale: float,
) -> float:
    device = next(model.parameters()).device
    vsum, vcount = 0.0, 0
    with torch.no_grad():
        for I_clean, E_true in val_loader:
            I_clean = I_clean.to(device)
            I_noisy = add_trace_noise_awgn(I_clean, float(val_snr_db))
            E_pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
            vloss = trace_l1_sum_batch_torch(frog(E_pred), I_clean) / scale
            b = I_clean.shape[0]
            vsum += vloss.item() * b
            vcount += b
    return vsum / max(vcount, 1)

def train_multires_trace_only_early_stop(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    frog: FROGNet,
    *,
    trace_scale: float,
    max_epochs: int,
    patience: int,
    lr: float,
    train_snr_db_range: tuple[float, float],
    val_snr_db: float,
    early_stop_metric: str = "pulse",
    verbose: bool = True,
):
    """Train loss = trace L1 / scale only.

    early_stop_metric: 'pulse' -> val pulse L1 @ val_snr_db (default);
                       'trace' -> val trace L1 / scale @ val_snr_db.
    """
    early_stop_metric = str(early_stop_metric).lower()
    if early_stop_metric not in {"pulse", "trace"}:
        raise ValueError(f"early_stop_metric must be 'pulse' or 'trace', got {early_stop_metric!r}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = TrainHistory()
    device = next(model.parameters()).device
    scale = max(float(trace_scale), 1e-8)

    best_val_pulse = float("inf")
    best_val_trace = float("inf")
    best_val_stop = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    stopped_epoch = 0

    snr_lo, snr_hi = float(train_snr_db_range[0]), float(train_snr_db_range[1])

    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for I_clean, E_true in train_loader:
            I_clean = I_clean.to(device)
            E_true = E_true.to(device)
            snr = float(np.random.uniform(snr_lo, snr_hi))
            I_noisy = add_trace_noise_awgn(I_clean, snr)
            optimizer.zero_grad(set_to_none=True)
            E_pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
            trace_l1 = trace_l1_sum_batch_torch(frog(E_pred), I_clean)
            loss = trace_l1 / scale
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
        val_pulse_l1 = vsum / max(vcount, 1)
        val_trace_l1 = _val_trace_loss(
            model, val_loader, frog, val_snr_db=float(val_snr_db), scale=scale
        )
        history.val_l1_pulses.append(val_pulse_l1)

        val_stop = val_pulse_l1 if early_stop_metric == "pulse" else val_trace_l1

        if val_stop < best_val_stop:
            best_val_stop = val_stop
            best_val_pulse = val_pulse_l1
            best_val_trace = val_trace_l1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(
                f"  epoch {epoch + 1:03d}/{max_epochs}  "
                f"train_trace_loss={history.train_losses[-1]:.5f}  "
                f"val_pulse_L1@{val_snr_db:.0f}dB={val_pulse_l1:.5f}  "
                f"val_trace_loss@{val_snr_db:.0f}dB={val_trace_l1:.5f}",
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

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_l1": best_val_pulse,
        "best_val_trace_loss": best_val_trace,
        "best_val_early_stop": best_val_stop,
        "early_stop_metric": early_stop_metric,
        "stopped_epoch": stopped_epoch,
        "trace_scale": scale,
    }


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
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--meta-json", default=str(DEFAULT_META))
    parser.add_argument("--sweep-output", default=str(DEFAULT_SWEEP))
    parser.add_argument(
        "--early-stop-metric",
        choices=("pulse", "trace"),
        default="pulse",
        help="Validation metric for early stopping and best checkpoint selection",
    )
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.early_stop_metric == "trace" and args.checkpoint == str(DEFAULT_CKPT):
        args.checkpoint = str(DEFAULT_CKPT_VALTRACE)
    if args.early_stop_metric == "trace" and args.meta_json == str(DEFAULT_META):
        args.meta_json = str(DEFAULT_META_VALTRACE)
    if args.early_stop_metric == "trace" and args.sweep_output == str(DEFAULT_SWEEP):
        args.sweep_output = str(DEFAULT_SWEEP_VALTRACE)

    ckpt_path = Path(args.checkpoint)
    meta_path = Path(args.meta_json)
    sweep_path = Path(args.sweep_output)

    if ckpt_path.exists() and sweep_path.exists() and not args.force:
        print(f"Exists: {ckpt_path} and {sweep_path} (use --force to retrain)", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    n = 64
    train_snr_range = (float(args.train_snr_min), float(args.train_snr_max))

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    grid = stochastic_pulse_config_data_c(n=n)
    bundle = build_stochastic_frog_dataloaders(
        n_train=int(args.n_train),
        n_val=max(int(args.n_val), 64),
        n_test=max(int(args.n_test), 64),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
        grid=grid,
    )
    val_loader = subset_loader(bundle.val_loader, int(args.n_val))
    test_loader = subset_loader(bundle.test_loader, int(args.n_test))
    frog = FROGNet(num_delay_steps=n).to(device)

    t0 = time.time()
    if ckpt_path.exists() and not args.force:
        print(f"Loading checkpoint: {ckpt_path}", flush=True)
        meta = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(n, device, model_name="multires")
        model.load_state_dict(meta["model_state_dict"])
        trace_scale = float(meta.get("trace_scale", 1.0))
        result = {
            "best_epoch": int(meta["best_epoch"]),
            "best_val_l1": float(meta["best_val_l1"]),
            "stopped_epoch": int(meta["stopped_epoch"]),
            "trace_scale": trace_scale,
            "history": meta.get("history"),
        }
    else:
        print(
            f"Data C trace-only: t_center_std={grid.t_center_std_fs} fs  "
            f"train={int(args.n_train)} val={int(args.n_val)} test={int(args.n_test)}",
            flush=True,
        )
        print(
            f"train SNR [{train_snr_range[0]:.0f}, {train_snr_range[1]:.0f}] dB; "
            f"early stop on val {args.early_stop_metric} @ {args.val_snr_db:.0f} dB",
            flush=True,
        )
        cal_model = build_model(n, device, model_name="multires")
        trace_scale = calibrate_trace_scale(cal_model, frog, bundle.train_loader, device=device)
        del cal_model
        print(f"trace_scale = {trace_scale:.4f}", flush=True)

        model = build_model(n, device, model_name="multires")
        train_out = train_multires_trace_only_early_stop(
            model,
            bundle.train_loader,
            val_loader,
            frog,
            trace_scale=trace_scale,
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            lr=float(args.lr),
            train_snr_db_range=train_snr_range,
            val_snr_db=float(args.val_snr_db),
            early_stop_metric=str(args.early_stop_metric),
        )
        result = train_out
        hist = train_out["history"]
        payload = {
            "model_state_dict": model.state_dict(),
            "loss_type": "trace_only",
            "early_stop_metric": str(args.early_stop_metric),
            "best_epoch": train_out["best_epoch"],
            "stopped_epoch": train_out["stopped_epoch"],
            "best_val_l1": train_out["best_val_l1"],
            "best_val_trace_loss": train_out["best_val_trace_loss"],
            "best_val_early_stop": train_out["best_val_early_stop"],
            "trace_scale": train_out["trace_scale"],
            "train_losses": hist.train_losses,
            "val_l1_pulses": hist.val_l1_pulses,
            "train_snr_db_range": list(train_snr_range),
            "val_snr_db": float(args.val_snr_db),
            "n_train": int(args.n_train),
            "n_val": int(args.n_val),
            "n_test": int(args.n_test),
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "t_center_std_fs": float(grid.t_center_std_fs),
            "pulse_kind": "stochastic_data_c",
        }
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, ckpt_path)
        meta_path.write_text(
            json.dumps(
                {k: v for k, v in payload.items() if k != "model_state_dict"},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved checkpoint: {ckpt_path}", flush=True)
        print(
            f"  best_epoch={train_out['best_epoch']}  "
            f"best_val_{args.early_stop_metric}={train_out['best_val_early_stop']:.5f}  "
            f"(val pulse L1={train_out['best_val_l1']:.5f})",
            flush=True,
        )

    if not args.skip_sweep and (not sweep_path.exists() or args.force):
        print(f"\nSNR sweep (trace-only model, n={int(args.n_test)}) …", flush=True)
        if not hasattr(model, "eval"):
            model = build_model(n, device, model_name="multires")
            meta = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(meta["model_state_dict"])
        model.eval()
        sweep = run_cnn_snr_sweep(
            model,
            test_loader,
            SNR_SWEEP_DB,
            experiment_name=f"data_c: Multires 2K trace-only (early stop={args.early_stop_metric})",
            verbose=True,
        )
        save_cnn_sweep(sweep_path, sweep)
        print(f"Saved sweep: {sweep_path}", flush=True)

    print(f"\nDone in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
