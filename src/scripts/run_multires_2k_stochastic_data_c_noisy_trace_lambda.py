"""Multires 2K + trace loss on stochastic Data C — λ search (same protocol as Experiment A)."""

from __future__ import annotations

import argparse
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

import copy

from data_generation import stochastic_pulse_config_data_c
from dataset_utils import build_stochastic_frog_dataloaders
from evaluate_cnn import mean_l1_cnn_at_snr, mean_metric_cnn_at_snr, run_cnn_snr_sweep, save_cnn_sweep
from frog_reconstruction_model import extract_pulse_prediction
from frognet import FROGNet
from pulse_metrics import (
    best_l1_ambiguity,
    best_similarity_error_ambiguity,
    pulse_packed_l1_loss_torch,
    unpack_packed_field,
)
from trace_noise import add_trace_noise_awgn
from train import TrainHistory, build_model

SNR_SWEEP_DB = np.arange(-10.0, 31.0, 5.0)


def trace_l1_sum_batch_torch(i_pred: torch.Tensor, i_ref: torch.Tensor) -> torch.Tensor:
    return (i_pred - i_ref).abs().flatten(1).sum(dim=-1).mean()


def calibrate_trace_scale(
    model: torch.nn.Module,
    frog: FROGNet,
    loader,
    *,
    device: torch.device,
    n_batches: int = 8,
) -> float:
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
        return float(64 * 64 / (2 * 64))
    return float(np.median(ratios))


def _batch_mean_best_l1_ambiguity(
    E_pred: torch.Tensor, E_true: torch.Tensor
) -> float:
    """Mean packed L1 over FROG ambiguities for a packed [B, 2N] batch."""
    e_pred = E_pred.detach().cpu().numpy()
    e_true = E_true.detach().cpu().numpy()
    vals = [
        best_l1_ambiguity(unpack_packed_field(e_pred[i]), unpack_packed_field(e_true[i]))
        for i in range(e_pred.shape[0])
    ]
    return float(np.mean(vals)) if vals else float("nan")


def train_multires_noisy_trace_loss_early_stop(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    frog: FROGNet,
    *,
    lam: float,
    trace_scale: float,
    max_epochs: int,
    patience: int,
    lr: float,
    train_snr_db_range: tuple[float, float],
    val_snr_db: float,
    verbose: bool = True,
):
    """
    Train with composite loss ``pulse_L1 + λ·trace_L1/scale``.

    Logged / displayed metrics are **pulse L1 only**:
      - train: raw packed pulse L1 (no ambiguity)
      - val: raw packed pulse L1 **and** best-ambiguity pulse L1

    Early-stop / λ* selection still use raw val pulse L1 (no ambiguity).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = TrainHistory()
    val_l1_amb_hist: list[float] = []
    device = next(model.parameters()).device
    scale = max(float(trace_scale), 1e-8)

    best_val = float("inf")
    best_val_amb = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    stopped_epoch = 0

    snr_lo, snr_hi = float(train_snr_db_range[0]), float(train_snr_db_range[1])

    for epoch in range(max_epochs):
        model.train()
        running_pulse = 0.0
        n_seen = 0
        for I_clean, E_true in train_loader:
            I_clean = I_clean.to(device)
            E_true = E_true.to(device)
            snr = float(np.random.uniform(snr_lo, snr_hi))
            I_noisy = add_trace_noise_awgn(I_clean, snr)
            optimizer.zero_grad(set_to_none=True)
            E_pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
            pulse_l1 = pulse_packed_l1_loss_torch(E_pred, E_true)
            loss = pulse_l1
            if float(lam) > 0.0:
                trace_l1 = trace_l1_sum_batch_torch(frog(E_pred), I_clean)
                loss = loss + float(lam) * (trace_l1 / scale)
            loss.backward()
            optimizer.step()
            b = I_clean.shape[0]
            # Log pulse L1 only (not the composite physics loss).
            running_pulse += pulse_l1.item() * b
            n_seen += b
        history.train_losses.append(running_pulse / max(n_seen, 1))

        model.eval()
        vsum, vsum_amb, vcount = 0.0, 0.0, 0
        with torch.no_grad():
            for I_clean, E_true in val_loader:
                I_clean = I_clean.to(device)
                E_true = E_true.to(device)
                I_noisy = add_trace_noise_awgn(I_clean, float(val_snr_db))
                E_pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
                vloss = pulse_packed_l1_loss_torch(E_pred, E_true)
                b = I_clean.shape[0]
                vsum += vloss.item() * b
                vsum_amb += _batch_mean_best_l1_ambiguity(E_pred, E_true) * b
                vcount += b
        val_l1 = vsum / max(vcount, 1)
        val_l1_amb = vsum_amb / max(vcount, 1)
        history.val_l1_pulses.append(val_l1)
        val_l1_amb_hist.append(val_l1_amb)

        if val_l1 < best_val:
            best_val = val_l1
            best_val_amb = val_l1_amb
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(
                f"  lam={float(lam):.4f}  epoch {epoch + 1:03d}/{max_epochs}  "
                f"train_pulse_L1={history.train_losses[-1]:.5f}  "
                f"val_L1@{val_snr_db:.0f}dB={val_l1:.5f}  "
                f"val_L1_amb={val_l1_amb:.5f}",
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
        "val_l1_amb": val_l1_amb_hist,
        "best_epoch": best_epoch,
        "best_val_l1": best_val,
        "best_val_l1_amb": best_val_amb,
        "stopped_epoch": stopped_epoch,
        "lam": float(lam),
        "trace_scale": scale,
    }


def subset_loader(base, n: int):
    from torch.utils.data import DataLoader, Subset

    n = min(int(n), len(base.dataset))
    return DataLoader(
        Subset(base.dataset, range(n)),
        batch_size=min(int(base.batch_size), n),
        shuffle=False,
    )


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
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=3.0)
    parser.add_argument("--lambda-steps", type=int, default=5)
    parser.add_argument(
        "--output",
        default="checkpoints/benchmark/stochastic_data_c_multires_2k_noisy_trace_lambda.npz",
    )
    parser.add_argument(
        "--meta-json",
        default="checkpoints/benchmark/stochastic_data_c_multires_2k_noisy_trace_lambda_meta.json",
    )
    parser.add_argument(
        "--sweep-output",
        default="checkpoints/benchmark/stochastic_data_c_multires_2k_noisy_trace_lambda_opt_sweep.npz",
    )
    parser.add_argument(
        "--baseline-sweep-output",
        default="checkpoints/benchmark/stochastic_data_c_multires_2k_noisy_trace_lambda_baseline_sweep.npz",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="checkpoints/benchmark/stochastic_data_c_multires_2k_noisy_trace_lambda",
    )
    parser.add_argument(
        "--opt-checkpoint",
        default="checkpoints/benchmark/stochastic_data_c_multires_2k_noisy_trace_lambda_opt.pt",
    )
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--test-snr-db", type=float, default=0.0)
    parser.add_argument(
        "--eval-output",
        default="checkpoints/benchmark/stochastic_data_c_multires_2k_trace_test_0db.npz",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        device = torch.device("cpu")
        print("device: cpu", flush=True)

    n = 64
    train_snr_range = (float(args.train_snr_min), float(args.train_snr_max))
    lambda_grid = np.linspace(float(args.lambda_min), float(args.lambda_max), int(args.lambda_steps))
    n_lam = len(lambda_grid)

    out_npz = _SRC / args.output
    out_meta = _SRC / args.meta_json
    sweep_out = _SRC / args.sweep_output
    baseline_sweep_out = _SRC / args.baseline_sweep_output
    ckpt_dir = _SRC / args.ckpt_dir
    opt_ckpt = _SRC / args.opt_checkpoint

    if out_npz.exists() and not args.force and not args.resume:
        d = np.load(out_npz)
        print(f"Results exist: {out_npz}", flush=True)
        print(f"  lambda_opt={float(d['lambda_opt']):.4f}  best_val={float(d['best_val_at_opt']):.5f}", flush=True)
        return

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    stoch_grid = stochastic_pulse_config_data_c(n=n)
    print(f"Data C: t_center_std={stoch_grid.t_center_std_fs} fs", flush=True)
    bundle = build_stochastic_frog_dataloaders(
        n_train=int(args.n_train),
        n_val=max(int(args.n_val), 64),
        n_test=max(int(args.n_test), 64),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
        grid=stoch_grid,
    )
    val_loader = subset_loader(bundle.val_loader, int(args.n_val))
    test_loader = subset_loader(bundle.test_loader, int(args.n_test))
    frog = FROGNet(num_delay_steps=n).to(device)

    cal_model = build_model(n, device, model_name="multires")
    trace_scale = calibrate_trace_scale(cal_model, frog, bundle.train_loader, device=device)
    del cal_model
    print(f"trace_scale = {trace_scale:.4f}", flush=True)
    print(
        f"stochastic pulses: {int(args.n_train)} train / {int(args.n_val)} val / {int(args.n_test)} test",
        flush=True,
    )
    print(
        f"train SNR in [{train_snr_range[0]:.0f}, {train_snr_range[1]:.0f}] dB; "
        f"val @ {args.val_snr_db:.0f} dB (n={int(args.n_val)})",
        flush=True,
    )

    best_val_l1 = np.full(n_lam, np.nan)
    best_epochs = np.full(n_lam, -1, dtype=np.int32)
    stopped_epochs = np.full(n_lam, -1, dtype=np.int32)
    run_log: list[dict] = []
    t0 = time.time()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for li, lam in enumerate(lambda_grid):
        ckpt_path = ckpt_dir / f"lam_{float(lam):.4f}.pt"
        if args.resume and ckpt_path.exists() and not args.force:
            meta = torch.load(ckpt_path, map_location=device, weights_only=False)
            best_val_l1[li] = float(meta["best_val_l1"])
            best_epochs[li] = int(meta["best_epoch"])
            stopped_epochs[li] = int(meta["stopped_epoch"])
            run_log.append(meta.get("log_entry", meta))
            print(
                f"lam={lam:.4f} (cached)  best_ep={best_epochs[li]}  "
                f"best_val={best_val_l1[li]:.5f}",
                flush=True,
            )
            continue

        print(f"\n--- lam = {lam:.4f} ({li + 1}/{n_lam}) ---", flush=True)
        model = build_model(n, device, model_name="multires")
        result = train_multires_noisy_trace_loss_early_stop(
            model,
            bundle.train_loader,
            val_loader,
            frog,
            lam=float(lam),
            trace_scale=trace_scale,
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            lr=float(args.lr),
            train_snr_db_range=train_snr_range,
            val_snr_db=float(args.val_snr_db),
        )
        best_val_l1[li] = result["best_val_l1"]
        best_epochs[li] = result["best_epoch"]
        stopped_epochs[li] = result["stopped_epoch"]

        log_entry = {
            "lam": float(lam),
            "best_epoch": result["best_epoch"],
            "stopped_epoch": result["stopped_epoch"],
            "best_val_l1": result["best_val_l1"],
            "best_val_l1_amb": float(result["best_val_l1_amb"]),
            "trace_scale": trace_scale,
            "train_losses": result["history"].train_losses,
            "val_l1_pulses": result["history"].val_l1_pulses,
            "val_l1_amb": list(result["val_l1_amb"]),
            "train_metric": "pulse_l1_raw",
            "val_metric": "pulse_l1_raw",
            "val_metric_amb": "pulse_l1_best_ambiguity",
            "train_snr_db_range": list(train_snr_range),
            "val_snr_db": float(args.val_snr_db),
            "n_val": int(args.n_val),
            "n_train": int(args.n_train),
            "n_test": int(args.n_test),
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "early_stop": result["stopped_epoch"] < int(args.max_epochs),
            "pulse_kind": "stochastic_data_c",
            "t_center_std_fs": float(stoch_grid.t_center_std_fs),
        }
        run_log.append(log_entry)
        torch.save(
            {"model_state_dict": model.state_dict(), **log_entry, "log_entry": log_entry},
            ckpt_path,
        )
        print(
            f"  lam={lam:.4f}  best_epoch={result['best_epoch']}  "
            f"best_val_L1={result['best_val_l1']:.5f}  "
            f"best_val_L1_amb={result['best_val_l1_amb']:.5f}",
            flush=True,
        )

    print(f"\nAll lambda checkpoints saved under {ckpt_dir}", flush=True)
    run_log = []
    best_val_l1 = np.full(n_lam, np.nan)
    best_val_l1_amb = np.full(n_lam, np.nan)
    best_epochs = np.full(n_lam, -1, dtype=np.int32)
    stopped_epochs = np.full(n_lam, -1, dtype=np.int32)
    for li, lam in enumerate(lambda_grid):
        ckpt_path = ckpt_dir / f"lam_{float(lam):.4f}.pt"
        meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        entry = meta.get("log_entry", {k: v for k, v in meta.items() if k != "model_state_dict"})
        run_log.append(entry)
        best_val_l1[li] = float(meta["best_val_l1"])
        best_val_l1_amb[li] = float(meta.get("best_val_l1_amb", np.nan))
        best_epochs[li] = int(meta["best_epoch"])
        stopped_epochs[li] = int(meta["stopped_epoch"])

    opt_idx = int(np.nanargmin(best_val_l1))
    baseline_idx = int(np.argmin(np.abs(lambda_grid - 0.0)))
    lambda_opt = float(lambda_grid[opt_idx])
    lambda_baseline = float(lambda_grid[baseline_idx])
    print(f"\nlambda* = {lambda_opt:.4f}  (best val L1 = {best_val_l1[opt_idx]:.5f})", flush=True)
    if np.isfinite(best_val_l1_amb[opt_idx]):
        print(
            f"  at λ*: best val L1 (amb) @ best-raw epoch = {best_val_l1_amb[opt_idx]:.5f}",
            flush=True,
        )
    print(
        f"baseline λ = {lambda_baseline:.4f}  (best val L1 = {best_val_l1[baseline_idx]:.5f})",
        flush=True,
    )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(run_log, indent=2), encoding="utf-8")

    np.savez(
        out_npz,
        lambda_grid=lambda_grid,
        best_val_l1=best_val_l1,
        best_val_l1_amb=best_val_l1_amb,
        best_epochs=best_epochs,
        stopped_epochs=stopped_epochs,
        lambda_opt=lambda_opt,
        lambda_opt_idx=opt_idx,
        lambda_baseline=lambda_baseline,
        lambda_baseline_idx=baseline_idx,
        best_val_at_opt=float(best_val_l1[opt_idx]),
        best_val_amb_at_opt=float(best_val_l1_amb[opt_idx]),
        best_val_at_baseline=float(best_val_l1[baseline_idx]),
        best_val_amb_at_baseline=float(best_val_l1_amb[baseline_idx]),
        trace_scale=trace_scale,
        n_train=int(args.n_train),
        n_val=int(args.n_val),
        n_test=int(args.n_test),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        train_snr_min=float(train_snr_range[0]),
        train_snr_max=float(train_snr_range[1]),
        val_snr_db=float(args.val_snr_db),
        seed=int(args.seed),
        pulse_kind="stochastic_data_c",
        t_center_std_fs=float(stoch_grid.t_center_std_fs),
        train_metric="pulse_l1_raw",
        val_metric="pulse_l1_raw",
        val_metric_amb="pulse_l1_best_ambiguity",
    )

    opt_src = ckpt_dir / f"lam_{lambda_opt:.4f}.pt"
    if opt_src.exists():
        opt_data = torch.load(opt_src, map_location=device, weights_only=False)
        opt_data["lambda_opt"] = lambda_opt
        opt_data["selected_by"] = "min_val_l1"
        torch.save(opt_data, opt_ckpt)
        print(f"Saved optimal checkpoint: {opt_ckpt}", flush=True)

    if not args.skip_sweep:
        baseline_src = ckpt_dir / f"lam_{lambda_baseline:.4f}.pt"

        if opt_src.exists():
            print(f"\nSNR sweep (λ*) on stochastic test (n={int(args.n_test)}) …", flush=True)
            model = build_model(n, device, model_name="multires")
            opt_data = torch.load(opt_src, map_location=device, weights_only=False)
            model.load_state_dict(opt_data["model_state_dict"])
            sweep = run_cnn_snr_sweep(
                model,
                test_loader,
                SNR_SWEEP_DB,
                experiment_name="stochastic_multires_2k_trace_lambda_opt",
                verbose=True,
            )
            save_cnn_sweep(sweep_out, sweep)
            print(f"Saved sweep: {sweep_out}", flush=True)

        if baseline_src.exists():
            print(
                f"\nSNR sweep (baseline λ={lambda_baseline:.4f}, no trace loss) …",
                flush=True,
            )
            model = build_model(n, device, model_name="multires")
            base_data = torch.load(baseline_src, map_location=device, weights_only=False)
            model.load_state_dict(base_data["model_state_dict"])
            sweep_base = run_cnn_snr_sweep(
                model,
                test_loader,
                SNR_SWEEP_DB,
                experiment_name="stochastic_multires_2k_baseline",
                verbose=True,
            )
            save_cnn_sweep(baseline_sweep_out, sweep_base)
            print(f"Saved baseline sweep: {baseline_sweep_out}", flush=True)

    eval_path = _SRC / args.eval_output
    if opt_src.exists():
        test_snr = float(args.test_snr_db)
        print(f"\nTest eval λ* @ {test_snr:.0f} dB SNR (n={int(args.n_test)}) …", flush=True)
        model = build_model(n, device, model_name="multires")
        opt_data = torch.load(opt_src, map_location=device, weights_only=False)
        model.load_state_dict(opt_data["model_state_dict"])
        model.eval()

        class _PulseModel(torch.nn.Module):
            def __init__(self, net: torch.nn.Module) -> None:
                super().__init__()
                self.net = net

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return extract_pulse_prediction(self.net(x))

        wrapped = _PulseModel(model)
        l1_m, l1_s = mean_l1_cnn_at_snr(wrapped, test_loader, test_snr, score_fn=best_l1_ambiguity)
        sim_m, sim_s = mean_metric_cnn_at_snr(
            wrapped, test_loader, test_snr, score_fn=best_similarity_error_ambiguity
        )
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            eval_path,
            test_snr_db=test_snr,
            n_test=int(args.n_test),
            seed=int(args.seed),
            t_center_std_fs=float(stoch_grid.t_center_std_fs),
            lambda_opt=lambda_opt,
            best_epoch=int(best_epochs[opt_idx]),
            best_val_l1=float(best_val_l1[opt_idx]),
            l1_amb_mean=l1_m,
            l1_amb_std=l1_s,
            sim_amb_mean=sim_m,
            sim_amb_std=sim_s,
        )
        print(f"Saved eval: {eval_path}", flush=True)
        print(f"  L1 (best amb)     = {l1_m:.4f} ± {l1_s:.4f}", flush=True)
        print(f"  SIMILARITY (best) = {sim_m:.4f} ± {sim_s:.4f}", flush=True)

    print(f"\nDone in {time.time() - t0:.0f}s", flush=True)
    print(f"Saved: {out_npz}", flush=True)
    print(f"Meta:  {out_meta}", flush=True)


if __name__ == "__main__":
    main()
