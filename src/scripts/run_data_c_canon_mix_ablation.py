"""Data C Multires-2K physics: λ∈{0,0.75} × 4 canonicalize mixes (phase × flip).

Method 1 (pulses_generator): phase @ t=0, flip by Re-area.
Method 2 (stochastic generator): phase @ t*, flip by |E|^2 energy.

Scenarios:
  (1) t0_re       — phase1 + flip1
  (2) t0_energy   — phase1 + flip2
  (3) tstar_re    — phase2 + flip1
  (4) tstar_energy — phase2 + flip2

Displayed metrics: train pulse L1, val pulse L1 raw, val pulse L1 best-amb.
λ* and early-stop selected by **val L1 best-ambiguity**.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_generation import stochastic_pulse_config_data_c
from dataset_utils import build_stochastic_frog_dataloaders
from frog_reconstruction_model import extract_pulse_prediction
from frognet import FROGNet
from pulse_metrics import (
    CANON_MIX_MODES,
    best_l1_ambiguity,
    pulse_packed_l1_loss_torch,
    unpack_packed_field,
)
from trace_noise import add_trace_noise_awgn
from train import TrainHistory, build_model

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_data_c_lam",
    SRC / "scripts" / "run_multires_2k_stochastic_data_c_noisy_trace_lambda.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

calibrate_trace_scale = _mod.calibrate_trace_scale
subset_loader = _mod.subset_loader
trace_l1_sum_batch_torch = _mod.trace_l1_sum_batch_torch
_batch_mean_best_l1_ambiguity = _mod._batch_mean_best_l1_ambiguity

OUT_DIR = SRC / "checkpoints" / "benchmark" / "data_c_canon_mix_ablation"
LAMBDA_GRID = np.array([0.0, 0.75], dtype=np.float64)
SCENARIOS = [
    {
        "id": 1,
        "mode": "t0_re",
        "label": "(1) phase=t0, flip=Re",
        "short": "t0+Re",
    },
    {
        "id": 2,
        "mode": "t0_energy",
        "label": "(2) phase=t0, flip=energy",
        "short": "t0+E",
    },
    {
        "id": 3,
        "mode": "tstar_re",
        "label": "(3) phase=t*, flip=Re",
        "short": "t*+Re",
    },
    {
        "id": 4,
        "mode": "tstar_energy",
        "label": "(4) phase=t*, flip=energy",
        "short": "t*+E",
    },
]


def train_early_stop_on_amb(
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
    """Like Data C trainer, but early-stop / best ckpt by val L1 best-ambiguity."""
    import copy

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = TrainHistory()
    val_l1_amb_hist: list[float] = []
    device = next(model.parameters()).device
    scale = max(float(trace_scale), 1e-8)

    best_val_amb = float("inf")
    best_val_raw = float("inf")
    best_train = float("inf")
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
            running_pulse += pulse_l1.item() * b
            n_seen += b
        train_pulse = running_pulse / max(n_seen, 1)
        history.train_losses.append(train_pulse)

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

        if val_l1_amb < best_val_amb:
            best_val_amb = val_l1_amb
            best_val_raw = val_l1
            best_train = train_pulse
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(
                f"  lam={float(lam):.4f}  epoch {epoch + 1:03d}/{max_epochs}  "
                f"train_pulse_L1={train_pulse:.5f}  "
                f"val_L1={val_l1:.5f}  val_L1_amb={val_l1_amb:.5f}",
                flush=True,
            )

        if epochs_no_improve >= patience:
            stopped_epoch = epoch + 1
            if verbose:
                print(
                    f"  early stop on val_L1_amb: patience={patience}, "
                    f"stopped={stopped_epoch}, best_ep={best_epoch}",
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
        "best_val_l1": best_val_raw,
        "best_val_l1_amb": best_val_amb,
        "best_train_pulse_l1": best_train,
        "stopped_epoch": stopped_epoch,
        "lam": float(lam),
        "trace_scale": scale,
    }


def run_scenario(
    scenario: dict,
    *,
    device: torch.device,
    force: bool,
    n_train: int,
    n_val: int,
    n_test: int,
    batch_size: int,
    seed: int,
    max_epochs: int,
    patience: int,
    lr: float,
    train_snr: tuple[float, float],
    val_snr_db: float,
) -> dict:
    mode = scenario["mode"]
    tag = f"scen{scenario['id']}_{mode}"
    scen_dir = OUT_DIR / tag
    scen_dir.mkdir(parents=True, exist_ok=True)
    summary_path = scen_dir / "summary.json"

    if summary_path.exists() and not force:
        print(f"[cache] {tag}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    print(f"\n======== {scenario['label']} | mode={mode} ========", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    grid = stochastic_pulse_config_data_c(n=64)
    bundle = build_stochastic_frog_dataloaders(
        n_train=n_train,
        n_val=max(n_val, 64),
        n_test=max(n_test, 64),
        batch_size=batch_size,
        seed=seed,
        device=device,
        grid=grid,
        canonicalize_mode=mode,
    )
    val_loader = subset_loader(bundle.val_loader, n_val)
    frog = FROGNet(num_delay_steps=64).to(device)
    frog.eval()

    cal_model = build_model(64, device, model_name="multires")
    trace_scale = calibrate_trace_scale(
        cal_model, frog, bundle.train_loader, device=device
    )
    del cal_model
    print(f"trace_scale={trace_scale:.4f}", flush=True)

    lam_runs: list[dict] = []
    for lam in LAMBDA_GRID:
        ckpt_path = scen_dir / f"lam_{float(lam):.4f}.pt"
        if ckpt_path.exists() and not force:
            meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            entry = meta.get("log_entry", meta)
            lam_runs.append(entry)
            print(
                f"[cache] lam={lam:.2f}  best_amb={float(entry['best_val_l1_amb']):.4f}",
                flush=True,
            )
            continue

        print(f"\n--- {mode} lam={lam:.4f} ---", flush=True)
        t0 = time.perf_counter()
        model = build_model(64, device, model_name="multires")
        result = train_early_stop_on_amb(
            model,
            bundle.train_loader,
            val_loader,
            frog,
            lam=float(lam),
            trace_scale=trace_scale,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            train_snr_db_range=train_snr,
            val_snr_db=val_snr_db,
        )
        wall = time.perf_counter() - t0
        entry = {
            "lam": float(lam),
            "best_epoch": int(result["best_epoch"]),
            "stopped_epoch": int(result["stopped_epoch"]),
            "best_train_pulse_l1": float(result["best_train_pulse_l1"]),
            "best_val_l1": float(result["best_val_l1"]),
            "best_val_l1_amb": float(result["best_val_l1_amb"]),
            "train_losses": list(result["history"].train_losses),
            "val_l1_pulses": list(result["history"].val_l1_pulses),
            "val_l1_amb": list(result["val_l1_amb"]),
            "wall_time_sec": wall,
            "trace_scale": float(trace_scale),
            "canonicalize_mode": mode,
            "scenario_id": scenario["id"],
            "scenario_label": scenario["label"],
        }
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                **entry,
                "log_entry": entry,
            },
            ckpt_path,
        )
        lam_runs.append(entry)
        print(
            f"  done lam={lam:.2f}  train={entry['best_train_pulse_l1']:.4f}  "
            f"val_raw={entry['best_val_l1']:.4f}  "
            f"val_amb={entry['best_val_l1_amb']:.4f}  wall={wall:.0f}s",
            flush=True,
        )

    # λ* by best-ambiguity val L1
    amb_vals = np.array([e["best_val_l1_amb"] for e in lam_runs], dtype=np.float64)
    opt_idx = int(np.argmin(amb_vals))
    chosen = lam_runs[opt_idx]
    summary = {
        "scenario_id": scenario["id"],
        "scenario_label": scenario["label"],
        "scenario_short": scenario["short"],
        "canonicalize_mode": mode,
        "phase_mode": CANON_MIX_MODES[mode][0],
        "flip_mode": CANON_MIX_MODES[mode][1],
        "lambda_opt": float(chosen["lam"]),
        "lambda_opt_selected_by": "min_val_l1_best_ambiguity",
        "best_train_pulse_l1": float(chosen["best_train_pulse_l1"]),
        "best_val_l1": float(chosen["best_val_l1"]),
        "best_val_l1_amb": float(chosen["best_val_l1_amb"]),
        "best_epoch": int(chosen["best_epoch"]),
        "stopped_epoch": int(chosen["stopped_epoch"]),
        "lambda_runs": lam_runs,
        "t_center_std_fs": float(grid.t_center_std_fs),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[{mode}] λ*={summary['lambda_opt']:.2f} by val_amb  "
        f"train={summary['best_train_pulse_l1']:.4f}  "
        f"val_raw={summary['best_val_l1']:.4f}  "
        f"val_amb={summary['best_val_l1_amb']:.4f}",
        flush=True,
    )
    return summary


def make_histograms(summaries: list[dict], out_png: Path) -> None:
    labels = [s["scenario_short"] for s in summaries]
    train = [s["best_train_pulse_l1"] for s in summaries]
    val_raw = [s["best_val_l1"] for s in summaries]
    val_amb = [s["best_val_l1_amb"] for s in summaries]
    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, vals, title, color in (
        (axes[0], train, "Train pulse L1 (no amb)", "C0"),
        (axes[1], val_raw, "Val pulse L1 raw (no amb)", "C1"),
        (axes[2], val_amb, "Val pulse L1 (best amb)", "C2"),
    ):
        bars = ax.bar(x, vals, width=0.6, color=color, edgecolor="k", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("L1")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ymin = min(vals)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            if abs(v - ymin) < 1e-12:
                bar.set_edgecolor("red")
                bar.set_linewidth(2.0)
    fig.suptitle(
        "Data C Multires 2K — canon mixes at λ* (selected by val L1 best-amb)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Grouped bar chart (all three metrics together)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width, train, width, label="train pulse L1", color="C0")
    ax.bar(x, val_raw, width, label="val L1 raw", color="C1")
    ax.bar(x + width, val_amb, width, label="val L1 best-amb", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{s['scenario_short']}\nλ*={s['lambda_opt']:.2f}" for s in summaries]
    )
    ax.set_ylabel("pulse L1")
    ax.set_title("Data C — train / val raw / val amb by canonicalize scenario")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    grouped = out_png.with_name(out_png.stem + "_grouped.png")
    fig.savefig(grouped, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histograms: {out_png}", flush=True)
    print(f"Saved grouped: {grouped}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-snr-db", type=float, default=15.0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print(f"λ grid: {LAMBDA_GRID}", flush=True)
    print("λ* / early-stop criterion: val pulse L1 best-ambiguity", flush=True)

    summaries = []
    for scen in SCENARIOS:
        summaries.append(
            run_scenario(
                scen,
                device=device,
                force=args.force,
                n_train=args.n_train,
                n_val=args.n_val,
                n_test=args.n_test,
                batch_size=args.batch_size,
                seed=args.seed,
                max_epochs=args.max_epochs,
                patience=args.patience,
                lr=args.lr,
                train_snr=(0.0, 30.0),
                val_snr_db=args.val_snr_db,
            )
        )

    master = {
        "lambda_grid": LAMBDA_GRID.tolist(),
        "selection_rule": "min_val_l1_best_ambiguity",
        "scenarios": summaries,
    }
    master_path = OUT_DIR / "master_summary.json"
    master_path.write_text(json.dumps(master, indent=2), encoding="utf-8")

    hist_path = OUT_DIR / "canon_mix_error_histograms.png"
    make_histograms(summaries, hist_path)

    print("\n===== SUMMARY (at λ* by val_amb) =====", flush=True)
    best_amb_scen = min(summaries, key=lambda s: s["best_val_l1_amb"])
    best_raw_scen = min(summaries, key=lambda s: s["best_val_l1"])
    best_train_scen = min(summaries, key=lambda s: s["best_train_pulse_l1"])
    for s in summaries:
        print(
            f"  {s['scenario_label']:32s}  λ*={s['lambda_opt']:.2f}  "
            f"train={s['best_train_pulse_l1']:.4f}  "
            f"val_raw={s['best_val_l1']:.4f}  "
            f"val_amb={s['best_val_l1_amb']:.4f}",
            flush=True,
        )
    print(
        f"\nLowest train:   {best_train_scen['scenario_label']} "
        f"({best_train_scen['best_train_pulse_l1']:.4f})",
        flush=True,
    )
    print(
        f"Lowest val raw: {best_raw_scen['scenario_label']} "
        f"({best_raw_scen['best_val_l1']:.4f})",
        flush=True,
    )
    print(
        f"Lowest val amb: {best_amb_scen['scenario_label']} "
        f"({best_amb_scen['best_val_l1_amb']:.4f})",
        flush=True,
    )
    print(f"Wrote {master_path}", flush=True)


if __name__ == "__main__":
    main()
