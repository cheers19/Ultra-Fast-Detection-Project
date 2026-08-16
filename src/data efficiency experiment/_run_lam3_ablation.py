"""Run λ=3 final-budget ablation for physics n=2498 (train + SNR sweep + summary)."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import setup_src_path  # noqa: F401

import data_c_amb_loss_diagnostics as diag
from data_generation import filtered_c1_pulse_config
from dataset_utils import build_filtered_c1_frog_dataloaders
from evaluate_cnn import load_cnn_sweep

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = HERE / "checkpoints" / "n2498_physics"
PLAIN_OUT = HERE / "checkpoints" / "n2498_plain"
OUT.mkdir(parents=True, exist_ok=True)

N_TRAIN = 2498
N_VAL = 200
N_TEST = 512
BATCH_SIZE = 300
SEED = 0
TRACE_SCALE = 8.0
EARLY_STOP_MODE = "steps"
TRAIN_SNR = (0.0, 30.0)
VAL_SNR = (0.0, 30.0)
SNR_SWEEP_DB = np.arange(-10.0, 31.0, 5.0)
SNAPSHOT_EVERY_FINAL = 2
SNAPSHOT_SNR_DB = 10.0
SNAPSHOT_VAL_INDEX = 0
SNAPSHOT_NOISE_SEED = 12345

STEPS_PER_EPOCH = math.ceil(N_TRAIN / BATCH_SIZE)
PATIENCE_STEPS = 25 * STEPS_PER_EPOCH
MAX_STEPS_FINAL = 200 * STEPS_PER_EPOCH
MAX_EPOCHS_FINAL = max(1, math.ceil(MAX_STEPS_FINAL / STEPS_PER_EPOCH))

_plain = json.loads((PLAIN_OUT / "campaign_summary.json").read_text(encoding="utf-8"))
LR_STAR = float(_plain["LR_star"])
_ls = json.loads((OUT / "lambda_screen_summary.json").read_text(encoding="utf-8"))
LAM_STAR = float(_ls["LAM_STAR"])

FORCE_RETRAIN = False
FORCE_TEST_SWEEP = False


def lam_tag(lam: float) -> str:
    s = f"{float(lam):g}".replace(".", "p").replace("-", "m")
    return f"n{N_TRAIN}_phys_lam{s}"


def load_meta(tag: str) -> dict:
    return json.loads((OUT / f"{tag}_meta.json").read_text(encoding="utf-8"))


def train_physics(tag: str, *, lam: float, role: str) -> None:
    hist_path = OUT / f"{tag}_history.npz"
    if hist_path.exists() and not FORCE_RETRAIN:
        print(f"skip {role}; using {hist_path}", flush=True)
        return
    print(
        f"=== {role}: {tag}  lam={lam:g}  lr={LR_STAR:g}  "
        f"max_steps={MAX_STEPS_FINAL}  patience_steps={PATIENCE_STEPS} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    result = diag.train_data_c_amb_diagnostics(
        pulse_loss_mode="raw",
        lam=float(lam),
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_test=N_TEST,
        batch_size=BATCH_SIZE,
        seed=SEED,
        max_epochs=int(MAX_EPOCHS_FINAL),
        patience=int(PATIENCE_STEPS),
        lr=float(LR_STAR),
        train_snr_db_range=TRAIN_SNR,
        val_snr_db_range=VAL_SNR,
        device=DEVICE,
        verbose=True,
        ambiguity_backend="legacy",
        trace_loss_ref="clean",
        loader_builder="filtered_c1",
        canonicalize_mode="t0",
        max_steps=int(MAX_STEPS_FINAL),
        fixed_trace_scale=TRACE_SCALE,
        snapshot_every=int(SNAPSHOT_EVERY_FINAL),
        snapshot_snr_db=SNAPSHOT_SNR_DB,
        snapshot_val_index=SNAPSHOT_VAL_INDEX,
        snapshot_noise_seed=SNAPSHOT_NOISE_SEED,
        early_stop_mode=EARLY_STOP_MODE,
    )
    result_save = {k: v for k, v in result.items() if k != "bundle"}
    result_save["role"] = role
    diag.save_run_artifacts(result_save, OUT, tag)
    meta = load_meta(tag)
    meta["role"] = role
    meta["wall_time_total_sec"] = float(time.perf_counter() - t0)
    (OUT / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


def main() -> None:
    print("device:", DEVICE, flush=True)
    print("LR_star", LR_STAR, "LAM_star", LAM_STAR, flush=True)

    LAM_COMPARE3 = 3.0
    TAG_COMPARE3 = lam_tag(LAM_COMPARE3) + "_final"
    TAG_FINAL = lam_tag(LAM_STAR) + "_final"
    print(f"Running ablation lam={LAM_COMPARE3:g} tag={TAG_COMPARE3}", flush=True)

    train_physics(TAG_COMPARE3, lam=LAM_COMPARE3, role="final_lam3_ablation")
    meta3 = load_meta(TAG_COMPARE3)
    print(
        f"lam=3 best_epoch={meta3['best_epoch']} best_score={meta3['best_score']:.6f} "
        f"stopped={meta3['stopped_epoch']}",
        flush=True,
    )

    bundle = build_filtered_c1_frog_dataloaders(
        n_train=max(N_TRAIN, 64),
        n_val=N_VAL,
        n_test=N_TEST,
        batch_size=BATCH_SIZE,
        seed=SEED,
        device=DEVICE,
        grid=filtered_c1_pulse_config(n=64),
        canonicalize_mode="t0",
    )
    sweep_path3 = OUT / f"{TAG_COMPARE3}_test_snr_sweep.npz"
    if FORCE_TEST_SWEEP or not sweep_path3.exists():
        print("Running test SNR sweep for lam=3...", flush=True)
        diag.run_and_save_test_snr_sweep(
            OUT / f"{TAG_COMPARE3}_model.pt",
            sweep_path3,
            test_loader=bundle.test_loader,
            snr_sweep_db=SNR_SWEEP_DB,
            device=DEVICE,
            experiment_name=f"Physics n={N_TRAIN} lam={LAM_COMPARE3:g} LR*={LR_STAR:g}",
        )
    else:
        print("skip sweep; using", sweep_path3, flush=True)

    sweep3 = load_cnn_sweep(sweep_path3)
    print(f"=== lam={LAM_COMPARE3:g} SNR sweep ===", flush=True)
    for snr, l1, sim in zip(sweep3.snr_sweep_db, sweep3.cnn_l1_amb_m, sweep3.cnn_sim_amb_m):
        print(f"  SNR={snr:6.1f}  L1_amb={l1:.4f}  SIM_amb={sim:.4f}", flush=True)

    # Overlay plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].errorbar(
        sweep3.snr_sweep_db,
        sweep3.cnn_l1_amb_m,
        yerr=sweep3.cnn_l1_amb_s,
        marker="o",
        label=f"lam={LAM_COMPARE3:g}",
    )
    axes[1].errorbar(
        sweep3.snr_sweep_db,
        sweep3.cnn_sim_amb_m,
        yerr=sweep3.cnn_sim_amb_s,
        marker="o",
        label=f"lam={LAM_COMPARE3:g}",
    )
    for path, label, marker in [
        (OUT / f"{TAG_FINAL}_test_snr_sweep.npz", f"lam*={LAM_STAR:g}", "s"),
        (OUT / f"{lam_tag(5.0)}_final_test_snr_sweep.npz", "lam=5", "^"),
    ]:
        if not path.exists():
            print("missing overlay:", path, flush=True)
            continue
        sw = load_cnn_sweep(path)
        axes[0].errorbar(
            sw.snr_sweep_db, sw.cnn_l1_amb_m, yerr=sw.cnn_l1_amb_s, marker=marker, label=label
        )
        axes[1].errorbar(
            sw.snr_sweep_db, sw.cnn_sim_amb_m, yerr=sw.cnn_sim_amb_s, marker=marker, label=label
        )
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("L1 (best-amb)")
    axes[0].set_title(f"Physics n={N_TRAIN}: L1 vs SNR")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("SIM error (best-amb)")
    axes[1].set_title(f"Physics n={N_TRAIN}: SIM vs SNR")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = OUT / "ablation_lam3_snr_overlay.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print("wrote", fig_path, flush=True)

    summary3 = {
        "n_train": N_TRAIN,
        "model": "physics",
        "role": "final_lam3_ablation",
        "lambda": LAM_COMPARE3,
        "lambda_star": LAM_STAR,
        "LR_star": LR_STAR,
        "trace_scale": TRACE_SCALE,
        "batch_size": BATCH_SIZE,
        "best_epoch": meta3["best_epoch"],
        "best_score_val_amb": meta3["best_score"],
        "stopped_epoch": meta3["stopped_epoch"],
        "global_step": meta3.get("global_step"),
        "wall_time_data_sec": meta3.get("wall_time_data_sec"),
        "wall_time_train_sec": meta3.get("wall_time_train_sec"),
        "device": meta3.get("device"),
        "data": "filtered_c1",
        "tag": TAG_COMPARE3,
        "compare_to_tag": TAG_FINAL,
        "snapshot_val_index": SNAPSHOT_VAL_INDEX,
        "snapshot_snr_db": SNAPSHOT_SNR_DB,
        "snapshot_noise_seed": SNAPSHOT_NOISE_SEED,
        "snr_sweep_l1_amb_m": [float(x) for x in sweep3.cnn_l1_amb_m],
        "snr_sweep_db": [float(x) for x in sweep3.snr_sweep_db],
    }
    out_json = OUT / "ablation_lam3_summary.json"
    out_json.write_text(json.dumps(summary3, indent=2), encoding="utf-8")
    print("wrote", out_json, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

