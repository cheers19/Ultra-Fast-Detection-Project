"""Retrain physics lambda=15 with extended max_steps so patience can fire.

Recommended budget (n=2498, B=300, K=9):
  patience = 25*K = 225
  previous final hit ceiling: best_step=1782 / max_steps=1800
  => use max_steps = 600*K = 5400  (600 epochs)
     allows a new best as late as step 5175 and still stop on patience.
"""
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
SNAPSHOT_EVERY = 2
SNAPSHOT_SNR_DB = 10.0
SNAPSHOT_VAL_INDEX = 0
SNAPSHOT_NOISE_SEED = 12345

LAM = 15.0
STEPS_PER_EPOCH = math.ceil(N_TRAIN / BATCH_SIZE)  # K=9
PATIENCE_STEPS = 25 * STEPS_PER_EPOCH  # 225
# Extended final budget: 600 epochs (was 200). Chosen so patience can stop the run.
MAX_STEPS = 600 * STEPS_PER_EPOCH  # 5400
MAX_EPOCHS = max(1, math.ceil(MAX_STEPS / STEPS_PER_EPOCH))

_plain = json.loads((PLAIN_OUT / "campaign_summary.json").read_text(encoding="utf-8"))
LR_STAR = float(_plain["LR_star"])

TAG = f"n{N_TRAIN}_phys_lam{int(LAM)}_final_ext"
FORCE_RETRAIN = False
FORCE_TEST_SWEEP = False


def load_meta(tag: str) -> dict:
    return json.loads((OUT / f"{tag}_meta.json").read_text(encoding="utf-8"))


def main() -> None:
    print("device:", DEVICE, flush=True)
    print(
        f"tag={TAG}  lam={LAM:g}  lr={LR_STAR:g}  "
        f"K={STEPS_PER_EPOCH}  patience_steps={PATIENCE_STEPS}  "
        f"max_steps={MAX_STEPS} (= {MAX_EPOCHS} ep)",
        flush=True,
    )

    hist_path = OUT / f"{TAG}_history.npz"
    if hist_path.exists() and not FORCE_RETRAIN:
        print("skip train; using", hist_path, flush=True)
    else:
        t0 = time.perf_counter()
        result = diag.train_data_c_amb_diagnostics(
            pulse_loss_mode="raw",
            lam=float(LAM),
            n_train=N_TRAIN,
            n_val=N_VAL,
            n_test=N_TEST,
            batch_size=BATCH_SIZE,
            seed=SEED,
            max_epochs=int(MAX_EPOCHS),
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
            max_steps=int(MAX_STEPS),
            fixed_trace_scale=TRACE_SCALE,
            snapshot_every=int(SNAPSHOT_EVERY),
            snapshot_snr_db=SNAPSHOT_SNR_DB,
            snapshot_val_index=SNAPSHOT_VAL_INDEX,
            snapshot_noise_seed=SNAPSHOT_NOISE_SEED,
            early_stop_mode=EARLY_STOP_MODE,
        )
        result_save = {k: v for k, v in result.items() if k != "bundle"}
        result_save["role"] = "final_lam15_extended"
        diag.save_run_artifacts(result_save, OUT, TAG)
        meta = load_meta(TAG)
        meta["role"] = "final_lam15_extended"
        meta["wall_time_total_sec"] = float(time.perf_counter() - t0)
        meta["note"] = (
            "Extended max_steps=600*K so stop is likely via patience, "
            "not the previous 200*K ceiling."
        )
        (OUT / f"{TAG}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2), flush=True)

    meta = load_meta(TAG)
    print(
        f"RESULT best_step={meta.get('best_step')}  best_epoch={meta.get('best_epoch')}  "
        f"global_step={meta.get('global_step')}  max_steps={meta.get('max_steps')}  "
        f"stopped_epoch={meta.get('stopped_epoch')}  best_score={meta.get('best_score')}",
        flush=True,
    )
    gs = int(meta.get("global_step") or 0)
    ms = int(meta.get("max_steps") or MAX_STEPS)
    if gs >= ms:
        print("WARNING: still hit max_steps; patience did not stop the run.", flush=True)
    else:
        print("OK: stopped before max_steps (patience-driven).", flush=True)

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
    sweep_path = OUT / f"{TAG}_test_snr_sweep.npz"
    if FORCE_TEST_SWEEP or not sweep_path.exists():
        print("Running SNR sweep...", flush=True)
        diag.run_and_save_test_snr_sweep(
            OUT / f"{TAG}_model.pt",
            sweep_path,
            test_loader=bundle.test_loader,
            snr_sweep_db=SNR_SWEEP_DB,
            device=DEVICE,
            experiment_name=f"Physics n={N_TRAIN} lam={LAM:g} extended",
        )
    sweep = load_cnn_sweep(sweep_path)
    print("=== SNR sweep ===", flush=True)
    for snr, l1, sim in zip(sweep.snr_sweep_db, sweep.cnn_l1_amb_m, sweep.cnn_sim_amb_m):
        print(f"  SNR={snr:6.1f}  L1_amb={l1:.4f}  SIM_amb={sim:.4f}", flush=True)

    summary = {
        "n_train": N_TRAIN,
        "model": "physics",
        "role": "final_lam15_extended",
        "lambda": LAM,
        "LR_star": LR_STAR,
        "trace_scale": TRACE_SCALE,
        "batch_size": BATCH_SIZE,
        "patience_steps": PATIENCE_STEPS,
        "max_steps": MAX_STEPS,
        "best_epoch": meta["best_epoch"],
        "best_step": meta.get("best_step"),
        "best_score_val_amb": meta["best_score"],
        "stopped_epoch": meta["stopped_epoch"],
        "global_step": meta.get("global_step"),
        "hit_max_steps": int(meta.get("global_step") or 0) >= int(meta.get("max_steps") or MAX_STEPS),
        "tag": TAG,
        "compare_to_tag": f"n{N_TRAIN}_phys_lam{int(LAM)}_final",
        "snr_sweep_db": [float(x) for x in sweep.snr_sweep_db],
        "snr_sweep_l1_amb_m": [float(x) for x in sweep.cnn_l1_amb_m],
    }
    out_json = OUT / "ablation_lam15_extended_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", out_json, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
