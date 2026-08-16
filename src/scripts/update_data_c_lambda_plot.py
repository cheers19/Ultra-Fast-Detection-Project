"""Patch Data C Section 5 notebook outputs with new pulse-only / amb curves."""
from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1]
NB_PATH = SRC / "stochastic_multires_2k_noisy_trace_lambda_experiments.ipynb"
BENCH = SRC / "checkpoints" / "benchmark"
META = BENCH / "stochastic_data_c_multires_2k_noisy_trace_lambda_meta.json"
RESULTS = BENCH / "stochastic_data_c_multires_2k_noisy_trace_lambda.npz"
VAL_SNR_DB = 15.0
N_VAL = 200


def _fig_b64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    run_log = json.loads(META.read_text(encoding="utf-8"))
    d = np.load(RESULTS)
    lambda_opt = float(d["lambda_opt"])
    lambda_grid = np.array([e["lam"] for e in run_log], dtype=np.float64)
    best_val = np.array([e["best_val_l1"] for e in run_log], dtype=np.float64)
    best_amb = np.array(
        [float(e.get("best_val_l1_amb", np.nan)) for e in run_log], dtype=np.float64
    )

    images: list[str] = []
    stdout: list[str] = []

    stdout.append(
        "Data C — Multires 2K + trace loss (λ search; displayed metrics = pulse L1 only)\n"
    )
    stdout.append(f"  t_center_std = {float(d['t_center_std_fs']):.2f} fs\n")
    stdout.append(f"  trace_scale = {float(d['trace_scale']):.4f}\n")
    stdout.append(
        f"  lambda* = {lambda_opt:.4f}  "
        f"best val L1@15dB (no amb) = {float(d['best_val_at_opt']):.5f}\n"
    )
    if "best_val_amb_at_opt" in d.files:
        stdout.append(
            f"  at λ*: val L1 amb @ best-raw epoch = {float(d['best_val_amb_at_opt']):.5f}\n"
        )
    stdout.append(
        f"  baseline lambda = {float(d['lambda_baseline']):.4f}  "
        f"best val L1 = {float(d['best_val_at_baseline']):.5f}\n"
    )

    # λ vs best val
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lambda_grid, best_val, "o-", lw=2, ms=8, label="best val L1 (no amb)")
    if np.any(np.isfinite(best_amb)):
        ax.plot(lambda_grid, best_amb, "s--", lw=2, ms=7, label="val L1 amb @ best-raw ep")
    ax.axvline(lambda_opt, color="C3", ls="--", lw=1.5, label=f"lambda* = {lambda_opt:.2f}")
    ax.set_xlabel("lambda (trace loss weight)")
    ax.set_ylabel(f"val pulse L1 @ {VAL_SNR_DB:.0f} dB")
    ax.set_title(f"lambda search — Data C (n_val={N_VAL})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    images.append(_fig_b64())

    stdout.append("\nPer-lambda summary (data-c):\n")
    for e in run_log:
        stdout.append(
            f"  lambda={e['lam']:.4f}  best_epoch={e['best_epoch']:3d}  "
            f"stopped={e['stopped_epoch']:3d}  "
            f"best_val_L1={e['best_val_l1']:.5f}  "
            f"best_val_L1_amb={float(e.get('best_val_l1_amb', float('nan'))):.5f}\n"
        )

    # training curves: train / val raw / val amb
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for e in run_log:
        ep = np.arange(1, len(e["val_l1_pulses"]) + 1)
        axes[0].plot(ep, e["train_losses"], label=f"lambda={e['lam']:.2g}")
        axes[1].plot(ep, e["val_l1_pulses"], label=f"lambda={e['lam']:.2g}")
        axes[2].plot(ep, e["val_l1_amb"], label=f"lambda={e['lam']:.2g}")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train pulse L1 (no amb)")
    axes[0].set_title("Train pulse L1 — data-c")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(f"val pulse L1 @ {VAL_SNR_DB:.0f} dB (no amb)")
    axes[1].set_title("Val pulse L1 (no amb) — data-c")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel(f"val pulse L1 @ {VAL_SNR_DB:.0f} dB (best amb)")
    axes[2].set_title("Val pulse L1 (best amb) — data-c")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    images.append(_fig_b64())

    stdout.append("\nPer-lambda val at best-raw epoch (data-c):\n")
    for e in run_log:
        be = int(e["best_epoch"]) - 1
        raw_at = float(e["val_l1_pulses"][be])
        amb_at = float(e["val_l1_amb"][be])
        stdout.append(
            f"  lambda={e['lam']:.4f}  best_ep={e['best_epoch']:3d}  "
            f"val_L1_raw={raw_at:.5f}  val_L1_amb={amb_at:.5f}\n"
        )

    # test @ 0 dB
    eval0 = np.load(BENCH / "stochastic_data_c_multires_2k_trace_test_0db.npz")
    stdout.append(
        f"\nTest @ {float(eval0['test_snr_db']):.0f} dB SNR  (n={int(eval0['n_test'])})\n"
    )
    stdout.append(
        f"\n  λ* = {lambda_opt:.4f}  (best val L1 @ 15 dB = {float(d['best_val_at_opt']):.5f})\n"
    )
    stdout.append(
        f"    L1 (best ambiguity)        = {float(eval0['l1_amb_mean']):.4f} ± "
        f"{float(eval0['l1_amb_std']):.4f}\n"
    )
    stdout.append(
        f"    SIMILARITY_ERROR (best amb) = {float(eval0['sim_amb_mean']):.4f} ± "
        f"{float(eval0['sim_amb_std']):.4f}\n"
    )
    lam_evals = [
        (0.75, "lam_0.7500.pt", "stochastic_data_c_multires_2k_trace_lam075_test_0db.npz"),
        (1.5, "lam_1.5000.pt", "stochastic_data_c_multires_2k_trace_lam150_test_0db.npz"),
        (2.25, "lam_2.2500.pt", "stochastic_data_c_multires_2k_trace_lam225_test_0db.npz"),
        (3.0, "lam_3.0000.pt", "stochastic_data_c_multires_2k_trace_lam300_test_0db.npz"),
    ]
    import torch

    ckpt_dir = BENCH / "stochastic_data_c_multires_2k_noisy_trace_lambda"
    for lam_val, ckpt_name, eval_name in lam_evals:
        eval_lam = np.load(BENCH / eval_name)
        ckpt_lam = torch.load(ckpt_dir / ckpt_name, map_location="cpu", weights_only=False)
        stdout.append(
            f"\n  λ = {lam_val:.2f}  (best_epoch={int(ckpt_lam['best_epoch'])}, "
            f"best val L1 @ 15 dB = {float(ckpt_lam['best_val_l1']):.5f}, "
            f"val L1 amb = {float(ckpt_lam.get('best_val_l1_amb', float('nan'))):.5f})\n"
        )
        stdout.append(
            f"    L1 (best ambiguity)        = {float(eval_lam['l1_amb_mean']):.4f} ± "
            f"{float(eval_lam['l1_amb_std']):.4f}\n"
        )
        stdout.append(
            f"    SIMILARITY_ERROR (best amb) = {float(eval_lam['sim_amb_mean']):.4f} ± "
            f"{float(eval_lam['sim_amb_std']):.4f}\n"
        )

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # Reset train flags
    src23 = "".join(nb["cells"][23]["source"])
    src23 = src23.replace("RUN_DATA_C_TRACE_TRAIN = True", "RUN_DATA_C_TRACE_TRAIN = False")
    src23 = src23.replace(
        "FORCE_DATA_C_TRACE_RETRAIN = True", "FORCE_DATA_C_TRACE_RETRAIN = False"
    )
    # split back to lines preserving newlines like notebook source
    nb["cells"][23]["source"] = [line + "\n" for line in src23.splitlines()]
    if nb["cells"][23]["source"] and not src23.endswith("\n"):
        nb["cells"][23]["source"][-1] = nb["cells"][23]["source"][-1].rstrip("\n")

    outputs = [
        {"output_type": "stream", "name": "stdout", "text": stdout},
    ]
    for b64 in images:
        outputs.append(
            {
                "output_type": "display_data",
                "data": {"image/png": b64, "text/plain": ["<Figure>"]},
                "metadata": {},
            }
        )
    # Keep stream after first figure for training-curve summary? Already all in one stdout.
    nb["cells"][24]["outputs"] = outputs
    nb["cells"][24]["execution_count"] = (nb["cells"][24].get("execution_count") or 0) + 1

    NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Updated notebook cell 24 outputs.")
    print("".join(stdout))


if __name__ == "__main__":
    main()
