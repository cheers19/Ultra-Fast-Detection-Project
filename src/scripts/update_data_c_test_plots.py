"""Update Data C Section 6/7 notebook outputs for refreshed 0/30 dB test evals."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1]
NB_PATH = SRC / "stochastic_multires_2k_noisy_trace_lambda_experiments.ipynb"
BENCH = SRC / "checkpoints" / "benchmark"
RESULTS = BENCH / "stochastic_data_c_multires_2k_noisy_trace_lambda.npz"
CKPT_DIR = BENCH / "stochastic_data_c_multires_2k_noisy_trace_lambda"


def _fig_b64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("ascii")


def collect(eval_opt_name, lam_evals):
    eval_opt = np.load(BENCH / eval_opt_name)
    lam = [0.0]
    l1_m = [float(eval_opt["l1_amb_mean"])]
    l1_s = [float(eval_opt["l1_amb_std"])]
    sim_m = [float(eval_opt["sim_amb_mean"])]
    sim_s = [float(eval_opt["sim_amb_std"])]
    for lam_val, _, eval_name in lam_evals:
        e = np.load(BENCH / eval_name)
        lam.append(float(lam_val))
        l1_m.append(float(e["l1_amb_mean"]))
        l1_s.append(float(e["l1_amb_std"]))
        sim_m.append(float(e["sim_amb_mean"]))
        sim_s.append(float(e["sim_amb_std"]))
    return map(np.asarray, (lam, l1_m, l1_s, sim_m, sim_s))


def plot_vs_lambda(test_snr, eval_opt_name, lam_evals, lambda_opt, n_test=512):
    lam, l1_m, l1_s, sim_m, sim_s = collect(eval_opt_name, lam_evals)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, y_m, y_s, ylabel in (
        (axes[0], l1_m, l1_s, "L1 (best ambiguity)"),
        (axes[1], sim_m, sim_s, "SIMILARITY_ERROR (best amb)"),
    ):
        ax.errorbar(lam, y_m, yerr=y_s, fmt="o-", lw=2, ms=8, capsize=4)
        ax.axvline(lambda_opt, color="C3", ls="--", lw=1.5, label=f"lambda* = {lambda_opt:.2f}")
        ax.set_xlabel("lambda (trace loss weight)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs lambda — test @ {test_snr:.0f} dB SNR (n={n_test})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Data C — Multires 2K + trace loss", y=1.02)
    fig.tight_layout()
    return _fig_b64()


def section_stdout(test_snr, eval_opt_name, lam_evals, d, lambda_opt):
    import torch

    lines = []
    eval_opt = np.load(BENCH / eval_opt_name)
    lines.append(f"\nTest @ {test_snr:.0f} dB SNR  (n={int(eval_opt['n_test'])})\n")
    lines.append(
        f"\n  λ* = {lambda_opt:.4f}  (best val L1 @ 15 dB = {float(d['best_val_at_opt']):.5f})\n"
    )
    lines.append(
        f"    L1 (best ambiguity)        = {float(eval_opt['l1_amb_mean']):.4f} ± "
        f"{float(eval_opt['l1_amb_std']):.4f}\n"
    )
    lines.append(
        f"    SIMILARITY_ERROR (best amb) = {float(eval_opt['sim_amb_mean']):.4f} ± "
        f"{float(eval_opt['sim_amb_std']):.4f}\n"
    )
    for lam_val, ckpt_name, eval_name in lam_evals:
        eval_lam = np.load(BENCH / eval_name)
        ckpt = torch.load(CKPT_DIR / ckpt_name, map_location="cpu", weights_only=False)
        lines.append(
            f"\n  λ = {lam_val:.2f}  (best_epoch={int(ckpt['best_epoch'])}, "
            f"best val L1 @ 15 dB = {float(ckpt['best_val_l1']):.5f})\n"
        )
        lines.append(
            f"    L1 (best ambiguity)        = {float(eval_lam['l1_amb_mean']):.4f} ± "
            f"{float(eval_lam['l1_amb_std']):.4f}\n"
        )
        lines.append(
            f"    SIMILARITY_ERROR (best amb) = {float(eval_lam['sim_amb_mean']):.4f} ± "
            f"{float(eval_lam['sim_amb_std']):.4f}\n"
        )
    return lines


def main() -> None:
    d = np.load(RESULTS)
    lambda_opt = float(d["lambda_opt"])
    lam0 = [
        (0.75, "lam_0.7500.pt", "stochastic_data_c_multires_2k_trace_lam075_test_0db.npz"),
        (1.5, "lam_1.5000.pt", "stochastic_data_c_multires_2k_trace_lam150_test_0db.npz"),
        (2.25, "lam_2.2500.pt", "stochastic_data_c_multires_2k_trace_lam225_test_0db.npz"),
        (3.0, "lam_3.0000.pt", "stochastic_data_c_multires_2k_trace_lam300_test_0db.npz"),
    ]
    lam30 = [
        (0.75, "lam_0.7500.pt", "stochastic_data_c_multires_2k_trace_lam075_test_30db.npz"),
        (1.5, "lam_1.5000.pt", "stochastic_data_c_multires_2k_trace_lam150_test_30db.npz"),
        (2.25, "lam_2.2500.pt", "stochastic_data_c_multires_2k_trace_lam225_test_30db.npz"),
        (3.0, "lam_3.0000.pt", "stochastic_data_c_multires_2k_trace_lam300_test_30db.npz"),
    ]

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # cell 27: test error vs lambda plots (0 & 30)
    img0 = plot_vs_lambda(0.0, "stochastic_data_c_multires_2k_trace_test_0db.npz", lam0, lambda_opt)
    img30 = plot_vs_lambda(
        30.0, "stochastic_data_c_multires_2k_trace_test_30db.npz", lam30, lambda_opt
    )
    nb["cells"][29]["outputs"] = [
        {
            "output_type": "display_data",
            "data": {"image/png": img0, "text/plain": ["<Figure>"]},
            "metadata": {},
        },
        {
            "output_type": "display_data",
            "data": {"image/png": img30, "text/plain": ["<Figure>"]},
            "metadata": {},
        },
    ]
    # Find correct cell index for plot_data_c_test_errors - was 29 in earlier grep? check
    # Actually from earlier: cell 29 had plot_data_c_test_errors_vs_lambda. Verify.
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if "plot_data_c_test_errors_vs_lambda" in src and "DATA_C_TRACE_EVAL_0DB" in src and c["cell_type"] == "code" and "def " not in src:
            nb["cells"][i]["outputs"] = [
                {
                    "output_type": "display_data",
                    "data": {"image/png": img0, "text/plain": ["<Figure>"]},
                    "metadata": {},
                },
                {
                    "output_type": "display_data",
                    "data": {"image/png": img30, "text/plain": ["<Figure>"]},
                    "metadata": {},
                },
            ]
            print(f"Updated plot cell {i}")

        if "DATA_C_TRACE_EVAL_30DB" in src and "eval_trace_30" in src and c["cell_type"] == "code":
            nb["cells"][i]["outputs"] = [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": section_stdout(
                        30.0,
                        "stochastic_data_c_multires_2k_trace_test_30db.npz",
                        lam30,
                        d,
                        lambda_opt,
                    ),
                }
            ]
            print(f"Updated 30dB print cell {i}")

    NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
