"""Regenerate Exp A SNR sweep plots with updated Multires 60K (145 epochs) and patch notebook."""
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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_cnn import load_cnn_sweep, plot_metric_curves
from pulse_metrics import snr_db_to_equivalent_n_pulses

NB_PATH = SRC / "stochastic_multires_2k_noisy_trace_lambda_experiments.ipynb"
BENCH = SRC / "checkpoints" / "benchmark"
N_TEST = 512
N_TRAIN_60K_A = 60000
META_60K = BENCH / "stochastic_constant_phase_multires_60k_baseline_meta.json"


def _fig_to_png_b64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_and_capture(exp, lambda_trace, pcgpa, extra_sweeps) -> list[str]:
    """Return list of PNG base64 strings for each figure produced."""
    sweep_opt = load_cnn_sweep(exp["opt_sweep"])
    sweep_base = load_cnn_sweep(exp["baseline_sweep"])
    sweep_opt.experiment_name = (
        f"{exp['short']}: Multires 2K + trace (λ={lambda_trace:.2g})"
    )
    sweep_base.experiment_name = f"{exp['short']}: Multires 2K (lambda=0)"
    curves = [sweep_base, sweep_opt]
    if extra_sweeps:
        curves.extend(extra_sweeps)
    snr = sweep_opt.snr_sweep_db
    fmts = ["--^", "-s", "-o", "-D", "-v", "-P"]
    tag = f"{exp['short']} test n={N_TEST}"

    images: list[str] = []

    def _plot_snr_metric(attr_m, attr_s, ylabel, title_suffix, pcgpa_attr_m=None, pcgpa_attr_s=None):
        series = [
            (
                getattr(r, attr_m),
                getattr(r, attr_s),
                fmts[i % len(fmts)],
                r.experiment_name,
            )
            for i, r in enumerate(curves)
        ]
        if pcgpa is not None:
            series.append(
                (
                    pcgpa[pcgpa_attr_m],
                    pcgpa[pcgpa_attr_s],
                    "-.D",
                    f"PCGPA (best amb., n={pcgpa['n_test']})",
                )
            )
        plot_metric_curves(
            snr,
            series,
            xlabel="trace SNR (dB)",
            ylabel=ylabel,
            title=f"{ylabel} vs trace SNR — {title_suffix} ({tag})",
        )
        images.append(_fig_to_png_b64())
        n_equiv = np.array([snr_db_to_equivalent_n_pulses(float(s)) for s in snr])
        plot_metric_curves(
            n_equiv,
            series,
            xlabel="equivalent pulse count N (log scale)",
            ylabel=ylabel,
            title=f"{ylabel} vs N — {title_suffix} ({tag})",
            xscale="log",
        )
        images.append(_fig_to_png_b64())

    _plot_snr_metric(
        "cnn_l1_amb_m",
        "cnn_l1_amb_s",
        "L1 (best ambiguity, mean ± std)",
        "best L1 ambiguity",
        pcgpa_attr_m="l1_m",
        pcgpa_attr_s="l1_s",
    )
    _plot_snr_metric(
        "cnn_sim_amb_m",
        "cnn_sim_amb_s",
        "SIMILARITY_ERROR (best ambiguity)",
        "best ambiguity",
        pcgpa_attr_m="sim_m",
        pcgpa_attr_s="sim_s",
    )
    return images


def main() -> None:
    meta60 = json.loads(META_60K.read_text(encoding="utf-8"))
    print("60K meta:", json.dumps(meta60, indent=2))

    exp = {
        "label": "Constant random phase",
        "short": "constant-phase",
        "opt_sweep": BENCH / "stochastic_multires_2k_noisy_trace_lambda_opt_sweep.npz",
        "baseline_sweep": BENCH
        / "stochastic_multires_2k_noisy_trace_lambda_baseline_sweep.npz",
        "multires_60k_sweep": BENCH
        / "stochastic_constant_phase_multires_60k_baseline_sweep.npz",
        "pcgpa_sweep": BENCH / "stochastic_constant_phase_pcgpa_snr_sweep.npz",
        "meta_json": BENCH / "stochastic_multires_2k_noisy_trace_lambda_meta.json",
    }

    run_log = json.loads(exp["meta_json"].read_text(encoding="utf-8"))
    best_val = np.array([e["best_val_l1"] for e in run_log], dtype=np.float64)
    lam_grid = np.array([e["lam"] for e in run_log], dtype=np.float64)
    lambda_trace = float(lam_grid[int(np.argmin(best_val))])
    print(f"lambda_trace (auto) = {lambda_trace}")

    pcgpa = None
    pcgpa_path = exp["pcgpa_sweep"]
    if pcgpa_path.is_file():
        z = np.load(pcgpa_path)
        n_pcgpa = int(z["pcgpa_n_test"]) if "pcgpa_n_test" in z else 32
        pcgpa = {
            "snr_sweep_db": z["snr_sweep_db"],
            "l1_m": z["pcgpa_l1_m"],
            "l1_s": z["pcgpa_l1_s"],
            "sim_m": z["pcgpa_sim_m"],
            "sim_s": z["pcgpa_sim_s"],
            "n_test": n_pcgpa,
        }
        print(f"Loaded PCGPA sweep n={n_pcgpa}")

    s60k = load_cnn_sweep(exp["multires_60k_sweep"])
    epochs = int(meta60.get("matched_2k_lam15_stopped_epoch", meta60.get("stopped_epoch", 145)))
    s60k.experiment_name = (
        f"{exp['short']}: Multires 60K (λ=0, n_train={N_TRAIN_60K_A}, "
        f"{epochs} ep, no early-stop)"
    )

    images = plot_and_capture(exp, lambda_trace, pcgpa, [s60k])
    print(f"Captured {len(images)} figures")

    # Quick numeric peek at high SNR
    print(
        "60K L1@30dB = "
        f"{float(s60k.cnn_l1_amb_m[-1]):.4f} ± {float(s60k.cnn_l1_amb_s[-1]):.4f}"
    )

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # Update markdown note (cell 14)
    nb["cells"][14]["source"] = [
        "### Experiment A — constant random phase\n",
        "\n",
        "Test-set SNR sweep: **Multires 2K** (λ=0) vs **Multires 2K + trace** "
        "(λ* from val L1) vs **Multires 60K** (λ=0, 60K train pulses, no trace loss) "
        "vs **PCGPA**.\n",
        "\n",
        f"**Multires 60K fair-epoch retrain:** trained for **{epochs} epochs** "
        "(matched to Multires 2K physical λ=1.5 `stopped_epoch`), "
        "**no early-stop / patience**, **final-epoch weights** "
        "(`restore_best=False`). Artifacts overwritten under "
        "`stochastic_constant_phase_multires_60k_baseline_*`.\n",
    ]

    # Update plot cell label (cell 16)
    nb["cells"][16]["source"] = [
        '_exp_a = EXPERIMENTS["constant_phase"]\n',
        'if not _exp_a["meta_json"].exists():\n',
        '    print(f"Skipping {_exp_a[\'label\']}: missing meta")\n',
        "else:\n",
        '    _pcgpa_a = load_or_compute_pcgpa_sweep("constant_phase")\n',
        "    _extra_60k = []\n",
        '    if _exp_a["multires_60k_sweep"].exists():\n',
        '        _s60k = load_cnn_sweep(_exp_a["multires_60k_sweep"])\n',
        "        _meta60 = {}\n",
        '        _m60p = BENCH / "stochastic_constant_phase_multires_60k_baseline_meta.json"\n',
        "        if _m60p.exists():\n",
        '            _meta60 = json.loads(_m60p.read_text(encoding="utf-8"))\n',
        "        _ep60 = int(\n",
        '            _meta60.get(\n',
        '                "matched_2k_lam15_stopped_epoch",\n',
        '                _meta60.get("stopped_epoch", 145),\n',
        "            )\n",
        "        )\n",
        "        _s60k.experiment_name = (\n",
        '            f"{_exp_a[\'short\']}: Multires 60K (λ=0, n_train={N_TRAIN_60K_A}, "\n',
        '            f"{_ep60} ep, no early-stop)"\n',
        "        )\n",
        "        _extra_60k.append(_s60k)\n",
        "        print(\n",
        '            f"60K meta: stopped={_meta60.get(\'stopped_epoch\')} "\n',
        '            f"final_val={_meta60.get(\'final_val_l1\')} "\n',
        '            f"restore_best={_meta60.get(\'restore_best\')}"\n',
        "        )\n",
        "    else:\n",
        '        print("Multires 60K curve omitted — train via cell above first.")\n',
        "    plot_snr_sweeps(\n",
        "        _exp_a,\n",
        '        _resolve_lambda_trace("constant_phase", _exp_a),\n',
        "        pcgpa=_pcgpa_a,\n",
        "        extra_sweeps=_extra_60k or None,\n",
        "    )\n",
    ]

    # Replace outputs of cell 16 with new figures + stdout
    stdout_lines = [
        f"60K meta: stopped={meta60.get('stopped_epoch')} "
        f"final_val={meta60.get('final_val_l1')} "
        f"restore_best={meta60.get('restore_best')}\n"
    ]
    if pcgpa is not None:
        stdout_lines.insert(
            0, f"Loaded PCGPA sweep (Data A) from {pcgpa_path} (n={pcgpa['n_test']})\n"
        )
    else:
        stdout_lines.insert(
            0, "RUN_PCGPA_SWEEP['constant_phase']=False — PCGPA curves omitted\n"
        )

    outputs = [
        {
            "output_type": "stream",
            "name": "stdout",
            "text": stdout_lines,
        }
    ]
    for b64 in images:
        outputs.append(
            {
                "output_type": "display_data",
                "data": {"image/png": b64, "text/plain": ["<Figure size ...>"]},
                "metadata": {},
            }
        )
    nb["cells"][16]["outputs"] = outputs
    nb["cells"][16]["execution_count"] = (nb["cells"][16].get("execution_count") or 0) + 1

    # Soft-update train cell 15 comment / defaults note at top
    train_src = "".join(nb["cells"][15].get("source", []))
    if "matched to Multires 2K" not in train_src:
        nb["cells"][15]["source"] = [
            "# Multires 60K Exp A — fair-epoch protocol (already applied offline):\n",
            "#   max_epochs=145 (= Multires 2K λ=1.5 stopped_epoch),\n",
            "#   patience disabled, --no-restore-best (final-epoch weights).\n",
            "# Re-run with FORCE_MULTIRES_60K_A_RETRAIN only if you need to retrain.\n",
            "\n",
        ] + nb["cells"][15]["source"]

    NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated notebook: {NB_PATH}")


if __name__ == "__main__":
    main()
