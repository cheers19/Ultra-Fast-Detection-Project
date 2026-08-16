"""Create Protocol v2 plain-vs-physics data-efficiency comparison notebook."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent / "v2_plain_vs_physics_data_efficiency_NB.ipynb"


def lines(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip("\n") + "\n"
    return [ln + "\n" for ln in text.split("\n")[:-1]]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": lines(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines(text),
    }


cells: list[dict] = []

cells.append(
    md(
        r"""# Protocol v2 — Plain vs Physics data efficiency

Aggregate **test SNR-sweep** best-ambiguity pulse L1 across SNR, then compare
plain vs physics as a function of \(n_{\mathrm{train}}\).

For each model / \(n\):

1. Load `{tag}_test_snr_sweep.npz` → `cnn_l1_amb_m` at each SNR.
2. Convert to **per-time-sample** L1: divide by \(N_t=64\)
   (stored L1 is a sum over packed time / Re–Im components; we normalize by 64
   as specified for this campaign plot).
3. Report mean and std of those per-SNR values **across the SNR grid**.
4. Plot mean (plain vs physics) vs \(n_{\mathrm{train}}\) (log-\(x\)).
5. Plot **best optimization step** (`best_step`) vs \(n_{\mathrm{train}}\)
   for the same official plain / physics checkpoints.

Sources:

| \(n\) | Plain | Physics |
|------:|-------|---------|
| 300, 866, 7207, 20794, 60000 | `checkpoints/v2/n*_plain/` | `checkpoints/v2/n*_physics/` |
| ~2048 (diagnostics “2k”) | `filtered_c1_multires_lam0` | `filtered_c1_multires_lam4p5` |

Additional filtered-C1 sections later in this notebook:

- **A.** Empirical convergence (SIM ≤ 0.05) for phys 2K + plain 60K + inset best steps
- **B.** Retrain plain 2K to epoch 105 and overlay train/val curves vs physics
- **C.** Random noisy TRACE example @ 10 dB for phys 2K / plain 2K / plain 60K
"""
    )
)

cells.append(
    code(
        r"""import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
except Exception:
    pass

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path.cwd()
if (HERE / "setup_src_path.py").exists():
    sys.path.insert(0, str(HERE))
    import setup_src_path  # noqa: F401
    EXP = HERE
    SRC = HERE.parent if HERE.name == "data efficiency experiment" else HERE
elif (HERE / "data efficiency experiment" / "setup_src_path.py").exists():
    EXP = HERE / "data efficiency experiment"
    SRC = HERE
    sys.path.insert(0, str(EXP))
    import setup_src_path  # noqa: F401
else:
    raise RuntimeError("Run from src/ or from data efficiency experiment/")

from evaluate_cnn import load_cnn_sweep

V2 = EXP / "checkpoints" / "v2"
DIAG = SRC / "checkpoints" / "benchmark" / "filtered_c1_multires_2k_diagnostics"
OUT_FIG = EXP / "figures" / "v2_comparisons"
OUT_FIG.mkdir(parents=True, exist_ok=True)

N_TIME = 64  # divide sweep L1 by this (per-time normalization)

print("EXP:", EXP.resolve())
print("SRC:", SRC.resolve())
print("V2:", V2.resolve())
print("DIAG:", DIAG.resolve())
"""
    )
)

cells.append(md("## Locate official SNR-sweep artifacts"))

cells.append(
    code(
        r"""def resolve_plain_sweep(n: int) -> Path:
    d = V2 / f"n{n}_plain"
    hits = sorted(d.glob("*_test_snr_sweep.npz"))
    if not hits:
        raise FileNotFoundError(f"No plain SNR sweep under {d}")
    if len(hits) > 1:
        # Prefer the lr1e-3 / official tag if present
        preferred = [p for p in hits if "lr1e-3" in p.name or "plain" in p.name]
        hits = preferred or hits
    return hits[0]


def resolve_physics_sweep(n: int) -> Path:
    d = V2 / f"n{n}_physics"
    # Prefer campaign_summary tag (extension winner when present)
    summary = d / "campaign_summary.json"
    if summary.exists():
        tag = json.loads(summary.read_text(encoding="utf-8"))["tag"]
        p = d / f"{tag}_test_snr_sweep.npz"
        if p.exists():
            return p
    # Prefer *ext*_test_snr_sweep, else any sweep
    ext = sorted(d.glob("*_ext*_test_snr_sweep.npz"))
    if ext:
        return ext[-1]
    hits = sorted(d.glob("*_test_snr_sweep.npz"))
    if not hits:
        raise FileNotFoundError(f"No physics SNR sweep under {d}")
    return hits[-1]


# Protocol-v2 ladder + diagnostics ~2k (actual n_train=2048 in meta)
PLAIN_NS = [300, 866, 2048, 7207, 20794, 60000]
PHYS_NS = [300, 866, 2048, 7207, 20794, 60000]

PLAIN_SWEEP = {
    300: None,
    866: None,
    2048: DIAG / "filtered_c1_multires_lam0_test_snr_sweep.npz",
    7207: None,
    20794: None,
    60000: None,
}
PHYS_SWEEP = {
    300: None,
    866: None,
    2048: DIAG / "filtered_c1_multires_lam4p5_test_snr_sweep.npz",
    7207: None,
    20794: None,
    60000: None,
}

for n in (300, 866, 7207, 20794, 60000):
    PLAIN_SWEEP[n] = resolve_plain_sweep(n)
    PHYS_SWEEP[n] = resolve_physics_sweep(n)

print("=== Plain sweeps ===")
for n in PLAIN_NS:
    p = PLAIN_SWEEP[n]
    print(f"  n={n:>5d}  exists={p.exists()}  {p}")
print("=== Physics sweeps ===")
for n in PHYS_NS:
    p = PHYS_SWEEP[n]
    print(f"  n={n:>5d}  exists={p.exists()}  {p}")
"""
    )
)

cells.append(
    md(
        r"""## Aggregate L1 best-amb across SNR (then \(\div 64\))

For each sweep: take `cnn_l1_amb_m` at every SNR, set
\(e(s)=\texttt{cnn\_l1\_amb\_m}(s)/64\), then

\[
\bar e=\mathrm{mean}_s\, e(s),\qquad
\sigma_e=\mathrm{std}_s\, e(s)\ \ (\texttt{ddof=0}).
\]
"""
    )
)

cells.append(
    code(
        r"""def aggregate_l1_amb_over_snr(sweep_path: Path, *, n_time: int = N_TIME) -> dict:
    sw = load_cnn_sweep(sweep_path)
    snr = np.asarray(sw.snr_sweep_db, dtype=float)
    l1 = np.asarray(sw.cnn_l1_amb_m, dtype=float) / float(n_time)
    if l1.size == 0:
        raise ValueError(f"Empty L1 series in {sweep_path}")
    return {
        "path": str(sweep_path),
        "experiment_name": sw.experiment_name,
        "snr_db": snr,
        "l1_amb_per_time": l1,
        "mean_over_snr": float(l1.mean()),
        "std_over_snr": float(l1.std(ddof=0)),
        "n_snr": int(l1.size),
    }


def build_rows(kind: str, sweep_map: dict[int, Path]) -> pd.DataFrame:
    rows = []
    for n, path in sweep_map.items():
        assert path is not None and path.exists(), path
        agg = aggregate_l1_amb_over_snr(path)
        rows.append(
            {
                "kind": kind,
                "n_train": int(n),
                "mean_l1_amb_per_time": agg["mean_over_snr"],
                "std_l1_amb_per_time": agg["std_over_snr"],
                "n_snr": agg["n_snr"],
                "sweep_path": agg["path"],
                "experiment_name": agg["experiment_name"],
            }
        )
    return pd.DataFrame(rows).sort_values("n_train").reset_index(drop=True)


df_plain = build_rows("plain", PLAIN_SWEEP)
df_phys = build_rows("physics", PHYS_SWEEP)
df = pd.concat([df_plain, df_phys], ignore_index=True)

print("=== Plain: mean±std of (L1_amb/64) across SNR ===")
display_cols = [
    "n_train",
    "mean_l1_amb_per_time",
    "std_l1_amb_per_time",
    "n_snr",
]
print(df_plain[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("\\n=== Physics: mean±std of (L1_amb/64) across SNR ===")
print(df_phys[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

summary_path = OUT_FIG / "plain_vs_physics_snr_mean_l1_summary.csv"
df.to_csv(summary_path, index=False)
print("\\nwrote", summary_path)
"""
    )
)

cells.append(
    md(
        r"""## Comparison plot

\(x = n_{\mathrm{train}}\) (log scale),
\(y = \mathrm{mean}_{\mathrm{SNR}}(\mathrm{L1}_{amb}/64)\).
Error bars = std of those per-SNR values across the SNR grid (not SEM of the test set).
"""
    )
)

cells.append(
    code(
        r"""fig, ax = plt.subplots(figsize=(8.5, 5.0))

for kind, marker, color, label in [
    ("plain", "o", "C0", "Plain (λ=0)"),
    ("physics", "s", "C1", "Physics (λ*)"),
]:
    sub = df[df["kind"] == kind].sort_values("n_train")
    ax.errorbar(
        sub["n_train"].to_numpy(),
        sub["mean_l1_amb_per_time"].to_numpy(),
        yerr=sub["std_l1_amb_per_time"].to_numpy(),
        fmt=f"-{marker}",
        color=color,
        capsize=4,
        lw=2,
        ms=8,
        label=label,
    )

ax.set_xscale("log")
ax.set_xlabel(r"$n_{\mathrm{train}}$")
ax.set_ylabel(r"mean over SNR of (L1 best-amb / 64)")
ax.set_title("Protocol v2: plain vs physics — SNR-averaged pulse L1 (best-amb)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
fig_path = OUT_FIG / "plain_vs_physics_mean_l1_amb_vs_ntrain.png"
fig.savefig(fig_path, dpi=160)
plt.show()
print("wrote", fig_path)

# Same comparison without error bars (means only)
fig, ax = plt.subplots(figsize=(8.5, 5.0))
for kind, marker, color, label in [
    ("plain", "o", "C0", r"Plain ($\lambda=0$)"),
    ("physics", "s", "C1", r"Physics ($\lambda^*$)"),
]:
    sub = df[df["kind"] == kind].sort_values("n_train")
    ax.plot(
        sub["n_train"].to_numpy(),
        sub["mean_l1_amb_per_time"].to_numpy(),
        f"-{marker}",
        color=color,
        lw=2,
        ms=8,
        label=label,
    )
ax.set_xscale("log")
ax.set_xlabel(r"$n_{\mathrm{train}}$")
ax.set_ylabel(r"mean over SNR of (L1 best-amb / 64)")
ax.set_title("Protocol v2: plain vs physics — SNR-averaged pulse L1 (best-amb)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
fig_path_noerr = OUT_FIG / "plain_vs_physics_mean_l1_amb_vs_ntrain_no_errorbars.png"
fig.savefig(fig_path_noerr, dpi=160)
plt.show()
print("wrote", fig_path_noerr)

# Side-by-side numeric comparison at matched n
merged = df_plain.merge(
    df_phys,
    on="n_train",
    suffixes=("_plain", "_phys"),
)
merged["delta_phys_minus_plain"] = (
    merged["mean_l1_amb_per_time_phys"] - merged["mean_l1_amb_per_time_plain"]
)
merged["rel_improve_vs_plain"] = (
    -merged["delta_phys_minus_plain"] / merged["mean_l1_amb_per_time_plain"]
)
print("\\n=== Matched-n comparison ===")
print(
    merged[
        [
            "n_train",
            "mean_l1_amb_per_time_plain",
            "mean_l1_amb_per_time_phys",
            "delta_phys_minus_plain",
            "rel_improve_vs_plain",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
)
merged.to_csv(OUT_FIG / "plain_vs_physics_matched_n.csv", index=False)
"""
    )
)

cells.append(
    md(
        r"""## Optional: per-SNR L1 curves (sanity)

Overlay SNR-sweep L1 (÷64) for each \(n\), plain vs physics.
"""
    )
)

cells.append(
    code(
        r"""ns_show = PLAIN_NS
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
axes = axes.ravel()
for ax, n in zip(axes, ns_show):
    sp = load_cnn_sweep(PLAIN_SWEEP[n])
    ph = load_cnn_sweep(PHYS_SWEEP[n])
    ax.plot(
        sp.snr_sweep_db,
        np.asarray(sp.cnn_l1_amb_m) / N_TIME,
        "-o",
        label="plain",
    )
    ax.plot(
        ph.snr_sweep_db,
        np.asarray(ph.cnn_l1_amb_m) / N_TIME,
        "-s",
        label="physics",
    )
    ax.set_title(f"n={n}")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("L1 amb / 64")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
for j in range(len(ns_show), len(axes)):
    axes[j].axis("off")
plt.suptitle("SNR sweeps (L1 best-amb / 64)")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md(
        r"""## Best optimization step vs \(n_{\mathrm{train}}\)

For each official checkpoint (same tags as the L1 comparison), read
`best_step` from `campaign_summary.json` when available, else from the
matching `*_meta.json`.

Diagnostics \(n=2048\) metas store `best_epoch` only; convert with
\(B=64\):

\[
\texttt{best\_step} \approx \texttt{best\_epoch}\cdot\lceil n_{\mathrm{train}}/B\rceil.
\]

Physics Band B/C use the **extension** winner’s `best_step` from
`campaign_summary.json` (screen + extension continuum).
"""
    )
)

cells.append(
    code(
        r"""import math


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_step_from_meta(meta: dict, *, n_train: int, batch_size: int = 64) -> int:
    if meta.get("best_step") is not None:
        return int(meta["best_step"])
    # diagnostics-era meta: only best_epoch
    if meta.get("best_epoch") is None:
        raise KeyError(f"no best_step/best_epoch in meta keys={list(meta)}")
    spe = int(math.ceil(float(n_train) / float(batch_size)))
    return int(meta["best_epoch"]) * spe


def resolve_plain_best_step(n: int) -> dict:
    if n == 2048:
        meta_p = DIAG / "filtered_c1_multires_lam0_meta.json"
        meta = _load_json(meta_p)
        n_tr = int(meta.get("n_train", n))
        return {
            "kind": "plain",
            "n_train": n_tr,
            "tag": meta.get("experiment_name") or meta_p.stem.replace("_meta", ""),
            "best_step": _best_step_from_meta(meta, n_train=n_tr),
            "best_epoch": meta.get("best_epoch"),
            "source": str(meta_p),
        }
    d = V2 / f"n{n}_plain"
    summary = d / "campaign_summary.json"
    if summary.exists():
        j = _load_json(summary)
        return {
            "kind": "plain",
            "n_train": int(j.get("n_train", n)),
            "tag": j["tag"],
            "best_step": int(j["best_step"]),
            "best_epoch": j.get("best_epoch"),
            "source": str(summary),
        }
    metas = sorted(d.glob("*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No plain meta/summary under {d}")
    meta = _load_json(metas[0])
    n_tr = int(meta.get("n_train", n))
    return {
        "kind": "plain",
        "n_train": n_tr,
        "tag": metas[0].stem.replace("_meta", ""),
        "best_step": _best_step_from_meta(meta, n_train=n_tr),
        "best_epoch": meta.get("best_epoch"),
        "source": str(metas[0]),
    }


def resolve_physics_best_step(n: int) -> dict:
    if n == 2048:
        meta_p = DIAG / "filtered_c1_multires_lam4p5_meta.json"
        meta = _load_json(meta_p)
        n_tr = int(meta.get("n_train", n))
        return {
            "kind": "physics",
            "n_train": n_tr,
            "tag": meta.get("experiment_name") or meta_p.stem.replace("_meta", ""),
            "best_step": _best_step_from_meta(meta, n_train=n_tr),
            "best_epoch": meta.get("best_epoch"),
            "source": str(meta_p),
        }
    d = V2 / f"n{n}_physics"
    summary = d / "campaign_summary.json"
    if summary.exists():
        j = _load_json(summary)
        return {
            "kind": "physics",
            "n_train": int(j.get("n_train", n)),
            "tag": j["tag"],
            "best_step": int(j["best_step"]),
            "best_epoch": j.get("best_epoch"),
            "source": str(summary),
        }
    # Prefer extension meta, else any
    ext = sorted(d.glob("*_ext*_meta.json"))
    metas = ext or sorted(d.glob("*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No physics meta/summary under {d}")
    meta = _load_json(metas[-1])
    n_tr = int(meta.get("n_train", n))
    return {
        "kind": "physics",
        "n_train": n_tr,
        "tag": metas[-1].stem.replace("_meta", ""),
        "best_step": _best_step_from_meta(meta, n_train=n_tr),
        "best_epoch": meta.get("best_epoch"),
        "source": str(metas[-1]),
    }


rows_bs = [resolve_plain_best_step(n) for n in PLAIN_NS]
rows_bs += [resolve_physics_best_step(n) for n in PHYS_NS]
df_best = pd.DataFrame(rows_bs).sort_values(["kind", "n_train"]).reset_index(drop=True)

print("=== Best optimization step (official checkpoints) ===")
print(
    df_best[["kind", "n_train", "best_step", "best_epoch", "tag"]].to_string(
        index=False
    )
)
best_csv = OUT_FIG / "plain_vs_physics_best_step_vs_ntrain.csv"
df_best.to_csv(best_csv, index=False)
print("wrote", best_csv)

fig, ax = plt.subplots(figsize=(8.5, 5.0))
for kind, marker, color, label in [
    ("plain", "o", "C0", r"Plain ($\lambda=0$)"),
    ("physics", "s", "C1", r"Physics ($\lambda^*$)"),
]:
    sub = df_best[df_best["kind"] == kind].sort_values("n_train")
    ax.plot(
        sub["n_train"].to_numpy(),
        sub["best_step"].to_numpy(),
        f"-{marker}",
        color=color,
        lw=2,
        ms=8,
        label=label,
    )
ax.set_xscale("log")
ax.set_xlabel(r"$n_{\mathrm{train}}$")
ax.set_ylabel(r"best optimization step")
ax.set_title("Protocol v2: best_step vs $n_{\\mathrm{train}}$ (plain vs physics)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
fig_bs = OUT_FIG / "plain_vs_physics_best_step_vs_ntrain.png"
fig.savefig(fig_bs, dpi=160)
plt.show()
print("wrote", fig_bs)

# Same comparison with log-y (highlights decade-scale differences)
fig, ax = plt.subplots(figsize=(8.5, 5.0))
for kind, marker, color, label in [
    ("plain", "o", "C0", r"Plain ($\lambda=0$)"),
    ("physics", "s", "C1", r"Physics ($\lambda^*$)"),
]:
    sub = df_best[df_best["kind"] == kind].sort_values("n_train")
    ax.plot(
        sub["n_train"].to_numpy(),
        sub["best_step"].to_numpy(),
        f"-{marker}",
        color=color,
        lw=2,
        ms=8,
        label=label,
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$n_{\mathrm{train}}$")
ax.set_ylabel(r"best optimization step")
ax.set_title(
    r"Protocol v2: best_step vs $n_{\mathrm{train}}$ (log–log; plain vs physics)"
)
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
fig_bs_log = OUT_FIG / "plain_vs_physics_best_step_vs_ntrain_logy.png"
fig.savefig(fig_bs_log, dpi=160)
plt.show()
print("wrote", fig_bs_log)

merged_bs = (
    df_best[df_best["kind"] == "plain"][["n_train", "best_step", "tag"]]
    .rename(columns={"best_step": "best_step_plain", "tag": "tag_plain"})
    .merge(
        df_best[df_best["kind"] == "physics"][["n_train", "best_step", "tag"]].rename(
            columns={"best_step": "best_step_phys", "tag": "tag_phys"}
        ),
        on="n_train",
    )
)
print("\\n=== Matched-n best_step ===")
print(merged_bs.to_string(index=False))
"""
    )
)

cells.append(
    md(
        r"""## Estimated convergence rate vs SNR (Gaussian assumption)

From `PROMPT_estimate_convergence_from_snr_sweep.txt`, using SNR-sweep
**best-ambiguity SIMILARITY_ERROR** mean/std:

\[
\widehat C(s)=\Phi\!\left(\frac{\tau-\mu(s)}{\sigma(s)}\right)
=P\bigl(e\le\tau\bigr)
\quad\text{under }e\sim\mathcal N(\mu(s),\sigma(s)^2).
\]

Here \(\tau=0.05\) (SIM ≤ 5%). Artefacts from
`filtered_c1_multires_2k_diagnostics`:

- **Multires 2K physics**: `filtered_c1_multires_lam4p5_test_snr_sweep.npz`
- **Multires 60K plain** (\(\lambda=0\); only 60K sweep in that notebook):
  `filtered_c1_multires_60k_lam0_test_snr_sweep.npz`

This is an estimate from \((\mu,\sigma)\) only — not the empirical
per-pulse fraction.
"""
    )
)

cells.append(
    code(
        r"""from math import erf, sqrt


def phi_standard_normal(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def gaussian_convergence_rate(
    mu: np.ndarray,
    sig: np.ndarray,
    *,
    tau: float,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sig = np.asarray(sig, dtype=float)
    out = np.empty_like(mu, dtype=float)
    for i, (m, s) in enumerate(zip(mu, sig)):
        if s > 0.0:
            out[i] = phi_standard_normal((tau - m) / s)
        elif m < tau:
            out[i] = 1.0
        elif m > tau:
            out[i] = 0.0
        else:
            out[i] = 0.5
    return np.clip(out, 0.0, 1.0)


TAU_SIM = 0.05  # SIMILARITY_ERROR <= 5%

SWEEP_PHYS_2K = DIAG / "filtered_c1_multires_lam4p5_test_snr_sweep.npz"
SWEEP_PLAIN_60K = DIAG / "filtered_c1_multires_60k_lam0_test_snr_sweep.npz"

conv_specs = [
    ("phys_2k", SWEEP_PHYS_2K, "Multires 2K physics ($\\lambda{=}4.5$)", "s", "C1"),
    ("plain_60k", SWEEP_PLAIN_60K, "Multires 60K plain ($\\lambda{=}0$)", "o", "C0"),
]

rows_conv = []
series_conv = {}
for key, path, label, marker, color in conv_specs:
    assert path.exists(), path
    sw = load_cnn_sweep(path)
    snr = np.asarray(sw.snr_sweep_db, dtype=float)
    mu = np.asarray(sw.cnn_sim_amb_m, dtype=float)
    sig = np.asarray(sw.cnn_sim_amb_s, dtype=float)
    c_hat = gaussian_convergence_rate(mu, sig, tau=TAU_SIM)
    series_conv[key] = {
        "label": label,
        "marker": marker,
        "color": color,
        "snr_db": snr,
        "mu": mu,
        "sig": sig,
        "c_hat": c_hat,
        "path": path,
    }
    for s, m, sg, c in zip(snr, mu, sig, c_hat):
        rows_conv.append(
            {
                "model": key,
                "label": label,
                "snr_db": float(s),
                "sim_amb_mu": float(m),
                "sim_amb_sig": float(sg),
                "tau": TAU_SIM,
                "c_hat": float(c),
                "c_hat_pct": 100.0 * float(c),
                "sweep_path": str(path),
            }
        )

df_conv = pd.DataFrame(rows_conv)
print(
    f"=== Gaussian estimate: P(SIM_amb <= {TAU_SIM:g}) vs SNR ==="
)
print("(from cnn_sim_amb_m / cnn_sim_amb_s; continuous CDF => <= and < coincide)")
for key, *_ in conv_specs:
    sub = df_conv[df_conv["model"] == key]
    print(f"\n-- {series_conv[key]['label']} --")
    print(
        sub[["snr_db", "sim_amb_mu", "sim_amb_sig", "c_hat_pct"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

conv_csv = OUT_FIG / "gaussian_conv_sim_leq0p05_phys2k_plain60k.csv"
df_conv.to_csv(conv_csv, index=False)
print("\nwrote", conv_csv)

fig, ax = plt.subplots(figsize=(8.5, 5.0))
for key, *_rest in conv_specs:
    sc = series_conv[key]
    ax.plot(
        sc["snr_db"],
        100.0 * sc["c_hat"],
        f"-{sc['marker']}",
        color=sc["color"],
        lw=2,
        ms=8,
        label=sc["label"],
    )
ax.set_xlabel("SNR (dB)")
ax.set_ylabel(rf"estimated convergence (%); SIM$_{{\mathrm{{amb}}}}$ $\leq$ {TAU_SIM:g}")
ax.set_title(
    "Gaussian estimate of convergence vs SNR\n"
    r"(from SNR-sweep $\mu\pm\sigma$ of best-amb SIMILARITY_ERROR)"
)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
fig_conv = OUT_FIG / "gaussian_conv_sim_leq0p05_phys2k_plain60k_vs_snr.png"
fig.savefig(fig_conv, dpi=160)
plt.show()
print("wrote", fig_conv)
"""
    )
)

cells.append(
    md(
        r"""## Notes (protocol-v2 ladder)

- \(n=2048\) comes from `filtered_c1_multires_2k_diagnostics` (meta `n_train=2048`).
- Y-error bars on the main L1 comparison plot are **std across SNR points**.
- Extra filtered-C1 plots below (empirical convergence, train-to-105,
  random examples) use the diagnostics Multires checkpoints.
"""
    )
)

# ---------------------------------------------------------------------------
# Section A: empirical convergence (phys 2K + plain 60K) + inset best_step
# ---------------------------------------------------------------------------
cells.append(
    md(
        r"""# Filtered C1 extras (from diagnostics checkpoints)

## A. Empirical convergence vs SNR (SIM ≤ 5%)

Same idea as `multires_2k_noisy_trace_lambda_experiments.ipynb` §3b,
but on **filtered C1** with threshold \(\tau=0.05\):

- Multires **2K physics** (`filtered_c1_multires_lam4p5`)
- Multires **60K plain** (`filtered_c1_multires_60k_lam0`)

**Test:** 512 held-out filtered-C1 pulses (`seed=0`, same split recipe as
diagnostics). At each SNR, fraction with best-amb SIMILARITY_ERROR
**≤ 0.05**. Y-axis labeled `convergence (%)`; limits start at the
minimum curve value (not forced to 0).

Inset: bar chart of **best optimization step**
(`best_epoch * ceil(n_train / 64)`) for each model, colors matched to
curves. Cache: `figures/v2_comparisons/filtered_c1_conv_sim_leq0p05.npz`.
Set `FORCE_CONV_RECOMPUTE=True` to recompute.
"""
    )
)

cells.append(
    code(
        r"""import math

import torch

import data_c_amb_loss_diagnostics as diag
from dataset_utils import build_filtered_c1_frog_dataloaders
from evaluate_cnn import per_pulse_similarity_amb_cnn_at_snr
from frog_reconstruction_model import extract_pulse_prediction
from pulse_metrics import (
    best_l1_ambiguity,
    best_l1_ambiguity_field,
    best_similarity_error_ambiguity,
    prepare_frog_trace_for_plot,
    unpack_packed_field,
    unwrap_phases_for_overlay,
)
from trace_noise import add_trace_noise_awgn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRAIN_2K, N_VAL, N_TEST = 2048, 200, 512
BATCH_SIZE = 64
SEED = 0
SNR_SWEEP_DB = np.arange(-10.0, 31.0, 5.0)
TAU_SIM_EMP = 0.05
FORCE_CONV_RECOMPUTE = False

TAG_PHYS_2K = "filtered_c1_multires_lam4p5"
TAG_PLAIN_2K = "filtered_c1_multires_lam0"
TAG_PLAIN_60K = "filtered_c1_multires_60k_lam0"
TAG_PLAIN_2K_TO105 = "filtered_c1_multires_lam0_to105"

CONV_CACHE = OUT_FIG / "filtered_c1_conv_sim_leq0p05.npz"

COLOR_PHYS = "C1"
COLOR_60K = "C0"
COLOR_PLAIN = "C0"

print("device:", DEVICE)
print("DIAG:", DIAG.resolve())
"""
    )
)

cells.append(
    code(
        r"""def _best_step_from_meta(meta: dict, *, batch_size: int = BATCH_SIZE) -> int:
    if meta.get("best_step") is not None:
        return int(meta["best_step"])
    n_tr = int(meta["n_train"])
    spe = int(math.ceil(float(n_tr) / float(batch_size)))
    return int(meta["best_epoch"]) * spe


def empirical_convergence_leq(
    model,
    loader,
    snr_sweep_db,
    *,
    tau: float,
) -> np.ndarray:
    fracs = []
    for snr_db in snr_sweep_db:
        print(f"  convergence @ {float(snr_db):.1f} dB …", flush=True)
        per = per_pulse_similarity_amb_cnn_at_snr(model, loader, float(snr_db))
        fracs.append(float(np.mean(per <= float(tau))))
    return np.asarray(fracs, dtype=np.float64)


if FORCE_CONV_RECOMPUTE or not CONV_CACHE.exists():
    # Test split uses seed+2; n_train/n_val sizes do not change test pulses.
    print("Building filtered-C1 test loader (n_test=512)…")
    bundle = build_filtered_c1_frog_dataloaders(
        n_train=1,
        n_val=1,
        n_test=N_TEST,
        batch_size=BATCH_SIZE,
        seed=SEED,
        device=DEVICE,
        canonicalize_mode="t0",
    )
    test_loader_conv = bundle.test_loader
    print("Computing empirical convergence — physics 2K…")
    model_phys_c = diag.load_trained_multires(
        DIAG / f"{TAG_PHYS_2K}_model.pt", device=DEVICE
    )
    frac_phys = empirical_convergence_leq(
        model_phys_c, test_loader_conv, SNR_SWEEP_DB, tau=TAU_SIM_EMP
    )
    print("Computing empirical convergence — plain 60K…")
    model_60k_c = diag.load_trained_multires(
        DIAG / f"{TAG_PLAIN_60K}_model.pt", device=DEVICE
    )
    frac_60k = empirical_convergence_leq(
        model_60k_c, test_loader_conv, SNR_SWEEP_DB, tau=TAU_SIM_EMP
    )
    np.savez(
        CONV_CACHE,
        snr_sweep_db=SNR_SWEEP_DB,
        threshold=np.asarray([TAU_SIM_EMP]),
        n_test=np.asarray([N_TEST]),
        phys_2k=frac_phys,
        plain_60k=frac_60k,
        tag_phys=np.asarray([TAG_PHYS_2K]),
        tag_60k=np.asarray([TAG_PLAIN_60K]),
    )
    print("wrote", CONV_CACHE)
else:
    print("using cached", CONV_CACHE)

zconv = np.load(CONV_CACHE, allow_pickle=True)
snr_emp = np.asarray(zconv["snr_sweep_db"], dtype=float)
frac_phys_emp = np.asarray(zconv["phys_2k"], dtype=float)
frac_60k_emp = np.asarray(zconv["plain_60k"], dtype=float)
print("threshold=", float(zconv["threshold"][0]), "n_test=", int(zconv["n_test"][0]))
print("phys 2K %:", (100.0 * frac_phys_emp).round(2))
print("60K   %:", (100.0 * frac_60k_emp).round(2))

meta_phys_2k = json.loads((DIAG / f"{TAG_PHYS_2K}_meta.json").read_text(encoding="utf-8"))
meta_60k = json.loads((DIAG / f"{TAG_PLAIN_60K}_meta.json").read_text(encoding="utf-8"))
best_step_phys = _best_step_from_meta(meta_phys_2k)
best_step_60k = _best_step_from_meta(meta_60k)
print(
    f"best_step phys2k={best_step_phys} (best_epoch={meta_phys_2k.get('best_epoch')}), "
    f"60k={best_step_60k} (best_epoch={meta_60k.get('best_epoch')})"
)

y_phys_pct = 100.0 * frac_phys_emp
y_60k_pct = 100.0 * frac_60k_emp
y_min = float(min(y_phys_pct.min(), y_60k_pct.min()))
y_max = float(max(y_phys_pct.max(), y_60k_pct.max()))
pad = max(1.0, 0.05 * (y_max - y_min + 1e-9))

fig, ax = plt.subplots(figsize=(9.0, 5.2))
ax.plot(
    snr_emp,
    y_phys_pct,
    "-s",
    color=COLOR_PHYS,
    lw=2,
    ms=8,
    label=r"Multires 2K physics ($\lambda{=}4.5$)",
)
ax.plot(
    snr_emp,
    y_60k_pct,
    "-o",
    color=COLOR_60K,
    lw=2,
    ms=8,
    label=r"Multires 60K plain ($\lambda{=}0$)",
)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("convergence (%)")
ax.set_title(
    rf"Filtered C1 — empirical convergence (SIM$_{{\mathrm{{amb}}}}$ $\leq$ {TAU_SIM_EMP:g}, "
    f"n_test={N_TEST})"
)
ax.set_ylim(y_min - pad, min(105.0, y_max + pad))
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")

# Inset: best optimization steps
ax_in = ax.inset_axes([0.12, 0.55, 0.32, 0.38])
bars = ax_in.bar(
    [0, 1],
    [best_step_phys, best_step_60k],
    color=[COLOR_PHYS, COLOR_60K],
    width=0.65,
)
ax_in.set_xticks([0, 1], ["2K phys", "60K"])
ax_in.set_ylabel("# steps till convergence")
ax_in.set_title("best step", fontsize=9)
ax_in.grid(True, axis="y", alpha=0.3)
for rect, val in zip(bars, [best_step_phys, best_step_60k]):
    ax_in.text(
        rect.get_x() + rect.get_width() / 2.0,
        rect.get_height(),
        f"{int(val)}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

plt.tight_layout()
fig_emp = OUT_FIG / "filtered_c1_emp_conv_sim_leq0p05_phys2k_plain60k.png"
fig.savefig(fig_emp, dpi=160)
plt.show()
print("wrote", fig_emp)
"""
    )
)

# ---------------------------------------------------------------------------
# Section B: retrain plain 2K to epoch 105 + overlay histories
# ---------------------------------------------------------------------------
cells.append(
    md(
        r"""## B. Retrain Multires 2K plain to epoch 105 + history overlays

Diagnostics plain 2K stopped early (~58 epochs). Retrain with the **same**
filtered-C1 protocol as `filtered_c1_multires_2k_diagnostics_NB.ipynb`:

- \(n_{\mathrm{train}}=2048\), \(n_{\mathrm{val}}=200\), \(n_{\mathrm{test}}=512\), seed=0
- \(B=64\), LR \(=10^{-3}\), train/val SNR U[0,30], `loader_builder=filtered_c1`,
  `canonicalize_mode=t0`, `pulse_loss_mode=raw`, `lam=0`, legacy ambiguity

but `max_epochs=105` and large `patience` so training continues to epoch 105.
Artifacts use a **new tag** (does not overwrite original `lam0`):
`filtered_c1_multires_lam0_to105` under the diagnostics folder.

Then plot (physics curve from existing `lam4p5` history):
1. train pulse L1 **raw** (no best-amb)
2. train TRACE L1
3. val pulse L1 **best-amb**
4. **train improvement (same scale)**: pulse L1 raw and TRACE L1,
   each divided by its epoch-1 value (as in `data_c_amb_loss_diagnostics_NB`)

Set `FORCE_RETRAIN_PLAIN_TO105=True` to force a fresh run.
"""
    )
)

cells.append(
    code(
        r"""FORCE_RETRAIN_PLAIN_TO105 = False
MAX_EPOCHS_TO105 = 105
PATIENCE_TO105 = 10_000  # effectively disable early stop before max_epochs
LR = 1e-3
TRAIN_SNR = (0.0, 30.0)
VAL_SNR = (0.0, 30.0)

hist_path_to105 = DIAG / f"{TAG_PLAIN_2K_TO105}_history.npz"
if FORCE_RETRAIN_PLAIN_TO105 or not hist_path_to105.exists():
    print(
        f"Training {TAG_PLAIN_2K_TO105} (lam=0, max_epochs={MAX_EPOCHS_TO105})…"
    )
    result = diag.train_data_c_amb_diagnostics(
        pulse_loss_mode="raw",
        lam=0.0,
        n_train=N_TRAIN_2K,
        n_val=N_VAL,
        n_test=N_TEST,
        batch_size=BATCH_SIZE,
        seed=SEED,
        max_epochs=MAX_EPOCHS_TO105,
        patience=PATIENCE_TO105,
        lr=LR,
        train_snr_db_range=TRAIN_SNR,
        val_snr_db_range=VAL_SNR,
        device=DEVICE,
        verbose=True,
        ambiguity_backend="legacy",
        trace_loss_ref="clean",
        loader_builder="filtered_c1",
        canonicalize_mode="t0",
    )
    result_save = {k: v for k, v in result.items() if k != "bundle"}
    diag.save_run_artifacts(result_save, DIAG, TAG_PLAIN_2K_TO105)
    print(
        "saved",
        TAG_PLAIN_2K_TO105,
        "best_epoch=",
        result["best_epoch"],
        "stopped_epoch=",
        result.get("stopped_epoch"),
    )
else:
    print("skip train; using", hist_path_to105)

hist_plain_to105 = diag.load_history(hist_path_to105)
hist_phys_diag = diag.load_history(DIAG / f"{TAG_PHYS_2K}_history.npz")
meta_to105 = json.loads(
    (DIAG / f"{TAG_PLAIN_2K_TO105}_meta.json").read_text(encoding="utf-8")
)
print("plain_to105 meta best_epoch=", meta_to105.get("best_epoch"),
      "stopped=", meta_to105.get("stopped_epoch"),
      "n_epochs_hist=", len(hist_plain_to105["train_pulse_l1_raw"]))
print("phys hist epochs=", len(hist_phys_diag["train_pulse_l1_raw"]))

# 1) train pulse L1 raw
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(
    hist_phys_diag["train_pulse_l1_raw"],
    color=COLOR_PHYS,
    lw=2,
    label=r"Physics 2K ($\lambda{=}4.5$)",
)
ax.plot(
    hist_plain_to105["train_pulse_l1_raw"],
    color=COLOR_PLAIN,
    lw=2,
    label=r"Plain 2K ($\lambda{=}0$), to epoch 105",
)
ax.set_xlabel("epoch")
ax.set_ylabel("train pulse L1 (raw)")
ax.set_title("Filtered C1 Multires 2K — train pulse L1 (no best-amb)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
p1 = OUT_FIG / "filtered_c1_train_pulse_l1_raw_phys_vs_plain_to105.png"
fig.savefig(p1, dpi=160)
plt.show()
print("wrote", p1)

# 2) train TRACE L1
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(
    hist_phys_diag["train_trace_l1"],
    color=COLOR_PHYS,
    lw=2,
    label=r"Physics 2K ($\lambda{=}4.5$)",
)
ax.plot(
    hist_plain_to105["train_trace_l1"],
    color=COLOR_PLAIN,
    lw=2,
    label=r"Plain 2K ($\lambda{=}0$), to epoch 105",
)
ax.set_xlabel("epoch")
ax.set_ylabel("train TRACE L1")
ax.set_title("Filtered C1 Multires 2K — train TRACE L1")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
p2 = OUT_FIG / "filtered_c1_train_trace_l1_phys_vs_plain_to105.png"
fig.savefig(p2, dpi=160)
plt.show()
print("wrote", p2)

# 3) val pulse L1 best-amb
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(
    hist_phys_diag["val_pulse_l1_amb"],
    color=COLOR_PHYS,
    lw=2,
    label=r"Physics 2K ($\lambda{=}4.5$)",
)
ax.plot(
    hist_plain_to105["val_pulse_l1_amb"],
    color=COLOR_PLAIN,
    lw=2,
    label=r"Plain 2K ($\lambda{=}0$), to epoch 105",
)
ax.set_xlabel("epoch")
ax.set_ylabel("val pulse L1 (best-amb)")
ax.set_title("Filtered C1 Multires 2K — val pulse L1 (best-amb)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
p3 = OUT_FIG / "filtered_c1_val_pulse_l1_amb_phys_vs_plain_to105.png"
fig.savefig(p3, dpi=160)
plt.show()
print("wrote", p3)

# Train improvement on a shared relative scale (÷ epoch-1), as in
# data_c_amb_loss_diagnostics_NB: pulse raw vs TRACE
def _epochs(y):
    return np.arange(1, len(y) + 1)


for hist, name, fname in [
    (
        hist_phys_diag,
        r"Multires 2K physics ($\lambda{=}4.5$)",
        "filtered_c1_phys2k_train_improvement_same_scale.png",
    ),
    (
        hist_plain_to105,
        r"Multires 2K plain ($\lambda{=}0$), to epoch 105",
        "filtered_c1_plain2k_to105_train_improvement_same_scale.png",
    ),
]:
    pulse = np.asarray(hist["train_pulse_l1_raw"], dtype=float)
    trace = np.asarray(hist["train_trace_l1"], dtype=float)
    ep = _epochs(pulse)
    pulse_rel = pulse / max(float(pulse[0]), 1e-8)
    trace_rel = trace / max(float(trace[0]), 1e-8)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, pulse_rel, "-", label="train pulse L1 (raw) / start")
    ax.plot(ep, trace_rel, "-", label="train TRACE L1 / start")
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("error / error at epoch 1")
    ax.set_title(f"{name}: train improvement (same scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    outp = OUT_FIG / fname
    fig.savefig(outp, dpi=160)
    plt.show()
    print("wrote", outp)
    print(
        f"  final pulse_rel={pulse_rel[-1]:.3f}, trace_rel={trace_rel[-1]:.3f} "
        f"(n_epochs={len(ep)})"
    )
"""
    )
)

# ---------------------------------------------------------------------------
# Section C: random example @ SNR=10 dB
# ---------------------------------------------------------------------------
cells.append(
    md(
        r"""## C. Random example @ SNR = 10 dB

Each run draws a **new** held-out filtered-C1 test index, adds AWGN at
10 dB, and shows clean/noisy TRACE plus \(|E(t)|\) / phase for:

- Multires 2K physics (`lam4p5`)
- Multires 2K plain (`lam0`)
- Multires 60K plain (`60k_lam0`)

Reconstructions are aligned with `best_l1_ambiguity_field`.
"""
    )
)

cells.append(
    code(
        r"""EXAMPLE_SNR_DB = 10.0

print("Building filtered-C1 loaders for example cell…")
bundle_ex = build_filtered_c1_frog_dataloaders(
    n_train=1,
    n_val=1,
    n_test=N_TEST,
    batch_size=BATCH_SIZE,
    seed=SEED,
    device=DEVICE,
    canonicalize_mode="t0",
)
test_loader_ex = bundle_ex.test_loader
t_axis = np.asarray(bundle_ex.t_vec, dtype=float)
w_vec = np.asarray(bundle_ex.w_vec, dtype=float)
dt = float(t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 1.0

_test_ds = test_loader_ex.dataset
EXAMPLE_INDEX = int(np.random.randint(0, len(_test_ds)))
I_clean, E_true_packed = _test_ds[EXAMPLE_INDEX]
I_clean = I_clean.to(DEVICE)
E_true_packed = E_true_packed.to(DEVICE)
I_noisy = add_trace_noise_awgn(I_clean.unsqueeze(0), EXAMPLE_SNR_DB).squeeze(0)
print(f"Drew test sample index {EXAMPLE_INDEX} / {len(_test_ds) - 1}")

model_phys = diag.load_trained_multires(DIAG / f"{TAG_PHYS_2K}_model.pt", device=DEVICE)
model_plain = diag.load_trained_multires(DIAG / f"{TAG_PLAIN_2K}_model.pt", device=DEVICE)
model_plain_60k = diag.load_trained_multires(
    DIAG / f"{TAG_PLAIN_60K}_model.pt", device=DEVICE
)

with torch.no_grad():
    x = I_noisy.unsqueeze(0).unsqueeze(0)
    E_phys = extract_pulse_prediction(model_phys(x)).squeeze(0).cpu().numpy()
    E_plain = extract_pulse_prediction(model_plain(x)).squeeze(0).cpu().numpy()
    E_plain_60k = extract_pulse_prediction(model_plain_60k(x)).squeeze(0).cpu().numpy()

e_true = unpack_packed_field(E_true_packed.detach().cpu().numpy())
raw_fields = {
    r"Physics 2K ($\lambda{=}4.5$)": unpack_packed_field(E_phys),
    r"Plain 2K ($\lambda{=}0$)": unpack_packed_field(E_plain),
    r"Plain 60K ($\lambda{=}0$)": unpack_packed_field(E_plain_60k),
}
reconstructions = {
    name: best_l1_ambiguity_field(e_raw, e_true) for name, e_raw in raw_fields.items()
}

print(f"\nExample sample {EXAMPLE_INDEX} @ SNR={EXAMPLE_SNR_DB:.0f} dB")
print(f"{'Algorithm':<28} {'L1_amb':>10} {'SIM_amb':>10}")
print("-" * 50)
for name, e_raw in raw_fields.items():
    l1_amb = best_l1_ambiguity(e_raw, e_true)
    sim_amb = best_similarity_error_ambiguity(e_raw, e_true)
    print(f"{name:<28} {l1_amb:10.4f} {sim_amb:10.4f}")

trace_c, tau_axis, omega_plot = prepare_frog_trace_for_plot(
    I_clean.detach().cpu().numpy(), num_points=len(t_axis), dt=dt
)
trace_n, _, _ = prepare_frog_trace_for_plot(
    I_noisy.detach().cpu().numpy(), num_points=len(t_axis), dt=dt
)
energy_plot = omega_plot * 4.135667696 / (2.0 * np.pi)

# Combined figure: TRACE + overlays for all three nets
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
im0 = axes[0, 0].imshow(
    trace_c,
    origin="lower",
    aspect="auto",
    extent=[tau_axis[0], tau_axis[-1], energy_plot[0], energy_plot[-1]],
    cmap="magma",
)
axes[0, 0].set_title("Clean TRACE")
axes[0, 0].set_xlabel("Delay τ [fs]")
axes[0, 0].set_ylabel("Relative energy [eV]")
fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

im1 = axes[0, 1].imshow(
    trace_n,
    origin="lower",
    aspect="auto",
    extent=[tau_axis[0], tau_axis[-1], energy_plot[0], energy_plot[-1]],
    cmap="magma",
)
axes[0, 1].set_title(f"Noisy TRACE ({EXAMPLE_SNR_DB:.0f} dB)")
axes[0, 1].set_xlabel("Delay τ [fs]")
axes[0, 1].set_ylabel("Relative energy [eV]")
fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

axes[1, 0].plot(t_axis, np.abs(e_true), "k-", lw=2.2, label="true")
for name, e_aligned in reconstructions.items():
    axes[1, 0].plot(t_axis, np.abs(e_aligned), lw=1.4, label=name)
axes[1, 0].set_title("|E(t)| after best-amb")
axes[1, 0].set_xlabel("Time [fs]")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

_ph_true_ref = None
for name, e_aligned in reconstructions.items():
    ph_true, ph_rec = unwrap_phases_for_overlay(e_aligned, e_true)
    if _ph_true_ref is None:
        _ph_true_ref = ph_true
        axes[1, 1].plot(t_axis, ph_true, "k-", lw=2.2, label="true")
    axes[1, 1].plot(t_axis, ph_rec, lw=1.4, label=name)
axes[1, 1].set_title("phase(E(t)) after best-amb")
axes[1, 1].set_xlabel("Time [fs]")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(
    f"Filtered C1 | sample {EXAMPLE_INDEX}, SNR={EXAMPLE_SNR_DB:.0f} dB",
    fontweight="bold",
)
plt.tight_layout()
fig_ex = OUT_FIG / f"filtered_c1_example_snr10_sample{EXAMPLE_INDEX}.png"
fig.savefig(fig_ex, dpi=140)
plt.show()
print("wrote", fig_ex)
"""
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "cells": cells,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("Wrote", OUT, "cells=", len(cells))
