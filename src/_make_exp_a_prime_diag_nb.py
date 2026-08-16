"""Generate exp_a_prime_diagnostics.ipynb"""
import json
from pathlib import Path

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": [],
}


def md(s: str) -> None:
    lines = s.strip("\n").split("\n")
    nb["cells"].append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [ln + "\n" for ln in lines],
        }
    )


def code(s: str) -> None:
    lines = s.strip("\n").split("\n")
    nb["cells"].append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [ln + "\n" for ln in lines],
        }
    )


md(
    """# Exp A′ diagnostics — why Multires fails on complex C1 + padded FROG

Follows Karpathy *Recipe* + Google DL Tuning Playbook:
isolate **one scientific factor at a time**, trust the pipeline first, then bias–variance, then ablations **inside** the A′ family.

## Fixed protocol
- **λ = 0** (pulse L1 only; no physics/trace loss)
- Train SNR ~ **U[0, 30] dB** (unchanged)
- **No T sweeps** (T = 53 fs; σ parameters stay T-calibrated / recalibrated only when `N_spikes` changes)
- Primary metric: **high-SNR** reconstruction (30 dB), not noise robustness

## Phases
| Phase | Goal |
|-------|------|
| 0 | Pipeline trust (overfit mini-batch, input-independent, loss@init) |
| 1 | Data / ambiguity / forward consistency |
| 2 | Bias–variance (`n_train`, capacity) |
| 3 | Ablations: FROG padded vs plain, `t*` vs `t0`, `N_spikes` |

Helpers: `exp_a_prime_diagnostics_lib.py`  
Artifacts: `checkpoints/benchmark/exp_a_prime_diagnostics/`
"""
)

code(
    """from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

SRC = Path.cwd()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import exp_a_prime_diagnostics_lib as diag

OUT = diag.ensure_out_dir()
print("device:", diag.get_device())
print("OUT:", OUT)

# Toggle which blocks to (re)run. Cached JSON/PT skips work unless FORCE=True.
FORCE = False
RUN_PHASE0 = True
RUN_PHASE1 = True
RUN_PHASE2 = True
RUN_PHASE3 = True
NTRAIN_SIZES = [512, 2048, 8192]  # edit freely
"""
)

md("## 0. Pipeline trust")

code(
    """if RUN_PHASE0:
    p0a = diag.phase0a_overfit_minibatch(force=FORCE)
    print("0a overfit near-zero?", p0a.get("ok_near_zero"), "final_L1=", p0a.get("final_train_l1"))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(p0a["train_losses"])
    ax.set_xlabel("epoch (subsampled)")
    ax.set_ylabel("train L1")
    ax.set_title("0a mini-batch overfit")
    ax.grid(True, alpha=0.3)
    plt.show()
else:
    print("skip phase 0a")
"""
)

code(
    """if RUN_PHASE0:
    p0b = diag.phase0b_input_independent(force=FORCE)
    p0c = diag.phase0c_loss_at_init()
    print("0b real better than zero-input?", p0b["real_better"])
    print("0c init L1 vs zero-pred:", p0c)
else:
    print("skip phase 0b/0c")
"""
)

md("## 1. Data stats, ambiguity, forward consistency")

code(
    """if RUN_PHASE1:
    p1a = diag.phase1a_pulse_stats()
    p1b = diag.phase1b_ambiguity_probe()
else:
    print("skip phase 1a/1b")
"""
)

md(
    """## 2. Bias–variance: `n_train` and capacity

If high-SNR error **falls with more data** → data-hungry / variance.  
If it **plateaus** → model bias or information/ambiguity limit.
"""
)

code(
    """if RUN_PHASE2:
    r_n = diag.run_phase2_ntrain_sweep(NTRAIN_SIZES, force=FORCE)
    print(diag.summarize_results(r_n))
    xs = [r.meta["n_train"] for r in r_n]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, [r.best_val_l1 for r in r_n], "o-", label="best val L1 @15dB")
    ax.plot(xs, [r.high_snr_l1_amb_mean for r in r_n], "s-", label="test L1_amb @30dB")
    ax.set_xscale("log")
    ax.set_xlabel("n_train")
    ax.set_ylabel("L1")
    ax.set_title("Phase 2 — n_train sweep (λ=0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
else:
    print("skip phase 2 n_train")
"""
)

code(
    """if RUN_PHASE2:
    r_c = diag.run_phase2_capacity_sweep(n_train=2048, force=FORCE)
    print(diag.summarize_results(r_c))
    labels = [r.meta["filters_per_branch"] for r in r_c]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(r_c))
    ax.bar(x - 0.15, [r.best_val_l1 for r in r_c], width=0.3, label="val L1")
    ax.bar(x + 0.15, [r.high_snr_l1_amb_mean for r in r_c], width=0.3, label="L1_amb@30dB")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in labels], rotation=15)
    ax.set_title("Phase 2 — capacity (n_train=2048)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.show()
else:
    print("skip phase 2 capacity")
"""
)

code(
    """# Forward consistency needs a trained 2K checkpoint from phase 2
if RUN_PHASE1:
    p1c = diag.phase1c_forward_consistency(
        train_name="phase2_ntrain_2048", force_train_if_missing=True
    )
    print(
        "Interpretation: high pulse error + low relative trace error → ambiguity; "
        "both high → weak map learning"
    )
else:
    print("skip phase 1c")
"""
)

md(
    """## 3. Single-factor ablations (A′ family only)

Each run changes **one** of: FROG mode, canonicalization, `N_spikes` (T fixed).
"""
)

code(
    """if RUN_PHASE3:
    r3 = diag.run_phase3_ablations(n_train=2048, force=FORCE)
    print(diag.summarize_results(r3))
    names = [r.name.replace("phase3_", "").replace("_n2048", "") for r in r3]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(r3))
    ax.bar(x, [r.high_snr_l1_amb_mean for r in r3], color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("test L1_amb @ 30 dB")
    ax.set_title("Phase 3 ablations")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()
else:
    print("skip phase 3")
"""
)

md("## Summary table")

code(
    """rows = []
for p in sorted(OUT.glob("phase*.json")):
    if p.name.startswith("phase0") or p.name.startswith("phase1"):
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    if "high_snr_l1_amb_mean" not in d:
        continue
    rows.append(
        (
            p.stem,
            d.get("best_val_l1"),
            d.get("high_snr_l1_amb_mean"),
            d.get("high_snr_sim_amb_mean"),
            d.get("best_epoch"),
        )
    )

hdr = f"{'run':42s} {'valL1':>8s} {'hiL1':>8s} {'hiSim':>8s} {'ep':>4s}"
print(hdr)
for r in rows:
    print(f"{r[0]:42s} {r[1]:8.4f} {r[2]:8.4f} {r[3]:8.4f} {r[4]:4d}")

print("\\nArtifacts in", OUT)
"""
)

md(
    """## How to read outcomes (next actions)

| Finding | Likely cause | Next fix direction |
|---------|--------------|--------------------|
| 0a cannot overfit | pipeline / loss / packing bug | fix before anything else |
| 0b zero ≈ real | label leak or dead input path | fix data wiring |
| more `n_train` helps a lot | variance / under-data | scale data (+ capacity) |
| capacity helps, data less | architectural bias | wider Multires / other arch |
| plain FROG ≪ padded error | spectral padding / crop issue | revisit FROGNetPadded settings |
| `t0` ≪ `t*` | peak canonicalization hurts regression | switch / rethink gauge |
| spikes 30 ≪ 300 | degrees-of-freedom / interference | capacity, representation, or better inductive bias — **not** "just use simple pulses" |
| FROG(Ê) good, Ê bad | inverse ambiguity | ambiguity-aware loss / constraints (later; still λ=0 here) |
"""
)

path = Path(__file__).resolve().parent / "exp_a_prime_diagnostics.ipynb"
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:03d}"
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", path, "cells", len(nb["cells"]))
