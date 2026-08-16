import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None
print("DRIVER_START", flush=True)

# --- cell ---

import os
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
import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path.cwd()
if (HERE / "setup_src_path.py").exists():
    sys.path.insert(0, str(HERE))
    import setup_src_path  # noqa: F401
    EXP = HERE
elif (HERE / "data efficiency experiment" / "setup_src_path.py").exists():
    EXP = HERE / "data efficiency experiment"
    sys.path.insert(0, str(EXP))
    import setup_src_path  # noqa: F401
else:
    raise RuntimeError("Run from src/ or from data efficiency experiment/")

import data_c_amb_loss_diagnostics as diag
from data_generation import filtered_c1_pulse_config
from dataset_utils import build_filtered_c1_frog_dataloaders
from evaluate_cnn import load_cnn_sweep
from frog_reconstruction_model import extract_pulse_prediction
from pulse_metrics import (
    best_l1_ambiguity,
    best_l1_ambiguity_field,
    best_similarity_error_ambiguity,
    prepare_frog_trace_for_plot,
    unpack_packed_field,
    unwrap_phases_for_overlay,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = EXP / "checkpoints" / "n2498_physics"
OUT.mkdir(parents=True, exist_ok=True)
PLAIN_OUT = EXP / "checkpoints" / "n2498_plain"

N_TRAIN = 2498
N_VAL = 200
N_TEST = 512
BATCH_SIZE = 300
SEED = 0  # shared val/test with plain
TRACE_SCALE = 8.0
EARLY_STOP_MODE = "steps"
TRAIN_SNR = (0.0, 30.0)
VAL_SNR = (0.0, 30.0)
SNR_SWEEP_DB = np.arange(-10.0, 31.0, 5.0)
# MUST match plain_multires_n2498_NB snapshots
SNAPSHOT_EVERY_FINAL = 2
SNAPSHOT_SNR_DB = 10.0
SNAPSHOT_VAL_INDEX = 0
SNAPSHOT_NOISE_SEED = 12345

STEPS_PER_EPOCH = math.ceil(N_TRAIN / BATCH_SIZE)  # K
PATIENCE_STEPS = 25 * STEPS_PER_EPOCH
MAX_STEPS_SCREEN = 100 * STEPS_PER_EPOCH
MAX_STEPS_FINAL = 200 * STEPS_PER_EPOCH
MAX_EPOCHS_SCREEN = max(1, math.ceil(MAX_STEPS_SCREEN / STEPS_PER_EPOCH))
MAX_EPOCHS_FINAL = max(1, math.ceil(MAX_STEPS_FINAL / STEPS_PER_EPOCH))

# Small-n prior (§5.2): start coarse at lambda >= 3
LAMBDA_COARSE = [3.0, 6.0, 12.0, 15.0]

# LR* from plain campaign
_plain_summary = PLAIN_OUT / "campaign_summary.json"
assert _plain_summary.exists(), f"Missing plain summary: {_plain_summary}"
_plain = json.loads(_plain_summary.read_text(encoding="utf-8"))
LR_STAR = float(_plain["LR_star"])

FORCE_RETRAIN = True
FORCE_TEST_SWEEP = True

print("device:", DEVICE)
print("EXP:", EXP.resolve())
print("OUT:", OUT.resolve())
print("plain summary:", _plain_summary.resolve())
print("LR* (from plain):", LR_STAR)
print(
    f"n_train={N_TRAIN}  B={BATCH_SIZE}  K={STEPS_PER_EPOCH}  "
    f"patience_steps={PATIENCE_STEPS} (=25 ep)  "
    f"screen max_steps={MAX_STEPS_SCREEN} (=100 ep)  "
    f"final max_steps={MAX_STEPS_FINAL} (=200 ep)"
)
print("trace_scale:", TRACE_SCALE)
print("lambda coarse:", LAMBDA_COARSE)
print(
    f"snapshots: val_index={SNAPSHOT_VAL_INDEX}  "
    f"snr={SNAPSHOT_SNR_DB:g}  noise_seed={SNAPSHOT_NOISE_SEED}"
)
print(f"early_stop: {EARLY_STOP_MODE}")


# --- cell ---

def lam_tag(lam: float) -> str:
    s = f"{float(lam):g}".replace(".", "p").replace("-", "m")
    return f"n{N_TRAIN}_phys_lam{s}"


def load_meta(tag: str) -> dict:
    return json.loads((OUT / f"{tag}_meta.json").read_text(encoding="utf-8"))


def train_physics(
    tag: str,
    *,
    lam: float,
    lr: float,
    snapshot_every: int,
    role: str,
    max_steps: int,
    max_epochs: int,
):
    hist_path = OUT / f"{tag}_history.npz"
    if hist_path.exists() and not FORCE_RETRAIN:
        print(f"skip {role}; using", hist_path)
        return None
    print(
        f"=== {role}: {tag}  lam={lam:g}  lr={lr:g}  "
        f"early_stop={EARLY_STOP_MODE}(patience_steps={PATIENCE_STEPS})  "
        f"max_steps={max_steps}  snap_every={snapshot_every} ==="
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
        max_epochs=int(max_epochs),
        patience=int(PATIENCE_STEPS),
        lr=float(lr),
        train_snr_db_range=TRAIN_SNR,
        val_snr_db_range=VAL_SNR,
        device=DEVICE,
        verbose=True,
        ambiguity_backend="legacy",
        trace_loss_ref="clean",
        loader_builder="filtered_c1",
        canonicalize_mode="t0",
        max_steps=int(max_steps),
        fixed_trace_scale=TRACE_SCALE,
        snapshot_every=int(snapshot_every),
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
    print(json.dumps(meta, indent=2))
    return result


def summarize_timings(hist, name):
    keys = [
        "timing_data_prep_sec",
        "timing_loss_data_fwd_sec",
        "timing_loss_reg_fwd_sec",
        "timing_total_backward_sec",
        "timing_optimizer_step_sec",
    ]
    print(f"=== {name}: mean±std over epochs (ms/batch) ===")
    for k in keys:
        if k not in hist:
            continue
        x = np.asarray(hist[k], dtype=float) * 1e3
        print(f"  {k.replace('timing_', ''):22s}  {x.mean():7.3f} ± {x.std():6.3f}")


def plot_pulse_curves(hist, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["train_pulse_l1_raw"], label="raw")
    axes[0].plot(hist["train_pulse_l1_amb"], label="best-amb")
    axes[0].set_title(f"{title_prefix}: train pulse L1")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(hist["val_pulse_l1_raw"], label="raw")
    axes[1].plot(hist["val_pulse_l1_amb"], label="best-amb")
    axes[1].set_title(f"{title_prefix}: val pulse L1 (SNR~U[0,30])")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_trace_curves(hist, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["train_trace_l1"])
    axes[0].set_title(f"{title_prefix}: train TRACE L1")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(hist["val_trace_l1"])
    axes[1].set_title(f"{title_prefix}: val TRACE L1 (SNR~U[0,30])")
    axes[1].set_xlabel("epoch")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_grad_norms(hist, title_prefix, lam=0.0):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(hist["grad_norm_data"], label=r"$||\nabla L_{\mathrm{data}}||$")
    ax.semilogy(hist["grad_norm_reg"], label=r"$||\nabla L_{\mathrm{reg}}||$")
    if "grad_norm_total" in hist and len(hist["grad_norm_total"]):
        ax.semilogy(hist["grad_norm_total"], label=r"$||\nabla L_{\mathrm{total}}||$")
    ax.set_xlabel("epoch")
    ax.set_title(f"{title_prefix}: gradient norms (λ={lam:g})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# --- cell ---

def screen_one_lam(lam: float, *, role: str = "lam_coarse") -> dict:
    tag = lam_tag(lam) + "_screen"
    train_physics(
        tag,
        lam=lam,
        lr=LR_STAR,
        snapshot_every=0,
        role=role,
        max_steps=MAX_STEPS_SCREEN,
        max_epochs=MAX_EPOCHS_SCREEN,
    )
    hist = diag.load_history(OUT / f"{tag}_history.npz")
    meta = load_meta(tag)
    plot_pulse_curves(hist, f"λ screen λ={lam:g}")
    plot_trace_curves(hist, f"λ screen λ={lam:g}")
    return {
        "lam": float(lam),
        "tag": tag,
        "role": role,
        "best_score": float(meta["best_score"]),
        "best_epoch": int(meta["best_epoch"]),
        "best_step": int(meta.get("best_step", -1)),
        "stopped_epoch": int(meta["stopped_epoch"]),
        "global_step": meta.get("global_step"),
        "wall_time_train_sec": meta.get("wall_time_train_sec"),
        "trace_scale": meta.get("trace_scale"),
    }


coarse_rows = []
for lam in LAMBDA_COARSE:
    coarse_rows.append(screen_one_lam(lam, role="lam_coarse"))

best_c = min(coarse_rows, key=lambda r: float(r["best_score"]))
LAM_COARSE_STAR = float(best_c["lam"])

fig, ax = plt.subplots(figsize=(8, 4))
for r in sorted(coarse_rows, key=lambda x: x["lam"]):
    h = diag.load_history(OUT / f"{r['tag']}_history.npz")
    ax.plot(h["val_pulse_l1_amb"], label=f"λ={r['lam']:g}")
ax.set_xlabel("epoch")
ax.set_ylabel("val pulse L1 (best-amb)")
ax.set_title("Coarse λ screen overlay — val best-amb")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n=== Coarse λ screening summary ===")
for r in sorted(coarse_rows, key=lambda x: x["lam"]):
    print(
        f"  λ={r['lam']:g}  best_score={r['best_score']:.6f}  "
        f"best_epoch={r['best_epoch']}  best_step={r['best_step']}  "
        f"stopped={r['stopped_epoch']}"
    )
print(f"\ncoarse λ* = {LAM_COARSE_STAR:g}")


# --- cell ---

def fine_candidates(lam_star: float, coarse_grid: list[float]) -> list[float]:
    g = sorted(float(x) for x in coarse_grid)
    # find neighbors in the coarse grid
    if lam_star <= g[0] + 1e-15:
        # edge: bisect toward next interior point only
        return [float(math.sqrt(g[0] * g[1]))]
    if lam_star >= g[-1] - 1e-15:
        return [float(math.sqrt(g[-2] * g[-1]))]
    i = min(range(len(g)), key=lambda j: abs(g[j] - lam_star))
    lo = g[i - 1] if i > 0 else None
    hi = g[i + 1] if i < len(g) - 1 else None
    out = []
    if lo is not None:
        out.append(float(math.sqrt(lo * lam_star)))
    if hi is not None:
        out.append(float(math.sqrt(lam_star * hi)))
    return out


fine_lams = fine_candidates(LAM_COARSE_STAR, LAMBDA_COARSE)
# drop near-duplicates of already screened
existing = {float(r["lam"]) for r in coarse_rows}
fine_lams = [
    lam
    for lam in fine_lams
    if all(abs(lam - e) / max(e, 1e-12) > 1e-6 for e in existing)
]
print("Fine λ candidates:", fine_lams)

fine_rows = []
for lam in fine_lams:
    fine_rows.append(screen_one_lam(lam, role="lam_fine"))

all_rows = coarse_rows + fine_rows
best = min(all_rows, key=lambda r: float(r["best_score"]))
LAM_STAR = float(best["lam"])

fig, ax = plt.subplots(figsize=(8, 4))
for r in sorted(all_rows, key=lambda x: x["lam"]):
    h = diag.load_history(OUT / f"{r['tag']}_history.npz")
    ax.plot(h["val_pulse_l1_amb"], label=f"λ={r['lam']:g}")
ax.set_xlabel("epoch")
ax.set_ylabel("val pulse L1 (best-amb)")
ax.set_title("All λ screens overlay — val best-amb")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n=== All λ screening summary ===")
for r in sorted(all_rows, key=lambda x: x["lam"]):
    print(
        f"  λ={r['lam']:g}  role={r['role']}  best_score={r['best_score']:.6f}  "
        f"best_epoch={r['best_epoch']}  stopped={r['stopped_epoch']}"
    )
print(f"\nλ* = {LAM_STAR:g}")
(OUT / "lambda_screen_summary.json").write_text(
    json.dumps(
        {
            "LR_STAR": LR_STAR,
            "LAM_STAR": LAM_STAR,
            "TRACE_SCALE": TRACE_SCALE,
            "coarse_rows": coarse_rows,
            "fine_rows": fine_rows,
            "rows": all_rows,
        },
        indent=2,
    ),
    encoding="utf-8",
)


# --- cell ---

TAG_FINAL = lam_tag(LAM_STAR) + "_final"
train_physics(
    TAG_FINAL,
    lam=LAM_STAR,
    lr=LR_STAR,
    snapshot_every=SNAPSHOT_EVERY_FINAL,
    role="final",
    max_steps=MAX_STEPS_FINAL,
    max_epochs=MAX_EPOCHS_FINAL,
)
hist = diag.load_history(OUT / f"{TAG_FINAL}_history.npz")
meta = load_meta(TAG_FINAL)
print(json.dumps(meta, indent=2))


# --- cell ---

plot_pulse_curves(hist, f"Physics n={N_TRAIN} λ*={LAM_STAR:g} LR*={LR_STAR:g}")
plot_trace_curves(hist, f"Physics n={N_TRAIN} λ*={LAM_STAR:g} LR*={LR_STAR:g}")
plot_grad_norms(hist, f"Physics n={N_TRAIN} λ*={LAM_STAR:g}", lam=LAM_STAR)
summarize_timings(hist, f"Physics n={N_TRAIN} final")
print(
    f"wall_time_data_sec={meta.get('wall_time_data_sec')}  "
    f"wall_time_train_sec={meta.get('wall_time_train_sec')}  "
    f"global_step={meta.get('global_step')}  trace_scale={meta.get('trace_scale')}"
)


# --- cell ---

snap_path = OUT / f"{TAG_FINAL}_snapshots.npz"
assert snap_path.exists(), snap_path
snap = np.load(snap_path)
epochs = snap["epoch"]
E_pred_all = snap["E_pred"]
if E_pred_all.ndim == 3:
    E_pred_all = E_pred_all[:, 0, :]
E_true = snap["E_true"]
if E_true.ndim == 3:
    E_true = E_true[0]
elif E_true.ndim == 2 and E_true.shape[0] == 1:
    E_true = E_true[0]
I_noisy = snap["I_noisy"]
if I_noisy.ndim == 4:
    I_noisy = I_noisy[0, 0]
elif I_noisy.ndim == 3:
    I_noisy = I_noisy[0]

pulse_cfg = filtered_c1_pulse_config(n=64)
dt = float(pulse_cfg.dt)
t_axis = np.linspace(-pulse_cfg.t_total_fs / 2.0, pulse_cfg.t_total_fs / 2.0, pulse_cfg.n)
e_true = unpack_packed_field(E_true)
I_show, tau_axis, omega_plot = prepare_frog_trace_for_plot(
    I_noisy, num_points=pulse_cfg.n, dt=dt
)
energy_ev = omega_plot * 4.135667696 / (2.0 * np.pi)
extent = [tau_axis[0], tau_axis[-1], energy_ev[0], energy_ev[-1]]

n_show = min(6, len(epochs))
show_idx = np.unique(np.linspace(0, len(epochs) - 1, n_show, dtype=int)).tolist()
fig, axes = plt.subplots(len(show_idx), 3, figsize=(12, 3.0 * len(show_idx)))
if len(show_idx) == 1:
    axes = np.array([axes])
for row, si in enumerate(show_idx):
    e_pred = unpack_packed_field(E_pred_all[si])
    e_al = best_l1_ambiguity_field(e_pred, e_true)
    amp_p, amp_t = np.abs(e_al), np.abs(e_true)
    ph_t, ph_p = unwrap_phases_for_overlay(e_al, e_true)
    axes[row, 0].imshow(
        I_show, aspect="auto", origin="lower", extent=extent, cmap="magma"
    )
    axes[row, 0].set_title(f"noisy TRACE  ep={int(epochs[si])}")
    axes[row, 0].set_xlabel("Delay τ [fs]")
    axes[row, 0].set_ylabel("Relative energy [eV]")
    axes[row, 1].plot(t_axis, amp_t, label="true")
    axes[row, 1].plot(t_axis, amp_p, label="pred")
    axes[row, 1].set_title("|E(t)|")
    axes[row, 1].set_xlabel("Time [fs]")
    axes[row, 1].set_ylabel("|E| [a.u.]")
    axes[row, 1].legend(fontsize=8)
    axes[row, 2].plot(t_axis, ph_t, label="true")
    axes[row, 2].plot(t_axis, ph_p, label="pred")
    axes[row, 2].set_title("phase")
    axes[row, 2].set_xlabel("Time [fs]")
    axes[row, 2].set_ylabel("phase [rad]")
    axes[row, 2].legend(fontsize=8)
plt.suptitle(
    f"Physics snapshot evolution @ SNR={SNAPSHOT_SNR_DB:g} dB "
    f"(same probe as plain)"
)
plt.tight_layout()
plt.show()

l1s, sims = [], []
for i in range(len(epochs)):
    e_pred = unpack_packed_field(E_pred_all[i])
    l1s.append(best_l1_ambiguity(e_pred, e_true))
    sims.append(best_similarity_error_ambiguity(e_pred, e_true))
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(epochs, l1s, color="C0")
ax.set_xlabel("epoch")
ax.set_ylabel("L1 (best-amb)")
ax.set_title("Fixed val sample: L1_amb vs training epoch")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(epochs, sims, color="C1")
ax.set_xlabel("epoch")
ax.set_ylabel("similarity error (best-amb)")
ax.set_title("Fixed val sample: SIM_amb vs training epoch")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --- cell ---

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
test_loader = bundle.test_loader

sweep_path = OUT / f"{TAG_FINAL}_test_snr_sweep.npz"
if FORCE_TEST_SWEEP or not sweep_path.exists():
    print("Running test SNR sweep...")
    diag.run_and_save_test_snr_sweep(
        OUT / f"{TAG_FINAL}_model.pt",
        sweep_path,
        test_loader=test_loader,
        snr_sweep_db=SNR_SWEEP_DB,
        device=DEVICE,
        experiment_name=f"Physics n={N_TRAIN} λ*={LAM_STAR:g} LR*={LR_STAR:g}",
    )
else:
    print("skip sweep; using", sweep_path)

sweep = load_cnn_sweep(sweep_path)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].errorbar(
    sweep.snr_sweep_db, sweep.cnn_l1_amb_m, yerr=sweep.cnn_l1_amb_s, marker="o"
)
axes[0].set_xlabel("SNR (dB)")
axes[0].set_ylabel("L1 (best-amb)")
axes[0].set_title(f"Physics n={N_TRAIN}: test L1 vs SNR")
axes[0].grid(True, alpha=0.3)
axes[1].errorbar(
    sweep.snr_sweep_db, sweep.cnn_sim_amb_m, yerr=sweep.cnn_sim_amb_s, marker="o"
)
axes[1].set_xlabel("SNR (dB)")
axes[1].set_ylabel("SIM error (best-amb)")
axes[1].set_title(f"Physics n={N_TRAIN}: test SIM vs SNR")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
for snr, l1, sim in zip(sweep.snr_sweep_db, sweep.cnn_l1_amb_m, sweep.cnn_sim_amb_m):
    print(f"  SNR={snr:6.1f}  L1_amb={l1:.4f}  SIM_amb={sim:.4f}")


# --- cell ---

model = diag.load_trained_multires(OUT / f"{TAG_FINAL}_model.pt", device=DEVICE)
model.eval()

pulse_cfg = filtered_c1_pulse_config(n=64)
dt = float(pulse_cfg.dt)
t_axis = np.linspace(-pulse_cfg.t_total_fs / 2.0, pulse_cfg.t_total_fs / 2.0, pulse_cfg.n)

I_clean, E_true = test_loader.dataset[0]
I_clean_b = I_clean.unsqueeze(0).to(DEVICE)
E_np = E_true.detach().cpu().numpy()
g = torch.Generator()
g.manual_seed(999)
noise = torch.randn(I_clean_b.shape, generator=g, dtype=I_clean_b.dtype)
pwr = I_clean_b.pow(2).mean().clamp_min(1e-12)
sigma = torch.sqrt(pwr / (10.0 ** (10.0 / 10.0)))
I_noisy = I_clean_b + sigma * noise.to(DEVICE)

with torch.no_grad():
    E_pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1))).squeeze(0).cpu().numpy()
e_true = unpack_packed_field(E_np)
e_pred = unpack_packed_field(E_pred)
e_al = best_l1_ambiguity_field(e_pred, e_true)
amp_p, amp_t = np.abs(e_al), np.abs(e_true)
ph_t, ph_p = unwrap_phases_for_overlay(e_al, e_true)
I_n, tau_axis, omega_plot = prepare_frog_trace_for_plot(
    I_noisy.squeeze(0).cpu().numpy(),
    num_points=pulse_cfg.n,
    dt=dt,
)
I_c, _, _ = prepare_frog_trace_for_plot(
    I_clean_b.squeeze(0).cpu().numpy(),
    num_points=pulse_cfg.n,
    dt=dt,
)
energy_ev = omega_plot * 4.135667696 / (2.0 * np.pi)
extent = [tau_axis[0], tau_axis[-1], energy_ev[0], energy_ev[-1]]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].imshow(I_c, aspect="auto", origin="lower", extent=extent, cmap="magma")
axes[0, 0].set_title("clean TRACE")
axes[0, 0].set_xlabel("Delay τ [fs]")
axes[0, 0].set_ylabel("Relative energy [eV]")
axes[0, 1].imshow(I_n, aspect="auto", origin="lower", extent=extent, cmap="magma")
axes[0, 1].set_title("noisy TRACE (SNR=10)")
axes[0, 1].set_xlabel("Delay τ [fs]")
axes[0, 1].set_ylabel("Relative energy [eV]")
axes[1, 0].plot(t_axis, amp_t, label="true")
axes[1, 0].plot(t_axis, amp_p, label="pred")
axes[1, 0].set_title("|E(t)|")
axes[1, 0].set_xlabel("Time [fs]")
axes[1, 0].set_ylabel("|E| [a.u.]")
axes[1, 0].legend()
axes[1, 1].plot(t_axis, ph_t, label="true")
axes[1, 1].plot(t_axis, ph_p, label="pred")
axes[1, 1].set_title("phase")
axes[1, 1].set_xlabel("Time [fs]")
axes[1, 1].set_ylabel("phase [rad]")
axes[1, 1].legend()
plt.suptitle(f"Physics n={N_TRAIN}  λ*={LAM_STAR:g}  LR*={LR_STAR:g}  test[0] @ 10 dB")
plt.tight_layout()
plt.show()

print("L1_amb:", best_l1_ambiguity(e_pred, e_true))
print("SIM_amb:", best_similarity_error_ambiguity(e_pred, e_true))


# --- cell ---

summary = {
    "n_train": N_TRAIN,
    "model": "physics",
    "lambda": LAM_STAR,
    "LR_star": LR_STAR,
    "trace_scale": TRACE_SCALE,
    "batch_size": BATCH_SIZE,
    "best_epoch": meta["best_epoch"],
    "best_score_val_amb": meta["best_score"],
    "stopped_epoch": meta["stopped_epoch"],
    "global_step": meta.get("global_step"),
    "wall_time_data_sec": meta.get("wall_time_data_sec"),
    "wall_time_train_sec": meta.get("wall_time_train_sec"),
    "device": meta.get("device"),
    "data": "filtered_c1",
    "tag": TAG_FINAL,
    "snapshot_val_index": SNAPSHOT_VAL_INDEX,
    "snapshot_snr_db": SNAPSHOT_SNR_DB,
    "snapshot_noise_seed": SNAPSHOT_NOISE_SEED,
}
(OUT / "campaign_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))

