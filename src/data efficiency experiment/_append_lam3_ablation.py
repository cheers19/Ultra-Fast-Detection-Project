"""Append λ=3 ablation chapter to physics_multires_n2498_NB.ipynb."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

NB = Path(__file__).with_name("physics_multires_n2498_NB.ipynb")
MARKER = "Ablation: full final-budget run with $\\lambda=3$"


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": [line + "\n" for line in src.strip("\n").split("\n")],
    }


def code(src: str) -> dict:
    lines = src.strip("\n").split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else []),
    }


CELLS = [
    md(
        r"""## Ablation: full final-budget run with $\lambda=3$

Same protocol as the $\lambda^*$ / $\lambda=5$ finals (LR*, `trace_scale=8`, `max_steps=200\cdot K`, snapshots every 2 epochs, held-out SNR sweep, test example), but with **$\lambda=3$**.

Artifacts use tag `n2498_phys_lam3_final` (does not overwrite $\lambda^*$ or $\lambda=5$). Respects `FORCE_RETRAIN` / `FORCE_TEST_SWEEP`."""
    ),
    code(
        r"""# Ensure λ* context exists even if earlier cells were skipped after a kernel restart
if "LAM_STAR" not in globals() or "TAG_FINAL" not in globals():
    _ls = json.loads((OUT / "lambda_screen_summary.json").read_text(encoding="utf-8"))
    LAM_STAR = float(_ls["LAM_STAR"])
    TAG_FINAL = lam_tag(LAM_STAR) + "_final"
if "meta" not in globals() and (OUT / f"{TAG_FINAL}_meta.json").exists():
    meta = load_meta(TAG_FINAL)

LAM_COMPARE3 = 3.0
TAG_COMPARE3 = lam_tag(LAM_COMPARE3) + "_final"
print(f"λ_compare={LAM_COMPARE3:g}  tag={TAG_COMPARE3}  LR*={LR_STAR:g}  (λ*={LAM_STAR:g})")

train_physics(
    TAG_COMPARE3,
    lam=LAM_COMPARE3,
    lr=LR_STAR,
    snapshot_every=SNAPSHOT_EVERY_FINAL,
    role="final_lam3_ablation",
    max_steps=MAX_STEPS_FINAL,
    max_epochs=MAX_EPOCHS_FINAL,
)
hist3 = diag.load_history(OUT / f"{TAG_COMPARE3}_history.npz")
meta3 = load_meta(TAG_COMPARE3)
print(json.dumps(meta3, indent=2))
"""
    ),
    md("### λ=3 — training curves"),
    code(
        r"""plot_pulse_curves(hist3, f"Physics n={N_TRAIN} λ={LAM_COMPARE3:g} LR*={LR_STAR:g}")
plot_trace_curves(hist3, f"Physics n={N_TRAIN} λ={LAM_COMPARE3:g} LR*={LR_STAR:g}")
plot_grad_norms(hist3, f"Physics n={N_TRAIN} λ={LAM_COMPARE3:g}", lam=LAM_COMPARE3)
summarize_timings(hist3, f"Physics n={N_TRAIN} λ={LAM_COMPARE3:g} final")
print(
    f"wall_time_data_sec={meta3.get('wall_time_data_sec')}  "
    f"wall_time_train_sec={meta3.get('wall_time_train_sec')}  "
    f"global_step={meta3.get('global_step')}  "
    f"best_epoch={meta3.get('best_epoch')}  best_score={meta3.get('best_score')}"
)
if "meta" in globals():
    print(
        f"Compare to λ*={LAM_STAR:g}: best_epoch={meta.get('best_epoch')}  "
        f"best_score={meta.get('best_score')}"
    )
if "meta5" in globals():
    print(
        f"Compare to λ=5: best_epoch={meta5.get('best_epoch')}  "
        f"best_score={meta5.get('best_score')}"
    )
"""
    ),
    md("### λ=3 — snapshot evolution (same val probe @ SNR=10 dB)"),
    code(
        r"""snap_path3 = OUT / f"{TAG_COMPARE3}_snapshots.npz"
assert snap_path3.exists(), snap_path3
snap = np.load(snap_path3)
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
    f"λ={LAM_COMPARE3:g} snapshot evolution @ SNR={SNAPSHOT_SNR_DB:g} dB "
    f"(same probe as plain / λ*)"
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
ax.set_title(f"λ={LAM_COMPARE3:g} fixed val sample: L1_amb vs epoch")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(epochs, sims, color="C1")
ax.set_xlabel("epoch")
ax.set_ylabel("similarity error (best-amb)")
ax.set_title(f"λ={LAM_COMPARE3:g} fixed val sample: SIM_amb vs epoch")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""
    ),
    md("### λ=3 — held-out test SNR sweep (+ overlay vs λ* / λ=5)"),
    code(
        r"""bundle = build_filtered_c1_frog_dataloaders(
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

sweep_path3 = OUT / f"{TAG_COMPARE3}_test_snr_sweep.npz"
if FORCE_TEST_SWEEP or not sweep_path3.exists():
    print("Running test SNR sweep for λ=3...")
    diag.run_and_save_test_snr_sweep(
        OUT / f"{TAG_COMPARE3}_model.pt",
        sweep_path3,
        test_loader=test_loader,
        snr_sweep_db=SNR_SWEEP_DB,
        device=DEVICE,
        experiment_name=f"Physics n={N_TRAIN} λ={LAM_COMPARE3:g} LR*={LR_STAR:g}",
    )
else:
    print("skip sweep; using", sweep_path3)

sweep3 = load_cnn_sweep(sweep_path3)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].errorbar(
    sweep3.snr_sweep_db, sweep3.cnn_l1_amb_m, yerr=sweep3.cnn_l1_amb_s, marker="o"
)
axes[0].set_xlabel("SNR (dB)")
axes[0].set_ylabel("L1 (best-amb)")
axes[0].set_title(f"λ={LAM_COMPARE3:g}: test L1 vs SNR")
axes[0].grid(True, alpha=0.3)
axes[1].errorbar(
    sweep3.snr_sweep_db, sweep3.cnn_sim_amb_m, yerr=sweep3.cnn_sim_amb_s, marker="o"
)
axes[1].set_xlabel("SNR (dB)")
axes[1].set_ylabel("SIM error (best-amb)")
axes[1].set_title(f"λ={LAM_COMPARE3:g}: test SIM vs SNR")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Overlay vs λ* and λ=5 (if available)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].errorbar(
    sweep3.snr_sweep_db,
    sweep3.cnn_l1_amb_m,
    yerr=sweep3.cnn_l1_amb_s,
    marker="o",
    label=f"λ={LAM_COMPARE3:g}",
)
axes[1].errorbar(
    sweep3.snr_sweep_db,
    sweep3.cnn_sim_amb_m,
    yerr=sweep3.cnn_sim_amb_s,
    marker="o",
    label=f"λ={LAM_COMPARE3:g}",
)
sweep_star_path = OUT / f"{TAG_FINAL}_test_snr_sweep.npz"
if sweep_star_path.exists():
    sweep_star = load_cnn_sweep(sweep_star_path)
    axes[0].errorbar(
        sweep_star.snr_sweep_db,
        sweep_star.cnn_l1_amb_m,
        yerr=sweep_star.cnn_l1_amb_s,
        marker="s",
        label=f"λ*={LAM_STAR:g}",
    )
    axes[1].errorbar(
        sweep_star.snr_sweep_db,
        sweep_star.cnn_sim_amb_m,
        yerr=sweep_star.cnn_sim_amb_s,
        marker="s",
        label=f"λ*={LAM_STAR:g}",
    )
sweep5_path = OUT / f"{lam_tag(5.0)}_final_test_snr_sweep.npz"
if sweep5_path.exists():
    sweep5_ov = load_cnn_sweep(sweep5_path)
    axes[0].errorbar(
        sweep5_ov.snr_sweep_db,
        sweep5_ov.cnn_l1_amb_m,
        yerr=sweep5_ov.cnn_l1_amb_s,
        marker="^",
        label="λ=5",
    )
    axes[1].errorbar(
        sweep5_ov.snr_sweep_db,
        sweep5_ov.cnn_sim_amb_m,
        yerr=sweep5_ov.cnn_sim_amb_s,
        marker="^",
        label="λ=5",
    )

axes[0].set_xlabel("SNR (dB)")
axes[0].set_ylabel("L1 (best-amb)")
axes[0].set_title(f"Physics n={N_TRAIN}: L1 vs SNR — λ={LAM_COMPARE3:g} vs others")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel("SNR (dB)")
axes[1].set_ylabel("SIM error (best-amb)")
axes[1].set_title(f"Physics n={N_TRAIN}: SIM vs SNR — λ={LAM_COMPARE3:g} vs others")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"=== λ={LAM_COMPARE3:g} SNR sweep ===")
for snr, l1, sim in zip(sweep3.snr_sweep_db, sweep3.cnn_l1_amb_m, sweep3.cnn_sim_amb_m):
    print(f"  SNR={snr:6.1f}  L1_amb={l1:.4f}  SIM_amb={sim:.4f}")
"""
    ),
    md("### λ=3 — test example @ SNR = 10 dB"),
    code(
        r"""model3 = diag.load_trained_multires(OUT / f"{TAG_COMPARE3}_model.pt", device=DEVICE)
model3.eval()

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
    E_pred = extract_pulse_prediction(model3(I_noisy.unsqueeze(1))).squeeze(0).cpu().numpy()
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
plt.suptitle(f"Physics n={N_TRAIN}  λ={LAM_COMPARE3:g}  LR*={LR_STAR:g}  test[0] @ 10 dB")
plt.tight_layout()
plt.show()

print("L1_amb:", best_l1_ambiguity(e_pred, e_true))
print("SIM_amb:", best_similarity_error_ambiguity(e_pred, e_true))
"""
    ),
    md("### λ=3 — ablation summary (separate JSON; does not overwrite λ*)"),
    code(
        r"""summary3 = {
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
}
out_json = OUT / "ablation_lam3_summary.json"
out_json.write_text(json.dumps(summary3, indent=2), encoding="utf-8")
print(json.dumps(summary3, indent=2))
print("wrote", out_json)
"""
    ),
]


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    already = any(
        c.get("cell_type") == "markdown" and MARKER in "".join(c.get("source", []))
        for c in nb["cells"]
    )
    if already:
        print("Ablation λ=3 chapter already present; skipping append.")
        return
    nb["cells"].extend(CELLS)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Appended {len(CELLS)} cells to {NB} (now {len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
