EXAMPLE_SNR_DB = 10.0

import time

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
# Canonical paper example (SNR=10 dB). Exact noisy trace stored under OUT_FIG.
EXAMPLE_INDEX = 75
EXAMPLE_CACHE = OUT_FIG / "example_snr10_sample75" / (
    "filtered_c1_example_snr10_sample75_arrays.npz"
)
I_clean, E_true_packed = _test_ds[EXAMPLE_INDEX]
I_clean = I_clean.to(DEVICE)
E_true_packed = E_true_packed.to(DEVICE)
if EXAMPLE_CACHE.exists():
    _ex = np.load(EXAMPLE_CACHE)
    assert int(_ex["EXAMPLE_INDEX"]) == EXAMPLE_INDEX
    I_noisy = torch.as_tensor(_ex["I_noisy"], device=DEVICE, dtype=I_clean.dtype)
    print(f"Loaded exact I_noisy from cache: {EXAMPLE_CACHE}")
else:
    I_noisy = add_trace_noise_awgn(I_clean.unsqueeze(0), EXAMPLE_SNR_DB).squeeze(0)
    print("WARN: example cache missing; drew a NEW AWGN realization")
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
# Generic short keys for files / console only (not used as figure titles).
raw_fields = {
    "physics_2k": unpack_packed_field(E_phys),
    "plain_2k": unpack_packed_field(E_plain),
    "plain_60k": unpack_packed_field(E_plain_60k),
}

amb_rows = []
for key, e_raw in raw_fields.items():
    t0 = time.perf_counter()
    l1_amb = best_l1_ambiguity(e_raw, e_true) / 64.0  # per-time L1
    t_l1_ms = 1000.0 * (time.perf_counter() - t0)

    t0 = time.perf_counter()
    sim_amb = best_similarity_error_ambiguity(e_raw, e_true)
    t_sim_ms = 1000.0 * (time.perf_counter() - t0)

    amb_rows.append(
        {
            "key": key,
            "l1_amb": float(l1_amb),
            "sim_amb": float(sim_amb),
            "t_l1_ms": float(t_l1_ms),
            "t_sim_ms": float(t_sim_ms),
        }
    )

reconstructions = {
    key: best_l1_ambiguity_field(e_raw, e_true) for key, e_raw in raw_fields.items()
}

print(f"\nExample sample {EXAMPLE_INDEX} @ SNR={EXAMPLE_SNR_DB:.0f} dB")
print(f"{'Network':<12} {'L1_amb/64':>10} {'t_L1_ms':>10} {'SIM_amb':>10} {'t_SIM_ms':>10}")
print("-" * 56)
for row in amb_rows:
    print(
        f"{row['key']:<12} {row['l1_amb']:10.4f} {row['t_l1_ms']:10.2f} "
        f"{row['sim_amb']:10.4f} {row['t_sim_ms']:10.2f}"
    )

trace_c, tau_axis, omega_plot = prepare_frog_trace_for_plot(
    I_clean.detach().cpu().numpy(), num_points=len(t_axis), dt=dt
)
trace_n, _, _ = prepare_frog_trace_for_plot(
    I_noisy.detach().cpu().numpy(), num_points=len(t_axis), dt=dt
)
energy_plot = omega_plot * 4.135667696 / (2.0 * np.pi)
extent = [tau_axis[0], tau_axis[-1], energy_plot[0], energy_plot[-1]]


def _style_axes(ax):
    ax.tick_params(labelsize=TICK_FS)
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab.set_fontweight("bold")
        lab.set_fontsize(TICK_FS)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def _style_title(ax, title: str):
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=12)


# --- Clean TRACE (separate figure) ---
fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.imshow(trace_c, origin="lower", aspect="auto", extent=extent, cmap="magma")
_style_title(ax, "Clean Trace")
ax.set_xlabel("Delay τ [fs]", fontsize=AXIS_FS, fontweight="bold")
ax.set_ylabel("Relative energy [eV]", fontsize=AXIS_FS, fontweight="bold")
_style_axes(ax)
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
p_clean = OUT_FIG / f"filtered_c1_example_snr10_sample{EXAMPLE_INDEX}_trace_clean.png"
fig.savefig(p_clean, dpi=140)
plt.show()
print("wrote", p_clean)

# --- Noisy TRACE (separate figure) ---
fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.imshow(trace_n, origin="lower", aspect="auto", extent=extent, cmap="magma")
_style_title(ax, "Noisy Trace")
ax.set_xlabel("Delay τ [fs]", fontsize=AXIS_FS, fontweight="bold")
ax.set_ylabel("Relative energy [eV]", fontsize=AXIS_FS, fontweight="bold")
_style_axes(ax)
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
p_noisy = OUT_FIG / f"filtered_c1_example_snr10_sample{EXAMPLE_INDEX}_trace_noisy.png"
fig.savefig(p_noisy, dpi=140)
plt.show()
print("wrote", p_noisy)

# --- One reconstruction figure per network: |E| + phase on same axes ---
_RECON_TITLES = {
    "physics_2k": "Physics 2K",
    "plain_2k": "Plain 2K",
    "plain_60k": "Plain 60K",
}
for key, e_aligned in reconstructions.items():
    ph_true, ph_rec = unwrap_phases_for_overlay(e_aligned, e_true)
    amp_true = np.abs(e_true)
    amp_rec = np.abs(e_aligned)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax2 = ax.twinx()

    (l1,) = ax.plot(t_axis, amp_true, color="C0", lw=2.4, label="Amplitude")
    (l2,) = ax.plot(
        t_axis, amp_rec, color="C0", lw=2.0, ls="--", label="Amplitude (rec)"
    )
    (l3,) = ax2.plot(t_axis, ph_true, color="C1", lw=2.4, label="Phase")
    (l4,) = ax2.plot(
        t_axis, ph_rec, color="C1", lw=2.0, ls="--", label="Phase (rec)"
    )

    _style_title(ax, _RECON_TITLES.get(key, key))
    ax.set_xlabel("Time [fs]", fontsize=AXIS_FS, fontweight="bold")
    ax.set_ylabel("Amplitude", fontsize=AXIS_FS, fontweight="bold", color="C0")
    ax2.set_ylabel("Phase [rad]", fontsize=AXIS_FS, fontweight="bold", color="C1")
    ax.tick_params(axis="y", labelcolor="C0", labelsize=TICK_FS)
    ax2.tick_params(axis="y", labelcolor="C1", labelsize=TICK_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab.set_fontweight("bold")
        lab.set_fontsize(TICK_FS)
    for lab in list(ax2.get_yticklabels()):
        lab.set_fontweight("bold")
        lab.set_fontsize(TICK_FS)
    ax.grid(True, alpha=0.3)

    leg = ax.legend(
        [l1, l2, l3, l4],
        ["Amplitude", "Amplitude (rec)", "Phase", "Phase (rec)"],
        loc="best",
        fontsize=LEGEND_FS,
        framealpha=0.9,
        prop={"weight": "bold", "size": LEGEND_FS},
    )

    plt.tight_layout()
    p_rec = (
        OUT_FIG
        / f"filtered_c1_example_snr10_sample{EXAMPLE_INDEX}_recon_{key}.png"
    )
    fig.savefig(p_rec, dpi=140)
    plt.show()
    print("wrote", p_rec, f"({key})")
