from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

PLANCK_CONSTANT_FS_EV = 4.135667696


@dataclass
class StochasticPulseConfig:
    """SASE-style multi-spike pulses (``stochastic_pulses_generator_NB.ipynb`` defaults)."""

    n: int = 64
    t_total_fs: float = 100.0
    n_spikes: int = 30
    central_energy_ev: float = 10000.0
    coherence_time_fs: float = 2.5
    delta_energy_ev_range: float = 6.5 * 0.0001
    t_center_std_fs: float = 5.73
    amplitude_min: float = 0.75
    amplitude_max: float = 1.0

    @property
    def dt(self) -> float:
        return self.t_total_fs / self.n


def stochastic_pulse_config_data_c(n: int = 64) -> StochasticPulseConfig:
    """Experiment A (modified C1) with wider spike-centre spread — Data C."""
    return StochasticPulseConfig(n=n, t_center_std_fs=9.5)


@dataclass
class FilteredC1PulseConfig:
    """
    C1 envelope with per-spike Δω, then spectral Gaussian filter.

    Matches ``c1_pulse_independent_NB.ipynb`` filtered-C1 section defaults.
    """

    n: int = 64
    t_total_fs: float = 70.0
    n_spikes: int = 500
    central_energy_ev: float = 10000.0
    coherence_time_fs: float = 1.0
    t_center_std_fs: float = 5.73
    delta_energy_ev_range: float = 0.00065
    fwhm_ev: float = 0.45
    amplitude_min: float = 0.75
    amplitude_max: float = 1.0
    use_delta_omega: bool = True

    @property
    def dt(self) -> float:
        return self.t_total_fs / max(self.n - 1, 1)


def filtered_c1_pulse_config(n: int = 64) -> FilteredC1PulseConfig:
    """Defaults from ``c1_pulse_independent_NB.ipynb`` filtered C1."""
    return FilteredC1PulseConfig(n=n)


def generate_pulses_filtered_c1(
    n_pulses: int,
    config: FilteredC1PulseConfig | None = None,
    seed: int | None = None,
    *,
    canonicalize_mode: str = "t0",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate filtered-C1 pulses: multi-spike C1 (+Δω) → FFT × G(E) → IFFT → canonicalize.

    Returns ``(pulses_t, pulses_w, t, omega)`` like ``generate_pulses_stochastic``.
    """
    from pulse_metrics import (
        CANON_MIX_MODES,
        canonicalize_field,
        canonicalize_field_mixed,
        canonicalize_field_tstar,
    )

    allowed = ("t0", "tstar", *CANON_MIX_MODES.keys())
    if canonicalize_mode not in allowed:
        raise ValueError(
            f"canonicalize_mode must be one of {allowed} (got {canonicalize_mode!r})"
        )

    cfg = config or FilteredC1PulseConfig()
    rng = np.random.default_rng(seed)
    n = int(cfg.n)
    t = np.linspace(-cfg.t_total_fs / 2.0, cfg.t_total_fs / 2.0, n)
    dt = cfg.t_total_fs / max(n - 1, 1)
    omega = np.fft.fftshift(np.fft.fftfreq(n, dt)) * 2.0 * np.pi
    energy_ev_rel = np.fft.fftshift(np.fft.fftfreq(n, d=dt) * PLANCK_CONSTANT_FS_EV)
    zero_index = n // 2

    omega_0 = cfg.central_energy_ev * (2.0 * np.pi) / PLANCK_CONSTANT_FS_EV
    delta_omega_max = (cfg.delta_energy_ev_range / cfg.central_energy_ev) * omega_0
    sigma_e = cfg.fwhm_ev / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    g_e = np.exp(-(energy_ev_rel**2) / (2.0 * sigma_e**2))

    pulses_t: list[np.ndarray] = []
    for _ in range(int(n_pulses)):
        f_t = np.zeros(n, dtype=complex)
        for _ in range(int(cfg.n_spikes)):
            t_center = float(rng.normal(loc=0.0, scale=cfg.t_center_std_fs))
            amplitude = float(rng.uniform(cfg.amplitude_min, cfg.amplitude_max))
            random_phase = float(rng.uniform(-np.pi, np.pi))
            dwi = (
                float(rng.uniform(-delta_omega_max, delta_omega_max))
                if cfg.use_delta_omega
                else 0.0
            )
            dt_k = t - t_center
            env = np.exp(-(dt_k**2) / (4.0 * cfg.coherence_time_fs**2))
            f_t += amplitude * env * np.exp(1j * (dwi * dt_k + random_phase))

        f_v = np.fft.fftshift(np.fft.fft(f_t))
        f_v_filt = f_v * g_e
        f_t = np.fft.ifft(np.fft.ifftshift(f_v_filt))

        if canonicalize_mode == "tstar":
            f_t = canonicalize_field_tstar(f_t)
        elif canonicalize_mode in CANON_MIX_MODES:
            phase_mode, flip_mode = CANON_MIX_MODES[canonicalize_mode]
            f_t = canonicalize_field_mixed(
                f_t,
                phase_mode=phase_mode,
                flip_mode=flip_mode,
                zero_index=zero_index,
            )
        else:
            f_t = canonicalize_field(f_t, zero_index=zero_index)
        pulses_t.append(np.asarray(f_t, dtype=np.complex128))

    pulses_t_arr = np.asarray(pulses_t)
    pulses_w = np.fft.fftshift(np.fft.fft(pulses_t_arr, axis=-1), axes=-1)
    return pulses_t_arr, pulses_w, t, omega


def stochastic_pulse_config_notebook_modified_c1(n: int = 64) -> StochasticPulseConfig:
    """
    Modified-C1 defaults matching ``stochastic_pulses_generator_NB.ipynb``:
    ``T=53`` fs, ``N_spikes=300``, ``σ_spike=0.025·T``, peak-spread from
    ``compute_sigma_t_center(T, N_spikes, 0.62)``.
    """
    from spectral_grid import compute_sigma_t_center

    t_total = 53.0
    n_spikes = 300
    pulse_temporal_fraction = 0.62
    return StochasticPulseConfig(
        n=n,
        t_total_fs=t_total,
        n_spikes=n_spikes,
        coherence_time_fs=0.025 * t_total,
        t_center_std_fs=float(
            compute_sigma_t_center(t_total, n_spikes, pulse_temporal_fraction)
        ),
        delta_energy_ev_range=6.5 * 0.0001,
    )


@dataclass
class SmoothedPhaseStochasticPulseConfig:
    """
    SASE multi-spike pulses with per-spike smoothed white-noise phase
    (``stochastic_pulses_generator_NB.ipynb`` third field model).
    """

    n: int = 64
    t_total_fs: float = 100.0
    n_spikes: int = 30
    central_energy_ev: float = 10000.0
    coherence_time_fs: float = 2.3
    delta_energy_ev_range: float = 6.5 * 0.0001
    t_center_std_fs: float = 11.0
    amplitude_min: float = 0.75
    amplitude_max: float = 1.0
    phase_noise_std: float = 0.2 * np.pi
    phase_smooth_sigma: float = 1.6

    @property
    def dt(self) -> float:
        return self.t_total_fs / self.n


def generate_pulses_gaussian(
    n_pulses: int,
    dt: float,
    sigma_omega: float,
    num_points: int = 64,
    sigma: float = 1.6,
    phase_scale: float = np.pi,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random synthetic pulses in frequency and time domains.

    Returns:
        pulses_t: complex array [N, num_points] in time domain
        pulses_w: complex array [N, num_points] in frequency domain
        t: time axis [num_points]
        omega: angular-frequency axis [num_points]
    """
    rng = np.random.default_rng(seed)
    pulses_t = []
    pulses_w = []

    t = np.arange(-num_points // 2, num_points // 2) * dt
    omega = np.fft.fftshift(np.fft.fftfreq(num_points, dt)) * 2 * np.pi

    s_omega = np.exp(-(omega**2) / (2.0 * sigma_omega**2))
    amp_omega = np.sqrt(s_omega)
    zero_index = num_points // 2

    for _ in range(n_pulses):
        random_noise = rng.normal(size=num_points)
        phi_omega = gaussian_filter(random_noise, sigma=sigma)

        max_abs = np.max(np.abs(phi_omega))
        if max_abs != 0:
            phi_omega = (phi_omega / max_abs) * phase_scale

        e_omega = amp_omega * np.exp(1j * phi_omega)
        e_t = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(e_omega)))

        # Canonicalization (kept consistent with the notebook logic).
        phase_at_t0 = np.angle(e_t[zero_index])
        e_t = e_t * np.exp(-1j * phase_at_t0)
        left_area = np.sum(np.real(e_t[:zero_index]))
        right_area = np.sum(np.real(e_t[zero_index + 1 :]))
        if right_area > left_area:
            e_t = np.flip(e_t).conj()
            # Re-align global phase after ambiguity-removal flip to preserve phase(t=0)=0.
            e_t = e_t * np.exp(-1j * np.angle(e_t[zero_index]))

        e_t = e_t / (np.linalg.norm(e_t) + 1e-12)

        pulses_t.append(e_t)
        pulses_w.append(e_omega)

    return np.array(pulses_t), np.array(pulses_w), t, omega


def generate_pulses_stochastic(
    n_pulses: int,
    config: StochasticPulseConfig | None = None,
    seed: int | None = None,
    *,
    canonicalize_mode: str = "t0",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate SASE-style stochastic pulses (modified Formula C1, no linear phase term).

    Returns the same tuple layout as ``generate_pulses_gaussian``.

    ``canonicalize_mode``:
      - ``\"t0\"``: grid-centered ``canonicalize_field`` (legacy training default)
      - ``\"tstar\"``: peak-anchored ``canonicalize_field_tstar`` (generator notebook)
      - ``\"t0_re\"`` / ``\"t0_energy\"`` / ``\"tstar_re\"`` / ``\"tstar_energy\"``:
        mixed phase-anchor × flip heuristics (see ``canonicalize_field_mixed``)
    """
    from pulse_metrics import (
        CANON_MIX_MODES,
        canonicalize_field,
        canonicalize_field_mixed,
        canonicalize_field_tstar,
    )

    allowed = ("t0", "tstar", *CANON_MIX_MODES.keys())
    if canonicalize_mode not in allowed:
        raise ValueError(
            f"canonicalize_mode must be one of {allowed} (got {canonicalize_mode!r})"
        )

    cfg = config or StochasticPulseConfig()
    rng = np.random.default_rng(seed)
    n = cfg.n
    t = np.linspace(-cfg.t_total_fs / 2.0, cfg.t_total_fs / 2.0, n)
    # Match linspace spacing used by the generator notebook (not t_total/n).
    dt = cfg.t_total_fs / max(n - 1, 1)
    omega = np.fft.fftshift(np.fft.fftfreq(n, dt)) * 2.0 * np.pi
    zero_index = n // 2

    pulses_t: list[np.ndarray] = []
    for _ in range(n_pulses):
        f_t = np.zeros(n, dtype=complex)
        for _ in range(cfg.n_spikes):
            t_center = float(rng.normal(loc=0.0, scale=cfg.t_center_std_fs))
            amplitude = float(rng.uniform(cfg.amplitude_min, cfg.amplitude_max))
            random_phase = float(rng.uniform(-np.pi, np.pi))
            spike = (
                amplitude
                * np.exp(-((t - t_center) ** 2) / (4.0 * cfg.coherence_time_fs**2))
                * np.exp(1j * random_phase)
            )
            f_t += spike
        if canonicalize_mode == "tstar":
            f_t = canonicalize_field_tstar(f_t)
        elif canonicalize_mode in CANON_MIX_MODES:
            phase_mode, flip_mode = CANON_MIX_MODES[canonicalize_mode]
            f_t = canonicalize_field_mixed(
                f_t,
                phase_mode=phase_mode,
                flip_mode=flip_mode,
                zero_index=zero_index,
            )
        else:
            f_t = canonicalize_field(f_t, zero_index=zero_index)
        pulses_t.append(f_t)

    pulses_t_arr = np.asarray(pulses_t)
    pulses_w = np.fft.fftshift(np.fft.fft(pulses_t_arr, axis=-1), axes=-1)
    return pulses_t_arr, pulses_w, t, omega


def generate_pulses_stochastic_smoothed_phase(
    n_pulses: int,
    config: SmoothedPhaseStochasticPulseConfig | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate SASE-style pulses with per-spike smoothed white-noise phase on the time grid.

    For each spike k: draw theta_k ~ U[-pi, pi], then xi_k(t) ~ N(theta_k, sigma_phi^2),
    smooth with ``gaussian_filter``, and multiply the modified-C1 envelope by exp(i * phi_k(t)).
    """
    from pulse_metrics import canonicalize_field

    cfg = config or SmoothedPhaseStochasticPulseConfig()
    rng = np.random.default_rng(seed)
    n = cfg.n
    t = np.linspace(-cfg.t_total_fs / 2.0, cfg.t_total_fs / 2.0, n)
    dt = cfg.dt
    omega = np.fft.fftshift(np.fft.fftfreq(n, dt)) * 2.0 * np.pi
    zero_index = n // 2

    pulses_t: list[np.ndarray] = []
    for _ in range(n_pulses):
        f_t = np.zeros(n, dtype=complex)
        for _ in range(cfg.n_spikes):
            t_center = float(rng.normal(loc=0.0, scale=cfg.t_center_std_fs))
            amplitude = float(rng.uniform(cfg.amplitude_min, cfg.amplitude_max))
            random_phase = float(rng.uniform(-np.pi, np.pi))
            white_noise = rng.normal(random_phase, cfg.phase_noise_std, n)
            phase_smoothed = gaussian_filter(white_noise, sigma=cfg.phase_smooth_sigma)
            spike = (
                amplitude
                * np.exp(-((t - t_center) ** 2) / (4.0 * cfg.coherence_time_fs**2))
                * np.exp(1j * phase_smoothed)
            )
            f_t += spike
        f_t = canonicalize_field(f_t, zero_index=zero_index)
        pulses_t.append(f_t)

    pulses_t_arr = np.asarray(pulses_t)
    pulses_w = np.fft.fftshift(np.fft.fft(pulses_t_arr, axis=-1), axes=-1)
    return pulses_t_arr, pulses_w, t, omega


def phase_t_unwrapped_at_zero(pulse_t: np.ndarray) -> np.ndarray:
    """
    Unwrapped phase with φ(t=0)=0 at the center sample (``num_points // 2``).

    Use this for plots. ``np.unwrap(np.angle(E))`` alone can show 2π at t=0
    even when ``angle(E[t=0]) == 0``; ``unwrap(angle(E) - angle(E[t=0]))`` is
    also wrong because ``unwrap`` accumulates from the left edge.
    """
    pulse_t = np.asarray(pulse_t)
    z = pulse_t.size // 2
    ph = np.unwrap(np.angle(pulse_t))
    return ph - ph[z]


def phase_t_unwrapped_at_index(pulse_t: np.ndarray, index: int) -> np.ndarray:
    """Unwrapped phase with φ = 0 at sample ``index`` (e.g. ``t_*`` after peak canonicalization)."""
    pulse_t = np.asarray(pulse_t)
    z = int(index)
    ph = np.unwrap(np.angle(pulse_t))
    return ph - ph[z]
