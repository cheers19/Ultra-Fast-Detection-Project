"""PCGPA pulse retrieval for SHG-FROG (FROGNet delay/shift model + pypret PCGPA).

Primary entry point: ``reconstruct_pcgpa`` on traces shaped ``[N_omega, N_tau]``
as produced by ``frognet.FROGNet``.

Requires vendored ``pypret`` at ``vendor/pypret`` (not on PyPI for all platforms).
"""

from __future__ import annotations

PCGPA_API_VERSION = 6  # initial_guess_mode incl. random_initial (no dataset σ_ω prior)

import sys
from pathlib import Path
from typing import Literal

InitialGuessMode = Literal[
    "dataset_sigma",
    "trace_marginal_sqrt2",
    "random_width",
    "random_initial",
]

import numpy as np

_VENDOR = Path(__file__).resolve().parent / "vendor" / "pypret"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))


def reload_from_disk():
    """Re-load this module from ``pcgpa_reconstruct.py`` on disk (for Jupyter)."""
    import importlib.util

    path = Path(__file__).resolve()
    spec = importlib.util.spec_from_file_location("pcgpa_reconstruct", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pcgpa_reconstruct"] = mod
    spec.loader.exec_module(mod)
    if getattr(mod, "PCGPA_API_VERSION", 0) < 2:
        raise RuntimeError(f"Outdated {path}; need PCGPA_API_VERSION >= 2")
    return mod


def _ensure_pypret():
    import pypret  # noqa: F401

    return pypret


def _build_delay_indices(n_t: int, num_delay_steps: int) -> np.ndarray:
    delays = np.linspace(-n_t // 2, n_t // 2, num_delay_steps)
    return np.round(delays).astype(np.int64)


def _shift_with_zeros(e: np.ndarray, shift: int) -> np.ndarray:
    n_t = e.shape[-1]
    out = np.zeros_like(e)
    if shift == 0:
        return e.copy()
    if abs(shift) >= n_t:
        return out
    if shift > 0:
        out[..., shift:] = e[..., : n_t - shift]
    else:
        k = -shift
        out[..., : n_t - k] = e[..., k:]
    return out


def frognet_g_matrix(
    e_t: np.ndarray,
    num_delay_steps: int | None = None,
) -> np.ndarray:
    """SHG gate product G(t, tau) = E(t) E(t-tau), shape (N_t, N_tau)."""
    e_t = np.asarray(e_t, dtype=np.complex128)
    n_t = e_t.shape[-1]
    n_tau = int(num_delay_steps) if num_delay_steps is not None else n_t
    delays = _build_delay_indices(n_t, n_tau)
    delayed = np.stack(
        [_shift_with_zeros(e_t, int(tau)) for tau in delays],
        axis=-1,
    )
    return e_t[:, None] * delayed


def frognet_forward(
    e_t: np.ndarray,
    *,
    ft=None,
    num_delay_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SHG-FROG trace matching ``FROGNet`` intensities, using ``ft.forward`` when
    ``ft`` is a pypret ``FourierTransform`` (needed for consistent PCGPA).

    Returns
    -------
    I_trace : (N_omega, N_tau)
    g_w : complex spectrogram before |.|^2
    """
    g = frognet_g_matrix(e_t, num_delay_steps=num_delay_steps)
    n_tau = g.shape[1]
    if ft is None:
        g_w = np.fft.fft(g, axis=0)
        i_trace = g_w.real**2 + g_w.imag**2
        return i_trace.astype(np.float64), g_w

    n_t = g.shape[0]
    g_w = np.zeros((n_t, n_tau), dtype=np.complex128)
    for m in range(n_tau):
        g_w[:, m] = ft.forward(g[:, m])
    i_trace = np.abs(g_w) ** 2
    return i_trace.astype(np.float64), g_w


class FrogNetSHG_PNPS:
    """
    Minimal PNPS shim: FROGNet ``E(t-tau)`` product gate + pypret Fourier grid.

    Exposes ``scheme == 'shg-frog'`` for ``PCGPARetriever``.
    """

    scheme = "shg-frog"
    process = "shg"
    method = "frog"
    parameter_name = "delay"
    parameter_unit = "s"

    def __init__(self, n_points: int = 64, dt: float = 1.0, center_wl: float = 800e-9):
        _ensure_pypret()
        from pypret import FourierTransform, Pulse

        self.ft = FourierTransform(n_points, dt=dt)
        pulse = Pulse(self.ft, center_wl)
        self.process_w = pulse.w
        self.w0 = pulse.w0
        self.w = pulse.w
        self.t = pulse.t
        self.parameter = pulse.t
        self.Smk: np.ndarray | None = None
        self.Tmn: np.ndarray | None = None
        self._tmp: dict = {}

    def measure(self, sk: np.ndarray) -> np.ndarray:
        sn = self.ft.forward(sk)
        return sn.real**2 + sn.imag**2

    def calculate(self, spectrum: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        parameter = np.atleast_1d(parameter)
        e = self.ft.backward(np.asarray(spectrum, dtype=np.complex128))
        g = frognet_g_matrix(e, parameter.size)
        m_delays, n_t = g.shape[1], g.shape[0]
        smk = np.zeros((m_delays, n_t), dtype=np.complex128)
        tmn = np.zeros((m_delays, n_t), dtype=np.float64)
        for m in range(m_delays):
            smk[m, :] = g[:, m]
            # Match ``FROGNet`` (numpy FFT along time), not pypret's phased FT.
            tmn[m, :] = np.abs(np.fft.fft(g[:, m])) ** 2
        self.Smk = smk
        self.Tmn = tmn
        return tmn.squeeze() if m_delays == 1 else tmn


def make_pypret_setup(
    n_points: int = 64,
    dt: float = 1.0,
    center_wl: float = 800e-9,
) -> tuple[object, object, FrogNetSHG_PNPS]:
    """Fourier grid + ``FrogNetSHG_PNPS`` (delay axis = ``pulse.t``)."""
    _ensure_pypret()
    from pypret import Pulse

    pnps = FrogNetSHG_PNPS(n_points, dt=dt, center_wl=center_wl)
    pulse = Pulse(pnps.ft, center_wl)
    return pnps.ft, pulse, pnps


def _project_smk_numpy(measured: np.ndarray, smk: np.ndarray) -> np.ndarray:
    """Intensity projection using numpy FFT (consistent with ``FROGNet``)."""
    smn = np.fft.fft(smk, axis=-1)
    abs_smn = np.abs(smn)
    target = np.sqrt(np.maximum(measured, 0.0))
    smn2 = np.zeros_like(smn)
    mask = abs_smn > 1e-15
    smn2[mask] = smn[mask] / abs_smn[mask] * target[mask]
    smn2[~mask] = target[~mask]
    return np.fft.ifft(smn2, axis=-1)


class _FrogNetPCGPARetriever:
    """PCGPA retriever with numpy-FFT projection for ``FrogNetSHG_PNPS``."""

    def __init__(self, pnps: FrogNetSHG_PNPS, **kwargs):
        from pypret.retrieval import Retriever

        self._retriever = Retriever(pnps, "pcgpa", **kwargs)
        self._retriever._project = _project_smk_numpy

    def retrieve(self, measurement, initial_guess, weights=None, **kwargs):
        return self._retriever.retrieve(measurement, initial_guess, weights, **kwargs)

    def result(self, pulse_original=None, full=True):
        return self._retriever.result(pulse_original, full=full)


def _pcgpa_retrieve_with_early_stop(
    wrapper: _FrogNetPCGPARetriever,
    measurement,
    initial_guess,
    *,
    patience: int,
) -> None:
    """Run PCGPA iterations; stop when trace error fails to improve for ``patience`` steps."""
    inner = wrapper._retriever
    inner._retrieve_begin(measurement, initial_guess, None)
    o = inner.options
    res = inner._result
    rs = inner._retrieval_state
    spectrum = inner.initial_guess.copy()
    R = res.trace_error
    for i in range(o.maxiter):
        if inner.logging and inner.log is not None:
            if rs.approximate_error:
                R = inner.trace_error(spectrum, store=False)
            inner.log.trace_error.append(R)
        R, new_spectrum = inner._retrieve_step(i, spectrum.copy())
        if R < res.trace_error:
            res.trace_error = R
            res.approximate_error = rs.approximate_error
            res.spectrum[:] = spectrum
            rs.steps_since_improvement = 0
        else:
            rs.steps_since_improvement += 1
        spectrum[:] = new_spectrum
        if rs.steps_since_improvement >= patience:
            break
    inner._retrieve_end()


from pulse_metrics import (
    best_delta_e_ambiguity,
    best_l1_ambiguity,
    best_similarity_error_ambiguity,
    delta_e_numpy,
    l1_packed_mae,
    pack_complex_field,
    similarity_error_numpy,
    unpack_packed_field,
)


def fwhm_t_transform_limited(sigma_omega: float) -> float:
    """
    Transform-limited intensity FWHM in the same units as ``pulse.t`` / ``dt``.

    Matches ``generate_pulses_gaussian``: spectral envelope std ``sigma_omega``
    (rad / time unit). ``pypret.random_gaussian`` expects intensity FWHM in
    ``pulse.t`` units (attoseconds as plain floats in this project).
    """
    sigma_omega = float(sigma_omega)
    if sigma_omega <= 0:
        raise ValueError("sigma_omega must be positive")
    sigma_field = 1.0 / sigma_omega
    return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_field)


def default_sigma_omega_from_dt(dt: float) -> float:
    """Same convention as ``reconstruction_snr_experiments``: ``0.05 * (2π/dt)``."""
    dt = float(dt)
    if dt <= 0:
        raise ValueError("dt must be positive")
    return 0.05 * (2.0 * np.pi / dt)


def _sample_fwhm_t_for_restart(
    rng: np.random.Generator,
    fwhm_nom: float,
    *,
    scale_range: tuple[float, float] = (0.5, 2.0),
) -> float:
    lo, hi = (float(scale_range[0]), float(scale_range[1]))
    if lo <= 0 or hi <= 0 or lo > hi:
        raise ValueError("fwhm_scale_range must be positive with lo <= hi")
    return float(fwhm_nom * rng.uniform(lo, hi))


def _random_sigma_omega(rng: np.random.Generator, dt: float) -> float:
    """Random spectral envelope std in rad / time unit (independent of dataset value)."""
    span = 2.0 * np.pi / float(dt)
    return float(rng.uniform(0.01 * span, 0.25 * span))


def _sigma_omega_from_noisy_trace_marginal(
    trace: np.ndarray,
    omega_axis: np.ndarray,
    *,
    dt: float,
    fallback_sigma: float,
) -> float:
    """
    Gaussian fit to the spectral marginal of a noisy trace; return ``√2 × σ_fit``.
    """
    from scipy.optimize import curve_fit

    from pulse_metrics import frog_trace_marginals, prepare_frog_trace_for_plot

    trace_plot, _, _ = prepare_frog_trace_for_plot(
        trace, omega_axis=omega_axis, dt=dt
    )
    spec, _ = frog_trace_marginals(trace_plot)
    omega = np.asarray(omega_axis, dtype=float)
    fallback = float(fallback_sigma)

    def _gauss(w, amp, sigma):
        return amp * np.exp(-0.5 * (w / sigma) ** 2)

    try:
        popt, _ = curve_fit(
            _gauss,
            omega,
            spec,
            p0=(float(np.max(spec)), fallback),
            bounds=([0.0, 1e-12], [np.inf, np.inf]),
            maxfev=10_000,
        )
        sigma_fit = float(popt[1])
    except (RuntimeError, ValueError):
        sigma_fit = fallback
    return float(np.sqrt(2.0) * sigma_fit)


def _fwhm_t_for_initial_guess(
    mode: InitialGuessMode,
    *,
    rng: np.random.Generator,
    fwhm_nom: float,
    fwhm_scale_range: tuple[float, float],
    n_restarts: int,
    attempt: int,
    dt: float,
    trace: np.ndarray | None = None,
    omega_axis: np.ndarray | None = None,
    sigma_omega_fallback: float | None = None,
    trace_sigma_omega: float | None = None,
) -> float:
    if mode == "dataset_sigma":
        if n_restarts == 1:
            return float(fwhm_nom)
        return _sample_fwhm_t_for_restart(
            rng, fwhm_nom, scale_range=fwhm_scale_range
        )
    if mode == "trace_marginal_sqrt2":
        if trace_sigma_omega is None:
            raise ValueError("trace_sigma_omega required for trace_marginal_sqrt2")
        return fwhm_t_transform_limited(trace_sigma_omega)
    if mode == "random_width":
        return fwhm_t_transform_limited(_random_sigma_omega(rng, dt))
    if mode == "random_initial":
        # Random FWHM jitter around dt-only nominal — no dataset ``sigma_omega`` prior.
        nom = fwhm_t_transform_limited(default_sigma_omega_from_dt(dt))
        return _sample_fwhm_t_for_restart(
            rng, nom, scale_range=fwhm_scale_range
        )
    raise ValueError(f"unknown initial_guess_mode: {mode!r}")


def _initial_spectrum_guess(
    pulse,
    rng: np.random.Generator,
    fwhm_t: float,
    *,
    phase_max: float = 0.3 * np.pi,
    jitter_scale: float = 0.05,
) -> np.ndarray:
    """Random Gaussian spectrum: envelope width ``fwhm_t``, random phase, small jitter."""
    from pypret.random_pulse import random_gaussian

    n = int(pulse.N)
    random_gaussian(pulse, float(fwhm_t), float(phase_max))
    jitter = float(jitter_scale) * (
        rng.normal(size=n) + 1j * rng.normal(size=n)
    )
    return np.asarray(pulse.spectrum + jitter, dtype=np.complex128)


def reconstruct_pcgpa(
    trace: np.ndarray,
    *,
    n_points: int | None = None,
    dt: float = 1.0,
    maxiter: int = 300,
    early_stop_patience: int | None = None,
    decomposition: Literal["power", "svd"] = "power",
    initial_spectrum: np.ndarray | None = None,
    reference_spectrum: np.ndarray | None = None,
    n_restarts: int = 1,
    rng: np.random.Generator | None = None,
    sigma_omega: float | None = None,
    fwhm_t: float | None = None,
    fwhm_scale_range: tuple[float, float] = (0.5, 2.0),
    phase_max: float = 0.3 * np.pi,
    jitter_scale: float = 0.05,
    initial_guess_mode: InitialGuessMode = "dataset_sigma",
    omega_axis: np.ndarray | None = None,
) -> np.ndarray:
    """
    PCGPA retrieval from FROGNet-layout trace ``[N_omega, N_tau]``.

    Do not pass the true pulse as ``initial_spectrum`` for benchmarking — use a
    random guess (default). ``reference_spectrum`` is only for optional
    post-hoc alignment via pypret ``pulse_error`` (not used in SNR sweeps).

    ``initial_guess_mode`` controls the Gaussian spectral envelope width:

    - ``dataset_sigma`` (default): ``fwhm_t`` from ``sigma_omega``; jitter across
      restarts via ``fwhm_scale_range``.
    - ``trace_marginal_sqrt2``: fit the noisy trace spectral marginal, use
      ``√2 × σ_fit`` (requires ``omega_axis``, e.g. ``w_vec``).
    - ``random_width``: random ``σ_ω`` per restart, independent of dataset / trace.
    - ``random_initial``: random Gaussian envelope width (``fwhm_scale_range`` jitter
      around dt-only nominal); ignores passed ``sigma_omega``.

    ``n_restarts`` > 1 runs PCGPA from independent random guesses and keeps the
    result with the lowest trace error (recommended).

    Returns
    -------
    e_t : complex temporal field on the pypret time grid, shape (N,)
    """
    from pypret.mesh_data import MeshData

    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")

    i_meas = np.asarray(trace, dtype=np.float64)
    if i_meas.ndim != 2:
        raise ValueError("trace must be 2D [N_omega, N_tau]")
    n_omega, n_tau = i_meas.shape
    n = int(n_points) if n_points is not None else n_omega
    if n != n_omega or n != n_tau:
        raise ValueError("trace must be square [N, N] matching FROGNet layout")

    ft, pulse, pnps = make_pypret_setup(n, dt=dt)
    # pypret PCGPA expects (delay, frequency); FROGNet is (omega, tau)
    data = i_meas.T.copy()
    measurement = MeshData(data, pnps.parameter, pnps.process_w)

    rng = rng or np.random.default_rng()
    if fwhm_t is not None:
        fwhm_nom = float(fwhm_t)
    elif sigma_omega is not None:
        fwhm_nom = fwhm_t_transform_limited(sigma_omega)
    else:
        fwhm_nom = fwhm_t_transform_limited(default_sigma_omega_from_dt(dt))

    if initial_guess_mode == "trace_marginal_sqrt2" and omega_axis is None:
        raise ValueError("omega_axis is required when initial_guess_mode='trace_marginal_sqrt2'")

    sigma_fallback = (
        float(sigma_omega)
        if sigma_omega is not None
        else default_sigma_omega_from_dt(dt)
    )
    trace_sigma_omega: float | None = None
    if initial_guess_mode == "trace_marginal_sqrt2":
        trace_sigma_omega = _sigma_omega_from_noisy_trace_marginal(
            i_meas,
            omega_axis,
            dt=dt,
            fallback_sigma=sigma_fallback,
        )

    best_spec: np.ndarray | None = None
    best_trace_err = np.inf

    for attempt in range(n_restarts):
        if initial_spectrum is not None and n_restarts == 1:
            guess = np.asarray(initial_spectrum, dtype=np.complex128)
        else:
            fwhm_try = _fwhm_t_for_initial_guess(
                initial_guess_mode,
                rng=rng,
                fwhm_nom=fwhm_nom,
                fwhm_scale_range=fwhm_scale_range,
                n_restarts=n_restarts,
                attempt=attempt,
                dt=dt,
                trace=i_meas,
                omega_axis=omega_axis,
                sigma_omega_fallback=sigma_fallback,
                trace_sigma_omega=trace_sigma_omega,
            )
            guess = _initial_spectrum_guess(
                pulse,
                rng,
                fwhm_try,
                phase_max=phase_max,
                jitter_scale=jitter_scale,
            )

        retriever = _FrogNetPCGPARetriever(
            pnps,
            maxiter=maxiter,
            decomposition=decomposition,
            verbose=False,
        )
        if early_stop_patience is not None:
            _pcgpa_retrieve_with_early_stop(
                retriever,
                measurement,
                guess,
                patience=int(early_stop_patience),
            )
        else:
            retriever.retrieve(measurement, guess)
        res = retriever.result()
        if res.trace_error < best_trace_err:
            best_trace_err = float(res.trace_error)
            best_spec = res.pulse_retrieved.copy()

    spec_rec = best_spec
    if reference_spectrum is not None:
        from pypret.pulse_error import pulse_error

        _, spec_rec = pulse_error(
            spec_rec,
            np.asarray(reference_spectrum, dtype=np.complex128),
            ft,
            dot_ambiguity=True,
        )
    return ft.backward(spec_rec)


# Backward-compatible alias
reconstruct_pcgpa_frognet = reconstruct_pcgpa


def trace_from_field(
    e_t: np.ndarray,
    *,
    n_points: int = 64,
    dt: float = 1.0,
) -> np.ndarray:
    """Forward trace ``[N_omega, N_tau]`` for a temporal field."""
    _, _, pnps = make_pypret_setup(n_points, dt=dt)
    tmn = pnps.calculate(pnps.ft.forward(np.asarray(e_t, dtype=np.complex128)), pnps.parameter)
    return np.asarray(tmn).T


def temporal_field_to_initial_spectrum(
    e_t: np.ndarray,
    *,
    n_points: int | None = None,
    dt: float = 1.0,
) -> np.ndarray:
    """Map a temporal field to the pypret spectral grid used by ``reconstruct_pcgpa``."""
    e_t = np.asarray(e_t, dtype=np.complex128).ravel()
    n = int(n_points) if n_points is not None else e_t.size
    if n != e_t.size:
        raise ValueError(f"n_points={n} must match e_t.size={e_t.size}")
    ft, _, _ = make_pypret_setup(n, dt=dt)
    return ft.forward(e_t)


def _pcgpa_subsample_indices(
    batch_size: int, n_subsample: int | None, seed: int
) -> np.ndarray:
    idx = np.arange(batch_size)
    if n_subsample is not None and n_subsample < batch_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(batch_size, size=n_subsample, replace=False)
    return idx


def _pcgpa_rng_for_pulse(seed: int, pulse_index: int, snr_db: float) -> np.random.Generator:
    tag = (seed + int(pulse_index) * 10007 + int(round(snr_db * 10))) % (2**32 - 1)
    return np.random.default_rng(tag)


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def mean_metrics_at_snr_pcgpa(
    i_clean_batch: np.ndarray,
    e_true_batch: np.ndarray,
    snr_db: float,
    *,
    add_noise_fn,
    dt: float = 1.0,
    sigma_omega: float | None = None,
    maxiter: int = 300,
    early_stop_patience: int | None = None,
    n_subsample: int | None = None,
    seed: int = 0,
    n_restarts: int = 3,
    use_best_ambiguity: bool = True,
    show_progress: bool = False,
    fwhm_scale_range: tuple[float, float] = (0.5, 2.0),
    initial_guess_mode: InitialGuessMode = "dataset_sigma",
    omega_axis: np.ndarray | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Mean/std SIMILARITY_ERROR and L1 with **one** PCGPA reconstruction per pulse.

    Returns ``((sim_mean, sim_std), (l1_mean, l1_std))``.
    """
    import torch

    i_clean = np.asarray(i_clean_batch, dtype=np.float64)
    e_true = np.asarray(e_true_batch, dtype=np.float64)
    idx = _pcgpa_subsample_indices(i_clean.shape[0], n_subsample, seed)

    sim_per: list[float] = []
    l1_per: list[float] = []
    iterator = idx
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(
                idx,
                desc=f"PCGPA/{initial_guess_mode} @ {float(snr_db):.0f} dB",
                leave=False,
            )
        except ImportError:
            pass

    for i in iterator:
        i_c = torch.as_tensor(i_clean[i])
        i_n = add_noise_fn(i_c.unsqueeze(0), float(snr_db)).squeeze(0).numpy()
        e_ref = unpack_packed_field(e_true[i])
        e_ref_packed = e_true[i]
        e_rec = reconstruct_pcgpa(
            i_n,
            dt=dt,
            maxiter=maxiter,
            early_stop_patience=early_stop_patience,
            n_restarts=n_restarts,
            rng=_pcgpa_rng_for_pulse(seed, int(i), float(snr_db)),
            sigma_omega=sigma_omega,
            fwhm_scale_range=fwhm_scale_range,
            initial_guess_mode=initial_guess_mode,
            omega_axis=omega_axis,
        )
        if use_best_ambiguity:
            sim_per.append(best_similarity_error_ambiguity(e_rec, e_ref))
            l1_per.append(
                l1_packed_mae(e_rec, e_ref_packed, use_best_ambiguity=True)
            )
        else:
            sim_per.append(similarity_error_numpy(e_rec, e_ref))
            l1_per.append(
                l1_packed_mae(e_rec, e_ref_packed, use_best_ambiguity=False)
            )

    return _mean_std(sim_per), _mean_std(l1_per)


def mean_delta_e_at_snr_pcgpa(
    i_clean_batch: np.ndarray,
    e_true_batch: np.ndarray,
    snr_db: float,
    *,
    add_noise_fn,
    dt: float = 1.0,
    sigma_omega: float | None = None,
    maxiter: int = 300,
    n_subsample: int | None = None,
    seed: int = 0,
    n_restarts: int = 3,
    use_best_ambiguity: bool = True,
    fwhm_scale_range: tuple[float, float] = (0.5, 2.0),
) -> tuple[float, float]:
    """
    Mean/std δE for PCGPA over pulses.

    Random initial guesses only (never the ground-truth pulse). Uses
    ``n_restarts`` and, by default, ``use_best_ambiguity=True``: align the
    recovered pulse to truth via flip/conjugate/shift before δE (simulation eval).

    Parameters
    ----------
    i_clean_batch : (B, N, N) numpy or torch-convertible
    e_true_batch : (B, 2N) packed real/imag
    add_noise_fn : callable (I_clean_tensor, snr_db) -> I_noisy tensor
    n_restarts : independent PCGPA runs; keep lowest trace-error result
    use_best_ambiguity : minimize δE over flip/conjugate/shift vs. ground truth
    """
    import torch

    i_clean = np.asarray(i_clean_batch, dtype=np.float64)
    e_true = np.asarray(e_true_batch, dtype=np.float64)
    idx = _pcgpa_subsample_indices(i_clean.shape[0], n_subsample, seed)

    per: list[float] = []
    for i in idx:
        i_c = torch.as_tensor(i_clean[i])
        i_n = add_noise_fn(i_c.unsqueeze(0), float(snr_db)).squeeze(0).numpy()
        e_ref = unpack_packed_field(e_true[i])
        e_rec = reconstruct_pcgpa(
            i_n,
            dt=dt,
            maxiter=maxiter,
            n_restarts=n_restarts,
            rng=_pcgpa_rng_for_pulse(seed, int(i), float(snr_db)),
            sigma_omega=sigma_omega,
            fwhm_scale_range=fwhm_scale_range,
        )
        if use_best_ambiguity:
            per.append(best_delta_e_ambiguity(e_rec, e_ref))
        else:
            per.append(delta_e_numpy(e_rec, e_ref))
    return _mean_std(per)


def mean_similarity_at_snr_pcgpa(
    i_clean_batch: np.ndarray,
    e_true_batch: np.ndarray,
    snr_db: float,
    *,
    add_noise_fn,
    dt: float = 1.0,
    sigma_omega: float | None = None,
    maxiter: int = 300,
    n_subsample: int | None = None,
    seed: int = 0,
    n_restarts: int = 3,
    use_best_ambiguity: bool = True,
    show_progress: bool = False,
    fwhm_scale_range: tuple[float, float] = (0.5, 2.0),
    initial_guess_mode: InitialGuessMode = "dataset_sigma",
    omega_axis: np.ndarray | None = None,
) -> tuple[float, float]:
    """Mean/std SIMILARITY_ERROR (= 1 - cos δE) for PCGPA over pulses."""
    (sim, _), _ = mean_metrics_at_snr_pcgpa(
        i_clean_batch,
        e_true_batch,
        snr_db,
        add_noise_fn=add_noise_fn,
        dt=dt,
        sigma_omega=sigma_omega,
        maxiter=maxiter,
        n_subsample=n_subsample,
        seed=seed,
        n_restarts=n_restarts,
        use_best_ambiguity=use_best_ambiguity,
        show_progress=show_progress,
        fwhm_scale_range=fwhm_scale_range,
        initial_guess_mode=initial_guess_mode,
        omega_axis=omega_axis,
    )
    return sim


def mean_l1_at_snr_pcgpa(
    i_clean_batch: np.ndarray,
    e_true_batch: np.ndarray,
    snr_db: float,
    *,
    add_noise_fn,
    dt: float = 1.0,
    sigma_omega: float | None = None,
    maxiter: int = 300,
    n_subsample: int | None = None,
    seed: int = 0,
    n_restarts: int = 3,
    use_best_ambiguity: bool = True,
    show_progress: bool = False,
    fwhm_scale_range: tuple[float, float] = (0.5, 2.0),
    initial_guess_mode: InitialGuessMode = "dataset_sigma",
    omega_axis: np.ndarray | None = None,
) -> tuple[float, float]:
    """Mean/std per-pulse L1 (sum over packed Re/Im) for PCGPA over pulses."""
    _, (l1, _) = mean_metrics_at_snr_pcgpa(
        i_clean_batch,
        e_true_batch,
        snr_db,
        add_noise_fn=add_noise_fn,
        dt=dt,
        sigma_omega=sigma_omega,
        maxiter=maxiter,
        n_subsample=n_subsample,
        seed=seed,
        n_restarts=n_restarts,
        use_best_ambiguity=use_best_ambiguity,
        show_progress=show_progress,
        fwhm_scale_range=fwhm_scale_range,
        initial_guess_mode=initial_guess_mode,
        omega_axis=omega_axis,
    )
    return l1
