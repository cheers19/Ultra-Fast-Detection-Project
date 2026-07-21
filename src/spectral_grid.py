"""Spectral grid helpers (Mathematica-aligned target resolution + legacy FFT grid)."""

from __future__ import annotations

import numpy as np


def compute_sigma_t_center(
    t_total: float,
    n_spikes: int,
    pulse_temporal_fraction: float,
) -> float:
    if n_spikes < 2:
        raise ValueError(
            f"n_spikes must be >= 2 (got {n_spikes}); "
            "sigma_t_center uses sqrt(log2(N_spikes)) which requires N_spikes > 1."
        )
    if pulse_temporal_fraction <= 0:
        raise ValueError(
            f"pulse_temporal_fraction must be > 0 (got {pulse_temporal_fraction})."
        )
    pulse_effective_width = pulse_temporal_fraction * t_total
    return pulse_effective_width / (2.0 * np.sqrt(np.log2(n_spikes)))


def compute_de_new_target(
    h: float,
    sigma_t_center: float,
    fraction_from_nyquist: float,
) -> float:
    if not (np.isfinite(sigma_t_center) and sigma_t_center > 0):
        raise ValueError(
            f"sigma_t_center must be finite and > 0 (got {sigma_t_center!r})."
        )
    if fraction_from_nyquist <= 0:
        raise ValueError(
            f"fraction_from_nyquist must be > 0 (got {fraction_from_nyquist!r})."
        )
    return fraction_from_nyquist * h / (4.0 * np.pi * sigma_t_center)


def compute_legacy_spectral_grid(
    n_points: int,
    t_total: float,
    h: float,
) -> tuple[float, float, float]:
    """Legacy full N-point FFT grid (pre upsample/crop), for display only."""
    dt = t_total / (n_points - 1)
    energy = np.fft.fftshift(np.fft.fftfreq(n_points, d=dt) * h)
    de_old = float(np.mean(np.diff(energy)))
    e_extreme_old = float(np.max(np.abs(energy)))
    return dt, de_old, e_extreme_old


def build_spectral_plot_grid(
    dt: float,
    n_spectral_points: int,
    de_new_target: float,
    h: float,
) -> dict:
    """
    Derive n_fft from target bin spacing, crop central bins, return axes + metrics.
    """
    denom = de_new_target * dt
    if not (np.isfinite(denom) and denom > 0):
        raise ValueError(
            f"Cannot derive N_FFT: de_new_target*dt must be finite and > 0 "
            f"(de_new_target={de_new_target!r}, dt={dt!r}). "
            "Check N_SPIKES >= 2 and PULSE_TEMPORAL_FRACTION > 0."
        )
    n_fft_est = h / denom
    if not np.isfinite(n_fft_est):
        raise ValueError(
            f"Cannot derive N_FFT: h/(de_new_target*dt) is not finite "
            f"(value={n_fft_est!r})."
        )
    n_fft = int(round(n_fft_est))
    if n_fft % 2:
        n_fft += 1  # even FFT length (symmetric crop around DC)
    n_fft = max(n_fft, n_spectral_points)

    energy_hi = np.fft.fftshift(np.fft.fftfreq(n_fft, d=dt) * h)
    spec_center = n_fft // 2
    spec_half = n_spectral_points // 2
    spec_idx = slice(spec_center - spec_half, spec_center + spec_half)
    energy_ev_relative = energy_hi[spec_idx]

    de_new_actual = float(np.mean(np.diff(energy_ev_relative)))
    e_extreme_new = float(np.max(np.abs(energy_ev_relative)))

    return {
        "n_fft": n_fft,
        "spec_idx": spec_idx,
        "energy_ev_relative": energy_ev_relative,
        "de_new_target": de_new_target,
        "de_new_actual": de_new_actual,
        "e_extreme_new": e_extreme_new,
    }


def fft_to_plot_spectrum(f_t: np.ndarray, n_fft: int, spec_idx: slice) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft(f_t, n=n_fft))[spec_idx]
