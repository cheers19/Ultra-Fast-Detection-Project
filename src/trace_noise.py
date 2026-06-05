"""
Trace-domain noise injection (pluggable; independent of ``utils.py``).

SNR convention (amplitude, not power)
-------------------------------------
Per trace pixel the **signal** is the SHG-FROG **intensity** ``I`` (non-negative).
The scalar signal level is the **mean over all pixels**:

    mu_s = mean(I_clean)

For a requested trace SNR in dB, the **linear amplitude SNR** is

    rho = 10^(SNR_dB / 20)

Independent AWGN on every pixel uses standard deviation

    sigma_n = mu_s / rho

so that rho = mu_s / sigma_n (ratio of mean signal to noise std).

Multi-pulse scaling (used for equivalent-N plots, not inside this module):

    mean(I)_N = N * mean(I)_1
    mean(sigma)_N = sqrt(N) * mean(sigma)_1
    rho_N = rho_1 * sqrt(N)
    N = (rho_N / rho_1)^2

Call signature used by notebooks:
``noisy = add_trace_noise(trace_clean, snr_db)``.
"""

from __future__ import annotations

import math

import torch


def snr_db_to_linear(snr_db: float) -> float:
    """Linear **amplitude** SNR from decibels: ``rho = 10^(snr_db/20)``."""
    return 10.0 ** (float(snr_db) / 20.0)


def snr_linear_to_db(snr_linear: float) -> float:
    """Decibels from linear amplitude SNR: ``SNR_dB = 20 log10(rho)``."""
    rho = float(snr_linear)
    if rho <= 0.0:
        raise ValueError("snr_linear must be positive")
    return 20.0 * math.log10(rho)


def trace_mean_signal_amplitude(trace: torch.Tensor) -> torch.Tensor:
    """Mean FROG intensity over all pixels (signal level for SNR)."""
    return torch.mean(trace)


def awgn_std_for_snr(trace_clean: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    AWGN standard deviation per pixel for target amplitude SNR.

    ``sigma_n = mean(I) / 10^(SNR_dB/20)``.
    """
    mu_s = trace_mean_signal_amplitude(trace_clean)
    rho = snr_db_to_linear(snr_db)
    return mu_s / rho


def add_trace_noise_awgn(trace_clean: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Additive white Gaussian noise on FROG intensity.

    Each pixel: ``I_noisy = I_clean + n``, ``n ~ N(0, sigma_n^2)`` with
    ``sigma_n = mean(I_clean) / 10^(SNR_dB/20)``.
    """
    sigma_n = awgn_std_for_snr(trace_clean, snr_db)
    noise = torch.randn_like(trace_clean) * sigma_n
    return trace_clean + noise
