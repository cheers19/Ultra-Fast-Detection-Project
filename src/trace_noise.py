"""
Trace-domain noise injection (pluggable; independent of ``utils.py``).

Replace or extend with Poisson / mixed models as needed; keep the same call
signature used by the training notebook:
``noisy = add_trace_noise(trace_clean, snr_db)``.
"""

from __future__ import annotations

import torch


def add_trace_noise_awgn(trace_clean: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Additive white Gaussian noise on intensity.

    SNR (dB) defined from mean squared signal vs. noise variance (same structure
    as common FROG denoising setups):

        SNR_linear = mean(I^2) / sigma_n^2
    """
    signal_power = torch.mean(trace_clean**2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(trace_clean) * torch.sqrt(noise_power)
    return trace_clean + noise
