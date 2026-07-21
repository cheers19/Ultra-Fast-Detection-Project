"""SHG-FROG forward model with zero-padded FFT and central frequency crop."""

from __future__ import annotations

import torch
import torch.nn as nn


class FROGNetPadded(nn.Module):
    """
    Same SHG-FROG pipeline as ``FROGNet``, but FFT along time uses ``n_fft`` bins
    and returns the central ``n_spectral_points`` frequency rows (display order).
    """

    def __init__(
        self,
        num_delay_steps: int,
        n_fft: int,
        n_spectral_points: int,
    ):
        super().__init__()
        if num_delay_steps <= 0:
            raise ValueError("num_delay_steps must be positive.")
        if n_fft < n_spectral_points:
            raise ValueError("n_fft must be >= n_spectral_points.")
        self.num_delay_steps = int(num_delay_steps)
        self.n_fft = int(n_fft)
        self.n_spectral_points = int(n_spectral_points)
        self._spec_center = self.n_fft // 2
        self._spec_half = self.n_spectral_points // 2

    @staticmethod
    def _to_complex(e_t: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(e_t):
            return e_t
        if e_t.size(-1) % 2 != 0:
            raise ValueError(
                "Real-valued input must have even last dimension: [real(N), imag(N)]."
            )
        half = e_t.size(-1) // 2
        return torch.complex(e_t[..., :half], e_t[..., half:])

    def _build_delay_indices(self, n_t: int, device: torch.device) -> torch.Tensor:
        delays = torch.linspace(
            -n_t // 2,
            n_t // 2,
            steps=self.num_delay_steps,
            device=device,
        )
        return torch.round(delays).to(torch.long)

    @staticmethod
    def _shift_with_zeros(e_complex: torch.Tensor, shift: int) -> torch.Tensor:
        _, n_t = e_complex.shape
        out = torch.zeros_like(e_complex)
        if shift == 0:
            return e_complex
        if abs(shift) >= n_t:
            return out
        if shift > 0:
            out[:, shift:] = e_complex[:, : n_t - shift]
        else:
            k = -shift
            out[:, : n_t - k] = e_complex[:, k:]
        return out

    def forward(self, e_t: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        i_trace : [B, N_spectral, N_tau] intensity, fftshifted along omega.
        """
        e_complex = self._to_complex(e_t)
        if e_complex.dim() != 2:
            raise ValueError("Input must be 2D tensor: [batch, time].")

        _, n_t = e_complex.shape
        delays = self._build_delay_indices(n_t=n_t, device=e_complex.device)

        delayed_fields = torch.stack(
            [self._shift_with_zeros(e_complex, shift=int(tau.item())) for tau in delays],
            dim=-1,
        )
        g_t_tau = e_complex.unsqueeze(-1) * delayed_fields
        g_w_tau = torch.fft.fft(g_t_tau, n=self.n_fft, dim=1)

        i_trace = g_w_tau.real.pow(2) + g_w_tau.imag.pow(2)
        i_trace = torch.fft.fftshift(i_trace, dim=1)

        c0 = self._spec_center - self._spec_half
        c1 = self._spec_center + self._spec_half
        return i_trace[:, c0:c1, :]
