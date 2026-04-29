import torch
import torch.nn as nn


class FROGNet(nn.Module):
    """
    Non-parametric differentiable SHG-FROG forward model.

    Implements steps 1-6 from the paper:
    1) receive E(t)
    2) create delayed replicas E(t-tau_j)
    3) multiply E(t) * E(t-tau_j)
    4) FFT along time
    5) intensity |.|^2
    6) collect into I(omega, tau)

    Notes for requirements:
    - Differentiable (requirement 8): built from torch ops with autograd support.
    - Constant/non-learned (requirement 10): no trainable parameters.
    """

    def __init__(self, num_delay_steps: int):
        super().__init__()
        if num_delay_steps <= 0:
            raise ValueError("num_delay_steps must be positive.")
        self.num_delay_steps = int(num_delay_steps)

    @staticmethod
    def _to_complex(e_t: torch.Tensor) -> torch.Tensor:
        """
        Accept either:
        - complex tensor [..., N]
        - stacked real/imag tensor [..., 2N], where first half is real and second half imag
        Returns complex tensor [..., N].
        """
        if torch.is_complex(e_t):
            return e_t

        # This even-length check is only for packed real/imag inputs.
        # Complex inputs return above and bypass this branch.
        if e_t.size(-1) % 2 != 0:
            raise ValueError(
                "Real-valued input must have even last dimension: [real(N), imag(N)]."
            )

        half = e_t.size(-1) // 2
        real = e_t[..., :half]
        imag = e_t[..., half:]
        return torch.complex(real, imag)

    def _build_delay_indices(self, n_t: int, device: torch.device) -> torch.Tensor:
        """
        Build integer delay indices tau_j over [-n_t//2, ..., n_t//2] with length num_delay_steps.
        """
        delays = torch.linspace(
            -n_t // 2,
            n_t // 2,
            steps=self.num_delay_steps,
            device=device,
        )
        return torch.round(delays).to(torch.long)

    @staticmethod
    def _shift_with_zeros(e_complex: torch.Tensor, shift: int) -> torch.Tensor:
        """
        Non-circular temporal shift with zero padding.
        Returns E(t-shift) on the sampled grid.
        """
        bsz, n_t = e_complex.shape
        out = torch.zeros_like(e_complex)

        if shift == 0:
            return e_complex
        if abs(shift) >= n_t:
            return out

        if shift > 0:
            # Right shift: out[t] = in[t-shift], left edge is zero-padded.
            out[:, shift:] = e_complex[:, : n_t - shift]
        else:
            # Left shift: out[t] = in[t-shift], right edge is zero-padded.
            k = -shift
            out[:, : n_t - k] = e_complex[:, k:]
        return out

    def forward(self, e_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e_t: shape [B, N] complex OR [B, 2N] real/imag-packed.

        Returns:
            i_trace: SHG-FROG trace with shape [B, N_omega, N_tau]
                     where N_omega=N and N_tau=num_delay_steps.
        """
        # Step 1: input pulse E(t)
        e_complex = self._to_complex(e_t)
        if e_complex.dim() != 2:
            raise ValueError("Input must be 2D tensor: [batch, time].")

        _, n_t = e_complex.shape
        delays = self._build_delay_indices(n_t=n_t, device=e_complex.device)

        # Step 2: delayed replicas E(t - tau_j) for all j (non-circular, zero-padded)
        delayed_fields = torch.stack(
            [self._shift_with_zeros(e_complex, shift=int(tau.item())) for tau in delays],
            dim=-1,
        )  # [B, N_t, N_tau]

        # Step 3: nonlinear SHG product field G_j(t) = E(t) * E(t - tau_j)
        e_expanded = e_complex.unsqueeze(-1)  # [B, N_t, 1]
        g_t_tau = e_expanded * delayed_fields  # [B, N_t, N_tau]

        # Step 4: Fourier transform over time axis t -> omega
        g_w_tau = torch.fft.fft(g_t_tau, dim=1)  # [B, N_omega, N_tau]

        # Step 5: spectral intensity I(omega, tau) = |G~(omega, tau)|^2
        i_trace = g_w_tau.real.pow(2) + g_w_tau.imag.pow(2)

        # Step 6: return the 2D trace matrix I(omega, tau)
        return i_trace


if __name__ == "__main__":
    # Minimal smoke test
    torch.manual_seed(0)
    bsz, n_t, n_tau = 2, 64, 64
    e_realimag = torch.randn(bsz, 2 * n_t, dtype=torch.float32, requires_grad=True)

    frognet = FROGNet(num_delay_steps=n_tau)
    i_out = frognet(e_realimag)  # [B, N_omega, N_tau]

    loss = i_out.mean()
    loss.backward()  # proves gradients pass through the whole model

    print("Output shape:", tuple(i_out.shape))
    print("Has learnable parameters:", any(p.requires_grad for p in frognet.parameters()))
