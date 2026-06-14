"""Pulse reconstruction metrics: δE (complex overlap) and L1 (packed Re/Im).

L1 convention (DeepFROG-style): sum of |errors| over 2N Re/Im samples per pulse
(no mean over 2N). Training averages only over batch.
"""

from __future__ import annotations

import numpy as np

from data_generation import phase_t_unwrapped_at_zero

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def _shift_field_zeros(e: np.ndarray, shift: int) -> np.ndarray:
    """Non-circular shift with zero padding (same convention as ``FROGNet`` / PCGPA)."""
    e = np.asarray(e, dtype=np.complex128).ravel().copy()
    n = e.size
    shift = int(shift)
    if shift == 0:
        return e
    if abs(shift) >= n:
        return np.zeros_like(e)
    out = np.zeros_like(e)
    if shift > 0:
        out[shift:] = e[: n - shift]
    else:
        k = -shift
        out[: n - k] = e[k:]
    return out


def _ambiguity_bases(e_rec: np.ndarray) -> list[np.ndarray]:
    """FROG field ambiguities without time shift: identity, conjugate, flip+conjugate."""
    e = np.asarray(e_rec, dtype=np.complex128).ravel()
    return [e, e.conj(), np.flip(e).conj()]


def _best_shift_by_amplitude(
    e_rec: np.ndarray,
    e_true: np.ndarray,
    *,
    max_shift: int | None = None,
) -> int:
    """Integer shift (zero-padded) maximizing overlap of ``|E_rec|`` with ``|E_true|``."""
    e_rec = np.asarray(e_rec, dtype=np.complex128).ravel()
    e_true = np.asarray(e_true, dtype=np.complex128).ravel()
    n = e_true.size
    if e_rec.size != n:
        raise ValueError("e_rec and e_true must have the same length")
    lim = (n - 1) if max_shift is None else min(int(max_shift), n - 1)
    a_ref = np.abs(e_true)
    ref_norm = float(np.linalg.norm(a_ref)) + 1e-30
    best_k = 0
    best_score = -np.inf
    for k in range(-lim, lim + 1):
        a_shift = np.abs(_shift_field_zeros(e_rec, k))
        denom = float(np.linalg.norm(a_shift)) + 1e-30
        score = float(np.dot(a_shift, a_ref)) / (denom * ref_norm)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _aligned_ambiguity_candidates(
    e_rec: np.ndarray,
    e_true: np.ndarray,
    *,
    max_shift: int | None = None,
) -> list[np.ndarray]:
    """
    FROG ambiguity variants aligned to truth before metric evaluation.

    For each of {E, E*, flip(E)*}: find best zero-padded time shift via |E|
    correlation, then return the shifted field.
    """
    e_true = np.asarray(e_true, dtype=np.complex128).ravel()
    return [
        _shift_field_zeros(
            base,
            _best_shift_by_amplitude(base, e_true, max_shift=max_shift),
        )
        for base in _ambiguity_bases(e_rec)
    ]


def _frog_ambiguity_candidates(e_rec: np.ndarray, n: int) -> list[np.ndarray]:
    """Legacy exhaustive zero-pad shifts (used only when truth is unavailable)."""
    e_rec = np.asarray(e_rec, dtype=np.complex128).ravel()
    if e_rec.size != n:
        raise ValueError("e_rec length must match n")
    candidates: list[np.ndarray] = []
    for base in _ambiguity_bases(e_rec):
        for k in range(-(n - 1), n):
            candidates.append(_shift_field_zeros(base, k))
    return candidates


def pack_complex_field(e_t: np.ndarray) -> np.ndarray:
    e_t = np.asarray(e_t)
    return np.concatenate([e_t.real, e_t.imag]).astype(np.float32)


def unpack_packed_field(e_packed: np.ndarray) -> np.ndarray:
    e_packed = np.asarray(e_packed)
    half = e_packed.shape[-1] // 2
    return e_packed[..., :half] + 1j * e_packed[..., half:]


def packed_batch_to_complex(E_packed) -> np.ndarray:
    """[B, 2N] float (torch or numpy) -> [B, N] complex numpy."""
    if torch is not None and isinstance(E_packed, torch.Tensor):
        x = E_packed.detach().cpu().numpy()
    else:
        x = np.asarray(E_packed)
    half = x.shape[-1] // 2
    return x[..., :half] + 1j * x[..., half:]


def canonicalize_field(
    e_t: np.ndarray,
    *,
    zero_index: int | None = None,
) -> np.ndarray:
    """Match ``data_generation`` global phase / time-flip conventions."""
    e = np.asarray(e_t, dtype=np.complex128).copy()
    n = e.shape[-1]
    z = n // 2 if zero_index is None else int(zero_index)
    phase_at_t0 = np.angle(e[z])
    e *= np.exp(-1j * phase_at_t0)
    left = np.sum(e[:z].real)
    right = np.sum(e[z + 1 :].real)
    if right > left:
        e = np.flip(e).conj()
        e *= np.exp(-1j * np.angle(e[z]))
    nrm = np.linalg.norm(e)
    if nrm > 0:
        e /= nrm
    return e


def delta_e_numpy(e_rec: np.ndarray, e_true: np.ndarray) -> float:
    """Complex overlap error δE (radians), phase-invariant."""
    e_rec = np.asarray(e_rec, dtype=np.complex128).ravel()
    e_true = np.asarray(e_true, dtype=np.complex128).ravel()
    dot = np.abs(np.vdot(e_rec, e_true))
    denom = np.linalg.norm(e_rec) * np.linalg.norm(e_true)
    return float(np.arccos(np.clip(dot / (denom + 1e-30), -1.0, 1.0)))


def similarity_error_from_delta_e(delta_e: float) -> float:
    """SIMILARITY_ERROR = 1 - cos(δE); lower is more similar."""
    return float(1.0 - np.cos(float(delta_e)))


def similarity_error_numpy(e_rec: np.ndarray, e_true: np.ndarray) -> float:
    """SIMILARITY_ERROR for direct alignment (no ambiguity search)."""
    return similarity_error_from_delta_e(delta_e_numpy(e_rec, e_true))


def best_delta_e_ambiguity(e_rec: np.ndarray, e_true: np.ndarray) -> float:
    """δE after conj/flip bases + |E|-guided zero-pad shift vs. truth."""
    e_true = np.asarray(e_true, dtype=np.complex128).ravel()
    return min(
        delta_e_numpy(c, e_true) for c in _aligned_ambiguity_candidates(e_rec, e_true)
    )


def best_similarity_error_ambiguity(e_rec: np.ndarray, e_true: np.ndarray) -> float:
    """Min SIMILARITY_ERROR over FROG ambiguity variants."""
    return similarity_error_from_delta_e(best_delta_e_ambiguity(e_rec, e_true))


def best_ambiguity_field(e_rec: np.ndarray, e_true: np.ndarray) -> np.ndarray:
    """Recovered field variant that minimizes δE vs. truth (for plotting)."""
    e_true = np.asarray(e_true, dtype=np.complex128).ravel()
    return min(
        _aligned_ambiguity_candidates(e_rec, e_true),
        key=lambda c: delta_e_numpy(c, e_true),
    )


def l1_packed_sum_numpy(e_rec: np.ndarray, e_true_packed: np.ndarray) -> float:
    """Sum |error| over packed Re/Im (raw alignment, no ambiguity search)."""
    e_r = np.asarray(e_rec, dtype=np.complex128).ravel()
    e_t = unpack_packed_field(e_true_packed)
    return float(np.abs(pack_complex_field(e_r) - pack_complex_field(e_t)).sum())


def _l1_packed_vs_true_packed(e_rec: np.ndarray, true_packed: np.ndarray) -> float:
    e_r = np.asarray(e_rec, dtype=np.complex128).ravel()
    return float(np.abs(pack_complex_field(e_r) - true_packed).sum())


def global_phase_min_l1(
    e_rec: np.ndarray,
    e_true: np.ndarray,
    *,
    n_phase: int = 128,
) -> float:
    """
    Phase φ that minimizes L1 on packed Re/Im for ``e_rec * exp(iφ)`` vs. ``e_true``.

    FROG intensity is unchanged by a global phase; this only affects the L1 metric.
    """
    e_rec = np.asarray(e_rec, dtype=np.complex128).ravel()
    e_true = np.asarray(e_true, dtype=np.complex128).ravel()
    true_packed = pack_complex_field(e_true)
    phis = np.linspace(0.0, 2.0 * np.pi, int(n_phase), endpoint=False)
    rot = e_rec[:, np.newaxis] * np.exp(1j * phis[np.newaxis, :])
    pred = np.concatenate([rot.real, rot.imag], axis=0)
    return float(phis[int(np.argmin(np.abs(pred - true_packed[:, np.newaxis]).sum(axis=0)))])


def apply_global_phase(e_t: np.ndarray, phi: float) -> np.ndarray:
    """Multiply the field by ``exp(i * phi)``."""
    return np.asarray(e_t, dtype=np.complex128) * np.exp(1j * float(phi))


def best_l1_ambiguity(e_rec: np.ndarray, e_true: np.ndarray) -> float:
    """
    Minimum L1 (packed Re/Im) over FROG ambiguities and a global phase on the recovery.

    Per candidate: conj/flip base, |E|-guided zero-pad shift, then φ ∈ [0, 2π) for L1.
    """
    e_true = np.asarray(e_true, dtype=np.complex128).ravel()
    true_packed = pack_complex_field(e_true)
    best = np.inf
    for c in _aligned_ambiguity_candidates(e_rec, e_true):
        phi = global_phase_min_l1(c, e_true)
        val = _l1_packed_vs_true_packed(apply_global_phase(c, phi), true_packed)
        if val < best:
            best = val
    return float(best)


def best_l1_ambiguity_field(e_rec: np.ndarray, e_true: np.ndarray) -> np.ndarray:
    """Field variant minimizing L1 (|E| shift + conj/flip + global phase)."""
    e_true = np.asarray(e_true, dtype=np.complex128).ravel()
    true_packed = pack_complex_field(e_true)
    best_c = None
    best_phi = 0.0
    best_val = np.inf
    for c in _aligned_ambiguity_candidates(e_rec, e_true):
        phi = global_phase_min_l1(c, e_true)
        val = _l1_packed_vs_true_packed(apply_global_phase(c, phi), true_packed)
        if val < best_val:
            best_val = val
            best_c = c
            best_phi = phi
    assert best_c is not None
    return apply_global_phase(best_c, best_phi)


def l1_packed_mae(
    e_rec: np.ndarray,
    e_true_packed: np.ndarray,
    *,
    use_best_ambiguity: bool = True,
    canonicalize: bool = False,
) -> float:
    """
    Sum |error| over packed Re/Im (‖E_pred − E_true‖₁ per pulse, no /2N).

    Default: ``use_best_ambiguity=True`` (conj/flip, |E| zero-pad shift, global phase).
  """
    e_r = np.asarray(e_rec, dtype=np.complex128).ravel()
    if use_best_ambiguity:
        e_t = unpack_packed_field(e_true_packed).ravel()
        return best_l1_ambiguity(e_r, e_t)
    e_t = unpack_packed_field(e_true_packed)
    if canonicalize:
        e_r = canonicalize_field(e_r)
        e_t = canonicalize_field(e_t)
    return float(np.abs(pack_complex_field(e_r) - pack_complex_field(e_t)).sum())


def prepare_frog_trace_for_plot(
    trace: np.ndarray,
    *,
    omega_axis: np.ndarray | None = None,
    num_points: int | None = None,
    dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shift omega for display (FROGNet raw FFT order → centered), with tau/omega axes.

    Matches ``pulses_generator_NB.ipynb``: ``fftshift(trace, axes=0)``,
    symmetric ``tau_axis``, ``extent`` for ``imshow(..., cmap='magma')``.
    """
    trace = np.asarray(trace)
    trace_plot = np.fft.fftshift(trace, axes=0)
    if omega_axis is not None:
        omega_plot = np.fft.fftshift(np.asarray(omega_axis, dtype=float))
        n = omega_plot.size
    else:
        if num_points is None or dt is None:
            raise ValueError("provide omega_axis or both num_points and dt")
        n = int(num_points)
        omega_plot = np.fft.fftshift(np.fft.fftfreq(n, dt)) * (2.0 * np.pi)
    num_tau = trace.shape[-1]
    tau_samples = np.linspace(-n // 2, n // 2, num_tau)
    tau_axis = tau_samples * float(dt) if dt is not None else tau_samples
    return trace_plot, tau_axis, omega_plot


def frog_trace_marginals(trace_plot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Marginals of a display-order FROG trace ``[N_omega, N_tau]``.

    Returns
    -------
    spectral_marginal : sum over delay (τ) → profile vs ω
    delay_marginal : sum over angular frequency (ω) → profile vs τ
    """
    trace_plot = np.asarray(trace_plot, dtype=np.float64)
    if trace_plot.ndim != 2:
        raise ValueError("trace_plot must be 2D [N_omega, N_tau]")
    spectral_marginal = trace_plot.sum(axis=1)
    delay_marginal = trace_plot.sum(axis=0)
    return spectral_marginal, delay_marginal


def phase_relative_to_center(e_t: np.ndarray, zero_index: int | None = None) -> np.ndarray:
    """Wrapped phase with φ=0 at ``zero_index`` (default N//2)."""
    e = np.asarray(e_t, dtype=np.complex128).ravel()
    z = e.size // 2 if zero_index is None else int(zero_index)
    return np.angle(e) - np.angle(e[z])


def unwrap_phases_for_overlay(
    e_rec: np.ndarray,
    e_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unwrapped phases for overlay plots (generation convention: φ(t=0)=0 on true).

    ``phase_true`` uses ``phase_t_unwrapped_at_zero``. ``phase_rec`` adds the
    relative phase ``unwrap(angle(e_rec·e_true*))`` anchored to 0 at t=0.
    """
    e_rec = np.asarray(e_rec, dtype=np.complex128).ravel()
    e_true = np.asarray(e_true, dtype=np.complex128).ravel()
    z = e_true.size // 2
    ph_true = phase_t_unwrapped_at_zero(e_true)
    d = np.unwrap(np.angle(e_rec * np.conj(e_true)))
    ph_rec = ph_true + (d - d[z])
    return ph_true, ph_rec


def delta_e_per_pulse_torch(E_rec, E_orig):
    """Complex overlap δE per sample; tensors [B, 2N] (Re then Im). Returns [B]."""
    if torch is None:
        raise ImportError("torch is required for delta_e_per_pulse_torch")
    half = E_rec.shape[-1] // 2
    Er_r, Er_i = E_rec[..., :half], E_rec[..., half:]
    Ei_r, Ei_i = E_orig[..., :half], E_orig[..., half:]
    dot_r = torch.sum(Er_r * Ei_r + Er_i * Ei_i, dim=-1)
    dot_i = torch.sum(Er_r * Ei_i - Er_i * Ei_r, dim=-1)
    norm_r = torch.sum(Er_r**2 + Er_i**2, dim=-1)
    norm_o = torch.sum(Ei_r**2 + Ei_i**2, dim=-1)
    abs_dot = torch.sqrt(dot_r**2 + dot_i**2)
    return torch.acos(torch.clamp(abs_dot / torch.sqrt(norm_r * norm_o), -1.0, 1.0))


def l1_packed_per_pulse_torch(E_pred, E_true):
    """Per-pulse L1 (sum |error| over Re/Im); tensors [B, 2N]. Returns [B]."""
    if torch is None:
        raise ImportError("torch is required for l1_packed_per_pulse_torch")
    return (E_pred - E_true).abs().sum(dim=-1)


def pulse_packed_l1_loss_torch(E_pred, E_true):
    """Training loss: sum over 2N per pulse, mean over batch."""
    if torch is None:
        raise ImportError("torch is required for pulse_packed_l1_loss_torch")
    return l1_packed_per_pulse_torch(E_pred, E_true).mean()


def snr_db_l1_loss_torch(snr_pred, snr_true) -> "torch.Tensor":
    """Mean L1 on SNR (dB); ``snr_true`` may be scalar or per-batch vector."""
    if torch is None:
        raise ImportError("torch is required for snr_db_l1_loss_torch")
    target = snr_true
    if not isinstance(target, torch.Tensor):
        target = torch.full_like(snr_pred, float(target))
    elif target.ndim == 0:
        target = target.expand_as(snr_pred)
    return (snr_pred - target).abs().mean()


def mean_delta_e_torch(E_rec, E_orig) -> float:
    """Batch mean δE for packed tensors [B, 2N]."""
    return float(delta_e_per_pulse_torch(E_rec, E_orig).mean().item())


def similarity_error_per_pulse_torch(E_rec, E_orig):
    """SIMILARITY_ERROR per sample; tensors [B, 2N]. Returns [B]."""
    if torch is None:
        raise ImportError("torch is required for similarity_error_per_pulse_torch")
    return 1.0 - torch.cos(delta_e_per_pulse_torch(E_rec, E_orig))


def mean_similarity_error_torch(E_rec, E_orig) -> float:
    """Batch mean SIMILARITY_ERROR for packed tensors [B, 2N]."""
    return float(similarity_error_per_pulse_torch(E_rec, E_orig).mean().item())


def snr_db_to_equivalent_n_pulses(
    snr_db: float,
    *,
    efficiency: float = 1e-12,
    photons_per_pulse: float = 1e12,
    pn1_over_ps1: float = 80.0,
) -> float:
    """
    Map trace SNR (dB) to equivalent pulse count ``N_eq``.

    Uses **amplitude** SNR throughout (``rho = 10^(SNR_dB/20)``):

    - Single-pulse reference: ``A_{s1} = eta * N_ph``,
      noise std ``sigma_{n1} = pn1_over_ps1 * A_{s1}``,
      ``rho_1 = A_{s1} / sigma_{n1} = 1 / pn1_over_ps1``.
    - ``N`` measurements: ``rho_N = rho_1 * sqrt(N)``  =>  ``N_eq = (rho_N / rho_1)^2``.
    """
    from trace_noise import snr_db_to_linear

    a_s1 = efficiency * photons_per_pulse
    snr1_linear = a_s1 / (pn1_over_ps1 * a_s1)
    snr_linear = snr_db_to_linear(snr_db)
    return float((snr_linear / snr1_linear) ** 2)


def trace_l1_sum_numpy(i_rec: np.ndarray, i_ref: np.ndarray) -> float:
    """L1 on trace: sum of |I_rec - I_ref| over all pixels (same convention as pulse L1 sum)."""
    a = np.asarray(i_rec, dtype=np.float64)
    b = np.asarray(i_ref, dtype=np.float64)
    return float(np.sum(np.abs(a - b)))
