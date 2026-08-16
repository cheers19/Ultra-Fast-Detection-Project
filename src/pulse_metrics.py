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


def _peak_amplitude_index(e_t: np.ndarray) -> int:
    return int(np.argmax(np.abs(np.asarray(e_t, dtype=np.complex128).ravel())))


def _remove_global_phase_at_index(e_t: np.ndarray, index: int) -> np.ndarray:
    e = np.asarray(e_t, dtype=np.complex128).copy()
    idx = int(index)
    e *= np.exp(-1j * np.angle(e[idx]))
    return e


def canonicalize_field_tstar(
    e_t: np.ndarray,
    *,
    return_t_star: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """
    Ambiguity removal anchored at the temporal peak ``t_* = argmax_t |E(t)|``.

    1. ``t_*`` = index of ``|E|`` maximum.
    2. ``E <- E * exp(-i * arg E(t_*))``  (global phase).
    3. If more than half of ``|E|^2`` lies at ``t > t_*``: flip+conjugate, then repeat step 2
       at the new peak.
    4. ``E <- E / ||E||_2``  (global scale).
    """
    e = np.asarray(e_t, dtype=np.complex128).copy()

    t_star = _peak_amplitude_index(e)
    e = _remove_global_phase_at_index(e, t_star)

    e2 = np.abs(e) ** 2
    if float(np.sum(e2[t_star + 1 :])) > 0.5 * float(np.sum(e2)):
        e = np.flip(e).conj()
        t_star = _peak_amplitude_index(e)
        e = _remove_global_phase_at_index(e, t_star)

    nrm = np.linalg.norm(e)
    if nrm > 0:
        e /= nrm

    if return_t_star:
        return e, t_star
    return e


def canonicalize_field_mixed(
    e_t: np.ndarray,
    *,
    phase_mode: str = "t0",
    flip_mode: str = "re",
    zero_index: int | None = None,
) -> np.ndarray:
    """
    Mix phase-anchor and flip heuristics from the two generator notebooks.

    ``phase_mode``:
      - ``\"t0\"``: zero global phase at grid center (method 1 / ``canonicalize_field``)
      - ``\"tstar\"``: zero global phase at peak |E| (method 2 / ``canonicalize_field_tstar``)

    ``flip_mode``:
      - ``\"re\"``: flip+conj if Re-area to the right of the phase anchor exceeds the left
        (method 1)
      - ``\"energy\"``: flip+conj if more than half of |E|^2 lies after the phase anchor
        (method 2)

    Always finishes with L2 normalization. Does not remove time-shift ambiguity.
    """
    if phase_mode not in ("t0", "tstar"):
        raise ValueError(f"phase_mode must be 't0' or 'tstar' (got {phase_mode!r})")
    if flip_mode not in ("re", "energy"):
        raise ValueError(f"flip_mode must be 're' or 'energy' (got {flip_mode!r})")

    e = np.asarray(e_t, dtype=np.complex128).copy()
    n = e.shape[-1]
    z_grid = n // 2 if zero_index is None else int(zero_index)

    def _phase_anchor(field: np.ndarray) -> int:
        if phase_mode == "tstar":
            return _peak_amplitude_index(field)
        return z_grid

    anchor = _phase_anchor(e)
    e = _remove_global_phase_at_index(e, anchor)

    do_flip = False
    if flip_mode == "re":
        left = float(np.sum(e[:anchor].real))
        right = float(np.sum(e[anchor + 1 :].real))
        do_flip = right > left
    else:
        e2 = np.abs(e) ** 2
        do_flip = float(np.sum(e2[anchor + 1 :])) > 0.5 * float(np.sum(e2))

    if do_flip:
        e = np.flip(e).conj()
        anchor = _phase_anchor(e)
        e = _remove_global_phase_at_index(e, anchor)

    nrm = np.linalg.norm(e)
    if nrm > 0:
        e /= nrm
    return e


# Named mixes used by Data C canonicalization ablation:
# (1) phase=t0, flip=re      → method 1 / method 1
# (2) phase=t0, flip=energy  → method 1 / method 2
# (3) phase=tstar, flip=re   → method 2 / method 1
# (4) phase=tstar, flip=energy → method 2 / method 2
CANON_MIX_MODES: dict[str, tuple[str, str]] = {
    "t0_re": ("t0", "re"),
    "t0_energy": ("t0", "energy"),
    "tstar_re": ("tstar", "re"),
    "tstar_energy": ("tstar", "energy"),
}


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


# ---------------------------------------------------------------------------
# Fast best-ambiguity (batched GPU + FFT |E| shift). Legacy API above is kept.
# ---------------------------------------------------------------------------


def _best_shift_by_amplitude_fft(
    e_rec: np.ndarray,
    e_true: np.ndarray,
    *,
    max_shift: int | None = None,
) -> int:
    """Same objective as ``_best_shift_by_amplitude``, via linear FFT correlation."""
    e_rec = np.asarray(e_rec, dtype=np.complex128).ravel()
    e_true = np.asarray(e_true, dtype=np.complex128).ravel()
    n = e_true.size
    if e_rec.size != n:
        raise ValueError("e_rec and e_true must have the same length")
    lim = (n - 1) if max_shift is None else min(int(max_shift), n - 1)
    a = np.abs(e_rec)
    a_ref = np.abs(e_true)
    ref_norm = float(np.linalg.norm(a_ref)) + 1e-30
    n_fft = 2 * n - 1
    c = np.fft.irfft(np.fft.rfft(a_ref, n_fft) * np.conj(np.fft.rfft(a, n_fft)), n_fft)
    corr_full = np.concatenate([c[-(n - 1) :], c[:n]])  # lags -(n-1)..+(n-1)
    lags = np.arange(-(n - 1), n)
    a2 = a * a
    csum = np.concatenate([[0.0], np.cumsum(a2)])
    shift_norm = np.empty_like(corr_full)
    for i, k in enumerate(lags):
        if k >= 0:
            shift_norm[i] = np.sqrt(csum[n - k] - csum[0] + 1e-30)
        else:
            shift_norm[i] = np.sqrt(csum[n] - csum[-k] + 1e-30)
    scores = corr_full / (shift_norm * ref_norm)
    mask = np.abs(lags) <= lim
    best_i = int(np.argmax(np.where(mask, scores, -np.inf)))
    return int(lags[best_i])


def best_l1_ambiguity_params_fast(
    e_rec: np.ndarray,
    e_true: np.ndarray,
    *,
    n_phase: int = 128,
    max_shift: int | None = None,
) -> tuple[int, int, float]:
    """Return ``(base_kind, shift, phi)`` with FFT shift search (legacy-equivalent)."""
    e_true = np.asarray(e_true, dtype=np.complex128).ravel()
    true_packed = pack_complex_field(e_true)
    best_val = np.inf
    best = (0, 0, 0.0)
    for bi, base in enumerate(_ambiguity_bases(e_rec)):
        k = _best_shift_by_amplitude_fft(base, e_true, max_shift=max_shift)
        c = _shift_field_zeros(base, k)
        phi = global_phase_min_l1(c, e_true, n_phase=n_phase)
        val = _l1_packed_vs_true_packed(apply_global_phase(c, phi), true_packed)
        if val < best_val:
            best_val = val
            best = (bi, int(k), float(phi))
    return best


def best_l1_ambiguity_fast(
    e_rec: np.ndarray,
    e_true: np.ndarray,
    *,
    n_phase: int = 128,
    max_shift: int | None = None,
) -> float:
    """Fast best-ambiguity L1 (FFT |E| shift); same search space as ``best_l1_ambiguity``."""
    bi, k, phi = best_l1_ambiguity_params_fast(
        e_rec, e_true, n_phase=n_phase, max_shift=max_shift
    )
    base = _ambiguity_bases(e_rec)[bi]
    return _l1_packed_vs_true_packed(
        apply_global_phase(_shift_field_zeros(base, k), phi),
        pack_complex_field(np.asarray(e_true, dtype=np.complex128).ravel()),
    )


def best_l1_ambiguity_field_fast(
    e_rec: np.ndarray,
    e_true: np.ndarray,
    *,
    n_phase: int = 128,
    max_shift: int | None = None,
) -> np.ndarray:
    """Fast field variant of ``best_l1_ambiguity_field``."""
    bi, k, phi = best_l1_ambiguity_params_fast(
        e_rec, e_true, n_phase=n_phase, max_shift=max_shift
    )
    base = _ambiguity_bases(e_rec)[bi]
    return apply_global_phase(_shift_field_zeros(base, k), phi)


def best_l1_ambiguity_params_fast_torch(
    E_pred: "torch.Tensor",
    E_true: "torch.Tensor",
    *,
    n_phase: int = 128,
    max_shift: int | None = None,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Batched GPU best-ambiguity parameters.

    Returns
    -------
    base_kind : LongTensor [B]  (0=id, 1=conj, 2=flip+conj)
    shift : LongTensor [B]
    phi : Tensor [B]
    """
    if torch is None:
        raise ImportError("torch is required for best_l1_ambiguity_params_fast_torch")
    if E_pred.ndim != 2 or E_true.ndim != 2:
        raise ValueError("E_pred / E_true must be [B, 2N]")
    bsz, two_n = E_pred.shape
    n = two_n // 2
    device = E_pred.device
    dtype = E_pred.dtype

    e = torch.complex(E_pred[:, :n], E_pred[:, n:])
    e_t = torch.complex(E_true[:, :n], E_true[:, n:])
    bases = torch.stack([e, torch.conj(e), torch.flip(torch.conj(e), dims=[-1])], dim=1)
    a = torch.abs(bases)  # [B,3,N]
    a_ref = torch.abs(e_t)  # [B,N]
    lim = (n - 1) if max_shift is None else min(int(max_shift), n - 1)

    n_fft = 2 * n - 1
    A = torch.fft.rfft(a, n=n_fft)  # [B,3,F]
    R = torch.fft.rfft(a_ref, n=n_fft).unsqueeze(1)  # [B,1,F]
    c = torch.fft.irfft(R * torch.conj(A), n=n_fft)  # [B,3,n_fft]
    corr_full = torch.cat([c[..., -(n - 1) :], c[..., :n]], dim=-1)  # lags -(n-1)..+(n-1)

    a2 = a * a
    csum = torch.cumsum(a2, dim=-1)
    csum = torch.cat([torch.zeros(bsz, 3, 1, device=device, dtype=dtype), csum], dim=-1)
    lags = torch.arange(-(n - 1), n, device=device)
    # shift_norm[b,bi,i] for lag lags[i]
    # k>=0: sqrt(csum[..., n-k] - csum[..., 0])
    # k<0: sqrt(csum[..., n] - csum[..., -k])
    k_pos = lags.clamp(min=0)
    k_neg = (-lags).clamp(min=0)
    # gather: for each lag index
    idx_hi = (n - k_pos).long()  # for k>=0 use csum[n-k]; for k<0 unused
    idx_lo = k_neg.long()  # for k<0 use csum[-k]
    # Broadcast [2n-1] -> [B,3,2n-1]
    csum_exp = csum  # [B,3,N+1]
    # Build norms with a loop over lag dimension is OK for N=64; vectorize with gather
    norms_sq = torch.empty(bsz, 3, 2 * n - 1, device=device, dtype=dtype)
    for i, k in enumerate(range(-(n - 1), n)):
        if k >= 0:
            norms_sq[:, :, i] = csum_exp[:, :, n - k] - csum_exp[:, :, 0]
        else:
            norms_sq[:, :, i] = csum_exp[:, :, n] - csum_exp[:, :, -k]
    shift_norm = torch.sqrt(norms_sq + 1e-30)
    ref_norm = torch.linalg.vector_norm(a_ref, dim=-1).clamp_min(1e-30).view(bsz, 1, 1)
    scores = corr_full / (shift_norm * ref_norm)
    if lim < n - 1:
        scores = scores.masked_fill(lags.abs().view(1, 1, -1) > lim, -1e30)
    best_lag_idx = scores.argmax(dim=-1)  # [B,3]
    best_k = lags[best_lag_idx]  # [B,3]

    # Apply shifts (zero-pad) for each base, then phase grid L1
    phis = torch.arange(int(n_phase), device=device, dtype=dtype) * (
        2.0 * float(np.pi) / float(n_phase)
    )
    true_packed = E_true  # [B,2N]
    best_base = torch.zeros(bsz, dtype=torch.long, device=device)
    best_shift = torch.zeros(bsz, dtype=torch.long, device=device)
    best_phi = torch.zeros(bsz, dtype=dtype, device=device)
    best_val = torch.full((bsz,), float("inf"), device=device, dtype=dtype)

    for bi in range(3):
        base = bases[:, bi, :]  # [B,N]
        k_b = best_k[:, bi]  # [B]
        shifted = torch.zeros_like(base)
        for b in range(bsz):
            k = int(k_b[b].item())
            if k == 0:
                shifted[b] = base[b]
            elif abs(k) >= n:
                pass
            elif k > 0:
                shifted[b, k:] = base[b, : n - k]
            else:
                shifted[b, : n + k] = base[b, -k:]
        # phase sweep: [B, n_phase, N]
        rot = shifted.unsqueeze(1) * torch.exp(1j * phis.view(1, -1, 1))
        packed = torch.cat([rot.real, rot.imag], dim=-1)  # [B,P,2N]
        l1 = (packed - true_packed.unsqueeze(1)).abs().sum(dim=-1)  # [B,P]
        phi_idx = l1.argmin(dim=-1)
        val = l1.gather(1, phi_idx.unsqueeze(1)).squeeze(1)
        phi = phis[phi_idx]
        better = val < best_val
        best_val = torch.where(better, val, best_val)
        best_base = torch.where(better, torch.full_like(best_base, bi), best_base)
        best_shift = torch.where(better, k_b, best_shift)
        best_phi = torch.where(better, phi, best_phi)

    return best_base, best_shift, best_phi


def apply_ambiguity_params_torch(
    E_pred: "torch.Tensor",
    base_kind: "torch.Tensor",
    shift: "torch.Tensor",
    phi: "torch.Tensor",
) -> "torch.Tensor":
    """Apply per-sample ambiguity transforms to packed ``E_pred`` (differentiable w.r.t. E_pred)."""
    if torch is None:
        raise ImportError("torch is required")
    bsz, two_n = E_pred.shape
    n = two_n // 2
    out = []
    for i in range(bsz):
        e = torch.complex(E_pred[i, :n], E_pred[i, n:])
        bi = int(base_kind[i].item())
        k = int(shift[i].item())
        if bi == 1:
            e = torch.conj(e)
        elif bi == 2:
            e = torch.flip(torch.conj(e), dims=[0])
        if k == 0:
            pass
        elif abs(k) >= n:
            e = torch.zeros_like(e)
        elif k > 0:
            out_e = torch.zeros_like(e)
            out_e[k:] = e[: n - k]
            e = out_e
        else:
            out_e = torch.zeros_like(e)
            out_e[: n + k] = e[-k:]
            e = out_e
        e = e * torch.exp(1j * phi[i].to(dtype=E_pred.dtype))
        out.append(torch.cat([e.real, e.imag], dim=0))
    return torch.stack(out, dim=0)


def align_pred_best_l1_ambiguity_torch_fast(
    E_pred: "torch.Tensor",
    E_true: "torch.Tensor",
    *,
    n_phase: int = 128,
    max_shift: int | None = None,
) -> "torch.Tensor":
    """
    Align packed predictions via fast batched best-ambiguity (FFT shift + GPU phase).

    Discrete choice has no grad; applied transforms keep grad through ``E_pred``.
    """
    with torch.no_grad():
        base_kind, shift, phi = best_l1_ambiguity_params_fast_torch(
            E_pred.detach(),
            E_true.detach(),
            n_phase=n_phase,
            max_shift=max_shift,
        )
    return apply_ambiguity_params_torch(E_pred, base_kind, shift, phi)


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
