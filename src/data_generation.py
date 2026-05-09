from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


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
