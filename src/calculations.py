import numpy as np
from scipy import integrate

class UltrafastPulse:
    def __init__(self, w0, tau, zeta, kapa=None, alphaw=None, method="analytical"):
        # Physical parameters
        self.w0 = w0
        self.tau = tau
        self.zeta = zeta
        self.kapa = kapa
        self.alphaw = alphaw
        self.method = method  # preferred calculation method

    def get_envelope(self, xp, y, t):
        """Compute the pulse envelope in its coordinate system."""
        spatial_part = np.exp(-(xp/self.w0)**2) * np.exp(-(y/self.w0)**2)
        temporal_part = np.exp(-(t/self.tau)**2) * np.exp(-1j * self.zeta * (t**2 / self.tau**2))
        return spatial_part * temporal_part

    def calculate_norm(self, transform):
        """Compute the normalization factor using the selected method."""
        if self.method == "analytical":
            # Analytical formula from Mathematica (note use of np.abs and np.pi)
            # Note: verify this expression matches your reference exactly
            term1 = np.pi**1.5 * (self.w0**2 * self.tau) / (2 * np.sqrt(2))
            # Per your reference, there is dependence on the cosine of the angles:
            cos_tp = np.cos(transform.theta_p)
            norm = term1 * np.abs(cos_tp) 
            return norm

        else:  # numerical method
            def integrand(t, y, x):
                z_surface = -x * np.tan(transform.theta_p)
                xp, _ = transform.crystal_to_pulse(x, z_surface)
                field = self.get_envelope(xp, y, t)
                return np.abs(field)**2

            # Finite bounds (9x the characteristic width)
            t_lim = 9 * self.tau
            y_lim = 9 * self.w0
            x_lim = 9 * self.w0 / np.abs(np.cos(transform.theta_p))

            norm, _ = integrate.tplquad(integrand, 
                                        -x_lim, x_lim, 
                                        lambda x: -y_lim, lambda x: y_lim, 
                                        lambda x, y: -t_lim, lambda x, y: t_lim)
            return norm
        
def get_dipole(self, x, z, t, transform, norm):
        """Compute the nonlinear source (dipole) at a point in space and time."""
        # 1. Compute the pump field at the specific point (z_surface = z)
        # Assumes the user passes the current z inside the crystal
        pump = self.get_pump(x, y=0, z=z, t=t, transform=transform, norm=norm)
        
        # 2. Absorption/decay term as a function of depth z
        # Use cosine of incidence angle to correct the optical path length
        absorption = np.exp(-self.alphaw * z / np.cos(transform.theta_p))
        
        # 3. Combine per the formula: i * kapa * pump^2 * absorption
        # The 1j factor represents the additional phase from the nonlinear process
        return 1j * self.kapa * (pump**2) * absorption
