import sympy as sp
from Analytic_and_Numeric import UltrafastPulse  # import the class from pulses.py

# Now you can run the code you wanted
pulse = UltrafastPulse(w0=0.1, tau=100, zeta=0.5)

# Define the geometric context
z_surf = -pulse.sp_x * sp.tan(pulse.sp_theta_p)
xp_sym = pulse.sp_x * sp.cos(pulse.sp_theta_p) - z_surf * sp.sin(pulse.sp_theta_p)

# Get the envelope and print it (in a regular script, prefer pprint or print)
envelope_sym = pulse.get_envelope(xp_sym, pulse.sp_y, pulse.sp_t, library=sp)

print("Simplified Pulse Envelope Expression:")
sp.pprint(sp.simplify(envelope_sym))

# 1. Compute intensity (Abs^2) - with the assumptions, the imaginary part should vanish
intensity_sym = sp.simplify(sp.Abs(envelope_sym)**2)

print("Symbolic Intensity (expecting no imaginary parts):")
sp.pprint(intensity_sym)

# 2. Attempt integration over all space and time
norm_analytical = sp.integrate(intensity_sym, 
                               (pulse.sp_x, -sp.oo, sp.oo), 
                               (pulse.sp_y, -sp.oo, sp.oo), 
                               (pulse.sp_t, -sp.oo, sp.oo))

print("\nAnalytical Norm Result:")
sp.pprint(norm_analytical)
