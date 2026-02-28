import sympy as sp
from Analytic_and_Numeric import UltrafastPulse  # ייבוא המחלקה מהקובץ pulses.py

# עכשיו אפשר להריץ את הקוד שרצית
pulse = UltrafastPulse(w0=0.1, tau=100, zeta=0.5)

# הגדרת הקשר הגיאומטרי
z_surf = -pulse.sp_x * sp.tan(pulse.sp_theta_p)
xp_sym = pulse.sp_x * sp.cos(pulse.sp_theta_p) - z_surf * sp.sin(pulse.sp_theta_p)

# קבלת המעטפת והדפסה (בסקריפט רגיל כדאי להשתמש ב-pprint או print)
envelope_sym = pulse.get_envelope(xp_sym, pulse.sp_y, pulse.sp_t, library=sp)

print("Simplified Pulse Envelope Expression:")
sp.pprint(sp.simplify(envelope_sym))

# 1. חישוב העוצמה (Abs^2) - בזכות ההנחות האיבר המדומה אמור להתבטל
intensity_sym = sp.simplify(sp.Abs(envelope_sym)**2)

print("Symbolic Intensity (expecting no imaginary parts):")
sp.pprint(intensity_sym)

# 2. ניסיון אינטגרציה על פני כל המרחב והזמן
norm_analytical = sp.integrate(intensity_sym, 
                               (pulse.sp_x, -sp.oo, sp.oo), 
                               (pulse.sp_y, -sp.oo, sp.oo), 
                               (pulse.sp_t, -sp.oo, sp.oo))

print("\nAnalytical Norm Result:")
sp.pprint(norm_analytical)