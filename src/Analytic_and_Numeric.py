import sympy as sp
import numpy as np
from scipy import integrate

class Transform:
    def __init__(self, angle):
        self.theta_p = angle  # נחוץ לאנליטי 📐

    def crystal_to_pulse(self, x, z):
        # חישוב xp (הקואורדינטה במערכת הפולס) - נחוץ לנומרי 🔄
        xp = x * np.cos(self.theta_p) - z * np.sin(self.theta_p)
        zp = x * np.sin(self.theta_p) + z * np.cos(self.theta_p)
        return xp, zp  

class UltrafastPulse:
    def __init__(self, w0, tau, zeta, kapa=None, alphaw=None):
        # --- פרמטרים מספריים (NumPy) ---
        self.w0 = w0      # רדיוס האלומה 🎯
        self.tau = tau    # משך הפולס ⏱️
        self.zeta = zeta  # מקדם ה-Chirp 〰️
        self.kapa = kapa
        self.alphaw = alphaw # the Absorption Coefficient of the fundemntal field (at freq w). 

        # --- הגדרת סימבולים (SymPy) עם הנחות עבודה ---
        self.sp_w0, self.sp_tau = sp.symbols('w0 tau', real=True, positive=True)
        self.sp_x, self.sp_y, self.sp_t = sp.symbols('x y t', real=True)
        self.sp_zeta = sp.symbols('zeta', real=True)
        
        # הנחה מפורשת: theta_p הוא ממשי ושלילי 📐
        self.sp_theta_p = sp.symbols('theta_p', real=True, negative=True)

    def get_envelope(self, xp, y, t, library=np):
        exp = library.exp
        j = 1j if library is np else sp.I
        
        spatial_part = exp(-(xp/(self.w0 if library is np else self.sp_w0))**2) * \
                       exp(-(y/(self.w0 if library is np else self.sp_w0))**2)
        
        tau_var = self.tau if library is np else self.sp_tau
        zeta_var = self.zeta if library is np else self.sp_zeta
        
        temporal_part = exp(-(t/tau_var)**2) * exp(-j * zeta_var * (t**2 / tau_var**2))
        
        return spatial_part * temporal_part
    
    def calculate_intensity_analytic(self):
        # 1. הגדרת הגאומטריה של המשטח (z כפונקציה של x והזווית)
        z_surf = -self.sp_x * sp.tan(self.sp_theta_p)
        
        # 2. מעבר למערכת הצירים של הפולס (xp)
        # xp = x*cos(theta) - z*sin(theta)
        xp_sym = self.sp_x * sp.cos(self.sp_theta_p) - z_surf * sp.sin(self.sp_theta_p)
        
        # 3. קבלת השדה החשמלי הסימבולי ⚡
        field_sym = self.get_envelope(xp_sym, self.sp_y, self.sp_t, library=sp)
        
        print("\n--- Symbolic Envelope (field_sym) ---")
        sp.pprint(field_sym)
        
        # 4. חישוב העוצמה: I = E * E_conj 💥
        intensity_sym = sp.simplify(field_sym * sp.conjugate(field_sym))
        
        print("\n--- Symbolic Intensity (intensity_sym) ---")
        sp.pprint(intensity_sym)

        return intensity_sym

    def calculate_norm(self, transform):
        try:
                # --- נתיב 1: חישוב אנליטי (SymPy) ---
                # 1. השגת הביטוי הסימבולי של העוצמה
                intensity_sym = self.calculate_intensity_analytic()
                
                variables = [self.sp_x, self.sp_y, self.sp_t]
                current_expr = intensity_sym
                
                # 2. אינטגרציה מדורגת (Nested Integration)
                for var in variables:
                    raw_res = sp.integrate(current_expr, (var, -sp.oo, sp.oo))
                    
                    # חילוץ הענף המרכזי מתוך Piecewise למניעת RecursionError 🧹
                    if isinstance(raw_res, sp.Piecewise):
                        current_expr = raw_res.args[0][0]
                    else:
                        current_expr = raw_res
                
                # 3. הדפסת התוצאה האנליטית (לבקשתך) 🖨️
                print("\n--- Analytic Expression for Norm ---")
                sp.pprint(current_expr)
                print("------------------------------------\n")
                
                # 4. הצבת ערכים והפיכה ל-float
                final_norm = current_expr.subs({
                    self.sp_w0: self.w0,
                    self.sp_tau: self.tau,
                    self.sp_theta_p: transform.theta_p
                }).evalf()
                
                return float(final_norm)

        except Exception as e:
            print(f"Analytical integration failed: {e}")
            # אם נכשלנו, הקוד ימשיך לנתיב הנומרי למטה

        # --- נתיב 2: נתיב Fall-back: חישוב נומרי (SciPy) ---
        print("Switching to numerical integration...")
        
        # וידוא שהטרנספורמציה תומכת בחישוב נומרי
        if not hasattr(transform, 'crystal_to_pulse'): # מוודא שהאובייקט transform מכיל את הפונקציה ההכרחית crystal_to_pulse
            raise AttributeError("Numerical fallback failed: 'transform' object is missing 'crystal_to_pulse'. Check analytical path.")

        # --- נתיב נומרי מתוקן ---
        def integrand(t, y, x):
            z_surface = -x * np.tan(transform.theta_p) # מבחינה מתמטית, זוהי משוואה של ישר העובר דרך הראשית ($z = mx$), כאשר השיפוע $m$ נקבע על ידי הזווית $\theta_p$
            # שימוש במתודת הטרנספורמציה לחישוב הקואורדינטות המקומיות
            xp, _ = transform.crystal_to_pulse(x, z_surface) # describing the input pulse in the crystal coordinates: xp(x,z=z*tan(ThetaP))
            # הוספת self. לקריאה למתודה
            field_z0 = self.get_envelope(xp, y, t, library=np) # this is the field at the crystal surface! we don't propagate it. it's only for Flux (norm) calculation.
            return np.abs(field_z0)**2

        # הגדרת הגבולות בתוך המתודה
        t_lim = 9 * self.tau
        y_lim = 9 * self.w0
        x_lim = 9 * self.w0 / np.abs(np.cos(transform.theta_p)) # when illuminating the crystal at angle ThetaP, the beam width transforms from w to w/cos(ThetaP)

        norm_num, _ = integrate.tplquad(
            integrand, 
            -x_lim, x_lim, 
            lambda x: -y_lim, lambda x: y_lim, 
            lambda x, y: -t_lim, lambda x, y: t_lim
        )
        return norm_num
