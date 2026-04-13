import sympy as sp
import numpy as np
from scipy import integrate

class Transform:
    """
    Handles coordinate transformations between the crystal and the pulse frames.
    
    Attributes:
        theta_p (float): The tilt angle used for analytical and numerical coordinate transformations.
    """
    def __init__(self, angle):
        """
        Initializes the Transform class with a specific tilt angle.
        
        Args:
            angle (float): The tilt angle (theta_p) in radians.
        """
        self.theta_p = angle

    def crystal_to_pulse(self, x, z):
        """
        Calculates the coordinates in the pulse frame based on the crystal coordinates.
        
        Args:
            x (float or numpy.ndarray): The x-coordinate in the crystal frame.
            z (float or numpy.ndarray): The z-coordinate in the crystal frame.
            
        Returns:
            tuple: A tuple containing (xp, zp), the transformed coordinates in the pulse frame.
        """
        xp = x * np.cos(self.theta_p) - z * np.sin(self.theta_p)
        zp = x * np.sin(self.theta_p) + z * np.cos(self.theta_p)
        return xp, zp  

class UltrafastPulse:
    """
    Represents an ultrafast optical pulse and provides methods for 
    analytical and numerical intensity and norm calculations.
    
    Attributes:
        w0 (float): Beam waist radius.
        tau (float): Pulse duration.
        zeta (float): Chirp coefficient.
        kapa (float, optional): Additional pulse parameter.
        alphaw (float, optional): Absorption coefficient at frequency w.
    """
    def __init__(self, w0, tau, zeta, kapa=None, alphaw=None):
        """
        Initializes the pulse parameters for both numerical and symbolic computations.
        """
        # --- Numerical Parameters (NumPy) ---
        self.w0 = w0
        self.tau = tau
        self.zeta = zeta
        self.kapa = kapa
        self.alphaw = alphaw 

        # --- Symbolic Definition (SymPy) ---
        self.sp_w0, self.sp_tau = sp.symbols('w0 tau', real=True, positive=True)
        self.sp_x, self.sp_y, self.sp_t = sp.symbols('x y t', real=True)
        self.sp_zeta = sp.symbols('zeta', real=True)
        self.sp_theta_p = sp.symbols('theta_p', real=True, negative=True)

    def get_envelope(self, xp, y, t, library=np):
        """
        Calculates the complex electric field envelope of the pulse.
        
        Args:
            xp (Symbol/float): Transformed x-coordinate in the pulse frame.
            y (Symbol/float): y-coordinate.
            t (Symbol/float): time coordinate.
            library (module): The library to use for calculations (np or sp).
            
        Returns:
            Expression/Complex: The complex envelope value at the given coordinates.
        """
        exp = library.exp
        j = 1j if library is np else sp.I
        
        spatial_part = exp(-(xp/(self.w0 if library is np else self.sp_w0))**2) * \
                       exp(-(y/(self.w0 if library is np else self.sp_w0))**2)
        
        tau_var = self.tau if library is np else self.sp_tau
        zeta_var = self.zeta if library is np else self.sp_zeta
        
        temporal_part = exp(-(t/tau_var)**2) * exp(-j * zeta_var * (t**2 / tau_var**2))
        
        return spatial_part * temporal_part
    
    def calculate_intensity_analytic(self):
        """
        Derives the symbolic expression for the pulse intensity at the crystal surface.
        
        Returns:
            sympy.Expr: Simplified symbolic expression for I = |E|^2.
        """
        # 1. Geometry of the surface
        z_surf = -self.sp_x * sp.tan(self.sp_theta_p)
        
        # 2. Transformation to pulse frame
        xp_sym = self.sp_x * sp.cos(self.sp_theta_p) - z_surf * sp.sin(self.sp_theta_p)
        
        # 3. Symbolic Electric Field
        field_sym = self.get_envelope(xp_sym, self.sp_y, self.sp_t, library=sp)
        
        print("\n--- Symbolic Envelope (field_sym) ---")
        sp.pprint(field_sym)
        
        # 4. Intensity Calculation: I = E * E_conj
        intensity_sym = sp.simplify(field_sym * sp.conjugate(field_sym))
        
        print("\n--- Symbolic Intensity (intensity_sym) ---")
        sp.pprint(intensity_sym)

        return intensity_sym

    def calculate_norm(self, transform):
        """
        Calculates the pulse norm (energy flux) using analytical integration, 
        with a numerical fallback if the analytical path fails.
        
        Args:
            transform (Transform): The transformation object containing the tilt angle.
            
        Returns:
            float: The calculated norm of the pulse.
        """
        try:
                # --- Path 1: Analytical Integration (SymPy) ---
                intensity_sym = self.calculate_intensity_analytic()
                
                variables = [self.sp_x, self.sp_y, self.sp_t]
                current_expr = intensity_sym
                
                # Nested integration over x, y, and t
                for var in variables:
                    raw_res = sp.integrate(current_expr, (var, -sp.oo, sp.oo))
                    
                    if isinstance(raw_res, sp.Piecewise):
                        current_expr = raw_res.args[0][0]
                    else:
                        current_expr = raw_res
                
                print("\n--- Analytic Expression for Norm ---")
                sp.pprint(current_expr)
                print("------------------------------------\n")
                
                final_norm = current_expr.subs({
                    self.sp_w0: self.w0,
                    self.sp_tau: self.tau,
                    self.sp_theta_p: transform.theta_p
                }).evalf()
                
                return float(final_norm)

        except Exception as e:
            print(f"Analytical integration failed: {e}")

        # --- Path 2: Numerical Integration (SciPy) ---
        print("Switching to numerical integration...")
        
        if not hasattr(transform, 'crystal_to_pulse'):
            raise AttributeError("Numerical fallback failed: 'transform' object is missing 'crystal_to_pulse'.")

        def integrand(t, y, x):
            z_surface = -x * np.tan(transform.theta_p)
            xp, _ = transform.crystal_to_pulse(x, z_surface)
            field_z0 = self.get_envelope(xp, y, t, library=np)
            return np.abs(field_z0)**2

        # Integration limits based on parameters
        t_lim = 9 * self.tau
        y_lim = 9 * self.w0
        x_lim = 9 * self.w0 / np.abs(np.cos(transform.theta_p))

        norm_num, _ = integrate.tplquad(
            integrand, 
            -x_lim, x_lim, 
            lambda x: -y_lim, lambda x: y_lim, 
            lambda x, y: -t_lim, lambda x, y: t_lim
        )
        return norm_num