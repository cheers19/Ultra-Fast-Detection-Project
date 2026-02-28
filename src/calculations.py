import numpy as np
from scipy import integrate

class UltrafastPulse:
    def __init__(self, w0, tau, zeta, kapa=None, alphaw=None, method="analytical"):
        # פרמטרים פיזיקליים
        self.w0 = w0
        self.tau = tau
        self.zeta = zeta
        self.kapa = kapa
        self.alphaw = alphaw
        self.method = method # שמירת שיטת החישוב המועדפת

    def get_envelope(self, xp, y, t):
        """חישוב מעטפת הפולס במערכת הצירים שלו"""
        spatial_part = np.exp(-(xp/self.w0)**2) * np.exp(-(y/self.w0)**2)
        temporal_part = np.exp(-(t/self.tau)**2) * np.exp(-1j * self.zeta * (t**2 / self.tau**2))
        return spatial_part * temporal_part

    def calculate_norm(self, transform):
        """חישוב מקדם הנרמול בשיטה הנבחרת"""
        if self.method == "analytical":
            # הנוסחה האנליטית מ-Mathematica (שים לב לשימוש ב-np.abs ו-np.pi)
            # הערה: וודא שביטוי זה תואם בדיוק למה שקיבלת בתמונה
            term1 = np.pi**1.5 * (self.w0**2 * self.tau) / (2 * np.sqrt(2))
            # לפי התמונה שלך, יש שם תלות בקוסינוס של הזוויות:
            cos_tp = np.cos(transform.theta_p)
            norm = term1 * np.abs(cos_tp) 
            return norm

        else: # שיטה נומרית
            def integrand(t, y, x):
                z_surface = -x * np.tan(transform.theta_p)
                xp, _ = transform.crystal_to_pulse(x, z_surface)
                field = self.get_envelope(xp, y, t)
                return np.abs(field)**2

            # הגדרת גבולות סופיים (פי 9 מהרוחב האופייני)
            t_lim = 9 * self.tau
            y_lim = 9 * self.w0
            x_lim = 9 * self.w0 / np.abs(np.cos(transform.theta_p))

            norm, _ = integrate.tplquad(integrand, 
                                        -x_lim, x_lim, 
                                        lambda x: -y_lim, lambda x: y_lim, 
                                        lambda x, y: -t_lim, lambda x, y: t_lim)
            return norm
        
def get_dipole(self, x, z, t, transform, norm):
        """חישוב המקור הלא-ליניארי (Dipole) בנקודה במרחב ובזמן"""
        # 1. חישוב שדה הפמפום בנקודה הספציפית (z_surface = z)
        # אנחנו מניחים כאן שהמשתמש מעביר את ה-z הנוכחי בתוך הגביש
        pump = self.get_pump(x, y=0, z=z, t=t, transform=transform, norm=norm)
        
        # 2. חישוב איבר הבליעה/דעיכה לפי העומק z
        # שימוש בקוסינוס של זווית הפגיעה לתיקון המרחק שעובר האור
        absorption = np.exp(-self.alphaw * z / np.cos(transform.theta_p))
        
        # 3. שילוב הכל לפי הנוסחה: i * kapa * pump^2 * absorption
        # האיבר 1j מייצג את הפאזה הנוספת הנוצרת בתהליך הלא-ליניארי
        return 1j * self.kapa * (pump**2) * absorption