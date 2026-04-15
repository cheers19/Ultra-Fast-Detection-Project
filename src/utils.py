import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# פונקציית עזר: הזרקת רעש 
# ==========================================

def add_noise(I_clean, snr_db, noise_type="WGN"):
    """
    הזרקת רעש לתמונת ה-TRACE (לפי סוג הרעש המבוקש)
    """
    if noise_type == "WGN":
        # רעש גאוסיאני לבן (WGN) - כפי שבוצע במאמר
        signal_power = torch.mean(I_clean ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = torch.randn_like(I_clean) * torch.sqrt(noise_power)
        return I_clean + noise
    elif noise_type == "Poisson":
        # רעש פואסוני (מידע מחוץ למקורות)
        # נדרש סילום (Scaling) מתאים לעוצמת האות כדי לדמות SNR
        scale_factor = (10 ** (snr_db / 10.0)) / torch.mean(I_clean)
        noisy_I = torch.poisson(I_clean * scale_factor) / scale_factor
        return noisy_I

# ==========================================
# פונקציות עזר: מדדי הערכה 
# ==========================================

def calc_delta_E(E_rec, E_orig):
    """
    חישוב מדד הדמיון (Complex Overlap) - דלתא E
    E_rec, E_orig: טנזורים הכוללים חצי ממשי וחצי מדומה
    """
    # הפרדת ממשי ומדומה (בהנחה שחצי ראשון ממשי וחצי שני מדומה)
    half = E_rec.shape[-1] // 2
    E_rec_r, E_rec_i = E_rec[..., :half], E_rec[..., half:]
    E_orig_r, E_orig_i = E_orig[..., :half], E_orig[..., half:]

    # חישוב מכפלות פנימיות
    dot_rec_orig_r = torch.sum(E_rec_r * E_orig_r + E_rec_i * E_orig_i, dim=-1)
    dot_rec_orig_i = torch.sum(E_rec_r * E_orig_i - E_rec_i * E_orig_r, dim=-1)

    norm_rec = torch.sum(E_rec_r**2 + E_rec_i**2, dim=-1)
    norm_orig = torch.sum(E_orig_r**2 + E_orig_i**2, dim=-1)

    abs_dot = torch.sqrt(dot_rec_orig_r**2 + dot_rec_orig_i**2)
    # הנוסחה כפי שמופיעה במאמר: arccos(|<Er|Ei>| / sqrt(<Er|Er><Ei|Ei>))
    delta_E = torch.acos(torch.clamp(abs_dot / torch.sqrt(norm_rec * norm_orig), -1.0, 1.0))
    return torch.mean(delta_E).item()

def calc_delta_I(I_rec, I_orig):
    """חישוב שגיאת L1 מנורמלת לתמונת ה-TRACE"""
    l1_diff = torch.norm(I_rec - I_orig, p=1, dim=(-2,-1))
    l1_orig = torch.norm(I_orig, p=1, dim=(-2,-1))
    return torch.mean(l1_diff / l1_orig).item()