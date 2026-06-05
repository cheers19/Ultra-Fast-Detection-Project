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
        from trace_noise import add_trace_noise_awgn

        return add_trace_noise_awgn(I_clean, snr_db)
    elif noise_type == "Poisson":
        # Legacy Poisson scaling (not updated to amplitude-SNR convention)
        scale_factor = (10 ** (snr_db / 20.0)) / torch.mean(I_clean)
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
    from pulse_metrics import mean_delta_e_torch

    return mean_delta_e_torch(E_rec, E_orig)

def calc_delta_I(I_rec, I_orig):
    """חישוב שגיאת L1 מנורמלת לתמונת ה-TRACE"""
    l1_diff = torch.norm(I_rec - I_orig, p=1, dim=(-2,-1))
    l1_orig = torch.norm(I_orig, p=1, dim=(-2,-1))
    return torch.mean(l1_diff / l1_orig).item()