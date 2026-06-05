import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Helper: noise injection
# ==========================================

def add_noise(I_clean, snr_db, noise_type="WGN"):
    """
    Inject noise into the TRACE image (according to the requested noise type).
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
# Helper: evaluation metrics
# ==========================================

def calc_delta_E(E_rec, E_orig):
    """
    Compute the similarity metric (complex overlap) delta E.
    E_rec, E_orig: tensors with real part in the first half and imaginary in the second.
    """
    from pulse_metrics import mean_delta_e_torch

    return mean_delta_e_torch(E_rec, E_orig)

def calc_delta_I(I_rec, I_orig):
    """Compute normalized L1 error for the TRACE image."""
    l1_diff = torch.norm(I_rec - I_orig, p=1, dim=(-2,-1))
    l1_orig = torch.norm(I_orig, p=1, dim=(-2,-1))
    return torch.mean(l1_diff / l1_orig).item()