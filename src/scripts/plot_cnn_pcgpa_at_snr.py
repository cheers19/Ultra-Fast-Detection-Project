"""Plot CNN vs PCGPA reconstruction vs true pulse at a fixed SNR."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataset_utils import PulseGridConfig, build_frog_dataloaders
from pulse_metrics import (
    best_l1_ambiguity_field,
    l1_packed_mae,
    packed_batch_to_complex,
    unpack_packed_field,
    unwrap_phases_for_overlay,
)
from trace_noise import add_trace_noise_awgn
from train import load_checkpoint

import pcgpa_reconstruct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/large_60k_cnn_large.pt")
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument("--pulse-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pcgpa-maxiter", type=int, default=200)
    parser.add_argument("--pcgpa-restarts", type=int, default=3)
    parser.add_argument(
        "--output",
        default="checkpoints/large_60k_cnn_large_vs_pcgpa_m10dB.png",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = PulseGridConfig(n=64, t_total=250.0)
    bundle = build_frog_dataloaders(
        n_train=2048,
        n_val=512,
        n_test=512,
        batch_size=64,
        seed=args.seed,
        device=device,
        grid=grid,
    )
    I_test = bundle.test_loader.dataset.tensors[0]
    E_test = bundle.test_loader.dataset.tensors[1]
    t_axis = bundle.t_vec

    model, ckpt = load_checkpoint(_SRC / args.checkpoint, device)
    exp_name = ckpt.get("experiment_name", Path(args.checkpoint).stem)

    pcgpa_mod = pcgpa_reconstruct.reload_from_disk()
    reconstruct_pcgpa = pcgpa_mod.reconstruct_pcgpa
    pcgpa_rng = pcgpa_mod._pcgpa_rng_for_pulse

    snr_db = float(args.snr_db)
    i = int(args.pulse_index)

    I_clean = I_test[i : i + 1]
    I_noisy = add_trace_noise_awgn(I_clean, snr_db)
    with torch.no_grad():
        E_pred = model(I_noisy.unsqueeze(1).to(device))

    e_true = unpack_packed_field(E_test[i].cpu().numpy())
    e_rec = packed_batch_to_complex(E_pred.cpu())[0]
    e_rec_amb = best_l1_ambiguity_field(e_rec, e_true)

    i_noisy_np = I_noisy.squeeze(0).cpu().numpy()
    e_pcgpa = reconstruct_pcgpa(
        i_noisy_np,
        dt=grid.dt,
        sigma_omega=grid.resolved_sigma_omega,
        maxiter=args.pcgpa_maxiter,
        n_restarts=args.pcgpa_restarts,
        rng=pcgpa_rng(args.seed, i, snr_db),
    )
    e_pcgpa_amb = best_l1_ambiguity_field(e_pcgpa, e_true)

    e_true_packed = E_test[i].cpu().numpy()
    l1_cnn = l1_packed_mae(e_rec_amb, e_true_packed, use_best_ambiguity=False)
    l1_pcgpa = l1_packed_mae(e_pcgpa_amb, e_true_packed, use_best_ambiguity=False)

    ph_true, _ = unwrap_phases_for_overlay(e_rec_amb, e_true)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for col, e_r, color, name, val in (
        (0, e_rec_amb, "r", "CNN", l1_cnn),
        (1, e_pcgpa_amb, "b", "PCGPA", l1_pcgpa),
    ):
        axes[0, col].plot(t_axis, np.abs(e_true), "k-", lw=1.5, label="|E_true|")
        axes[0, col].plot(t_axis, np.abs(e_r), color + "--", lw=1.2, label=f"|E_{name}|")
        axes[0, col].set_title(f"{name} — best L1 amb. ({val:.4f})")
        axes[0, col].set_ylabel("|E(t)|")
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)

        _, ph_rec = unwrap_phases_for_overlay(e_r, e_true)
        axes[1, col].plot(t_axis, ph_true, "k-", lw=1.5, label="phase true")
        axes[1, col].plot(t_axis, ph_rec, color + "--", lw=1.2, label=f"phase {name}")
        axes[1, col].set_xlabel("Time [as]")
        axes[1, col].set_ylabel("phase (rad)")
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle(
        f"{exp_name} vs PCGPA @ {snr_db:.0f} dB — pulse {i} (best L1 ambiguity)",
        y=1.02,
    )
    plt.tight_layout()

    out = _SRC / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"L1 @ {snr_db:.0f} dB, pulse {i}: CNN={l1_cnn:.5f}  PCGPA={l1_pcgpa:.5f}")


if __name__ == "__main__":
    main()
