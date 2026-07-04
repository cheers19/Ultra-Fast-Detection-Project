"""SNR sweep for one lambda checkpoint (smoothed-phase stochastic Multires 2K)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from data_generation import SmoothedPhaseStochasticPulseConfig
from dataset_utils import build_smoothed_phase_stochastic_frog_dataloaders
from evaluate_cnn import run_cnn_snr_sweep, save_cnn_sweep
from train import build_model

SNR_SWEEP_DB = np.arange(-10.0, 31.0, 5.0)
CKPT_DIR = _SRC / "checkpoints/benchmark/stochastic_smoothed_phase_multires_2k_noisy_trace_lambda"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument(
        "--output",
        default=str(
            _SRC
            / "checkpoints/benchmark/stochastic_smoothed_phase_multires_2k_noisy_trace_lambda_lam075_sweep.npz"
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frog_device = torch.device("cpu")
    ckpt_path = CKPT_DIR / f"lam_{float(args.lam):.4f}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    bundle = build_smoothed_phase_stochastic_frog_dataloaders(
        n_train=1,
        n_val=1,
        n_test=int(args.n_test),
        batch_size=32,
        seed=int(args.seed),
        device=frog_device,
        grid=SmoothedPhaseStochasticPulseConfig(n=64),
    )
    I_test, E_test = bundle.test_loader.dataset.tensors
    from torch.utils.data import DataLoader, TensorDataset

    test_loader = DataLoader(
        TensorDataset(I_test.to(device), E_test.to(device)),
        batch_size=32,
        shuffle=False,
    )

    model = build_model(64, device, model_name="multires")
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    sweep = run_cnn_snr_sweep(
        model,
        test_loader,
        SNR_SWEEP_DB,
        experiment_name=f"smoothed-phase Multires 2K + trace (lambda={float(args.lam):.2g})",
        verbose=True,
    )
    out = save_cnn_sweep(args.output, sweep)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
