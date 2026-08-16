"""Re-evaluate Data C Multires-2K λ checkpoints at 0 dB and 30 dB test SNR."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_generation import stochastic_pulse_config_data_c
from dataset_utils import build_stochastic_frog_dataloaders
from evaluate_cnn import mean_l1_cnn_at_snr, mean_metric_cnn_at_snr
from frog_reconstruction_model import extract_pulse_prediction
from pulse_metrics import best_l1_ambiguity, best_similarity_error_ambiguity
from train import build_model

BENCH = SRC / "checkpoints" / "benchmark"
CKPT_DIR = BENCH / "stochastic_data_c_multires_2k_noisy_trace_lambda"
SEED = 0
N_TEST = 512
N = 64

LAM_EVALS = [
    (0.0, "lam_0.0000.pt", "stochastic_data_c_multires_2k_trace_test_{snr}db.npz"),
    (0.75, "lam_0.7500.pt", "stochastic_data_c_multires_2k_trace_lam075_test_{snr}db.npz"),
    (1.5, "lam_1.5000.pt", "stochastic_data_c_multires_2k_trace_lam150_test_{snr}db.npz"),
    (2.25, "lam_2.2500.pt", "stochastic_data_c_multires_2k_trace_lam225_test_{snr}db.npz"),
    (3.0, "lam_3.0000.pt", "stochastic_data_c_multires_2k_trace_lam300_test_{snr}db.npz"),
]


class _PulseModel(torch.nn.Module):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return extract_pulse_prediction(self.net(x))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    grid = stochastic_pulse_config_data_c(n=N)
    bundle = build_stochastic_frog_dataloaders(
        n_train=1,
        n_val=1,
        n_test=N_TEST,
        batch_size=64,
        seed=SEED,
        device=device,
        grid=grid,
    )
    test_loader = bundle.test_loader

    for test_snr in (0.0, 30.0):
        snr_tag = int(test_snr)
        print(f"\n=== Test @ {test_snr:.0f} dB ===", flush=True)
        for lam, ckpt_name, eval_tmpl in LAM_EVALS:
            ckpt_path = CKPT_DIR / ckpt_name
            eval_path = BENCH / eval_tmpl.format(snr=snr_tag)
            meta = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = build_model(N, device, model_name="multires")
            model.load_state_dict(meta["model_state_dict"])
            model.eval()
            wrapped = _PulseModel(model)
            l1_m, l1_s = mean_l1_cnn_at_snr(
                wrapped, test_loader, test_snr, score_fn=best_l1_ambiguity
            )
            sim_m, sim_s = mean_metric_cnn_at_snr(
                wrapped, test_loader, test_snr, score_fn=best_similarity_error_ambiguity
            )
            np.savez(
                eval_path,
                test_snr_db=test_snr,
                n_test=N_TEST,
                seed=SEED,
                t_center_std_fs=float(grid.t_center_std_fs),
                lam=float(lam),
                best_epoch=int(meta["best_epoch"]),
                best_val_l1=float(meta["best_val_l1"]),
                best_val_l1_amb=float(meta.get("best_val_l1_amb", np.nan)),
                l1_amb_mean=l1_m,
                l1_amb_std=l1_s,
                sim_amb_mean=sim_m,
                sim_amb_std=sim_s,
            )
            print(
                f"  λ={lam:.2f}  L1_amb={l1_m:.4f}±{l1_s:.4f}  "
                f"SIM={sim_m:.4f}±{sim_s:.4f}  -> {eval_path.name}",
                flush=True,
            )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
