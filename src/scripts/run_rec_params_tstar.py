"""Train recommended pulse params with canonicalize=tstar (not t0)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import exp_a_prime_diagnostics_lib as diag

T = 53.0
N_SPIKES = 1000
SIGMA_SPIKE = 0.045 * T  # ~2.385 fs
F_PULSE = 0.62
N_TRAIN = 2048
NAME = f"phase3_rec_spikes{N_SPIKES}_sigma045_fpulse062_tstar_n{N_TRAIN}"


def main() -> None:
    print(f"T={T} N_spikes={N_SPIKES} sigma_spike={SIGMA_SPIKE:.4f} f_pulse={F_PULSE}")
    print(f"canonicalize=tstar  name={NAME}")
    out = diag.run_named_train(
        NAME,
        n_train=N_TRAIN,
        n_spikes=N_SPIKES,
        coherence_time_fs=SIGMA_SPIKE,
        pulse_temporal_fraction=F_PULSE,
        canonicalize_mode="tstar",
        frog_mode="padded",
        force=False,
        verbose=True,
    )
    print(
        f"RESULT name={out.name} valL1={out.best_val_l1:.4f} "
        f"hiL1={out.high_snr_l1_amb_mean:.4f} hiSim={out.high_snr_sim_amb_mean:.4f} "
        f"ep={out.best_epoch} wall={out.wall_sec:.1f}s"
    )


if __name__ == "__main__":
    main()
