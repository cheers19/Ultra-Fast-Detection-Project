"""Rebuild summary NPZ / meta / opt checkpoint from per-lambda .pt files."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

CKPT_DIR = _SRC / "checkpoints/benchmark/multires_2k_noisy_trace_lambda"
OUT_NPZ = _SRC / "checkpoints/benchmark/multires_2k_noisy_trace_lambda.npz"
OUT_META = _SRC / "checkpoints/benchmark/multires_2k_noisy_trace_lambda_meta.json"
OPT_CKPT = _SRC / "checkpoints/benchmark/multires_2k_noisy_trace_lambda_opt.pt"


def main() -> None:
    paths = sorted(CKPT_DIR.glob("lam_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {CKPT_DIR}")

    entries: list[dict] = []
    best_vals: list[float] = []
    for p in paths:
        m = torch.load(p, map_location="cpu", weights_only=False)
        entry = m.get("log_entry", {k: v for k, v in m.items() if k != "model_state_dict"})
        entries.append(entry)
        best_vals.append(float(m["best_val_l1"]))
        print(
            f"lam={entry['lam']:.4f}  best_val={m['best_val_l1']:.5f}  "
            f"best_ep={m['best_epoch']}"
        )

    best_vals_arr = np.asarray(best_vals, dtype=np.float64)
    lambda_grid = np.asarray([e["lam"] for e in entries], dtype=np.float64)
    opt_i = int(np.argmin(best_vals_arr))
    lambda_opt = float(lambda_grid[opt_i])
    print(f"\nlambda* = {lambda_opt:.4f}  (best val L1 = {best_vals_arr[opt_i]:.5f})")

    np.savez(
        OUT_NPZ,
        lambda_grid=lambda_grid,
        best_val_l1=best_vals_arr,
        best_epochs=np.asarray([e["best_epoch"] for e in entries], dtype=np.int32),
        stopped_epochs=np.asarray([e["stopped_epoch"] for e in entries], dtype=np.int32),
        lambda_opt=lambda_opt,
        lambda_opt_idx=opt_i,
        best_val_at_opt=float(best_vals_arr[opt_i]),
        trace_scale=float(entries[0]["trace_scale"]),
        n_train=int(entries[0].get("n_train", 2048)),
        n_val=int(entries[0].get("n_val", 200)),
        n_test=512,
        max_epochs=int(entries[0].get("max_epochs", 200)),
        patience=int(entries[0].get("patience", 25)),
        train_snr_min=float(entries[0]["train_snr_db_range"][0]),
        train_snr_max=float(entries[0]["train_snr_db_range"][1]),
        val_snr_db=float(entries[0]["val_snr_db"]),
        seed=0,
    )
    OUT_META.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    src = CKPT_DIR / f"lam_{lambda_opt:.4f}.pt"
    shutil.copy2(src, OPT_CKPT)
    print(f"Saved: {OUT_NPZ}")
    print(f"Saved: {OUT_META}")
    print(f"Saved: {OPT_CKPT}")


if __name__ == "__main__":
    main()
