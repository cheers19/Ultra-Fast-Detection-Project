"""Multires 2K baseline (λ=0, no trace loss) on stochastic Data C — test @ 0 dB SNR."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from data_generation import stochastic_pulse_config_data_c
from dataset_utils import build_stochastic_frog_dataloaders
from evaluate_cnn import mean_l1_cnn_at_snr, mean_metric_cnn_at_snr
from frog_reconstruction_model import extract_pulse_prediction
from pulse_metrics import best_l1_ambiguity, best_similarity_error_ambiguity
from train import build_model

_STOCH_SCRIPT = _SRC / "scripts" / "run_multires_2k_stochastic_noisy_trace_lambda.py"
_spec = importlib.util.spec_from_file_location("_stoch_lam", _STOCH_SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

subset_loader = _mod.subset_loader
train_multires_noisy_trace_loss_early_stop = _mod.train_multires_noisy_trace_loss_early_stop

DEFAULT_CKPT = _SRC / "checkpoints/benchmark/stochastic_data_c_multires_2k_baseline.pt"
DEFAULT_META = _SRC / "checkpoints/benchmark/stochastic_data_c_multires_2k_baseline_meta.json"
DEFAULT_EVAL = _SRC / "checkpoints/benchmark/stochastic_data_c_multires_2k_baseline_test_0db.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--train-snr-min", type=float, default=0.0)
    parser.add_argument("--train-snr-max", type=float, default=30.0)
    parser.add_argument("--val-snr-db", type=float, default=15.0)
    parser.add_argument("--test-snr-db", type=float, default=0.0)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--meta-json", default=str(DEFAULT_META))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    meta_path = Path(args.meta_json)
    eval_path = Path(args.eval_output)

    if ckpt_path.exists() and eval_path.exists() and not args.force:
        print(f"Exists: {ckpt_path} and {eval_path} (use --force to recompute)", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    n = 64
    train_snr_range = (float(args.train_snr_min), float(args.train_snr_max))

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    grid = stochastic_pulse_config_data_c(n=n)
    bundle = build_stochastic_frog_dataloaders(
        n_train=int(args.n_train),
        n_val=max(int(args.n_val), 64),
        n_test=max(int(args.n_test), 64),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
        grid=grid,
    )
    val_loader = subset_loader(bundle.val_loader, int(args.n_val))
    test_loader = subset_loader(bundle.test_loader, int(args.n_test))

    from frognet import FROGNet

    frog = FROGNet(num_delay_steps=n).to(device)

    t0 = time.time()
    if ckpt_path.exists() and not args.force:
        print(f"Loading checkpoint: {ckpt_path}", flush=True)
        meta = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(n, device, model_name="multires")
        model.load_state_dict(meta["model_state_dict"])
        result = {
            "best_epoch": int(meta["best_epoch"]),
            "best_val_l1": float(meta["best_val_l1"]),
            "stopped_epoch": int(meta["stopped_epoch"]),
            "history": meta.get("history"),
        }
    else:
        print(
            f"Data C: t_center_std={grid.t_center_std_fs} fs  "
            f"train={int(args.n_train)} val={int(args.n_val)} test={int(args.n_test)}",
            flush=True,
        )
        print(
            f"train SNR [{train_snr_range[0]:.0f}, {train_snr_range[1]:.0f}] dB; "
            f"val @ {args.val_snr_db:.0f} dB",
            flush=True,
        )
        model = build_model(n, device, model_name="multires")
        trace_scale = _mod.calibrate_trace_scale(
            build_model(n, device, model_name="multires"),
            frog,
            bundle.train_loader,
            device=device,
        )
        train_out = train_multires_noisy_trace_loss_early_stop(
            model,
            bundle.train_loader,
            val_loader,
            frog,
            lam=0.0,
            trace_scale=trace_scale,
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            lr=float(args.lr),
            train_snr_db_range=train_snr_range,
            val_snr_db=float(args.val_snr_db),
        )
        result = train_out
        hist = train_out["history"]
        payload = {
            "model_state_dict": model.state_dict(),
            "lam": 0.0,
            "best_epoch": train_out["best_epoch"],
            "stopped_epoch": train_out["stopped_epoch"],
            "best_val_l1": train_out["best_val_l1"],
            "trace_scale": train_out["trace_scale"],
            "train_losses": hist.train_losses,
            "val_l1_pulses": hist.val_l1_pulses,
            "train_snr_db_range": list(train_snr_range),
            "val_snr_db": float(args.val_snr_db),
            "n_train": int(args.n_train),
            "n_val": int(args.n_val),
            "n_test": int(args.n_test),
            "t_center_std_fs": float(grid.t_center_std_fs),
            "pulse_kind": "stochastic_data_c",
        }
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, ckpt_path)
        meta_path.write_text(
            json.dumps(
                {
                    "best_epoch": train_out["best_epoch"],
                    "stopped_epoch": train_out["stopped_epoch"],
                    "best_val_l1": train_out["best_val_l1"],
                    "lam": 0.0,
                    "t_center_std_fs": float(grid.t_center_std_fs),
                    "pulse_kind": "stochastic_data_c",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved checkpoint: {ckpt_path}", flush=True)

    model.eval()
    test_snr = float(args.test_snr_db)
    print(f"\nTest evaluation @ {test_snr:.0f} dB SNR (n={int(args.n_test)}) …", flush=True)

    class _PulseModel(torch.nn.Module):
        def __init__(self, net: torch.nn.Module) -> None:
            super().__init__()
            self.net = net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return extract_pulse_prediction(self.net(x))

    wrapped = _PulseModel(model)
    l1_m, l1_s = mean_l1_cnn_at_snr(
        wrapped,
        test_loader,
        test_snr,
        score_fn=best_l1_ambiguity,
    )
    sim_m, sim_s = mean_metric_cnn_at_snr(
        wrapped,
        test_loader,
        test_snr,
        score_fn=best_similarity_error_ambiguity,
    )

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        eval_path,
        test_snr_db=test_snr,
        n_test=int(args.n_test),
        seed=int(args.seed),
        t_center_std_fs=float(grid.t_center_std_fs),
        best_epoch=int(result["best_epoch"]),
        best_val_l1=float(result["best_val_l1"]),
        l1_amb_mean=l1_m,
        l1_amb_std=l1_s,
        sim_amb_mean=sim_m,
        sim_amb_std=sim_s,
    )
    print(f"Saved eval: {eval_path}", flush=True)
    print(f"  L1 (best amb)     = {l1_m:.4f} ± {l1_s:.4f}", flush=True)
    print(f"  SIMILARITY (best) = {sim_m:.4f} ± {sim_s:.4f}", flush=True)
    print(f"Done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
