"""Sanity ablation: Exp A pulses through new Multires-2K pipeline factors.

Isolates whether Exp A Multires results (~λ*=1.5, val L1≈2.53) break because of:
  - canonicalize t* vs t0
  - FROGNetPadded (improved spectral resolution) vs plain FROGNet
  - the combination (full new pipeline)

Pulse parameters stay at Exp A defaults (StochasticPulseConfig).
Training hyperparams match Exp A / updated-C1 physics runs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_generation import (
    PLANCK_CONSTANT_FS_EV,
    StochasticPulseConfig,
    generate_pulses_stochastic,
)
from dataset_utils import (
    FrogDatasetBundle,
    PulseGridConfig,
    _frog_traces_batched,
    build_notebook_c1_padded_frog,
    pack_pulses_complex,
)
from evaluate_cnn import run_cnn_snr_sweep, save_cnn_sweep
from frog_reconstruction_model import extract_pulse_prediction
from frognet import FROGNet
from spectral_grid import build_spectral_plot_grid, compute_de_new_target
from train import build_model

_STOCH = SRC / "scripts" / "run_multires_2k_stochastic_noisy_trace_lambda.py"
_spec = importlib.util.spec_from_file_location("_stoch_lam", _STOCH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

subset_loader = _mod.subset_loader
train_multires_noisy_trace_loss_early_stop = _mod.train_multires_noisy_trace_loss_early_stop
calibrate_trace_scale = _mod.calibrate_trace_scale
SNR_SWEEP_DB = _mod.SNR_SWEEP_DB

N_POINTS = 64
N_TRAIN = 2048
N_VAL = 200
N_TEST = 512
BATCH = 64
SEED = 0
LR = 1e-3
MAX_EPOCHS = 200
PATIENCE = 25
TRAIN_SNR = (0.0, 30.0)
VAL_SNR_DB = 15.0
LAMBDA_GRID = np.linspace(0.0, 3.0, 5)

# Spectral κ used by the updated-C1 "new pipeline" (not Exp A, which had no padding).
FRACTION_FROM_NYQUIST = 0.52

OUT_DIR = SRC / "checkpoints" / "benchmark" / "exp_a_prime_diagnostics"
SUMMARY_JSON = OUT_DIR / "exp_a_pipeline_ablation_summary.json"

# Cached original Exp A Multires physics (for reference comparison).
EXP_A_REF_NPZ = SRC / "checkpoints" / "benchmark" / "stochastic_multires_2k_noisy_trace_lambda.npz"

VARIANTS = {
    "plain_t0": {
        "frog": "plain",
        "canonicalize_mode": "t0",
        "note": "Exp A control: plain FROGNet + t0",
    },
    "plain_tstar": {
        "frog": "plain",
        "canonicalize_mode": "tstar",
        "note": "t* only (plain FROG)",
    },
    "padded_t0": {
        "frog": "padded",
        "canonicalize_mode": "t0",
        "note": "padded FROG only (t0)",
    },
    "padded_tstar": {
        "frog": "padded",
        "canonicalize_mode": "tstar",
        "note": "full new pipeline: padded FROG + t*",
    },
}


class _PulseModel(torch.nn.Module):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return extract_pulse_prediction(self.net(x))


def _spectral_from_sigma_tc(
    *,
    n: int,
    t_total_fs: float,
    sigma_t_c: float,
    fraction_from_nyquist: float,
    n_spectral_points: int = 64,
) -> dict:
    """Build padded-FFT spectral dict from Exp A's actual σ_{t,c} (not f_pulse formula)."""
    dt = t_total_fs / max(n - 1, 1)
    de_target = compute_de_new_target(
        PLANCK_CONSTANT_FS_EV, float(sigma_t_c), float(fraction_from_nyquist)
    )
    spectral = build_spectral_plot_grid(
        dt, n_spectral_points, de_target, PLANCK_CONSTANT_FS_EV
    )
    return {
        "dt": dt,
        "sigma_t_center": float(sigma_t_c),
        "de_new_target": float(de_target),
        "n_fft": int(spectral["n_fft"]),
        "n_spectral_points": int(n_spectral_points),
        "de_new_actual": float(spectral["de_new_actual"]),
        "e_extreme_new": float(spectral["e_extreme_new"]),
        "energy_ev_relative": spectral["energy_ev_relative"],
        "fraction_from_nyquist": float(fraction_from_nyquist),
    }


def build_exp_a_bundle(
    *,
    frog_kind: str,
    canonicalize_mode: str,
    device: torch.device,
    spectral: dict | None,
) -> tuple[FrogDatasetBundle, torch.nn.Module, dict | None]:
    grid = StochasticPulseConfig()  # Exp A defaults
    p_train_c, _, _, _ = generate_pulses_stochastic(
        n_pulses=N_TRAIN, config=grid, seed=SEED, canonicalize_mode=canonicalize_mode
    )
    p_val_c, _, _, _ = generate_pulses_stochastic(
        n_pulses=max(N_VAL, 64),
        config=grid,
        seed=SEED + 1,
        canonicalize_mode=canonicalize_mode,
    )
    p_test_c, _, t_vec, w_vec = generate_pulses_stochastic(
        n_pulses=max(N_TEST, 64),
        config=grid,
        seed=SEED + 2,
        canonicalize_mode=canonicalize_mode,
    )
    E_train = pack_pulses_complex(p_train_c)
    E_val = pack_pulses_complex(p_val_c)
    E_test = pack_pulses_complex(p_test_c)

    if frog_kind == "plain":
        frog = FROGNet(num_delay_steps=grid.n).to(device)
        frog.eval()
        used_spectral = None
    elif frog_kind == "padded":
        assert spectral is not None
        frog = build_notebook_c1_padded_frog(device, n=grid.n, spectral=spectral)
        used_spectral = spectral
    else:
        raise ValueError(frog_kind)

    I_train = _frog_traces_batched(frog, E_train, device)
    with torch.no_grad():
        I_val = frog(E_val.to(device)).cpu()
        I_test = frog(E_test.to(device)).cpu()

    E_val = E_val.to(device)
    E_test = E_test.to(device)
    I_val = I_val.to(device)
    I_test = I_test.to(device)

    train_loader = DataLoader(
        TensorDataset(I_train, E_train),
        batch_size=BATCH,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(I_val, E_val),
        batch_size=BATCH,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(I_test, E_test),
        batch_size=BATCH,
        shuffle=False,
        drop_last=False,
    )
    pseudo_grid = PulseGridConfig(n=grid.n, t_total=grid.t_total_fs)
    bundle = FrogDatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        t_vec=t_vec,
        w_vec=w_vec,
        grid=pseudo_grid,
    )
    return bundle, frog, used_spectral


def run_variant(name: str, *, force: bool = False, skip_sweep: bool = False) -> dict:
    cfg = VARIANTS[name]
    tag = f"exp_a_ablation_{name}"
    ckpt_dir = OUT_DIR / f"{tag}_lam_ckpts"
    meta_json = OUT_DIR / f"{tag}_meta.json"
    npz_path = OUT_DIR / f"{tag}.npz"
    opt_ckpt = OUT_DIR / f"{tag}_opt.pt"
    opt_sweep = OUT_DIR / f"{tag}_opt_snr_sweep.npz"
    base_sweep = OUT_DIR / f"{tag}_baseline_snr_sweep.npz"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if npz_path.exists() and meta_json.exists() and not force:
        d = np.load(npz_path)
        print(
            f"[cache] {tag}: lambda*={float(d['lambda_opt']):.4f} "
            f"best_val={float(d['best_val_at_opt']):.4f}",
            flush=True,
        )
        return {
            "variant": name,
            "note": cfg["note"],
            "lambda_opt": float(d["lambda_opt"]),
            "best_val_at_opt": float(d["best_val_at_opt"]),
            "lambda_baseline": float(d["lambda_baseline"]),
            "best_val_at_baseline": float(d["best_val_at_baseline"]),
            "best_val_l1": [float(x) for x in d["best_val_l1"]],
            "cached": True,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n======== {tag} | device={device} ========", flush=True)
    print(f"note: {cfg['note']}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    grid = StochasticPulseConfig()
    spectral = None
    if cfg["frog"] == "padded":
        spectral = _spectral_from_sigma_tc(
            n=grid.n,
            t_total_fs=float(grid.t_total_fs),
            sigma_t_c=float(grid.t_center_std_fs),
            fraction_from_nyquist=FRACTION_FROM_NYQUIST,
        )
        print(
            f"padded spectral: κ={FRACTION_FROM_NYQUIST} "
            f"σ_tc={grid.t_center_std_fs} N_FFT={spectral['n_fft']} "
            f"dE={spectral['de_new_actual']:.6f} eV",
            flush=True,
        )

    print(
        f"pulses: T={grid.t_total_fs} spikes={grid.n_spikes} "
        f"σ_spike={grid.coherence_time_fs} σ_tc={grid.t_center_std_fs} "
        f"canon={cfg['canonicalize_mode']} frog={cfg['frog']}",
        flush=True,
    )

    bundle, frog, used_spectral = build_exp_a_bundle(
        frog_kind=cfg["frog"],
        canonicalize_mode=cfg["canonicalize_mode"],
        device=device,
        spectral=spectral,
    )
    val_loader = subset_loader(bundle.val_loader, N_VAL)
    test_loader = subset_loader(bundle.test_loader, N_TEST)

    cal_model = build_model(N_POINTS, device, model_name="multires")
    trace_scale = calibrate_trace_scale(
        cal_model, frog, bundle.train_loader, device=device
    )
    del cal_model
    print(f"trace_scale = {trace_scale:.4f}", flush=True)

    n_lam = len(LAMBDA_GRID)
    best_val_l1 = np.full(n_lam, np.nan)
    best_epochs = np.full(n_lam, -1, dtype=np.int32)
    stopped_epochs = np.full(n_lam, -1, dtype=np.int32)
    wall_times_sec = np.full(n_lam, np.nan)
    run_log: list[dict] = []

    for li, lam in enumerate(LAMBDA_GRID):
        ckpt_path = ckpt_dir / f"lam_{float(lam):.4f}.pt"
        if ckpt_path.exists() and not force:
            meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            best_val_l1[li] = float(meta["best_val_l1"])
            best_epochs[li] = int(meta["best_epoch"])
            stopped_epochs[li] = int(meta["stopped_epoch"])
            wall_times_sec[li] = float(meta.get("wall_time_sec", np.nan))
            run_log.append(meta.get("log_entry", meta))
            print(
                f"[cache] lam={lam:.4f}  best_val={best_val_l1[li]:.5f}  "
                f"ep={best_epochs[li]}",
                flush=True,
            )
            continue

        print(f"\n--- {name} lam = {lam:.4f} ({li + 1}/{n_lam}) ---", flush=True)
        t_lam0 = time.perf_counter()
        model = build_model(N_POINTS, device, model_name="multires")
        result = train_multires_noisy_trace_loss_early_stop(
            model,
            bundle.train_loader,
            val_loader,
            frog,
            lam=float(lam),
            trace_scale=trace_scale,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            lr=LR,
            train_snr_db_range=TRAIN_SNR,
            val_snr_db=VAL_SNR_DB,
        )
        wall_lam = time.perf_counter() - t_lam0
        best_val_l1[li] = float(result["best_val_l1"])
        best_epochs[li] = int(result["best_epoch"])
        stopped_epochs[li] = int(result["stopped_epoch"])
        wall_times_sec[li] = wall_lam
        log_entry = {
            "lam": float(lam),
            "best_epoch": int(result["best_epoch"]),
            "stopped_epoch": int(result["stopped_epoch"]),
            "best_val_l1": float(result["best_val_l1"]),
            "wall_time_sec": wall_lam,
            "trace_scale": float(trace_scale),
            "train_losses": list(result["history"].train_losses),
            "val_l1_pulses": list(result["history"].val_l1_pulses),
            "variant": name,
            "frog": cfg["frog"],
            "canonicalize_mode": cfg["canonicalize_mode"],
            "n_fft": int(used_spectral["n_fft"]) if used_spectral else grid.n,
            "de_new_actual": (
                float(used_spectral["de_new_actual"]) if used_spectral else None
            ),
            "fraction_from_nyquist": FRACTION_FROM_NYQUIST if used_spectral else None,
            "t_total_fs": float(grid.t_total_fs),
            "n_spikes": int(grid.n_spikes),
            "coherence_time_fs": float(grid.coherence_time_fs),
            "t_center_std_fs": float(grid.t_center_std_fs),
        }
        run_log.append(log_entry)
        torch.save(
            {"model_state_dict": model.state_dict(), **log_entry, "log_entry": log_entry},
            ckpt_path,
        )
        print(
            f"  lam={lam:.4f}  best_ep={result['best_epoch']}  "
            f"best_val={result['best_val_l1']:.5f}  wall={wall_lam:.1f}s",
            flush=True,
        )

    # Reload consistently from checkpoints
    run_log = []
    for li, lam in enumerate(LAMBDA_GRID):
        meta = torch.load(
            ckpt_dir / f"lam_{float(lam):.4f}.pt",
            map_location="cpu",
            weights_only=False,
        )
        entry = meta.get(
            "log_entry", {k: v for k, v in meta.items() if k != "model_state_dict"}
        )
        run_log.append(entry)
        best_val_l1[li] = float(meta["best_val_l1"])
        best_epochs[li] = int(meta["best_epoch"])
        stopped_epochs[li] = int(meta["stopped_epoch"])
        wall_times_sec[li] = float(meta.get("wall_time_sec", np.nan))

    opt_idx = int(np.nanargmin(best_val_l1))
    base_idx = int(np.argmin(np.abs(LAMBDA_GRID - 0.0)))
    lambda_opt = float(LAMBDA_GRID[opt_idx])
    lambda_base = float(LAMBDA_GRID[base_idx])
    print(
        f"\n[{name}] lambda* = {lambda_opt:.4f}  "
        f"(best val L1 = {best_val_l1[opt_idx]:.5f})",
        flush=True,
    )
    print(
        f"[{name}] baseline λ = {lambda_base:.4f}  "
        f"(best val L1 = {best_val_l1[base_idx]:.5f})",
        flush=True,
    )

    np.savez(
        npz_path,
        lambda_grid=LAMBDA_GRID,
        best_val_l1=best_val_l1,
        best_epochs=best_epochs,
        stopped_epochs=stopped_epochs,
        wall_times_sec=wall_times_sec,
        lambda_opt=lambda_opt,
        lambda_opt_idx=opt_idx,
        lambda_baseline=lambda_base,
        best_val_at_opt=float(best_val_l1[opt_idx]),
        best_val_at_baseline=float(best_val_l1[base_idx]),
        trace_scale=float(trace_scale),
        n_fft=int(used_spectral["n_fft"]) if used_spectral else grid.n,
        fraction_from_nyquist=FRACTION_FROM_NYQUIST if used_spectral else np.nan,
    )
    meta_json.write_text(
        json.dumps(
            {
                "variant": name,
                "note": cfg["note"],
                "runs": run_log,
                "lambda_opt": lambda_opt,
                "lambda_baseline": lambda_base,
                "pulse_params": {
                    "t_total_fs": float(grid.t_total_fs),
                    "n_spikes": int(grid.n_spikes),
                    "coherence_time_fs": float(grid.coherence_time_fs),
                    "t_center_std_fs": float(grid.t_center_std_fs),
                    "canonicalize_mode": cfg["canonicalize_mode"],
                    "frog": cfg["frog"],
                    "n_fft": int(used_spectral["n_fft"]) if used_spectral else grid.n,
                    "de_new_actual": (
                        float(used_spectral["de_new_actual"]) if used_spectral else None
                    ),
                    "fraction_from_nyquist": (
                        FRACTION_FROM_NYQUIST if used_spectral else None
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    opt_src = ckpt_dir / f"lam_{lambda_opt:.4f}.pt"
    opt_data = torch.load(opt_src, map_location=device, weights_only=False)
    opt_data["lambda_opt"] = lambda_opt
    torch.save(opt_data, opt_ckpt)

    if not skip_sweep:
        print(f"\n[{name}] SNR sweep (λ*={lambda_opt:.4f}) …", flush=True)
        model = build_model(N_POINTS, device, model_name="multires")
        model.load_state_dict(opt_data["model_state_dict"])
        model.eval()
        sweep = run_cnn_snr_sweep(
            _PulseModel(model),
            test_loader,
            SNR_SWEEP_DB,
            experiment_name=f"{tag}_opt",
            verbose=True,
        )
        save_cnn_sweep(opt_sweep, sweep)

        print(f"\n[{name}] SNR sweep (baseline λ={lambda_base:.4f}) …", flush=True)
        base_data = torch.load(
            ckpt_dir / f"lam_{lambda_base:.4f}.pt",
            map_location=device,
            weights_only=False,
        )
        model = build_model(N_POINTS, device, model_name="multires")
        model.load_state_dict(base_data["model_state_dict"])
        model.eval()
        sweep_b = run_cnn_snr_sweep(
            _PulseModel(model),
            test_loader,
            SNR_SWEEP_DB,
            experiment_name=f"{tag}_baseline",
            verbose=True,
        )
        save_cnn_sweep(base_sweep, sweep_b)

    return {
        "variant": name,
        "note": cfg["note"],
        "lambda_opt": lambda_opt,
        "best_val_at_opt": float(best_val_l1[opt_idx]),
        "lambda_baseline": lambda_base,
        "best_val_at_baseline": float(best_val_l1[base_idx]),
        "best_val_l1": [float(x) for x in best_val_l1],
        "cached": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = None
    if EXP_A_REF_NPZ.exists():
        d = np.load(EXP_A_REF_NPZ)
        ref = {
            "lambda_opt": float(d["lambda_opt"]),
            "best_val_at_opt": float(d["best_val_at_opt"]),
            "lambda_baseline": float(d["lambda_baseline"]),
            "best_val_at_baseline": float(d["best_val_at_baseline"]),
            "best_val_l1": [float(x) for x in d["best_val_l1"]],
            "source": str(EXP_A_REF_NPZ),
        }
        print(
            f"Exp A reference: λ*={ref['lambda_opt']:.4f} "
            f"val*={ref['best_val_at_opt']:.4f} "
            f"λ0={ref['best_val_at_baseline']:.4f}",
            flush=True,
        )

    results = []
    for name in args.variants:
        results.append(
            run_variant(name, force=args.force, skip_sweep=args.skip_sweep)
        )

    summary = {
        "exp_a_reference": ref,
        "fraction_from_nyquist_padded": FRACTION_FROM_NYQUIST,
        "pulse_config": "StochasticPulseConfig() Exp A defaults",
        "variants": results,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n===== ABLATION SUMMARY =====", flush=True)
    if ref is not None:
        print(
            f"Exp A cached: λ*={ref['lambda_opt']:.2f}  "
            f"val*={ref['best_val_at_opt']:.4f}  "
            f"λ0={ref['best_val_at_baseline']:.4f}",
            flush=True,
        )
    for r in results:
        print(
            f"{r['variant']:14s}  λ*={r['lambda_opt']:.2f}  "
            f"val*={r['best_val_at_opt']:.4f}  "
            f"λ0={r['best_val_at_baseline']:.4f}  "
            f"| {r['note']}",
            flush=True,
        )
    print(f"Wrote {SUMMARY_JSON}", flush=True)


if __name__ == "__main__":
    main()
