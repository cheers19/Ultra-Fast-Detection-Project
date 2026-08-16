"""Diagnostics for Exp A′ (notebook C1 + padded FROG) Multires failure modes.

Protocol (fixed):
  - pulse L1 loss only (λ = 0, no physics/trace loss)
  - train SNR ~ U[0, 30] dB
  - no T sweeps (T = 53 fs kept; other ablations allowed)
  - primary interest: high-SNR reconstruction quality
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_generation import (
    StochasticPulseConfig,
    generate_pulses_stochastic,
    stochastic_pulse_config_notebook_modified_c1,
)
from dataset_utils import (
    FrogDatasetBundle,
    PulseGridConfig,
    _frog_traces_batched,
    build_notebook_c1_padded_frog,
    notebook_c1_spectral_fft_params,
    pack_pulses_complex,
)
from evaluate_cnn import per_pulse_amb_l1_and_sim_cnn_at_snr
from frog_reconstruction_model import TraceToPulseMultires, extract_pulse_prediction
from frognet import FROGNet
from pulse_metrics import (
    best_l1_ambiguity,
    best_similarity_error_ambiguity,
    packed_batch_to_complex,
    pulse_packed_l1_loss_torch,
    unpack_packed_field,
)
from spectral_grid import compute_sigma_t_center
from trace_noise import add_trace_noise_awgn
from train import TrainHistory, train_trace_to_pulse_early_stopping

SRC = Path(__file__).resolve().parent
OUT_DIR = SRC / "checkpoints" / "benchmark" / "exp_a_prime_diagnostics"
N = 64
T_TOTAL_FS = 53.0
TRAIN_SNR = (0.0, 30.0)
VAL_SNR_DB = 15.0
HIGH_SNR_DB = 30.0


@dataclass
class RunResult:
    name: str
    best_epoch: int
    stopped_epoch: int
    best_val_l1: float
    train_losses: list[float]
    val_l1_pulses: list[float]
    high_snr_l1_amb_mean: float
    high_snr_l1_amb_std: float
    high_snr_sim_amb_mean: float
    high_snr_sim_amb_std: float
    wall_sec: float
    meta: dict[str, Any]


def ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_grid(
    *,
    n_spikes: int = 300,
    coherence_time_fs: float | None = None,
    pulse_temporal_fraction: float = 0.62,
) -> StochasticPulseConfig:
    """A′ grid at fixed T; σ_t,c from notebook rule (unless overridden)."""
    base = stochastic_pulse_config_notebook_modified_c1(n=N)
    sigma_spike = (
        float(coherence_time_fs)
        if coherence_time_fs is not None
        else 0.025 * T_TOTAL_FS
    )
    return StochasticPulseConfig(
        n=N,
        t_total_fs=T_TOTAL_FS,
        n_spikes=int(n_spikes),
        coherence_time_fs=sigma_spike,
        t_center_std_fs=float(
            compute_sigma_t_center(
                T_TOTAL_FS, int(n_spikes), float(pulse_temporal_fraction)
            )
        ),
        delta_energy_ev_range=base.delta_energy_ev_range,
    )


def build_frog(
    device: torch.device,
    *,
    frog_mode: str = "padded",
    n_fft: int | None = None,
    n_spikes: int = 300,
    pulse_temporal_fraction: float = 0.62,
) -> nn.Module:
    if frog_mode == "padded":
        spectral = notebook_c1_spectral_fft_params(
            n=N,
            t_total_fs=T_TOTAL_FS,
            n_spikes=int(n_spikes),
            pulse_temporal_fraction=float(pulse_temporal_fraction),
        )
        if n_fft is not None:
            spectral = dict(spectral)
            spectral["n_fft"] = int(n_fft)
        return build_notebook_c1_padded_frog(device, n=N, spectral=spectral)
    if frog_mode == "plain":
        frog = FROGNet(num_delay_steps=N).to(device)
        frog.eval()
        return frog
    raise ValueError(f"unknown frog_mode={frog_mode!r}")


def build_bundle(
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    n_spikes: int = 300,
    coherence_time_fs: float | None = None,
    pulse_temporal_fraction: float = 0.62,
    canonicalize_mode: str = "tstar",
    frog_mode: str = "padded",
    n_fft: int | None = None,
) -> tuple[FrogDatasetBundle, nn.Module, StochasticPulseConfig]:
    grid = make_grid(
        n_spikes=n_spikes,
        coherence_time_fs=coherence_time_fs,
        pulse_temporal_fraction=pulse_temporal_fraction,
    )
    frog = build_frog(
        device,
        frog_mode=frog_mode,
        n_fft=n_fft,
        n_spikes=n_spikes,
        pulse_temporal_fraction=pulse_temporal_fraction,
    )

    p_train, _, _, _ = generate_pulses_stochastic(
        n_pulses=n_train, config=grid, seed=seed, canonicalize_mode=canonicalize_mode
    )
    p_val, _, _, _ = generate_pulses_stochastic(
        n_pulses=n_val, config=grid, seed=seed + 1, canonicalize_mode=canonicalize_mode
    )
    p_test, _, t_vec, w_vec = generate_pulses_stochastic(
        n_pulses=n_test, config=grid, seed=seed + 2, canonicalize_mode=canonicalize_mode
    )

    E_train = pack_pulses_complex(p_train)
    E_val = pack_pulses_complex(p_val)
    E_test = pack_pulses_complex(p_test)

    I_train = _frog_traces_batched(frog, E_train, device)
    with torch.no_grad():
        I_val = frog(E_val.to(device)).cpu()
        I_test = frog(E_test.to(device)).cpu()

    train_loader = DataLoader(
        TensorDataset(I_train, E_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(I_val, E_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(I_test, E_test),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    bundle = FrogDatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        t_vec=t_vec,
        w_vec=w_vec,
        grid=PulseGridConfig(n=N, t_total=T_TOTAL_FS),
    )
    return bundle, frog, grid


def build_multires(
    device: torch.device,
    *,
    filters_per_branch: tuple[int, ...] = (8, 16, 32),
) -> nn.Module:
    model = TraceToPulseMultires(
        out_dim=2 * N, filters_per_branch=filters_per_branch
    ).to(device)
    model(torch.zeros(1, 1, N, N, device=device))
    return model


class _PulseModel(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return extract_pulse_prediction(self.net(x))


def eval_high_snr(
    model: nn.Module,
    loader: DataLoader,
    *,
    snr_db: float = HIGH_SNR_DB,
) -> dict[str, float]:
    wrap = _PulseModel(model)
    l1, sim = per_pulse_amb_l1_and_sim_cnn_at_snr(wrap, loader, snr_db)
    return {
        "high_snr_l1_amb_mean": float(l1.mean()),
        "high_snr_l1_amb_std": float(l1.std(ddof=0)),
        "high_snr_sim_amb_mean": float(sim.mean()),
        "high_snr_sim_amb_std": float(sim.std(ddof=0)),
    }


def train_lam0(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    max_epochs: int,
    patience: int,
    lr: float,
    verbose: bool = True,
):
    return train_trace_to_pulse_early_stopping(
        model,
        train_loader,
        val_loader,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        train_snr_db_range=TRAIN_SNR,
        val_snr_db=VAL_SNR_DB,
        verbose=verbose,
    )


def run_named_train(
    name: str,
    *,
    n_train: int,
    n_val: int = 200,
    n_test: int = 256,
    batch_size: int = 64,
    seed: int = 0,
    max_epochs: int = 120,
    patience: int = 25,
    lr: float = 1e-3,
    n_spikes: int = 300,
    coherence_time_fs: float | None = None,
    pulse_temporal_fraction: float = 0.62,
    canonicalize_mode: str = "tstar",
    frog_mode: str = "padded",
    n_fft: int | None = None,
    filters_per_branch: tuple[int, ...] = (8, 16, 32),
    force: bool = False,
    verbose: bool = True,
) -> RunResult:
    ensure_out_dir()
    cache = OUT_DIR / f"{name}.json"
    ckpt = OUT_DIR / f"{name}.pt"
    if cache.exists() and ckpt.exists() and not force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        print(f"[cache] {name}: best_val={data['best_val_l1']:.4f}  "
              f"highSNR_L1={data['high_snr_l1_amb_mean']:.4f}")
        return RunResult(**{k: data[k] for k in RunResult.__dataclass_fields__})

    device = get_device()
    set_seed(seed)
    print(f"\n=== {name} ===", flush=True)
    print(
        f"  n_train={n_train} spikes={n_spikes} "
        f"σ_spike={coherence_time_fs} f_pulse={pulse_temporal_fraction} "
        f"canon={canonicalize_mode} frog={frog_mode} fpb={filters_per_branch}",
        flush=True,
    )
    t0 = time.perf_counter()
    bundle, _, grid = build_bundle(
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        batch_size=min(batch_size, n_train),
        seed=seed,
        device=device,
        n_spikes=n_spikes,
        coherence_time_fs=coherence_time_fs,
        pulse_temporal_fraction=pulse_temporal_fraction,
        canonicalize_mode=canonicalize_mode,
        frog_mode=frog_mode,
        n_fft=n_fft,
    )
    model = build_multires(device, filters_per_branch=filters_per_branch)
    result = train_lam0(
        model,
        bundle.train_loader,
        bundle.val_loader,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        verbose=verbose,
    )
    metrics = eval_high_snr(model, bundle.test_loader, snr_db=HIGH_SNR_DB)
    wall = time.perf_counter() - t0
    meta = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_spikes": int(grid.n_spikes),
        "t_total_fs": float(grid.t_total_fs),
        "coherence_time_fs": float(grid.coherence_time_fs),
        "t_center_std_fs": float(grid.t_center_std_fs),
        "pulse_temporal_fraction": float(pulse_temporal_fraction),
        "canonicalize_mode": canonicalize_mode,
        "frog_mode": frog_mode,
        "n_fft": n_fft,
        "filters_per_branch": list(filters_per_branch),
        "lr": lr,
        "max_epochs": max_epochs,
        "patience": patience,
        "train_snr": list(TRAIN_SNR),
        "val_snr_db": VAL_SNR_DB,
        "high_snr_db": HIGH_SNR_DB,
        "seed": seed,
        "device": str(device),
    }
    out = RunResult(
        name=name,
        best_epoch=int(result.best_epoch),
        stopped_epoch=int(result.stopped_epoch),
        best_val_l1=float(result.best_val_l1),
        train_losses=list(result.history.train_losses),
        val_l1_pulses=list(result.history.val_l1_pulses),
        wall_sec=float(wall),
        meta=meta,
        **metrics,
    )
    torch.save(
        {"model_state_dict": model.state_dict(), "result": asdict(out)},
        ckpt,
    )
    cache.write_text(json.dumps(asdict(out), indent=2), encoding="utf-8")
    print(
        f"  done: best_val_L1={out.best_val_l1:.4f}  "
        f"highSNR@{HIGH_SNR_DB:.0f}dB L1_amb={out.high_snr_l1_amb_mean:.4f}  "
        f"wall={wall:.1f}s",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------


def phase0a_overfit_minibatch(
    *,
    n_examples: int = 4,
    max_epochs: int = 800,
    lr: float = 1e-3,
    seed: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    """Karpathy: overfit a tiny batch — expect near-zero train L1 if pipeline is sane."""
    ensure_out_dir()
    cache = OUT_DIR / "phase0a_overfit.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        print(f"[cache] phase0a: final_train_L1={data['final_train_l1']:.6f}")
        return data

    device = get_device()
    set_seed(seed)
    bundle, _, _ = build_bundle(
        n_train=max(n_examples, 8),
        n_val=8,
        n_test=8,
        batch_size=n_examples,
        seed=seed,
        device=device,
    )
    I, E = next(iter(bundle.train_loader))
    I, E = I[:n_examples].to(device), E[:n_examples].to(device)
    # Overfit on FIXED noisy traces (still SNR in [0,30], but frozen) — isolates fit capacity.
    snr_fixed = 20.0
    I_noisy = add_trace_noise_awgn(I, snr_fixed)

    model = build_multires(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    t0 = time.perf_counter()
    model.train()
    for ep in range(max_epochs):
        opt.zero_grad(set_to_none=True)
        pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
        loss = pulse_packed_l1_loss_torch(pred, E)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if (ep + 1) % 100 == 0 or ep == 0:
            print(f"  overfit ep {ep+1:4d}/{max_epochs}  L1={losses[-1]:.6f}", flush=True)

    with torch.no_grad():
        pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
        final = float(pulse_packed_l1_loss_torch(pred, E).item())
        # also report best-ambiguity L1 (fairer for visualization)
        rec = packed_batch_to_complex(pred.cpu())
        true = packed_batch_to_complex(E.cpu())
        amb = float(
            np.mean([best_l1_ambiguity(rec[i], true[i]) for i in range(n_examples)])
        )

    out = {
        "n_examples": n_examples,
        "snr_fixed_db": snr_fixed,
        "max_epochs": max_epochs,
        "final_train_l1": final,
        "final_train_l1_amb": amb,
        "min_train_l1": float(min(losses)),
        "train_losses": losses[:: max(1, len(losses) // 200)],
        "wall_sec": time.perf_counter() - t0,
        "ok_near_zero": final < 0.2,
    }
    cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"phase0a: final_L1={final:.6f} amb={amb:.6f} ok_near_zero={out['ok_near_zero']}",
        flush=True,
    )
    return out


def phase0b_input_independent(
    *,
    n_train: int = 512,
    max_epochs: int = 40,
    patience: int = 15,
    seed: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    """Train on real traces vs zeroed traces; real should win."""
    ensure_out_dir()
    cache = OUT_DIR / "phase0b_input_indep.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        print(f"[cache] phase0b: real={data['real_best_val']:.4f} zero={data['zero_best_val']:.4f}")
        return data

    device = get_device()
    set_seed(seed)
    bundle, _, _ = build_bundle(
        n_train=n_train, n_val=128, n_test=128, batch_size=64, seed=seed, device=device
    )

    model_real = build_multires(device)
    r_real = train_lam0(
        model_real, bundle.train_loader, bundle.val_loader,
        max_epochs=max_epochs, patience=patience, lr=1e-3, verbose=True,
    )

    # Zeroed-input loader: same labels, I=0
    I_tr, E_tr = bundle.train_loader.dataset.tensors
    I_va, E_va = bundle.val_loader.dataset.tensors
    zero_train = DataLoader(
        TensorDataset(torch.zeros_like(I_tr), E_tr), batch_size=64, shuffle=True
    )
    zero_val = DataLoader(
        TensorDataset(torch.zeros_like(I_va), E_va), batch_size=64, shuffle=False
    )
    model_zero = build_multires(device)
    r_zero = train_lam0(
        model_zero, zero_train, zero_val,
        max_epochs=max_epochs, patience=patience, lr=1e-3, verbose=True,
    )

    out = {
        "real_best_val": float(r_real.best_val_l1),
        "zero_best_val": float(r_zero.best_val_l1),
        "real_better": float(r_real.best_val_l1) < float(r_zero.best_val_l1),
        "n_train": n_train,
    }
    cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"phase0b: real={out['real_best_val']:.4f} zero={out['zero_best_val']:.4f} "
        f"real_better={out['real_better']}",
        flush=True,
    )
    return out


def phase0c_loss_at_init(*, seed: int = 0, n: int = 128) -> dict[str, Any]:
    ensure_out_dir()
    device = get_device()
    set_seed(seed)
    bundle, _, _ = build_bundle(
        n_train=n, n_val=n, n_test=n, batch_size=64, seed=seed, device=device
    )
    model = build_multires(device)
    model.eval()
    losses = []
    with torch.no_grad():
        for I, E in bundle.val_loader:
            I = I.to(device)
            E = E.to(device)
            I_n = add_trace_noise_awgn(I, VAL_SNR_DB)
            pred = extract_pulse_prediction(model(I_n.unsqueeze(1)))
            losses.append(float(pulse_packed_l1_loss_torch(pred, E).item()))
    # Rough baseline: predict zero field
    zero_losses = []
    with torch.no_grad():
        for _, E in bundle.val_loader:
            E = E.to(device)
            z = torch.zeros_like(E)
            zero_losses.append(float(pulse_packed_l1_loss_torch(z, E).item()))
    out = {
        "init_val_l1_mean": float(np.mean(losses)),
        "zero_pred_l1_mean": float(np.mean(zero_losses)),
    }
    print(f"phase0c: init_L1={out['init_val_l1_mean']:.4f} zero_pred={out['zero_pred_l1_mean']:.4f}")
    (OUT_DIR / "phase0c_init.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Phase 1 — data / ambiguity / forward consistency
# ---------------------------------------------------------------------------


def phase1a_pulse_stats(*, n: int = 512, seed: int = 0) -> dict[str, Any]:
    ensure_out_dir()
    device = get_device()
    set_seed(seed)
    grid = make_grid(n_spikes=300)
    pulses, _, t, _ = generate_pulses_stochastic(
        n_pulses=n, config=grid, seed=seed, canonicalize_mode="tstar"
    )
    amp = np.abs(pulses)
    # effective support: fraction of samples with |E| > 0.1 * peak
    peaks = amp.max(axis=1, keepdims=True) + 1e-12
    support = (amp > 0.1 * peaks).mean(axis=1)
    # spectral bandwidth proxy via FFT
    specs = np.fft.fftshift(np.fft.fft(pulses, axis=-1), axes=-1)
    s_pow = np.abs(specs) ** 2
    s_pow = s_pow / (s_pow.sum(axis=1, keepdims=True) + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=float(t[1] - t[0])))
    f_mean = (s_pow * freqs).sum(axis=1)
    f_var = (s_pow * (freqs - f_mean[:, None]) ** 2).sum(axis=1)

    frog = build_frog(device, frog_mode="padded")
    E = pack_pulses_complex(pulses)
    I = _frog_traces_batched(frog, E, device).numpy()
    out = {
        "n": n,
        "amp_peak_mean": float(amp.max(axis=1).mean()),
        "support_frac_mean": float(support.mean()),
        "support_frac_std": float(support.std()),
        "spec_rms_bw_mean": float(np.sqrt(f_var).mean()),
        "trace_energy_mean": float((I**2).sum(axis=(1, 2)).mean()),
        "trace_dynamic_range_db_mean": float(
            np.mean(10 * np.log10((I.max(axis=(1, 2)) + 1e-12) / (I.mean(axis=(1, 2)) + 1e-12)))
        ),
        "grid": {
            "n_spikes": grid.n_spikes,
            "t_total_fs": grid.t_total_fs,
            "coherence_time_fs": grid.coherence_time_fs,
            "t_center_std_fs": grid.t_center_std_fs,
        },
    }
    (OUT_DIR / "phase1a_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def phase1b_ambiguity_probe(
    *,
    n: int = 400,
    seed: int = 0,
    top_k: int = 20,
) -> dict[str, Any]:
    """Find pairs with similar traces but dissimilar pulses (near-collisions)."""
    ensure_out_dir()
    device = get_device()
    set_seed(seed)
    grid = make_grid()
    pulses, _, _, _ = generate_pulses_stochastic(
        n_pulses=n, config=grid, seed=seed, canonicalize_mode="tstar"
    )
    frog = build_frog(device, frog_mode="padded")
    E = pack_pulses_complex(pulses)
    I = _frog_traces_batched(frog, E, device).numpy().reshape(n, -1)
    # subsample pairwise among random pairs for speed
    rng = np.random.default_rng(seed)
    n_pairs = min(5000, n * (n - 1) // 2)
    i1 = rng.integers(0, n, size=n_pairs)
    i2 = rng.integers(0, n, size=n_pairs)
    mask = i1 != i2
    i1, i2 = i1[mask], i2[mask]

    dI = np.linalg.norm(I[i1] - I[i2], axis=1)
    dE = np.array(
        [
            best_l1_ambiguity(pulses[a], pulses[b])
            for a, b in zip(i1, i2)
        ],
        dtype=np.float64,
    )
    # collision score: large dE / (dI + eps)
    score = dE / (dI + 1e-8)
    order = np.argsort(-score)[:top_k]
    out = {
        "n": n,
        "n_pairs": int(len(dI)),
        "dI_mean": float(dI.mean()),
        "dE_mean": float(dE.mean()),
        "corr_dI_dE": float(np.corrcoef(dI, dE)[0, 1]),
        "top_collision_score_mean": float(score[order].mean()),
        "top_pairs": [
            {
                "i": int(i1[j]),
                "j": int(i2[j]),
                "dI": float(dI[j]),
                "dE_l1_amb": float(dE[j]),
                "score": float(score[j]),
            }
            for j in order[:10]
        ],
    }
    (OUT_DIR / "phase1b_ambiguity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"phase1b: corr(dI,dE)={out['corr_dI_dE']:.3f}  "
        f"top_collision_score_mean={out['top_collision_score_mean']:.3f}"
    )
    return out


def phase1c_forward_consistency(
    *,
    train_name: str = "phase2_ntrain_2048",
    snr_db: float = HIGH_SNR_DB,
    n_show: int = 64,
    seed: int = 0,
    force_train_if_missing: bool = True,
) -> dict[str, Any]:
    """If FROG(Ê)≈I but Ê≠E → ambiguity; if both bad → model not learning map."""
    ensure_out_dir()
    ckpt = OUT_DIR / f"{train_name}.pt"
    if not ckpt.exists():
        if force_train_if_missing:
            run_named_train(
                train_name,
                n_train=2048,
                n_val=200,
                n_test=256,
                max_epochs=100,
                patience=25,
                seed=seed,
            )
        else:
            raise FileNotFoundError(ckpt)

    device = get_device()
    set_seed(seed + 99)
    bundle, frog, _ = build_bundle(
        n_train=64, n_val=64, n_test=n_show, batch_size=64, seed=seed + 99, device=device
    )
    model = build_multires(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    pulse_l1 = []
    trace_l1 = []
    with torch.no_grad():
        for I_clean, E_true in bundle.test_loader:
            I_clean = I_clean.to(device)
            E_true = E_true.to(device)
            I_n = add_trace_noise_awgn(I_clean, snr_db)
            E_pred = extract_pulse_prediction(model(I_n.unsqueeze(1)))
            I_pred = frog(E_pred)
            rec = packed_batch_to_complex(E_pred.cpu())
            true = packed_batch_to_complex(E_true.cpu())
            for i in range(rec.shape[0]):
                pulse_l1.append(float(best_l1_ambiguity(rec[i], true[i])))
            # trace L1 relative to clean (high SNR → nearly clean)
            t_err = (I_pred - I_clean).abs().flatten(1).sum(dim=-1).cpu().numpy()
            trace_l1.extend([float(x) for x in t_err])

    out = {
        "train_name": train_name,
        "snr_db": snr_db,
        "pulse_l1_amb_mean": float(np.mean(pulse_l1)),
        "pulse_l1_amb_std": float(np.std(pulse_l1)),
        "trace_l1_mean": float(np.mean(trace_l1)),
        "trace_l1_std": float(np.std(trace_l1)),
        "trace_over_pulse_ratio": float(np.mean(trace_l1) / (np.mean(pulse_l1) + 1e-8)),
    }
    (OUT_DIR / "phase1c_forward.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# Phase 2 / 3 orchestration
# ---------------------------------------------------------------------------


def run_phase2_ntrain_sweep(
    sizes: list[int] | None = None,
    *,
    force: bool = False,
) -> list[RunResult]:
    sizes = sizes or [512, 2048, 8192]
    results = []
    for n in sizes:
        results.append(
            run_named_train(
                f"phase2_ntrain_{n}",
                n_train=n,
                n_val=200,
                n_test=256,
                max_epochs=120 if n <= 2048 else 100,
                patience=25,
                force=force,
            )
        )
    return results


def run_phase2_capacity_sweep(*, n_train: int = 2048, force: bool = False) -> list[RunResult]:
    configs = {
        "tiny": (4, 8, 16),
        "default": (8, 16, 32),
        "wide": (16, 32, 64),
    }
    results = []
    for tag, fpb in configs.items():
        results.append(
            run_named_train(
                f"phase2_cap_{tag}_n{n_train}",
                n_train=n_train,
                filters_per_branch=fpb,
                max_epochs=100,
                patience=25,
                force=force,
            )
        )
    return results


def run_phase3_ablations(*, n_train: int = 2048, force: bool = False) -> list[RunResult]:
    jobs = [
        dict(name=f"phase3_frog_padded_n{n_train}", frog_mode="padded"),
        dict(name=f"phase3_frog_plain_n{n_train}", frog_mode="plain"),
        dict(name=f"phase3_canon_tstar_n{n_train}", canonicalize_mode="tstar"),
        dict(name=f"phase3_canon_t0_n{n_train}", canonicalize_mode="t0"),
        dict(name=f"phase3_spikes_30_n{n_train}", n_spikes=30),
        dict(name=f"phase3_spikes_100_n{n_train}", n_spikes=100),
        dict(name=f"phase3_spikes_300_n{n_train}", n_spikes=300),
        dict(name=f"phase3_spikes_800_n{n_train}", n_spikes=800),
    ]
    # dedupe: padded/tstar/300 already overlap — still fine, cache handles it
    results = []
    for job in jobs:
        kwargs = dict(job)
        name = kwargs.pop("name")
        results.append(
            run_named_train(
                name,
                n_train=n_train,
                n_val=200,
                n_test=256,
                max_epochs=100,
                patience=25,
                force=force,
                **kwargs,
            )
        )
    return results


def summarize_results(results: list[RunResult]) -> str:
    lines = [
        f"{'name':40s}  {'valL1':>8s}  {'hiL1':>8s}  {'hiSim':>8s}  {'ep':>4s}"
    ]
    for r in results:
        lines.append(
            f"{r.name:40s}  {r.best_val_l1:8.4f}  {r.high_snr_l1_amb_mean:8.4f}  "
            f"{r.high_snr_sim_amb_mean:8.4f}  {r.best_epoch:4d}"
        )
    return "\n".join(lines)


def run_all(
    *,
    run_phase0: bool = True,
    run_phase1: bool = True,
    run_phase2: bool = True,
    run_phase3: bool = True,
    force: bool = False,
    ntrain_sizes: list[int] | None = None,
) -> dict[str, Any]:
    ensure_out_dir()
    summary: dict[str, Any] = {}
    if run_phase0:
        summary["phase0a"] = phase0a_overfit_minibatch(force=force)
        summary["phase0b"] = phase0b_input_independent(force=force)
        summary["phase0c"] = phase0c_loss_at_init()
    if run_phase1:
        summary["phase1a"] = phase1a_pulse_stats()
        summary["phase1b"] = phase1b_ambiguity_probe()
    if run_phase2:
        r2 = run_phase2_ntrain_sweep(ntrain_sizes, force=force)
        summary["phase2_ntrain"] = [asdict(x) for x in r2]
        # capacity after a 2K model exists
        r2c = run_phase2_capacity_sweep(n_train=2048, force=force)
        summary["phase2_capacity"] = [asdict(x) for x in r2c]
        print("\nPhase 2 n_train:\n" + summarize_results(r2))
        print("\nPhase 2 capacity:\n" + summarize_results(r2c))
    if run_phase1:
        summary["phase1c"] = phase1c_forward_consistency(
            train_name="phase2_ntrain_2048", force_train_if_missing=True
        )
    if run_phase3:
        r3 = run_phase3_ablations(n_train=2048, force=force)
        summary["phase3"] = [asdict(x) for x in r3]
        print("\nPhase 3 ablations:\n" + summarize_results(r3))

    out_path = OUT_DIR / "summary_all.json"
    # strip huge loss curves from top-level dump
    slim = json.loads(json.dumps(summary, default=str))
    for key in ("phase2_ntrain", "phase2_capacity", "phase3"):
        if key in slim:
            for item in slim[key]:
                item.pop("train_losses", None)
                item.pop("val_l1_pulses", None)
    if "phase0a" in slim:
        slim["phase0a"].pop("train_losses", None)
    out_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return summary


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-phase0", action="store_true")
    p.add_argument("--skip-phase1", action="store_true")
    p.add_argument("--skip-phase2", action="store_true")
    p.add_argument("--skip-phase3", action="store_true")
    p.add_argument("--ntrain", type=int, nargs="*", default=None)
    args = p.parse_args()
    run_all(
        run_phase0=not args.skip_phase0,
        run_phase1=not args.skip_phase1,
        run_phase2=not args.skip_phase2,
        run_phase3=not args.skip_phase3,
        force=args.force,
        ntrain_sizes=args.ntrain,
    )
