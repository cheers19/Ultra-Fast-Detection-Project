"""Benchmark training, SNR sweeps, and plotting for Multires / PCGPA comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt  # noqa: F401 — used by plot_metric_curves
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_utils import PulseGridConfig, build_frog_dataloaders
from frog_reconstruction_model import extract_pulse_prediction
from evaluate_cnn import plot_metric_curves
from frognet import FROGNet
from pcgpa_reconstruct import (
    reconstruct_pcgpa,
    temporal_field_to_initial_spectrum,
)
from pulse_metrics import (
    best_l1_ambiguity,
    best_l1_ambiguity_field,
    best_similarity_error_ambiguity,
    l1_packed_mae,
    packed_batch_to_complex,
    snr_db_to_equivalent_n_pulses,
    trace_l1_sum_numpy,
    unpack_packed_field,
)
from train import (
    EarlyStopTrainResult,
    TrainConfig,
    build_model,
    load_checkpoint,
    save_checkpoint,
    snr_curriculum_db_range,
    train_trace_to_pulse_early_stopping,
)
from trace_noise import add_trace_noise_awgn


DEFAULT_SNR_HEAD_LOSS_WEIGHT = 0.05

MetricKind = Literal["pulse_l1", "trace_l1", "similarity_error"]


@dataclass
class BenchmarkCurve:
    label: str
    mean: np.ndarray
    std: np.ndarray


@dataclass
class BenchmarkSweep:
    snr_sweep_db: np.ndarray
    pulse_l1: BenchmarkCurve
    trace_l1: BenchmarkCurve
    similarity_error: BenchmarkCurve | None = None
    meta: dict | None = None


def train_trace_model_early_stop(
    *,
    model_name: str,
    n_train: int,
    checkpoint_path: str | Path,
    experiment_name: str,
    device: torch.device,
    seed: int = 0,
    n_val: int = 512,
    n_test: int = 512,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 15,
    train_snr_db_range: tuple[float, float] = (0.0, 30.0),
    val_snr_db: float = 15.0,
    n: int = 64,
    add_noise_fn: Callable = add_trace_noise_awgn,
    snr_range_fn: Callable[[int, int], tuple[float, float]] | None = None,
    snr_loss_weight: float = 0.0,
    verbose: bool = True,
) -> tuple[nn.Module, EarlyStopTrainResult]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    bundle = build_frog_dataloaders(
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        batch_size=batch_size,
        seed=seed,
        device=device,
        grid=PulseGridConfig(n=n),
    )
    model = build_model(n, device, model_name=model_name)
    cfg = TrainConfig(
        n=n,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        batch_size=batch_size,
        epochs=max_epochs,
        lr=lr,
        train_snr_db_range=train_snr_db_range,
        val_snr_db=val_snr_db,
        seed=seed,
        snr_loss_weight=snr_loss_weight,
        checkpoint_path=str(checkpoint_path),
        experiment_name=experiment_name,
        model_name=model_name,
        device=str(device),
    )
    result = train_trace_to_pulse_early_stopping(
        model,
        bundle.train_loader,
        bundle.val_loader,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        train_snr_db_range=train_snr_db_range,
        val_snr_db=val_snr_db,
        add_noise_fn=add_noise_fn,
        snr_range_fn=snr_range_fn,
        snr_loss_weight=snr_loss_weight,
        verbose=verbose,
    )
    extra: dict = {
        "best_epoch": result.best_epoch,
        "best_val_l1": result.best_val_l1,
        "stopped_epoch": result.stopped_epoch,
    }
    if snr_range_fn is not None:
        extra["snr_curriculum"] = True
        extra["snr_curriculum_fn"] = getattr(snr_range_fn, "__name__", "custom")
    if snr_loss_weight > 0.0:
        extra["snr_loss_weight"] = float(snr_loss_weight)
        extra["predicts_snr"] = True
    payload = save_checkpoint(
        checkpoint_path,
        model,
        cfg,
        result.history,
        extra=extra,
    )
    if verbose:
        print(f"Saved {payload} (best epoch {result.best_epoch})")
    return model, result


def train_multires_early_stop(
    *,
    n_train: int,
    checkpoint_path: str | Path,
    experiment_name: str,
    device: torch.device,
    seed: int = 0,
    n_val: int = 512,
    n_test: int = 512,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 15,
    train_snr_db_range: tuple[float, float] = (0.0, 30.0),
    val_snr_db: float = 15.0,
    n: int = 64,
    add_noise_fn: Callable = add_trace_noise_awgn,
    snr_range_fn: Callable[[int, int], tuple[float, float]] | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, EarlyStopTrainResult]:
    return train_trace_model_early_stop(
        model_name="multires",
        n_train=n_train,
        checkpoint_path=checkpoint_path,
        experiment_name=experiment_name,
        device=device,
        seed=seed,
        n_val=n_val,
        n_test=n_test,
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        train_snr_db_range=train_snr_db_range,
        val_snr_db=val_snr_db,
        n=n,
        add_noise_fn=add_noise_fn,
        snr_range_fn=snr_range_fn,
        verbose=verbose,
    )


def train_multires_curriculum_early_stop(
    *,
    n_train: int = 6_000,
    checkpoint_path: str | Path = Path("checkpoints/benchmark/multires_6k_curriculum_es.pt"),
    experiment_name: str = "multires_6k_curriculum_es",
    device: torch.device,
    seed: int = 0,
    **kwargs,
) -> tuple[nn.Module, EarlyStopTrainResult]:
    """Multires with 3-phase SNR curriculum: 20–30 → 10–25 → 0–30 dB."""
    return train_multires_early_stop(
        n_train=n_train,
        checkpoint_path=checkpoint_path,
        experiment_name=experiment_name,
        device=device,
        seed=seed,
        snr_range_fn=snr_curriculum_db_range,
        **kwargs,
    )


def train_unet_early_stop(
    *,
    n_train: int = 6_000,
    checkpoint_path: str | Path = Path("checkpoints/benchmark/unet_6k_es.pt"),
    experiment_name: str = "unet_6k_es",
    device: torch.device,
    seed: int = 0,
    **kwargs,
) -> tuple[nn.Module, EarlyStopTrainResult]:
    return train_trace_model_early_stop(
        model_name="unet",
        n_train=n_train,
        checkpoint_path=checkpoint_path,
        experiment_name=experiment_name,
        device=device,
        seed=seed,
        **kwargs,
    )


def train_multires_snr_early_stop(
    *,
    n_train: int = 6_000,
    checkpoint_path: str | Path = Path("checkpoints/benchmark/multires_6k_snr_es.pt"),
    experiment_name: str = "multires_6k_snr_es",
    device: torch.device,
    seed: int = 0,
    snr_loss_weight: float = DEFAULT_SNR_HEAD_LOSS_WEIGHT,
    **kwargs,
) -> tuple[nn.Module, EarlyStopTrainResult]:
    return train_trace_model_early_stop(
        model_name="multires_snr",
        n_train=n_train,
        checkpoint_path=checkpoint_path,
        experiment_name=experiment_name,
        device=device,
        seed=seed,
        snr_loss_weight=snr_loss_weight,
        **kwargs,
    )


def train_unet_snr_early_stop(
    *,
    n_train: int = 6_000,
    checkpoint_path: str | Path = Path("checkpoints/benchmark/unet_6k_snr_es.pt"),
    experiment_name: str = "unet_6k_snr_es",
    device: torch.device,
    seed: int = 0,
    snr_loss_weight: float = DEFAULT_SNR_HEAD_LOSS_WEIGHT,
    **kwargs,
) -> tuple[nn.Module, EarlyStopTrainResult]:
    return train_trace_model_early_stop(
        model_name="unet_snr",
        n_train=n_train,
        checkpoint_path=checkpoint_path,
        experiment_name=experiment_name,
        device=device,
        seed=seed,
        snr_loss_weight=snr_loss_weight,
        **kwargs,
    )


MULTIRES_60K_PRETRAINED = Path("checkpoints/large_60k_multires_50ep.pt")


def load_or_train_benchmark_models(
    *,
    bench_dir: Path,
    device: torch.device,
    seed: int = 0,
    n_val: int = 512,
    n_test: int = 512,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 15,
    train_snr_db_range: tuple[float, float] = (0.0, 30.0),
    val_snr_db: float = 15.0,
    n: int = 64,
    add_noise_fn: Callable = add_trace_noise_awgn,
    force_retrain: set[str] | frozenset[str] = frozenset(),
    multires_60k_checkpoint: Path | None = None,
    verbose: bool = True,
) -> dict[str, nn.Module]:
    """Load pretrained Multires 60K; early-stop train 6K and 2K only."""
    trained: dict[str, nn.Module] = {}
    ckpt_60k = Path(multires_60k_checkpoint or MULTIRES_60K_PRETRAINED)
    if not ckpt_60k.is_file():
        raise FileNotFoundError(
            f"Pretrained Multires 60K not found: {ckpt_60k}. "
            "Train it separately (e.g. scripts/train_trace_to_pulse.py)."
        )
    model_60k, meta_60k = load_checkpoint(ckpt_60k, device)
    trained["multires_60k"] = model_60k
    if verbose:
        print(f"Loaded multires_60k (pretrained, no retrain): {ckpt_60k}")

    bench_dir.mkdir(parents=True, exist_ok=True)
    train_specs = {
        "multires_6k": {"n_train": 6_000, "ckpt": bench_dir / "multires_6k_es.pt"},
        "multires_2k": {"n_train": 2048, "ckpt": bench_dir / "multires_2k_es.pt"},
    }
    for name, spec in train_specs.items():
        ckpt = spec["ckpt"]
        if ckpt.is_file() and name not in force_retrain:
            model, meta = load_checkpoint(ckpt, device)
            trained[name] = model
            if verbose:
                print(f"Loaded {name}: {ckpt} (best epoch {meta.get('best_epoch', '?')})")
            continue
        if verbose:
            print(
                f"Training {name} (n_train={spec['n_train']}, "
                f"max_epochs={max_epochs}, patience={patience}) …"
            )
        model, res = train_multires_early_stop(
            n_train=spec["n_train"],
            checkpoint_path=ckpt,
            experiment_name=name,
            device=device,
            seed=seed,
            n_val=n_val,
            n_test=n_test,
            batch_size=batch_size,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            train_snr_db_range=train_snr_db_range,
            val_snr_db=val_snr_db,
            n=n,
            add_noise_fn=add_noise_fn,
            verbose=verbose,
        )
        trained[name] = model
        if verbose:
            print(
                f"  stopped @ epoch {res.stopped_epoch}, "
                f"best epoch {res.best_epoch}, val L1={res.best_val_l1:.5f}"
            )
    return trained


def _frog_trace_from_field(e_t: np.ndarray, frog: FROGNet, device: torch.device) -> np.ndarray:
    e = np.asarray(e_t, dtype=np.complex128).ravel()
    packed = np.concatenate([e.real, e.imag]).astype(np.float32)
    with torch.no_grad():
        i_t = frog(torch.from_numpy(packed).unsqueeze(0).to(device))
    return i_t.squeeze(0).cpu().numpy()


def _curve(label: str, m: list[float], s: list[float]) -> BenchmarkCurve:
    return BenchmarkCurve(label=label, mean=np.asarray(m), std=np.asarray(s))


def sweep_multires_at_snr(
    model: nn.Module,
    loader: DataLoader,
    snr_db: float,
    *,
    frog: FROGNet,
    device: torch.device,
    add_noise_fn: Callable = add_trace_noise_awgn,
) -> tuple[list[float], list[float], list[float]]:
    model.eval()
    pulse_l1: list[float] = []
    trace_l1: list[float] = []
    sim_err: list[float] = []
    with torch.no_grad():
        for I_clean, E_true in loader:
            I_noisy = add_noise_fn(I_clean, float(snr_db))
            E_pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
            rec = packed_batch_to_complex(E_pred)
            true_c = packed_batch_to_complex(E_true)
            for i in range(rec.shape[0]):
                e_ref = true_c[i]
                pulse_l1.append(float(best_l1_ambiguity(rec[i], e_ref)))
                sim_err.append(float(best_similarity_error_ambiguity(rec[i], e_ref)))
                i_n = I_noisy[i].cpu().numpy()
                i_rec = _frog_trace_from_field(rec[i], frog, device)
                trace_l1.append(float(trace_l1_sum_numpy(i_rec, i_n)))
    return pulse_l1, trace_l1, sim_err


def run_multires_benchmark_sweep(
    model: nn.Module,
    test_loader: DataLoader,
    snr_sweep_db: np.ndarray,
    *,
    label: str,
    frog: FROGNet,
    device: torch.device,
    add_noise_fn: Callable = add_trace_noise_awgn,
    verbose: bool = True,
) -> BenchmarkSweep:
    pl_m, pl_s, tl_m, tl_s, sim_m, sim_s = [], [], [], [], [], []
    for snr_db in snr_sweep_db:
        if verbose:
            print(f"{label} @ {float(snr_db):.1f} dB …")
        p_per, t_per, s_per = sweep_multires_at_snr(
            model, test_loader, float(snr_db), frog=frog, device=device, add_noise_fn=add_noise_fn
        )
        p_arr = np.asarray(p_per, dtype=np.float64)
        t_arr = np.asarray(t_per, dtype=np.float64)
        s_arr = np.asarray(s_per, dtype=np.float64)
        pl_m.append(float(p_arr.mean()))
        pl_s.append(float(p_arr.std(ddof=0)))
        tl_m.append(float(t_arr.mean()))
        tl_s.append(float(t_arr.std(ddof=0)))
        sim_m.append(float(s_arr.mean()))
        sim_s.append(float(s_arr.std(ddof=0)))
    return BenchmarkSweep(
        snr_sweep_db=np.asarray(snr_sweep_db, dtype=np.float64),
        pulse_l1=_curve(label, pl_m, pl_s),
        trace_l1=_curve(label, tl_m, tl_s),
        similarity_error=_curve(label, sim_m, sim_s),
    )


def sweep_pcgpa_at_snr(
    i_clean_batch: np.ndarray,
    e_true_batch: np.ndarray,
    snr_db: float,
    *,
    add_noise_fn: Callable,
    dt: float,
    frog: FROGNet,
    device: torch.device,
    pcgpa_maxiter: int,
    pcgpa_early_stop_patience: int | None,
    n_subsample: int | None,
    seed: int,
    n_restarts: int = 3,
    sigma_omega: float | None = None,
    show_progress: bool = False,
) -> tuple[list[float], list[float], list[float]]:
    from pcgpa_reconstruct import _pcgpa_rng_for_pulse, _pcgpa_subsample_indices

    i_clean = np.asarray(i_clean_batch, dtype=np.float64)
    e_true = np.asarray(e_true_batch, dtype=np.float64)
    idx = _pcgpa_subsample_indices(i_clean.shape[0], n_subsample, seed)
    pulse_l1: list[float] = []
    trace_l1: list[float] = []
    sim_err: list[float] = []
    iterator = idx
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(idx, desc=f"PCGPA @ {float(snr_db):.0f} dB", leave=False)
        except ImportError:
            pass

    for i in iterator:
        i_c = torch.as_tensor(i_clean[i])
        i_n = add_noise_fn(i_c.unsqueeze(0), float(snr_db)).squeeze(0).numpy()
        e_ref_packed = e_true[i]
        e_ref = unpack_packed_field(e_ref_packed)
        e_rec = reconstruct_pcgpa(
            i_n,
            dt=dt,
            maxiter=pcgpa_maxiter,
            early_stop_patience=pcgpa_early_stop_patience,
            n_restarts=n_restarts,
            rng=_pcgpa_rng_for_pulse(seed, int(i), float(snr_db)),
            sigma_omega=sigma_omega,
        )
        pulse_l1.append(float(l1_packed_mae(e_rec, e_ref_packed, use_best_ambiguity=True)))
        sim_err.append(float(best_similarity_error_ambiguity(e_rec, e_ref)))
        i_rec = _frog_trace_from_field(e_rec, frog, device)
        trace_l1.append(float(trace_l1_sum_numpy(i_rec, i_n)))

    return pulse_l1, trace_l1, sim_err


def run_pcgpa_benchmark_sweep(
    i_clean_batch: np.ndarray,
    e_true_batch: np.ndarray,
    snr_sweep_db: np.ndarray,
    *,
    label: str = "PCGPA",
    add_noise_fn: Callable,
    dt: float,
    frog: FROGNet,
    device: torch.device,
    pcgpa_maxiter: int = 200,
    pcgpa_early_stop_patience: int | None = 50,
    n_subsample: int | None = 32,
    seed: int = 0,
    n_restarts: int = 3,
    sigma_omega: float | None = None,
    verbose: bool = True,
) -> BenchmarkSweep:
    pl_m, pl_s, tl_m, tl_s, sim_m, sim_s = [], [], [], [], [], []
    for snr_db in snr_sweep_db:
        if verbose:
            print(f"{label} @ {float(snr_db):.1f} dB …")
        p_per, t_per, s_per = sweep_pcgpa_at_snr(
            i_clean_batch,
            e_true_batch,
            float(snr_db),
            add_noise_fn=add_noise_fn,
            dt=dt,
            frog=frog,
            device=device,
            pcgpa_maxiter=pcgpa_maxiter,
            pcgpa_early_stop_patience=pcgpa_early_stop_patience,
            n_subsample=n_subsample,
            seed=seed,
            n_restarts=n_restarts,
            sigma_omega=sigma_omega,
            show_progress=verbose,
        )
        p_arr = np.asarray(p_per, dtype=np.float64)
        t_arr = np.asarray(t_per, dtype=np.float64)
        s_arr = np.asarray(s_per, dtype=np.float64)
        pl_m.append(float(p_arr.mean()))
        pl_s.append(float(p_arr.std(ddof=0)))
        tl_m.append(float(t_arr.mean()))
        tl_s.append(float(t_arr.std(ddof=0)))
        sim_m.append(float(s_arr.mean()))
        sim_s.append(float(s_arr.std(ddof=0)))
    return BenchmarkSweep(
        snr_sweep_db=np.asarray(snr_sweep_db, dtype=np.float64),
        pulse_l1=_curve(label, pl_m, pl_s),
        trace_l1=_curve(label, tl_m, tl_s),
        similarity_error=_curve(label, sim_m, sim_s),
        meta={"pcgpa_maxiter": pcgpa_maxiter, "early_stop_patience": pcgpa_early_stop_patience},
    )


def sweep_hybrid_at_snr(
    model: nn.Module,
    i_clean_batch: np.ndarray,
    e_true_batch: np.ndarray,
    snr_db: float,
    *,
    add_noise_fn: Callable,
    dt: float,
    frog: FROGNet,
    device: torch.device,
    pcgpa_maxiter: int,
    pcgpa_early_stop_patience: int | None,
    n_subsample: int | None,
    seed: int,
    show_progress: bool = False,
) -> tuple[list[float], list[float], list[float]]:
    from pcgpa_reconstruct import _pcgpa_subsample_indices

    i_clean = np.asarray(i_clean_batch, dtype=np.float64)
    e_true = np.asarray(e_true_batch, dtype=np.float64)
    idx = _pcgpa_subsample_indices(i_clean.shape[0], n_subsample, seed)
    dev = device or next(model.parameters()).device
    pulse_l1: list[float] = []
    trace_l1: list[float] = []
    sim_err: list[float] = []
    iterator = idx
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(idx, desc=f"Hybrid @ {float(snr_db):.0f} dB", leave=False)
        except ImportError:
            pass

    model.eval()
    with torch.no_grad():
        for i in iterator:
            i_c = torch.as_tensor(i_clean[i], dtype=torch.float32, device=dev)
            i_noisy = add_noise_fn(i_c.unsqueeze(0), float(snr_db))
            i_n = i_noisy.squeeze(0).cpu().numpy()
            e_cnn = packed_batch_to_complex(model(i_noisy.unsqueeze(1).to(dev)))[0]
            e_ref = unpack_packed_field(e_true[i])
            e_cnn_init = best_l1_ambiguity_field(e_cnn, e_ref)
            guess = temporal_field_to_initial_spectrum(e_cnn_init, dt=dt)
            e_rec = reconstruct_pcgpa(
                i_n,
                dt=dt,
                maxiter=pcgpa_maxiter,
                early_stop_patience=pcgpa_early_stop_patience,
                n_restarts=1,
                initial_spectrum=guess,
            )
            pulse_l1.append(float(best_l1_ambiguity(e_rec, e_ref)))
            sim_err.append(float(best_similarity_error_ambiguity(e_rec, e_ref)))
            i_rec = _frog_trace_from_field(e_rec, frog, device)
            trace_l1.append(float(trace_l1_sum_numpy(i_rec, i_n)))

    return pulse_l1, trace_l1, sim_err


def run_hybrid_benchmark_sweep(
    model: nn.Module,
    i_clean_batch: np.ndarray,
    e_true_batch: np.ndarray,
    snr_sweep_db: np.ndarray,
    *,
    label: str,
    add_noise_fn: Callable,
    dt: float,
    frog: FROGNet,
    device: torch.device,
    pcgpa_maxiter: int = 200,
    pcgpa_early_stop_patience: int | None = 50,
    n_subsample: int | None = 32,
    seed: int = 0,
    verbose: bool = True,
) -> BenchmarkSweep:
    pl_m, pl_s, tl_m, tl_s, sim_m, sim_s = [], [], [], [], [], []
    for snr_db in snr_sweep_db:
        if verbose:
            print(f"{label} @ {float(snr_db):.1f} dB …")
        p_per, t_per, s_per = sweep_hybrid_at_snr(
            model,
            i_clean_batch,
            e_true_batch,
            float(snr_db),
            add_noise_fn=add_noise_fn,
            dt=dt,
            frog=frog,
            device=device,
            pcgpa_maxiter=pcgpa_maxiter,
            pcgpa_early_stop_patience=pcgpa_early_stop_patience,
            n_subsample=n_subsample,
            seed=seed,
            show_progress=verbose,
        )
        p_arr = np.asarray(p_per, dtype=np.float64)
        t_arr = np.asarray(t_per, dtype=np.float64)
        s_arr = np.asarray(s_per, dtype=np.float64)
        pl_m.append(float(p_arr.mean()))
        pl_s.append(float(p_arr.std(ddof=0)))
        tl_m.append(float(t_arr.mean()))
        tl_s.append(float(t_arr.std(ddof=0)))
        sim_m.append(float(s_arr.mean()))
        sim_s.append(float(s_arr.std(ddof=0)))
    return BenchmarkSweep(
        snr_sweep_db=np.asarray(snr_sweep_db, dtype=np.float64),
        pulse_l1=_curve(label, pl_m, pl_s),
        trace_l1=_curve(label, tl_m, tl_s),
        similarity_error=_curve(label, sim_m, sim_s),
    )


def save_benchmark_sweep(path: str | Path, sweep: BenchmarkSweep) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "snr_sweep_db": sweep.snr_sweep_db,
        "pulse_l1_m": sweep.pulse_l1.mean,
        "pulse_l1_s": sweep.pulse_l1.std,
        "trace_l1_m": sweep.trace_l1.mean,
        "trace_l1_s": sweep.trace_l1.std,
        "label": sweep.pulse_l1.label,
        "meta": sweep.meta or {},
    }
    if sweep.similarity_error is not None:
        payload["sim_m"] = sweep.similarity_error.mean
        payload["sim_s"] = sweep.similarity_error.std
    np.savez(out, **payload)
    return out.resolve()


def load_benchmark_sweep(path: str | Path) -> BenchmarkSweep:
    z = np.load(path, allow_pickle=True)
    label = str(z["label"])
    meta = z["meta"].item() if "meta" in z else None
    sim = None
    if "sim_m" in z and "sim_s" in z:
        sim = BenchmarkCurve(label=label, mean=z["sim_m"], std=z["sim_s"])
    return BenchmarkSweep(
        snr_sweep_db=z["snr_sweep_db"],
        pulse_l1=BenchmarkCurve(label=label, mean=z["pulse_l1_m"], std=z["pulse_l1_s"]),
        trace_l1=BenchmarkCurve(label=label, mean=z["trace_l1_m"], std=z["trace_l1_s"]),
        similarity_error=sim,
        meta=meta,
    )


def plot_benchmark_comparison(
    snr_sweep_db: np.ndarray,
    curves: list[BenchmarkSweep],
    *,
    metric: MetricKind,
    title_prefix: str,
    fmts: list[str] | None = None,
) -> None:
    fmts = fmts or ["-o", "--^", "-s", "-D", "-v", "-P"]
    if metric == "pulse_l1":
        ylabel = "L1 on pulse (best amb., sum over 2N)"
        attr = "pulse_l1"
    elif metric == "similarity_error":
        ylabel = "mean SIMILARITY_ERROR (1 − cos δE, best amb.)"
        attr = "similarity_error"
    else:
        ylabel = "L1 on trace (sum |I_rec − I_noisy|)"
        attr = "trace_l1"

    series = []
    for i, sw in enumerate(curves):
        c: BenchmarkCurve | None = getattr(sw, attr)
        if c is None:
            raise ValueError(
                f"Sweep {sw.pulse_l1.label!r} has no cached {metric}; "
                "recompute with FORCE_RECOMPUTE_SWEEPS=True."
            )
        series.append((c.mean, c.std, fmts[i % len(fmts)], c.label))

    plot_metric_curves(
        snr_sweep_db,
        series,
        xlabel="trace SNR (dB)",
        ylabel=ylabel,
        title=f"{title_prefix} vs trace SNR",
    )
    n_equiv = np.array([snr_db_to_equivalent_n_pulses(float(s)) for s in snr_sweep_db])
    plot_metric_curves(
        n_equiv,
        series,
        xlabel="equivalent pulse count N (log scale)",
        ylabel=ylabel,
        title=f"{title_prefix} vs N",
        xscale="log",
    )
