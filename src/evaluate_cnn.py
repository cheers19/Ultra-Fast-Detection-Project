"""CNN SNR sweep evaluation, caching, and plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pulse_metrics import (
    best_l1_ambiguity,
    best_l1_ambiguity_field,
    best_similarity_error_ambiguity,
    l1_packed_mae,
    l1_packed_per_pulse_torch,
    packed_batch_to_complex,
    similarity_error_numpy,
    snr_db_to_equivalent_n_pulses,
    unpack_packed_field,
)
from pcgpa_reconstruct import reconstruct_pcgpa, temporal_field_to_initial_spectrum
from trace_noise import add_trace_noise_awgn


@dataclass
class CnnSweepResult:
    snr_sweep_db: np.ndarray
    cnn_sim_raw_m: np.ndarray
    cnn_sim_raw_s: np.ndarray
    cnn_sim_amb_m: np.ndarray
    cnn_sim_amb_s: np.ndarray
    cnn_l1_raw_m: np.ndarray
    cnn_l1_raw_s: np.ndarray
    cnn_l1_amb_m: np.ndarray
    cnn_l1_amb_s: np.ndarray
    experiment_name: str = ""

    @property
    def n_equiv(self) -> np.ndarray:
        return np.array(
            [snr_db_to_equivalent_n_pulses(float(s)) for s in self.snr_sweep_db]
        )


@dataclass
class HybridPcgpaSweepResult:
    snr_sweep_db: np.ndarray
    l1_amb_m: np.ndarray
    l1_amb_s: np.ndarray
    sim_amb_m: np.ndarray
    sim_amb_s: np.ndarray
    maxiter: int
    label: str = "Multires → PCGPA"


def mean_multires_init_pcgpa_metrics_at_snr(
    model: nn.Module,
    i_clean_batch: np.ndarray,
    e_true_batch: np.ndarray,
    snr_db: float,
    *,
    add_noise_fn: Callable,
    dt: float,
    pcgpa_maxiter: int,
    n_subsample: int | None = None,
    seed: int = 0,
    use_best_ambiguity: bool = True,
    device: torch.device | None = None,
    show_progress: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Mean/std L1 and SIMILARITY_ERROR (best ambiguity) for Multires-init PCGPA."""
    from pcgpa_reconstruct import _pcgpa_subsample_indices

    i_clean = np.asarray(i_clean_batch, dtype=np.float64)
    e_true = np.asarray(e_true_batch, dtype=np.float64)
    idx = _pcgpa_subsample_indices(i_clean.shape[0], n_subsample, seed)
    dev = device or next(model.parameters()).device

    l1_per: list[float] = []
    sim_per: list[float] = []
    iterator = idx
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(
                idx,
                desc=f"Multires→PCGPA @ {float(snr_db):.0f} dB",
                leave=False,
            )
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
                n_restarts=1,
                initial_spectrum=guess,
            )
            if use_best_ambiguity:
                l1_per.append(float(best_l1_ambiguity(e_rec, e_ref)))
                sim_per.append(float(best_similarity_error_ambiguity(e_rec, e_ref)))
            else:
                l1_per.append(
                    float(l1_packed_mae(e_rec, e_true[i], use_best_ambiguity=False))
                )
                sim_per.append(float(similarity_error_numpy(e_rec, e_ref)))

    l1_arr = np.asarray(l1_per, dtype=np.float64)
    sim_arr = np.asarray(sim_per, dtype=np.float64)
    return (
        (float(l1_arr.mean()), float(l1_arr.std(ddof=0))),
        (float(sim_arr.mean()), float(sim_arr.std(ddof=0))),
    )


def mean_l1_multires_init_pcgpa_at_snr(
    model: nn.Module,
    i_clean_batch: np.ndarray,
    e_true_batch: np.ndarray,
    snr_db: float,
    *,
    add_noise_fn: Callable,
    dt: float,
    pcgpa_maxiter: int,
    n_subsample: int | None = None,
    seed: int = 0,
    use_best_ambiguity: bool = True,
    device: torch.device | None = None,
    show_progress: bool = False,
) -> tuple[float, float]:
    """Mean/std L1 for PCGPA seeded by Multires prediction after best-L1 ambiguity alignment."""
    (l1_m, l1_s), _ = mean_multires_init_pcgpa_metrics_at_snr(
        model,
        i_clean_batch,
        e_true_batch,
        snr_db,
        add_noise_fn=add_noise_fn,
        dt=dt,
        pcgpa_maxiter=pcgpa_maxiter,
        n_subsample=n_subsample,
        seed=seed,
        use_best_ambiguity=use_best_ambiguity,
        device=device,
        show_progress=show_progress,
    )
    return l1_m, l1_s


def run_multires_init_pcgpa_snr_sweep(
    model: nn.Module,
    i_clean_batch: np.ndarray,
    e_true_batch: np.ndarray,
    snr_sweep_db: np.ndarray,
    *,
    add_noise_fn: Callable,
    dt: float,
    pcgpa_maxiter: int,
    n_subsample: int | None = None,
    seed: int = 0,
    use_best_ambiguity: bool = True,
    device: torch.device | None = None,
    label: str = "Multires → PCGPA",
    verbose: bool = True,
) -> HybridPcgpaSweepResult:
    maxiter = int(pcgpa_maxiter)
    l1_m, l1_s = [], []
    sim_m, sim_s = [], []
    for snr_db in snr_sweep_db:
        if verbose:
            print(f"Multires→PCGPA sweep @ {float(snr_db):.1f} dB (maxiter={maxiter}) …")
        (lm, ls), (sm, ss) = mean_multires_init_pcgpa_metrics_at_snr(
            model,
            i_clean_batch,
            e_true_batch,
            float(snr_db),
            add_noise_fn=add_noise_fn,
            dt=dt,
            pcgpa_maxiter=maxiter,
            n_subsample=n_subsample,
            seed=seed,
            use_best_ambiguity=use_best_ambiguity,
            device=device,
            show_progress=verbose,
        )
        l1_m.append(lm)
        l1_s.append(ls)
        sim_m.append(sm)
        sim_s.append(ss)
    return HybridPcgpaSweepResult(
        snr_sweep_db=np.asarray(snr_sweep_db, dtype=np.float64),
        l1_amb_m=np.asarray(l1_m),
        l1_amb_s=np.asarray(l1_s),
        sim_amb_m=np.asarray(sim_m),
        sim_amb_s=np.asarray(sim_s),
        maxiter=maxiter,
        label=label,
    )


def save_hybrid_pcgpa_sweep(path: str | Path, result: HybridPcgpaSweepResult) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        snr_sweep_db=result.snr_sweep_db,
        l1_amb_m=result.l1_amb_m,
        l1_amb_s=result.l1_amb_s,
        sim_amb_m=result.sim_amb_m,
        sim_amb_s=result.sim_amb_s,
        maxiter=result.maxiter,
        label=result.label,
    )
    return out.resolve()


def load_hybrid_pcgpa_sweep(path: str | Path) -> HybridPcgpaSweepResult:
    z = np.load(path, allow_pickle=False)
    label = str(z["label"]) if "label" in z else "Multires → PCGPA"
    if "sim_amb_m" not in z:
        raise ValueError(
            f"Hybrid sweep cache {path} has no sim_amb_m — re-run "
            "run_multires_init_pcgpa_snr_sweep with FORCE_RECOMPUTE."
        )
    return HybridPcgpaSweepResult(
        snr_sweep_db=z["snr_sweep_db"],
        l1_amb_m=z["l1_amb_m"],
        l1_amb_s=z["l1_amb_s"],
        sim_amb_m=z["sim_amb_m"],
        sim_amb_s=z["sim_amb_s"],
        maxiter=int(z["maxiter"]),
        label=label,
    )


def mean_metric_cnn_at_snr(
    model: nn.Module,
    loader: DataLoader,
    snr_db: float,
    *,
    score_fn: Callable,
    add_noise_fn: Callable[[torch.Tensor, float], torch.Tensor] = add_trace_noise_awgn,
) -> tuple[float, float]:
    model.eval()
    per: list[float] = []
    with torch.no_grad():
        for I_clean, E_true in loader:
            I_noisy = add_noise_fn(I_clean, snr_db)
            E_pred = model(I_noisy.unsqueeze(1))
            rec = packed_batch_to_complex(E_pred)
            true = packed_batch_to_complex(E_true)
            for i in range(rec.shape[0]):
                per.append(float(score_fn(rec[i], true[i])))
    arr = np.asarray(per, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def mean_l1_cnn_at_snr(
    model: nn.Module,
    loader: DataLoader,
    snr_db: float,
    *,
    score_fn: Callable,
    truth: str = "complex",
    add_noise_fn: Callable[[torch.Tensor, float], torch.Tensor] = add_trace_noise_awgn,
) -> tuple[float, float]:
    model.eval()
    per: list[float] = []
    with torch.no_grad():
        for I_clean, E_true in loader:
            I_noisy = add_noise_fn(I_clean, snr_db)
            E_pred = model(I_noisy.unsqueeze(1))
            rec = packed_batch_to_complex(E_pred)
            true_c = packed_batch_to_complex(E_true)
            for i in range(rec.shape[0]):
                ref = E_true[i].cpu().numpy() if truth == "packed" else true_c[i]
                per.append(float(score_fn(rec[i], ref)))
    arr = np.asarray(per, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def l1_stats_at_snr(
    model: nn.Module,
    loader: DataLoader,
    snr_db: float,
    *,
    add_noise_fn: Callable[[torch.Tensor, float], torch.Tensor] = add_trace_noise_awgn,
) -> tuple[float, float]:
    model.eval()
    per_pulse: list[float] = []
    with torch.no_grad():
        for I_clean, E_true in loader:
            I_noisy = add_noise_fn(I_clean, snr_db)
            E_pred = model(I_noisy.unsqueeze(1))
            per_pulse.extend(l1_packed_per_pulse_torch(E_pred, E_true).cpu().tolist())
    arr = np.asarray(per_pulse, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def run_cnn_snr_sweep(
    model: nn.Module,
    test_loader: DataLoader,
    snr_sweep_db: np.ndarray,
    *,
    experiment_name: str = "",
    add_noise_fn: Callable[[torch.Tensor, float], torch.Tensor] = add_trace_noise_awgn,
    verbose: bool = True,
) -> CnnSweepResult:
    cnn_sim_raw_m, cnn_sim_raw_s = [], []
    cnn_sim_amb_m, cnn_sim_amb_s = [], []
    cnn_l1_raw_m, cnn_l1_raw_s = [], []
    cnn_l1_amb_m, cnn_l1_amb_s = [], []

    for snr_db in snr_sweep_db:
        if verbose:
            print(f"CNN sweep @ {float(snr_db):.1f} dB …")
        m, s = mean_metric_cnn_at_snr(
            model, test_loader, float(snr_db),
            score_fn=similarity_error_numpy, add_noise_fn=add_noise_fn,
        )
        cnn_sim_raw_m.append(m)
        cnn_sim_raw_s.append(s)
        m, s = mean_metric_cnn_at_snr(
            model, test_loader, float(snr_db),
            score_fn=best_similarity_error_ambiguity, add_noise_fn=add_noise_fn,
        )
        cnn_sim_amb_m.append(m)
        cnn_sim_amb_s.append(s)
        m, s = mean_l1_cnn_at_snr(
            model, test_loader, float(snr_db),
            score_fn=lambda rec, packed: l1_packed_mae(rec, packed, use_best_ambiguity=False),
            truth="packed", add_noise_fn=add_noise_fn,
        )
        cnn_l1_raw_m.append(m)
        cnn_l1_raw_s.append(s)
        m, s = mean_l1_cnn_at_snr(
            model, test_loader, float(snr_db),
            score_fn=best_l1_ambiguity, add_noise_fn=add_noise_fn,
        )
        cnn_l1_amb_m.append(m)
        cnn_l1_amb_s.append(s)

    return CnnSweepResult(
        snr_sweep_db=np.asarray(snr_sweep_db, dtype=np.float64),
        cnn_sim_raw_m=np.asarray(cnn_sim_raw_m),
        cnn_sim_raw_s=np.asarray(cnn_sim_raw_s),
        cnn_sim_amb_m=np.asarray(cnn_sim_amb_m),
        cnn_sim_amb_s=np.asarray(cnn_sim_amb_s),
        cnn_l1_raw_m=np.asarray(cnn_l1_raw_m),
        cnn_l1_raw_s=np.asarray(cnn_l1_raw_s),
        cnn_l1_amb_m=np.asarray(cnn_l1_amb_m),
        cnn_l1_amb_s=np.asarray(cnn_l1_amb_s),
        experiment_name=experiment_name,
    )


def save_cnn_sweep(path: str | Path, result: CnnSweepResult) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        snr_sweep_db=result.snr_sweep_db,
        cnn_sim_raw_m=result.cnn_sim_raw_m,
        cnn_sim_raw_s=result.cnn_sim_raw_s,
        cnn_sim_amb_m=result.cnn_sim_amb_m,
        cnn_sim_amb_s=result.cnn_sim_amb_s,
        cnn_l1_raw_m=result.cnn_l1_raw_m,
        cnn_l1_raw_s=result.cnn_l1_raw_s,
        cnn_l1_amb_m=result.cnn_l1_amb_m,
        cnn_l1_amb_s=result.cnn_l1_amb_s,
        experiment_name=result.experiment_name,
    )
    return out.resolve()


def load_cnn_sweep(path: str | Path) -> CnnSweepResult:
    z = np.load(path, allow_pickle=False)
    name = str(z["experiment_name"]) if "experiment_name" in z else ""
    return CnnSweepResult(
        snr_sweep_db=z["snr_sweep_db"],
        cnn_sim_raw_m=z["cnn_sim_raw_m"],
        cnn_sim_raw_s=z["cnn_sim_raw_s"],
        cnn_sim_amb_m=z["cnn_sim_amb_m"],
        cnn_sim_amb_s=z["cnn_sim_amb_s"],
        cnn_l1_raw_m=z["cnn_l1_raw_m"],
        cnn_l1_raw_s=z["cnn_l1_raw_s"],
        cnn_l1_amb_m=z["cnn_l1_amb_m"],
        cnn_l1_amb_s=z["cnn_l1_amb_s"],
        experiment_name=name,
    )


def plot_metric_curves(
    x,
    series: list[tuple[list | np.ndarray, list | np.ndarray, str, str]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    xscale: str | None = None,
) -> None:
    plt.figure(figsize=(7, 4))
    for mean, std, fmt, label in series:
        plt.errorbar(x, mean, yerr=std, fmt=fmt, label=label, capsize=4)
    if xscale:
        plt.xscale(xscale)
        plt.grid(True, which="both", alpha=0.3)
    else:
        plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_training_history(
    train_losses: list[float],
    val_l1_pulses: list[float],
    *,
    val_snr_db: float,
    title: str = "Training vs. validation loss",
) -> None:
    epochs_axis = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs_axis, train_losses, "-o", markersize=3, label="train (random SNR)")
    plt.plot(
        epochs_axis,
        val_l1_pulses,
        "-o",
        color="tab:red",
        markersize=3,
        label=f"val @ {val_snr_db:.1f} dB",
    )
    plt.xlabel("epoch")
    plt.ylabel("L1 (sum over 2N, mean over pulses)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_l1_snr_cnn_vs_pcgpa(
    snr_sweep_db: np.ndarray,
    cnn_results: list[CnnSweepResult],
    *,
    pcgpa_l1_m: list | np.ndarray,
    pcgpa_l1_s: list | np.ndarray,
    pcgpa_label: str,
    exclude_experiments: tuple[str, ...] | list[str] = (),
    extra_l1_series: list[tuple[list | np.ndarray, list | np.ndarray, str, str]] | None = None,
    use_best_l1_amb: bool = True,
    include_vs_n: bool = True,
) -> None:
    """L1 vs SNR (and vs N): CNN curves + optional extras + PCGPA."""
    skip = set(exclude_experiments)
    filtered = [r for r in cnn_results if r.experiment_name not in skip]
    if not filtered:
        raise ValueError("no CNN results left after exclude_experiments")

    fmts = ["-o", "--^", "-s", "-D", "-v", "-P", "-X"]
    attr_m = "cnn_l1_amb_m" if use_best_l1_amb else "cnn_l1_raw_m"
    attr_s = "cnn_l1_amb_s" if use_best_l1_amb else "cnn_l1_raw_s"
    suffix = "best L1 amb." if use_best_l1_amb else "raw"
    ylabel = "L1 (sum over 2N, mean over pulses)"
    series = [
        (
            getattr(res, attr_m),
            getattr(res, attr_s),
            fmts[i % len(fmts)],
            res.experiment_name or f"CNN {i + 1}",
        )
        for i, res in enumerate(filtered)
    ]
    if extra_l1_series:
        series.extend(extra_l1_series)
    series.append((pcgpa_l1_m, pcgpa_l1_s, "--s", pcgpa_label))

    excl_note = f" (excl. {', '.join(sorted(skip))})" if skip else ""
    plot_metric_curves(
        snr_sweep_db,
        series,
        xlabel="trace SNR (dB)",
        ylabel=ylabel,
        title=f"L1 vs trace SNR — CNN ({suffix}) + PCGPA{excl_note}",
    )
    if include_vs_n:
        n_equiv = np.array([snr_db_to_equivalent_n_pulses(float(s)) for s in snr_sweep_db])
        plot_metric_curves(
            n_equiv,
            series,
            xlabel="equivalent pulse count N (log scale)",
            ylabel=ylabel,
            title=f"L1 vs N — CNN ({suffix}) + PCGPA{excl_note}",
            xscale="log",
        )


def plot_sim_snr_cnn_vs_pcgpa(
    snr_sweep_db: np.ndarray,
    cnn_results: list[CnnSweepResult],
    *,
    pcgpa_sim_m: list | np.ndarray,
    pcgpa_sim_s: list | np.ndarray,
    pcgpa_label: str,
    exclude_experiments: tuple[str, ...] | list[str] = (),
    extra_sim_series: list[tuple[list | np.ndarray, list | np.ndarray, str, str]] | None = None,
    use_best_amb: bool = True,
    include_vs_n: bool = True,
) -> None:
    """SIMILARITY_ERROR vs SNR (and vs N): CNN curves + optional extras + PCGPA."""
    skip = set(exclude_experiments)
    filtered = [r for r in cnn_results if r.experiment_name not in skip]
    if not filtered:
        raise ValueError("no CNN results left after exclude_experiments")

    fmts = ["-o", "--^", "-s", "-D", "-v", "-P", "-X"]
    attr_m = "cnn_sim_amb_m" if use_best_amb else "cnn_sim_raw_m"
    attr_s = "cnn_sim_amb_s" if use_best_amb else "cnn_sim_raw_s"
    suffix = "best amb." if use_best_amb else "raw"
    ylabel = "mean SIMILARITY_ERROR (1 − cos δE)"
    series = [
        (
            getattr(res, attr_m),
            getattr(res, attr_s),
            fmts[i % len(fmts)],
            res.experiment_name or f"CNN {i + 1}",
        )
        for i, res in enumerate(filtered)
    ]
    if extra_sim_series:
        series.extend(extra_sim_series)
    series.append((pcgpa_sim_m, pcgpa_sim_s, "--s", pcgpa_label))

    excl_note = f" (excl. {', '.join(sorted(skip))})" if skip else ""
    plot_metric_curves(
        snr_sweep_db,
        series,
        xlabel="trace SNR (dB)",
        ylabel=ylabel,
        title=f"SIMILARITY_ERROR vs trace SNR — CNN ({suffix}) + PCGPA{excl_note}",
    )
    if include_vs_n:
        n_equiv = np.array([snr_db_to_equivalent_n_pulses(float(s)) for s in snr_sweep_db])
        plot_metric_curves(
            n_equiv,
            series,
            xlabel="equivalent pulse count N (log scale)",
            ylabel=ylabel,
            title=f"SIMILARITY_ERROR vs N — CNN ({suffix}) + PCGPA{excl_note}",
            xscale="log",
        )


def plot_cnn_snr_compare(
    cnn_results: list[CnnSweepResult],
    *,
    use_best_l1_amb: bool = True,
    include_vs_n: bool = True,
    exclude_experiments: tuple[str, ...] | list[str] = (),
) -> None:
    """Overlay CNN-only SNR sweeps (no PCGPA). Default: L1 with best ambiguity."""
    skip = set(exclude_experiments)
    filtered = [r for r in cnn_results if r.experiment_name not in skip]
    if not filtered:
        raise ValueError("no CNN results left after exclude_experiments")
    snr = filtered[0].snr_sweep_db
    fmts = ["-o", "--^", "-s", "-D", "-v", "-P", "-X"]
    attr_m = "cnn_l1_amb_m" if use_best_l1_amb else "cnn_l1_raw_m"
    attr_s = "cnn_l1_amb_s" if use_best_l1_amb else "cnn_l1_raw_s"
    suffix = "best L1 amb." if use_best_l1_amb else "raw"
    ylabel = "L1 (sum over 2N, mean over pulses)"
    series = [
        (
            getattr(res, attr_m),
            getattr(res, attr_s),
            fmts[i % len(fmts)],
            res.experiment_name or f"CNN {i + 1}",
        )
        for i, res in enumerate(filtered)
    ]
    excl_note = f" (excl. {', '.join(sorted(skip))})" if skip else ""
    plot_metric_curves(
        snr,
        series,
        xlabel="trace SNR (dB)",
        ylabel=ylabel,
        title=f"L1 vs trace SNR — CNN ({suffix}){excl_note}",
    )
    if include_vs_n:
        n_equiv = np.array([snr_db_to_equivalent_n_pulses(float(s)) for s in snr])
        plot_metric_curves(
            n_equiv,
            series,
            xlabel="equivalent pulse count N (log scale)",
            ylabel=ylabel,
            title=f"L1 vs N — CNN ({suffix}){excl_note}",
            xscale="log",
        )


def plot_standard_cnn_vs_pcgpa_suite(
    snr_sweep_db: np.ndarray,
    cnn_results: list[CnnSweepResult],
    *,
    pcgpa_sim_m: list | np.ndarray,
    pcgpa_sim_s: list | np.ndarray,
    pcgpa_l1_m: list | np.ndarray,
    pcgpa_l1_s: list | np.ndarray,
    pcgpa_label: str,
) -> None:
    """Eight comparison plots: sim/L1 vs SNR and vs N, raw/amb CNN curves + PCGPA."""
    fmts = ["-o", "--^", "-s", "-D", "-v"]
    sim_ylabel = "mean SIMILARITY_ERROR (1 − cos δE)"
    n_equiv = np.array([snr_db_to_equivalent_n_pulses(float(s)) for s in snr_sweep_db])

    def _cnn_series(attr_m: str, attr_s: str, suffix: str):
        out = []
        for i, res in enumerate(cnn_results):
            fmt = fmts[i % len(fmts)]
            label = f"{res.experiment_name or f'CNN {i+1}'} ({suffix})"
            out.append((getattr(res, attr_m), getattr(res, attr_s), fmt, label))
        return out

    pcgpa_sim = (pcgpa_sim_m, pcgpa_sim_s, "--s", pcgpa_label)
    pcgpa_l1 = (pcgpa_l1_m, pcgpa_l1_s, "--s", pcgpa_label)

    plot_metric_curves(
        snr_sweep_db,
        _cnn_series("cnn_sim_raw_m", "cnn_sim_raw_s", "raw") + [pcgpa_sim],
        xlabel="trace SNR (dB)",
        ylabel=sim_ylabel,
        title="SIMILARITY_ERROR vs trace SNR — CNN raw + PCGPA",
    )
    plot_metric_curves(
        snr_sweep_db,
        _cnn_series("cnn_sim_amb_m", "cnn_sim_amb_s", "best amb.") + [pcgpa_sim],
        xlabel="trace SNR (dB)",
        ylabel=sim_ylabel,
        title="SIMILARITY_ERROR vs trace SNR — CNN best amb. + PCGPA",
    )
    plot_metric_curves(
        snr_sweep_db,
        _cnn_series("cnn_l1_raw_m", "cnn_l1_raw_s", "raw") + [pcgpa_l1],
        xlabel="trace SNR (dB)",
        ylabel="L1 (sum over 2N, mean over pulses)",
        title="L1 vs trace SNR — CNN raw + PCGPA",
    )
    plot_metric_curves(
        snr_sweep_db,
        _cnn_series("cnn_l1_amb_m", "cnn_l1_amb_s", "best L1 amb.") + [pcgpa_l1],
        xlabel="trace SNR (dB)",
        ylabel="L1 (sum over 2N, mean over pulses)",
        title="L1 vs trace SNR — CNN best amb. + PCGPA",
    )
    plot_metric_curves(
        n_equiv,
        _cnn_series("cnn_sim_raw_m", "cnn_sim_raw_s", "raw") + [pcgpa_sim],
        xlabel="equivalent pulse count N (log scale)",
        ylabel=sim_ylabel,
        title="SIMILARITY_ERROR vs N — CNN raw + PCGPA",
        xscale="log",
    )
    plot_metric_curves(
        n_equiv,
        _cnn_series("cnn_sim_amb_m", "cnn_sim_amb_s", "best amb.") + [pcgpa_sim],
        xlabel="equivalent pulse count N (log scale)",
        ylabel=sim_ylabel,
        title="SIMILARITY_ERROR vs N — CNN best amb. + PCGPA",
        xscale="log",
    )
    plot_metric_curves(
        n_equiv,
        _cnn_series("cnn_l1_raw_m", "cnn_l1_raw_s", "raw") + [pcgpa_l1],
        xlabel="equivalent pulse count N (log scale)",
        ylabel="L1 (sum over 2N, mean over pulses)",
        title="L1 vs N — CNN raw + PCGPA",
        xscale="log",
    )
    plot_metric_curves(
        n_equiv,
        _cnn_series("cnn_l1_amb_m", "cnn_l1_amb_s", "best L1 amb.") + [pcgpa_l1],
        xlabel="equivalent pulse count N (log scale)",
        ylabel="L1 (sum over 2N, mean over pulses)",
        title="L1 vs N — CNN best amb. + PCGPA",
        xscale="log",
    )
