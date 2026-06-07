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
    best_similarity_error_ambiguity,
    l1_packed_mae,
    l1_packed_per_pulse_torch,
    packed_batch_to_complex,
    similarity_error_numpy,
    snr_db_to_equivalent_n_pulses,
)
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
