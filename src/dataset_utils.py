"""Simulated FROG trace datasets and DataLoaders for CNN training/eval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_generation import (
    PLANCK_CONSTANT_FS_EV,
    FilteredC1PulseConfig,
    SmoothedPhaseStochasticPulseConfig,
    StochasticPulseConfig,
    filtered_c1_pulse_config,
    generate_pulses_filtered_c1,
    generate_pulses_gaussian,
    generate_pulses_stochastic,
    generate_pulses_stochastic_smoothed_phase,
    stochastic_pulse_config_notebook_modified_c1,
)
from frognet import FROGNet
from frognet_padded import FROGNetPadded
from spectral_grid import (
    build_spectral_plot_grid,
    compute_de_new_target,
    compute_sigma_t_center,
)


def pack_pulses_complex(pulses_c: np.ndarray) -> torch.Tensor:
    """[B, N] complex -> [B, 2N] float32 (Re then Im)."""
    r = np.real(pulses_c).astype(np.float32)
    im = np.imag(pulses_c).astype(np.float32)
    return torch.from_numpy(np.concatenate([r, im], axis=-1))


@dataclass
class PulseGridConfig:
    n: int = 64
    t_total: float = 250.0
    sigma_omega: float | None = None
    sigma_gauss: float = 1.6
    phase_scale: float = np.pi

    @property
    def dt(self) -> float:
        return self.t_total / self.n

    @property
    def resolved_sigma_omega(self) -> float:
        if self.sigma_omega is not None:
            return self.sigma_omega
        return 0.05 * (2 * np.pi / self.dt)


@dataclass
class FrogDatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    t_vec: np.ndarray
    w_vec: np.ndarray
    grid: PulseGridConfig


def _frog_traces_batched(
    frog: torch.nn.Module,
    e_packed: torch.Tensor,
    device: torch.device,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Forward FROG model in chunks; return CPU float tensors."""
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, e_packed.shape[0], chunk_size):
            batch = e_packed[start : start + chunk_size].to(device)
            out.append(frog(batch).cpu())
    return torch.cat(out, dim=0)


def notebook_c1_spectral_fft_params(
    *,
    n: int = 64,
    t_total_fs: float = 53.0,
    n_spikes: int = 300,
    pulse_temporal_fraction: float = 0.62,
    fraction_from_nyquist: float = 0.673,
    n_spectral_points: int = 64,
) -> dict:
    """Match ``stochastic_pulses_generator_NB.ipynb`` padded-FFT spectral settings."""
    dt = t_total_fs / max(n - 1, 1)
    sigma_t_c = compute_sigma_t_center(t_total_fs, n_spikes, pulse_temporal_fraction)
    de_target = compute_de_new_target(
        PLANCK_CONSTANT_FS_EV, sigma_t_c, fraction_from_nyquist
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
    }


def build_notebook_c1_padded_frog(
    device: torch.device,
    *,
    n: int = 64,
    spectral: dict | None = None,
) -> FROGNetPadded:
    """``FROGNetPadded`` with notebook C1 spectral resolution (default ``N_FFT=128``)."""
    spec = spectral or notebook_c1_spectral_fft_params(n=n)
    frog = FROGNetPadded(
        num_delay_steps=n,
        n_fft=int(spec["n_fft"]),
        n_spectral_points=int(spec["n_spectral_points"]),
    ).to(device)
    frog.eval()
    return frog


def build_stochastic_padded_frog_dataloaders(
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    grid: StochasticPulseConfig | None = None,
    canonicalize_mode: str = "tstar",
    spectral: dict | None = None,
    pulse_temporal_fraction: float = 0.62,
    fraction_from_nyquist: float = 0.673,
) -> FrogDatasetBundle:
    """
    Modified-C1 pulses + padded FROG traces (generator-notebook protocol).

    Default grid: ``stochastic_pulse_config_notebook_modified_c1``;
    default canonicalization: peak-anchored ``tstar``.

    Pass ``spectral`` from ``notebook_c1_spectral_fft_params`` (or set
    ``pulse_temporal_fraction`` / ``fraction_from_nyquist``) so TRACE resolution
    matches ``stochastic_pulses_generator_NB.ipynb``.
    """
    grid = grid or stochastic_pulse_config_notebook_modified_c1()

    p_train_c, _, _, _ = generate_pulses_stochastic(
        n_pulses=n_train,
        config=grid,
        seed=seed,
        canonicalize_mode=canonicalize_mode,
    )
    p_val_c, _, _, _ = generate_pulses_stochastic(
        n_pulses=n_val,
        config=grid,
        seed=seed + 1,
        canonicalize_mode=canonicalize_mode,
    )
    p_test_c, _, t_vec, w_vec = generate_pulses_stochastic(
        n_pulses=n_test,
        config=grid,
        seed=seed + 2,
        canonicalize_mode=canonicalize_mode,
    )

    E_train = pack_pulses_complex(p_train_c)
    E_val = pack_pulses_complex(p_val_c)
    E_test = pack_pulses_complex(p_test_c)

    if spectral is None:
        spectral = notebook_c1_spectral_fft_params(
            n=grid.n,
            t_total_fs=float(grid.t_total_fs),
            n_spikes=int(grid.n_spikes),
            pulse_temporal_fraction=float(pulse_temporal_fraction),
            fraction_from_nyquist=float(fraction_from_nyquist),
        )
    frog = build_notebook_c1_padded_frog(device, n=grid.n, spectral=spectral)
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

    pseudo_grid = PulseGridConfig(n=grid.n, t_total=grid.t_total_fs)
    return FrogDatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        t_vec=t_vec,
        w_vec=w_vec,
        grid=pseudo_grid,
    )


def build_frog_dataloaders(
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    grid: PulseGridConfig | None = None,
) -> FrogDatasetBundle:
    grid = grid or PulseGridConfig()
    dt = grid.dt
    sigma_omega = grid.resolved_sigma_omega

    p_train_c, _, _, _ = generate_pulses_gaussian(
        n_pulses=n_train,
        dt=dt,
        sigma_omega=sigma_omega,
        num_points=grid.n,
        sigma=grid.sigma_gauss,
        phase_scale=grid.phase_scale,
        seed=seed,
    )
    p_val_c, _, _, _ = generate_pulses_gaussian(
        n_pulses=n_val,
        dt=dt,
        sigma_omega=sigma_omega,
        num_points=grid.n,
        sigma=grid.sigma_gauss,
        phase_scale=grid.phase_scale,
        seed=seed + 1,
    )
    p_test_c, _, t_vec, w_vec = generate_pulses_gaussian(
        n_pulses=n_test,
        dt=dt,
        sigma_omega=sigma_omega,
        num_points=grid.n,
        sigma=grid.sigma_gauss,
        phase_scale=grid.phase_scale,
        seed=seed + 2,
    )

    E_train = pack_pulses_complex(p_train_c)
    E_val = pack_pulses_complex(p_val_c)
    E_test = pack_pulses_complex(p_test_c)

    frog = FROGNet(num_delay_steps=grid.n).to(device)
    frog.eval()
    I_train = _frog_traces_batched(frog, E_train, device)
    with torch.no_grad():
        I_val = frog(E_val.to(device)).cpu()
        I_test = frog(E_test.to(device)).cpu()

    # Small val/test stay on device for notebook inference; train stays on CPU.
    E_val = E_val.to(device)
    E_test = E_test.to(device)
    I_val = I_val.to(device)
    I_test = I_test.to(device)

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
    return FrogDatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        t_vec=t_vec,
        w_vec=w_vec,
        grid=grid,
    )


def build_filtered_c1_frog_dataloaders(
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    grid: FilteredC1PulseConfig | None = None,
    canonicalize_mode: str = "t0",
) -> FrogDatasetBundle:
    """FROG dataloaders for spectrally filtered C1 pulses (``c1_pulse_independent_NB``)."""
    grid = grid or filtered_c1_pulse_config(n=64)

    p_train_c, _, _, _ = generate_pulses_filtered_c1(
        n_pulses=n_train,
        config=grid,
        seed=seed,
        canonicalize_mode=canonicalize_mode,
    )
    p_val_c, _, _, _ = generate_pulses_filtered_c1(
        n_pulses=n_val,
        config=grid,
        seed=seed + 1,
        canonicalize_mode=canonicalize_mode,
    )
    p_test_c, _, t_vec, w_vec = generate_pulses_filtered_c1(
        n_pulses=n_test,
        config=grid,
        seed=seed + 2,
        canonicalize_mode=canonicalize_mode,
    )

    E_train = pack_pulses_complex(p_train_c)
    E_val = pack_pulses_complex(p_val_c)
    E_test = pack_pulses_complex(p_test_c)

    frog = FROGNet(num_delay_steps=grid.n).to(device)
    frog.eval()
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
    pseudo_grid = PulseGridConfig(n=grid.n, t_total=grid.t_total_fs)
    return FrogDatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        t_vec=t_vec,
        w_vec=w_vec,
        grid=pseudo_grid,
    )


def build_stochastic_frog_dataloaders(
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    grid: StochasticPulseConfig | None = None,
    canonicalize_mode: str = "t0",
) -> FrogDatasetBundle:
    """Same as ``build_frog_dataloaders`` but with SASE stochastic pulses (fs time grid)."""
    grid = grid or StochasticPulseConfig()

    p_train_c, _, _, _ = generate_pulses_stochastic(
        n_pulses=n_train,
        config=grid,
        seed=seed,
        canonicalize_mode=canonicalize_mode,
    )
    p_val_c, _, _, _ = generate_pulses_stochastic(
        n_pulses=n_val,
        config=grid,
        seed=seed + 1,
        canonicalize_mode=canonicalize_mode,
    )
    p_test_c, _, t_vec, w_vec = generate_pulses_stochastic(
        n_pulses=n_test,
        config=grid,
        seed=seed + 2,
        canonicalize_mode=canonicalize_mode,
    )

    E_train = pack_pulses_complex(p_train_c)
    E_val = pack_pulses_complex(p_val_c)
    E_test = pack_pulses_complex(p_test_c)

    frog = FROGNet(num_delay_steps=grid.n).to(device)
    frog.eval()
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

    pseudo_grid = PulseGridConfig(n=grid.n, t_total=grid.t_total_fs)
    return FrogDatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        t_vec=t_vec,
        w_vec=w_vec,
        grid=pseudo_grid,
    )


def build_smoothed_phase_stochastic_frog_dataloaders(
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    grid: SmoothedPhaseStochasticPulseConfig | None = None,
) -> FrogDatasetBundle:
    """Same as ``build_stochastic_frog_dataloaders`` but with smoothed phase-noise SASE pulses."""
    grid = grid or SmoothedPhaseStochasticPulseConfig()

    p_train_c, _, _, _ = generate_pulses_stochastic_smoothed_phase(
        n_pulses=n_train, config=grid, seed=seed
    )
    p_val_c, _, _, _ = generate_pulses_stochastic_smoothed_phase(
        n_pulses=n_val, config=grid, seed=seed + 1
    )
    p_test_c, _, t_vec, w_vec = generate_pulses_stochastic_smoothed_phase(
        n_pulses=n_test, config=grid, seed=seed + 2
    )

    E_train = pack_pulses_complex(p_train_c)
    E_val = pack_pulses_complex(p_val_c)
    E_test = pack_pulses_complex(p_test_c)

    frog = FROGNet(num_delay_steps=grid.n).to(device)
    frog.eval()
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

    pseudo_grid = PulseGridConfig(n=grid.n, t_total=grid.t_total_fs)
    return FrogDatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        t_vec=t_vec,
        w_vec=w_vec,
        grid=pseudo_grid,
    )
