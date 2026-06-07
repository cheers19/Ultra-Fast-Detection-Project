"""Simulated FROG trace datasets and DataLoaders for CNN training/eval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_generation import generate_pulses_gaussian
from frognet import FROGNet


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

    E_train = pack_pulses_complex(p_train_c).to(device)
    E_val = pack_pulses_complex(p_val_c).to(device)
    E_test = pack_pulses_complex(p_test_c).to(device)

    frog = FROGNet(num_delay_steps=grid.n).to(device)
    frog.eval()
    with torch.no_grad():
        I_train = frog(E_train)
        I_val = frog(E_val)
        I_test = frog(E_test)

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
