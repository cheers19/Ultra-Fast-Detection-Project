"""Supervised training for TraceToPulseCNN with checkpoint I/O."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from frog_reconstruction_model import TraceToPulseCNN
from pulse_metrics import pulse_packed_l1_loss_torch
from trace_noise import add_trace_noise_awgn


@dataclass
class TrainConfig:
    n: int = 64
    n_train: int = 2048
    n_val: int = 512
    n_test: int = 512
    batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-3
    train_snr_db_range: tuple[float, float] = (0.0, 30.0)
    val_snr_db: float = 15.0
    seed: int = 0
    checkpoint_path: str = "checkpoints/baseline_2k.pt"
    experiment_name: str = "baseline_2k"
    device: str | None = None

    def resolve_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainHistory:
    train_losses: list[float] = field(default_factory=list)
    val_l1_pulses: list[float] = field(default_factory=list)


def build_model(n: int, device: torch.device) -> TraceToPulseCNN:
    model = TraceToPulseCNN(out_dim=2 * n).to(device)
    # LazyLinear needs one forward pass to initialize weights.
    model(torch.zeros(1, 1, n, n, device=device))
    return model


def train_trace_to_pulse(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    train_snr_db_range: tuple[float, float],
    val_snr_db: float,
    add_noise_fn: Callable[[torch.Tensor, float], torch.Tensor] = add_trace_noise_awgn,
    verbose: bool = True,
) -> TrainHistory:
    criterion = pulse_packed_l1_loss_torch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = TrainHistory()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for I_clean, E_true in train_loader:
            snr = float(np.random.uniform(train_snr_db_range[0], train_snr_db_range[1]))
            I_noisy = add_noise_fn(I_clean, snr)
            optimizer.zero_grad(set_to_none=True)
            E_pred = model(I_noisy.unsqueeze(1))
            loss = criterion(E_pred, E_true)
            loss.backward()
            optimizer.step()

            b = I_clean.shape[0]
            running += loss.item() * b
            n_seen += b
        history.train_losses.append(running / max(n_seen, 1))

        model.eval()
        vsum, vcount = 0.0, 0
        with torch.no_grad():
            for I_clean, E_true in val_loader:
                I_noisy = add_noise_fn(I_clean, val_snr_db)
                E_pred = model(I_noisy.unsqueeze(1))
                vloss = criterion(E_pred, E_true)
                b = I_clean.shape[0]
                vsum += vloss.item() * b
                vcount += b
        history.val_l1_pulses.append(vsum / max(vcount, 1))

        if verbose:
            print(
                f"epoch {epoch + 1:03d}/{epochs}  "
                f"train_L1={history.train_losses[-1]:.5f}  "
                f"val_L1@{val_snr_db:.1f}dB={history.val_l1_pulses[-1]:.5f}"
            )

    return history


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: TrainConfig,
    history: TrainHistory | None = None,
) -> Path:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "N": config.n,
        "train_snr_db_range": config.train_snr_db_range,
        "experiment_name": config.experiment_name,
        "train_config": asdict(config),
    }
    if history is not None:
        payload["train_losses"] = history.train_losses
        payload["val_l1_pulses"] = history.val_l1_pulses
    torch.save(payload, ckpt_path)
    return ckpt_path.resolve()


def load_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> tuple[TraceToPulseCNN, dict]:
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    n = int(ckpt["N"])
    model = build_model(n, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt
