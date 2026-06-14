"""Supervised training for TraceToPulseCNN with checkpoint I/O."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from frog_reconstruction_model import (
    MODEL_REGISTRY,
    TraceToPulseCNN,
    TraceToPulseWithSnrOutput,
    extract_pulse_prediction,
)
from pulse_metrics import pulse_packed_l1_loss_torch, snr_db_l1_loss_torch
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
    snr_loss_weight: float = 0.0
    seed: int = 0
    checkpoint_path: str = "checkpoints/baseline_2k.pt"
    experiment_name: str = "baseline_2k"
    model_name: str = "cnn"
    device: str | None = None

    def resolve_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainHistory:
    train_losses: list[float] = field(default_factory=list)
    val_l1_pulses: list[float] = field(default_factory=list)


@dataclass
class EarlyStopTrainResult:
    history: TrainHistory
    best_epoch: int
    best_val_l1: float
    stopped_epoch: int


class SnrRangeFn(Protocol):
    def __call__(self, epoch: int, max_epochs: int) -> tuple[float, float]: ...


def snr_curriculum_db_range(epoch: int, max_epochs: int) -> tuple[float, float]:
    """Three-phase SNR curriculum (0-based epoch index).

    Phase 1 — high SNR only; phase 2 — mid; phase 3 — full training range.
    """
    if max_epochs <= 0:
        return (0.0, 30.0)
    t1 = max(1, max_epochs // 3)
    t2 = max(t1 + 1, (2 * max_epochs) // 3)
    if epoch < t1:
        return (20.0, 30.0)
    if epoch < t2:
        return (10.0, 25.0)
    return (0.0, 30.0)


def build_model(
    n: int,
    device: torch.device,
    model_name: str = "cnn",
) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model_name={model_name!r}; choose from {list(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[model_name](out_dim=2 * n).to(device)
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
    snr_loss_weight: float = 0.0,
    verbose: bool = True,
) -> TrainHistory:
    criterion = pulse_packed_l1_loss_torch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = TrainHistory()

    device = next(model.parameters()).device

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for I_clean, E_true in train_loader:
            I_clean = I_clean.to(device)
            E_true = E_true.to(device)
            snr = float(np.random.uniform(train_snr_db_range[0], train_snr_db_range[1]))
            I_noisy = add_noise_fn(I_clean, snr)
            optimizer.zero_grad(set_to_none=True)
            out = model(I_noisy.unsqueeze(1))
            E_pred = extract_pulse_prediction(out)
            loss = criterion(E_pred, E_true)
            if snr_loss_weight > 0.0 and isinstance(out, TraceToPulseWithSnrOutput):
                loss = loss + float(snr_loss_weight) * snr_db_l1_loss_torch(out.snr_db, snr)
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
                I_clean = I_clean.to(device)
                E_true = E_true.to(device)
                I_noisy = add_noise_fn(I_clean, val_snr_db)
                E_pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
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


def train_trace_to_pulse_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    max_epochs: int,
    patience: int,
    lr: float,
    train_snr_db_range: tuple[float, float],
    val_snr_db: float,
    add_noise_fn: Callable[[torch.Tensor, float], torch.Tensor] = add_trace_noise_awgn,
    snr_range_fn: SnrRangeFn | None = None,
    snr_loss_weight: float = 0.0,
    verbose: bool = True,
) -> EarlyStopTrainResult:
    """Train with validation L1; restore best weights; stop after ``patience`` epochs without improvement."""
    criterion = pulse_packed_l1_loss_torch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = TrainHistory()
    device = next(model.parameters()).device

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    stopped_epoch = 0

    for epoch in range(max_epochs):
        epoch_snr_range = (
            snr_range_fn(epoch, max_epochs)
            if snr_range_fn is not None
            else train_snr_db_range
        )
        model.train()
        running = 0.0
        n_seen = 0
        for I_clean, E_true in train_loader:
            I_clean = I_clean.to(device)
            E_true = E_true.to(device)
            snr = float(np.random.uniform(epoch_snr_range[0], epoch_snr_range[1]))
            I_noisy = add_noise_fn(I_clean, snr)
            optimizer.zero_grad(set_to_none=True)
            out = model(I_noisy.unsqueeze(1))
            E_pred = extract_pulse_prediction(out)
            loss = criterion(E_pred, E_true)
            if snr_loss_weight > 0.0 and isinstance(out, TraceToPulseWithSnrOutput):
                loss = loss + float(snr_loss_weight) * snr_db_l1_loss_torch(out.snr_db, snr)
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
                I_clean = I_clean.to(device)
                E_true = E_true.to(device)
                I_noisy = add_noise_fn(I_clean, val_snr_db)
                E_pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
                vloss = criterion(E_pred, E_true)
                b = I_clean.shape[0]
                vsum += vloss.item() * b
                vcount += b
        val_l1 = vsum / max(vcount, 1)
        history.val_l1_pulses.append(val_l1)

        if val_l1 < best_val:
            best_val = float(val_l1)
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            snr_note = (
                f"  train_SNR=[{epoch_snr_range[0]:.0f},{epoch_snr_range[1]:.0f}]dB"
                if snr_range_fn is not None
                else ""
            )
            print(
                f"epoch {epoch + 1:03d}/{max_epochs}  "
                f"train_L1={history.train_losses[-1]:.5f}  "
                f"val_L1@{val_snr_db:.1f}dB={val_l1:.5f}"
                + snr_note
                + (f"  *best" if epoch + 1 == best_epoch else "")
            )

        stopped_epoch = epoch + 1
        if epochs_no_improve >= patience:
            if verbose:
                print(f"early stop @ epoch {stopped_epoch} (best epoch {best_epoch}, val_L1={best_val:.5f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return EarlyStopTrainResult(
        history=history,
        best_epoch=best_epoch,
        best_val_l1=best_val,
        stopped_epoch=stopped_epoch,
    )


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: TrainConfig,
    history: TrainHistory | None = None,
    extra: dict | None = None,
) -> Path:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "N": config.n,
        "model_name": config.model_name,
        "train_snr_db_range": config.train_snr_db_range,
        "experiment_name": config.experiment_name,
        "train_config": asdict(config),
    }
    if history is not None:
        payload["train_losses"] = history.train_losses
        payload["val_l1_pulses"] = history.val_l1_pulses
    if extra:
        payload.update(extra)
    torch.save(payload, ckpt_path)
    return ckpt_path.resolve()


def load_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict]:
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    n = int(ckpt["N"])
    model_name = ckpt.get("model_name", "cnn")
    if "train_config" in ckpt and "model_name" in ckpt["train_config"]:
        model_name = ckpt["train_config"].get("model_name", model_name)
    model = build_model(n, device, model_name=model_name)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt
