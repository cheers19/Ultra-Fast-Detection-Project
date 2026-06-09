"""
Reconstruction network: FROG-like intensity trace -> packed real/imag pulse.

Input expected by ``TraceToPulseCNN.forward``:
    ``x`` of shape ``[B, 1, H, W]`` (single-channel intensity), e.g. ``H=W=N`` for SHG-FROG.

Output:
    ``[B, 2 * N]`` with ``[..., :N] = Re(E)``, ``[..., N:] = Im(E)`` (same convention as ``FROGNet``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TraceToPulseCNN(nn.Module):
    """Baseline CNN (3 conv layers, 32 channels, FC 512)."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TraceToPulseCNNLarge(nn.Module):
    """Wider/deeper CNN: 64 channels, 4 conv layers, FC 1024."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "cnn": TraceToPulseCNN,
    "cnn_large": TraceToPulseCNNLarge,
}
