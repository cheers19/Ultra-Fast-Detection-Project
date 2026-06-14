"""
Reconstruction network: FROG-like intensity trace -> packed real/imag pulse.

Input expected by ``TraceToPulseCNN.forward``:
    ``x`` of shape ``[B, 1, H, W]`` (single-channel intensity), e.g. ``H=W=N`` for SHG-FROG.

Output:
    ``[B, 2 * N]`` with ``[..., :N] = Re(E)``, ``[..., N:] = Im(E)`` (same convention as ``FROGNet``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TraceToPulseWithSnrOutput:
    """Dual-head model output: packed pulse and trace SNR estimate (dB)."""

    pulse: torch.Tensor
    snr_db: torch.Tensor


def extract_pulse_prediction(
    out: torch.Tensor | TraceToPulseWithSnrOutput,
) -> torch.Tensor:
    """Return packed pulse tensor whether or not the model predicts SNR."""
    if isinstance(out, TraceToPulseWithSnrOutput):
        return out.pulse
    return out


class _SnrHeadFromMap(nn.Module):
    """Global-average pool on a feature map → scalar SNR (dB)."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)
# DeepFROG Multires branch kernel sizes (Szegedy-style multi-scale conv).
_MULTIRES_KERNELS = (11, 7, 5, 3)


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


class _MultiresSameSize(nn.Module):
    """Four parallel conv branches (kernels 11/7/5/3), same spatial size, concat channels."""

    def __init__(self, in_ch: int, filters_per_branch: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(in_ch, filters_per_branch, kernel_size=k, stride=1, padding=k // 2)
                for k in _MULTIRES_KERNELS
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class _DownsampleConv(nn.Module):
    """Stride-2 conv: halve spatial size, double channel count (DeepFROG Multires block)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _UNetConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UNetDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _UNetConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _UNetUp(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = _UNetConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = nn.functional.pad(
            x,
            [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2],
        )
        return self.conv(torch.cat([skip, x], dim=1))


class TraceToPulseUNet(nn.Module):
    """
    Compact U-Net for trace -> pulse: encoder-decoder with skip connections,
    then global average pool + FC head to packed Re/Im pulse.
    """

    def __init__(self, out_dim: int, base_ch: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.enc1 = _UNetConvBlock(1, c1)
        self.enc2 = _UNetDown(c1, c2)
        self.enc3 = _UNetDown(c2, c3)
        self.bottleneck = _UNetDown(c3, c4)
        self.up3 = _UNetUp(c4, c3, c3)
        self.up2 = _UNetUp(c3, c2, c2)
        self.up1 = _UNetUp(c2, c1, c1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)


class TraceToPulseMultires(nn.Module):
    """
    DeepFROG-style Multires CNN (2018): 3×(Multires + downsample) + FC 512.

    Six conv layers total; ReLU after each. ``filters_per_branch`` per stage: 8, 16, 32
    for N=64 traces (spatial 64→32→16→8 before flatten).
    """

    def __init__(self, out_dim: int, filters_per_branch: tuple[int, ...] = (8, 16, 32)) -> None:
        super().__init__()
        if len(filters_per_branch) != 3:
            raise ValueError("filters_per_branch must have length 3 for three Multires stages")

        layers: list[nn.Module] = []
        in_ch = 1
        for fpb in filters_per_branch:
            multires_out = 4 * fpb
            layers.append(_MultiresSameSize(in_ch, fpb))
            layers.append(nn.ReLU(inplace=True))
            layers.append(_DownsampleConv(multires_out, multires_out * 2))
            layers.append(nn.ReLU(inplace=True))
            in_ch = multires_out * 2

        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class TraceToPulseMultiresWithSnr(nn.Module):
    """Multires CNN with an auxiliary SNR (dB) head from shared features."""

    def __init__(self, out_dim: int, filters_per_branch: tuple[int, ...] = (8, 16, 32)) -> None:
        super().__init__()
        if len(filters_per_branch) != 3:
            raise ValueError("filters_per_branch must have length 3 for three Multires stages")

        layers: list[nn.Module] = []
        in_ch = 1
        for fpb in filters_per_branch:
            multires_out = 4 * fpb
            layers.append(_MultiresSameSize(in_ch, fpb))
            layers.append(nn.ReLU(inplace=True))
            layers.append(_DownsampleConv(multires_out, multires_out * 2))
            layers.append(nn.ReLU(inplace=True))
            in_ch = multires_out * 2

        self.features = nn.Sequential(*layers)
        self.snr_head = _SnrHeadFromMap(in_ch)
        self.pulse_head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> TraceToPulseWithSnrOutput:
        feat = self.features(x)
        return TraceToPulseWithSnrOutput(
            pulse=self.pulse_head(feat),
            snr_db=self.snr_head(feat),
        )


class TraceToPulseUNetWithSnr(nn.Module):
    """U-Net with an auxiliary SNR (dB) head from the decoder feature map."""

    def __init__(self, out_dim: int, base_ch: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.enc1 = _UNetConvBlock(1, c1)
        self.enc2 = _UNetDown(c1, c2)
        self.enc3 = _UNetDown(c2, c3)
        self.bottleneck = _UNetDown(c3, c4)
        self.up3 = _UNetUp(c4, c3, c3)
        self.up2 = _UNetUp(c3, c2, c2)
        self.up1 = _UNetUp(c2, c1, c1)
        self.snr_head = _SnrHeadFromMap(c1)
        self.pulse_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> TraceToPulseWithSnrOutput:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return TraceToPulseWithSnrOutput(
            pulse=self.pulse_head(d1),
            snr_db=self.snr_head(d1),
        )


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "cnn": TraceToPulseCNN,
    "cnn_large": TraceToPulseCNNLarge,
    "multires": TraceToPulseMultires,
    "multires_snr": TraceToPulseMultiresWithSnr,
    "unet": TraceToPulseUNet,
    "unet_snr": TraceToPulseUNetWithSnr,
}
