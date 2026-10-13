"""Frequency branch (PLAN 3.5, ADR-017): FFT magnitude + 8x8 block-DCT energy -> small CNN."""

from __future__ import annotations

import math

import torch
from torch import nn

BLOCK = 8
GRID = 128  # both maps are resized to GRID x GRID before the CNN


def dct_matrix(n: int = BLOCK) -> torch.Tensor:
    """Orthonormal DCT-II matrix ``(n, n)``."""
    k = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(n, dtype=torch.float32)[None, :]
    m = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    m[0] /= math.sqrt(2.0)
    return m


def frequency_maps(gray: torch.Tensor) -> torch.Tensor:
    """Two-channel map ``(B, 2, GRID, GRID)`` from grayscale images ``(B, H, W)``.

    Channel 0: log(1 + |FFT|), centred (fftshift). Channel 1: per 8x8 block, the energy of the
    DCT-II AC coefficients (sum of squares, log1p), i.e. block-level high-frequency content
    sensitive to JPEG blocking. Each channel is standardized per image.
    """
    b, h, w = gray.shape
    spectrum = torch.log1p(torch.fft.fftshift(torch.fft.fft2(gray.float()), dim=(-2, -1)).abs())
    hb, wb = h // BLOCK * BLOCK, w // BLOCK * BLOCK
    blocks = gray[:, :hb, :wb].float().reshape(b, hb // BLOCK, BLOCK, wb // BLOCK, BLOCK)
    d = dct_matrix().to(gray.device)
    coeffs = torch.einsum("ki,bxiyj,lj->bxkyl", d, blocks, d)
    energy = coeffs.pow(2).sum(dim=(2, 4)) - coeffs[:, :, 0, :, 0].pow(2)  # drop the DC term
    maps = [
        nn.functional.interpolate(
            m[:, None], size=(GRID, GRID), mode="bilinear", align_corners=False
        )
        for m in (spectrum, torch.log1p(energy))
    ]
    out = torch.cat(maps, dim=1)
    mean = out.mean(dim=(2, 3), keepdim=True)
    std = out.std(dim=(2, 3), keepdim=True) + 1e-6
    return (out - mean) / std


class FrequencyBranch(nn.Module):
    """4-block CNN (32-64-128-256, BN, ReLU, max-pool) + global average pooling -> 256-d."""

    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 2
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_dim = c_in

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Embedding ``(B, 256)`` from normalized RGB images ``(B, 3, H, W)``."""
        return self.net(frequency_maps(images.mean(dim=1))).mean(dim=(2, 3))
