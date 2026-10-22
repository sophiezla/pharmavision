"""Physically plausible attacks for E15 (PLAN 7.8), optimized by gradient descent on the
counterfeit logit of the detector (white box).

* ``color``: per-image global brightness, contrast, saturation (factors in [0.8, 1.2]) and hue
  rotation (+-0.05 turn, YIQ rotation), Adam (lr 0.1), 50 steps; parameters bounded by tanh.
* ``patch``: a square patch of 5% of the image area at a fixed location (top-left corner),
  pixel values free in [0, 1] (sigmoid), Adam (lr 0.1), 100 steps, random initialization.
"""

from __future__ import annotations

import math

import torch

from pharmachecked.adversarial.attacks import Normalized

COLOR_STEPS = 50
PATCH_STEPS = 100
PATCH_FRACTION = 0.05
BOUNDS = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05}

_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])


def _gray(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def apply_color(x01: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Differentiable colour transform; ``params`` ``(B, 4)`` already in their bounded ranges:
    brightness, contrast, saturation factors and hue shift (turns)."""
    b, c, s, h = (params[:, k].view(-1, 1, 1, 1) for k in range(4))
    x = x01 * b
    mean = _gray(x).mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * c + mean
    g = _gray(x)
    x = g + (x - g) * s
    m = _RGB2YIQ.to(x.device)
    yiq = torch.einsum("ij,bjhw->bihw", m, x)
    theta = 2 * math.pi * h[:, 0, 0, 0]
    cos, sin = torch.cos(theta).view(-1, 1, 1), torch.sin(theta).view(-1, 1, 1)
    i, q = yiq[:, 1], yiq[:, 2]
    yiq = torch.stack([yiq[:, 0], i * cos - q * sin, i * sin + q * cos], dim=1)
    rgb = torch.einsum("ij,bjhw->bihw", torch.linalg.inv(m), yiq)
    return rgb.clamp(0, 1)


def bounded(raw: torch.Tensor) -> torch.Tensor:
    """Map unconstrained ``(B, 4)`` parameters into their ranges."""
    scale = torch.tensor(list(BOUNDS.values()), device=raw.device)
    offset = torch.tensor([1.0, 1.0, 1.0, 0.0], device=raw.device)
    return offset + scale * torch.tanh(raw)


def color_attack(wrapped: Normalized, x01: torch.Tensor, steps: int = COLOR_STEPS) -> torch.Tensor:
    """Adversarial global colour change per image (minimizes the counterfeit logit)."""
    raw = torch.zeros(len(x01), 4, device=x01.device, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        wrapped.logit(apply_color(x01, bounded(raw))).sum().backward()
        opt.step()
    with torch.no_grad():
        return apply_color(x01, bounded(raw))


def patch_side(height: int, width: int, fraction: float = PATCH_FRACTION) -> int:
    """Side of the square patch covering at most ``fraction`` of the image."""
    return math.floor(math.sqrt(fraction * height * width))


def patch_attack(
    wrapped: Normalized, x01: torch.Tensor, steps: int = PATCH_STEPS, seed: int = 0
) -> torch.Tensor:
    """Adversarial square patch at the top-left corner, optimized per image."""
    side = patch_side(*x01.shape[-2:])
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(len(x01), 3, side, side, generator=g).to(x01.device).requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=0.1)

    def pasted() -> torch.Tensor:
        out = x01.clone()
        out[:, :, :side, :side] = torch.sigmoid(raw)
        return out

    for _ in range(steps):
        opt.zero_grad()
        wrapped.logit(pasted()).sum().backward()
        opt.step()
    with torch.no_grad():
        return pasted()
