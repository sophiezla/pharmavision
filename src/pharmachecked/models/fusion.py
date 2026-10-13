"""Branch fusion (PLAN 3.5): gated (softmax gate over branches) or concatenation."""

from __future__ import annotations

import torch
from torch import nn


class Fusion(nn.Module):
    """Project each branch to ``dim``; gated: sum_i gate_i * p_i, gate = softmax(MLP([p_1..p_k])).

    ``concat`` mode concatenates the projections and maps them back to ``dim`` (ablation); its
    gate weights are reported as uniform.
    """

    def __init__(self, in_dims: list[int], dim: int = 256, mode: str = "gated") -> None:
        super().__init__()
        if mode not in ("gated", "concat"):
            raise ValueError(f"unknown fusion mode {mode!r}")
        self.mode = mode
        self.proj = nn.ModuleList(nn.Linear(d, dim) for d in in_dims)
        k = len(in_dims)
        if mode == "gated":
            self.gate = nn.Sequential(nn.Linear(k * dim, dim), nn.ReLU(), nn.Linear(dim, k))
        else:
            self.mix = nn.Linear(k * dim, dim)

    def forward(self, embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(fused (B, dim), gate weights (B, k))``; gate rows sum to 1."""
        projected = [p(e) for p, e in zip(self.proj, embeddings, strict=True)]
        stacked = torch.stack(projected, dim=1)  # (B, k, dim)
        flat = stacked.flatten(1)
        if self.mode == "gated":
            gate = torch.softmax(self.gate(flat), dim=1)
            return (gate.unsqueeze(-1) * stacked).sum(dim=1), gate
        uniform = torch.full(stacked.shape[:2], 1.0 / stacked.shape[1], device=flat.device)
        return self.mix(flat), uniform
