"""Gated-attention multiple-instance pooling (Ilse et al., 2018)."""

from __future__ import annotations

import torch
from torch import nn


class GatedAttentionMIL(nn.Module):
    """Pool a bag of instance embeddings ``(B, N, D)`` into ``(B, D)`` with attention weights.

    a_n = softmax_n( w^T (tanh(V h_n) * sigmoid(U h_n)) ); bag = sum_n a_n h_n.
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, instances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(bag embedding (B, D), attention (B, N))``; attention rows sum to 1."""
        scores = self.w(torch.tanh(self.v(instances)) * torch.sigmoid(self.u(instances)))
        attention = torch.softmax(scores.squeeze(-1), dim=1)
        return (attention.unsqueeze(-1) * instances).sum(dim=1), attention
