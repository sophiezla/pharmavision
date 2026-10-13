"""Packaging-type router (PLAN 3.5, ADR-009): linear layer on the global embedding.

Classes are blister, packaging, both and other; ``not_a_drug`` images (and images without a
packaging type) get target -1 and contribute no router loss.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

ROUTER_CLASSES = ("blister", "packaging", "both", "other")


def router_targets(packaging_types: Sequence[str]) -> torch.Tensor:
    """Class index per packaging type, -1 when it is not a router class."""
    index = {name: k for k, name in enumerate(ROUTER_CLASSES)}
    return torch.tensor([index.get(t, -1) for t in packaging_types], dtype=torch.long)


class Router(nn.Module):
    """Linear packaging-type classifier on the global embedding."""

    def __init__(self, embedding_dim: int, n_classes: int = len(ROUTER_CLASSES)) -> None:
        super().__init__()
        self.linear = nn.Linear(embedding_dim, n_classes)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Router logits ``(B, n_classes)``."""
        return self.linear(embedding)
