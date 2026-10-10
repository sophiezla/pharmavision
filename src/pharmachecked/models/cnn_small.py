"""Small from-scratch CNN: three conv blocks, global average pooling, linear head (E1 tier 2)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from pharmachecked.models.registry import register


class SmallCNN(nn.Module):
    """Conv(3x3)-BN-ReLU-MaxPool x 3, global average pooling, one-logit linear head.

    Returns ``{"logits": (B,), "embedding": (B, channels[-1])}``; the logit is for the
    counterfeit class (label 1).
    """

    def __init__(self, channels: tuple[int, ...] = (32, 64, 128), dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 3
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c_in, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return logits and the pooled embedding."""
        embedding = self.pool(self.features(x)).flatten(1)
        return {"logits": self.head(self.dropout(embedding)).squeeze(1), "embedding": embedding}


@register("cnn_small")
def build(cfg: Any) -> SmallCNN:
    """Build from ``configs/model/cnn_small.yaml``."""
    return SmallCNN(tuple(cfg.channels), float(cfg.get("dropout", 0.0)))
