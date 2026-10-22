"""Concept bottleneck model (Koh et al., 2020; PLAN 6.5, E13).

backbone -> 7 concept logits -> sigmoid -> linear (logistic-regression) label head on the concept
probabilities only. The backbone settings come from the model config (EfficientNetV2-S with
LoRA r8, as the final model). Returns ``logits`` (task), ``concept_logits``, ``concept_probs``
and ``embedding`` (backbone features).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from pharmachecked.data.annotations import CONCEPTS
from pharmachecked.models.backbones import BackboneClassifier, make_backbone
from pharmachecked.models.registry import register


class ConceptBottleneck(nn.Module):
    """See the module docstring."""

    def __init__(self, backbone: BackboneClassifier, n_concepts: int = len(CONCEPTS)) -> None:
        super().__init__()
        # reuse BackboneClassifier for its LoRA / batch-norm handling; its head is replaced
        self.backbone = backbone.backbone
        self._wrapper_mode = backbone.mode
        dim = backbone.head.in_features
        self.concept_head = nn.Linear(dim, n_concepts)
        self.label_head = nn.Linear(n_concepts, 1)

    def train(self, mode: bool = True) -> ConceptBottleneck:
        """Keep batch-norm layers in eval mode under LoRA (as ``BackboneClassifier``)."""
        super().train(mode)
        if self._wrapper_mode == "lora":
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Task logits from the predicted concept probabilities."""
        embedding = self.backbone(x)
        concept_logits = self.concept_head(embedding)
        concept_probs = torch.sigmoid(concept_logits.float()).to(concept_logits.dtype)
        logits = self.label_head(concept_probs).squeeze(1)
        return {
            "logits": logits,
            "concept_logits": concept_logits,
            "concept_probs": concept_probs,
            "embedding": embedding,
        }


@register("concept_bottleneck")
def build(cfg: Any) -> ConceptBottleneck:
    """Build from ``configs/model/cbm_efficientnetv2_s.yaml``."""
    backbone, dim = make_backbone(cfg)
    wrapper = BackboneClassifier(
        backbone,
        dim,
        mode=str(cfg.get("mode", "full")),
        lora_targets=list(cfg.get("lora_targets") or []),
        lora_rank=int(cfg.get("lora_rank", 8)),
    )
    return ConceptBottleneck(wrapper)
