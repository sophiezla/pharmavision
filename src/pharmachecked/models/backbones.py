"""Pretrained backbones (timm / open_clip) with a one-logit head (E1 tiers 3-5, E2).

Modes:
    full          all parameters train (backbone at ``train.backbone_lr_mult`` x lr);
    linear_probe  backbone frozen and kept in eval mode; only the head trains;
    last_k        only the last ``trainable_blocks`` blocks and the final norm train (E2);
    lora          backbone frozen; LoRA adapters (Hu et al., 2022; via ``peft``) of rank
                  ``lora_rank`` on the config's ``lora_targets`` train with the head (E2).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from pharmachecked.models.registry import register

MODES = ("full", "linear_probe", "last_k", "lora")


def add_lora(backbone: nn.Module, targets: list[str], rank: int) -> nn.Module:
    """Wrap ``backbone`` with LoRA adapters (alpha = 2 x rank, no dropout); base weights frozen."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=targets)
    return get_peft_model(backbone, config)


def make_backbone(cfg: Any) -> tuple[nn.Module, int]:
    """Build the feature extractor named by a model config; return (module, embedding width).

    timm models are built without a classifier (pooled features). For open_clip the image tower
    is used, whose output is the projected CLIP image embedding.
    """
    pretrained = bool(cfg.get("pretrained", True))
    if cfg.source == "timm":
        import timm

        kwargs: dict[str, Any] = {}
        if "dinov2" in cfg.timm_name:
            kwargs["img_size"] = int(cfg.img_size)
        model = timm.create_model(cfg.timm_name, pretrained=pretrained, num_classes=0, **kwargs)
        return model, int(cfg.embedding_dim)
    if cfg.source == "open_clip":
        import open_clip

        clip = open_clip.create_model(
            cfg.open_clip_arch, pretrained=cfg.open_clip_pretrained if pretrained else None
        )
        return clip.visual, int(cfg.embedding_dim)
    raise ValueError(f"unknown backbone source {cfg.source!r}")


def _blocks(backbone: nn.Module) -> list[nn.Module]:
    """The backbone's sequential stages/blocks, for last-k fine-tuning."""
    for attr in ("blocks", "stages", "layers"):
        found = getattr(backbone, attr, None)
        if found is not None:
            return list(found)
    transformer = getattr(backbone, "transformer", None)  # open_clip ViT tower
    if transformer is not None and hasattr(transformer, "resblocks"):
        return list(transformer.resblocks)
    names = [n for n in ("layer1", "layer2", "layer3", "layer4") if hasattr(backbone, n)]
    if names:  # torchvision-style ResNet
        return [getattr(backbone, n) for n in names]
    raise ValueError(f"cannot find blocks in {type(backbone).__name__}")


class BackboneClassifier(nn.Module):
    """Backbone + dropout + linear head; returns ``{"logits": (B,), "embedding": (B, D)}``."""

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        mode: str = "full",
        head_dropout: float = 0.0,
        trainable_blocks: int = 2,
        lora_targets: list[str] | None = None,
        lora_rank: int = 8,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "lora":
            if not lora_targets:
                raise ValueError("mode lora needs lora_targets in the model config")
            backbone = add_lora(backbone, list(lora_targets), lora_rank)
        self.backbone = backbone
        self.mode = mode
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(embedding_dim, 1)
        if mode == "linear_probe":
            self.backbone.requires_grad_(False)
        elif mode == "last_k":
            self.backbone.requires_grad_(False)
            for block in _blocks(backbone)[-trainable_blocks:]:
                block.requires_grad_(True)
            for name in ("norm", "fc_norm", "ln_post", "head_norm", "norm_pre"):
                layer = getattr(backbone, name, None)
                if isinstance(layer, nn.Module):
                    layer.requires_grad_(True)

    def train(self, mode: bool = True) -> BackboneClassifier:
        """Keep frozen parts in eval mode.

        Linear probe: the whole backbone (no BN statistics updates, no dropout). LoRA: the
        batch-norm layers, so only the adapters (and the head) change during training.
        """
        super().train(mode)
        if self.mode == "linear_probe":
            self.backbone.eval()
        elif self.mode == "lora":
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return logits and the backbone embedding."""
        embedding = self.backbone(x)
        return {"logits": self.head(self.dropout(embedding)).squeeze(1), "embedding": embedding}


@register("backbone")
def build(cfg: Any) -> BackboneClassifier:
    """Build from a ``configs/model/<backbone>.yaml`` config."""
    backbone, dim = make_backbone(cfg)
    return BackboneClassifier(
        backbone,
        dim,
        mode=str(cfg.get("mode", "full")),
        head_dropout=float(cfg.get("head_dropout", 0.0)),
        trainable_blocks=int(cfg.get("trainable_blocks", 2)),
        lora_targets=list(cfg.get("lora_targets") or []),
        lora_rank=int(cfg.get("lora_rank", 8)),
    )
