"""Multi-branch model (PLAN 3.5, ADR-015, ADR-017).

Input: a ``tile_grid * 224`` px square (448 px / 2x2 by default).
- Global branch: the image resized to 224 px -> backbone (EfficientNetV2-S + LoRA r8, ADR-015).
- Local branch: ``tile_grid x tile_grid`` tiles of 224 px -> the same backbone (or a separate
  copy) -> gated-attention MIL.
- Frequency branch: FFT + block-DCT maps -> small CNN.
- Fusion: gated (default) or concat over the active branches.
- Router: linear on the global embedding -> 4 packaging classes (ADR-009); 4 expert heads on
  the fused embedding mixed by router probabilities (soft) or the argmax head (hard). Without a
  router, a single head.
Returns ``logits, embedding, gate_weights, router_probs, router_logits, mil_attention``.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from pharmachecked.models.backbones import add_lora, make_backbone
from pharmachecked.models.frequency import FrequencyBranch
from pharmachecked.models.fusion import Fusion
from pharmachecked.models.mil import GatedAttentionMIL
from pharmachecked.models.registry import register
from pharmachecked.models.router import ROUTER_CLASSES, Router

TILE = 224


class MultiBranch(nn.Module):
    """See the module docstring."""

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        tile_grid: int = 2,
        use_local: bool = True,
        use_frequency: bool = True,
        use_router: bool = True,
        fusion: str = "gated",
        routing: str = "soft",
        shared_local_backbone: bool = True,
        fused_dim: int = 256,
    ) -> None:
        super().__init__()
        if routing not in ("soft", "hard"):
            raise ValueError(f"unknown routing {routing!r}")
        self.backbone = backbone
        self.tile_grid = tile_grid
        self.use_local, self.use_frequency, self.use_router = use_local, use_frequency, use_router
        self.routing = routing
        in_dims = [embedding_dim]
        if use_local:
            self.local_backbone = None if shared_local_backbone else copy.deepcopy(backbone)
            self.mil = GatedAttentionMIL(embedding_dim)
            in_dims.append(embedding_dim)
        if use_frequency:
            self.frequency = FrequencyBranch()
            in_dims.append(self.frequency.out_dim)
        self.fusion = Fusion(in_dims, fused_dim, fusion)
        n_heads = len(ROUTER_CLASSES) if use_router else 1
        self.heads = nn.Linear(fused_dim, n_heads)  # one output per expert head
        if use_router:
            self.router = Router(embedding_dim)

    def _tiles(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        g = self.tile_grid
        tiles = x.reshape(b, c, g, TILE, g, TILE).permute(0, 2, 4, 1, 3, 5)
        return tiles.reshape(b * g * g, c, TILE, TILE)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass on ``(B, 3, tile_grid * 224, tile_grid * 224)`` images."""
        b = x.shape[0]
        global_in = nn.functional.interpolate(
            x, size=(TILE, TILE), mode="bilinear", antialias=True, align_corners=False
        )
        e_global = self.backbone(global_in)
        embeddings = [e_global]
        out: dict[str, torch.Tensor] = {}
        if self.use_local:
            local_net = self.local_backbone if self.local_backbone is not None else self.backbone
            instances = local_net(self._tiles(x)).reshape(b, self.tile_grid**2, -1)
            e_local, attention = self.mil(instances)
            embeddings.append(e_local)
            out["mil_attention"] = attention
        if self.use_frequency:
            embeddings.append(self.frequency(x))
        fused, gate = self.fusion(embeddings)
        head_logits = self.heads(fused)  # (B, n_heads)
        if self.use_router:
            router_logits = self.router(e_global)
            probs = torch.softmax(router_logits.float(), dim=1).to(head_logits.dtype)
            if self.routing == "hard":
                probs = nn.functional.one_hot(probs.argmax(dim=1), probs.shape[1]).to(probs.dtype)
            logits = (probs * head_logits).sum(dim=1)
            out.update(router_logits=router_logits, router_probs=probs)
        else:
            logits = head_logits.squeeze(1)
        out.update(logits=logits, embedding=fused, gate_weights=gate)
        return out


@register("multibranch")
def build(cfg: Any) -> MultiBranch:
    """Build from ``configs/model/multibranch.yaml`` (backbone settings under ``backbone``)."""
    bcfg = cfg.backbone
    backbone, dim = make_backbone(bcfg)
    if bcfg.get("mode") == "lora":
        backbone = add_lora(backbone, list(bcfg.lora_targets), int(bcfg.lora_rank))
    elif bcfg.get("mode") not in (None, "full"):
        raise ValueError(f"unsupported backbone mode {bcfg.mode!r} in the multi-branch model")
    return MultiBranch(
        backbone,
        dim,
        tile_grid=int(cfg.tile_grid),
        use_local=bool(cfg.use_local),
        use_frequency=bool(cfg.use_frequency),
        use_router=bool(cfg.use_router),
        fusion=str(cfg.fusion),
        routing=str(cfg.routing),
        shared_local_backbone=bool(cfg.shared_local_backbone),
    )
