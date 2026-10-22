"""Attention rollout for timm ViT backbones (Abnar & Zuidema, 2020; PLAN 6.1).

Per block, the head-averaged attention matrix is mixed with the identity (residual path),
``A = 0.5 I + 0.5 mean_h(attn)``, row-normalized, and the blocks are multiplied in order. The
class token's row over the patch tokens, reshaped to the patch grid and upsampled bilinearly,
is the map. Attention probabilities are captured from each block's ``attn_drop`` input, which
requires the unfused attention path (``fused_attn = False``, numerically the same).
"""

from __future__ import annotations

import torch
from torch import nn

from pharmachecked.interpret.cams import unwrap_backbone


def attention_rollout(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Rollout maps ``(B, H, W)`` for a model whose backbone is a timm VisionTransformer.

    Raises:
        ValueError: If the backbone has no transformer blocks with ``attn.attn_drop``.
    """
    vit = unwrap_backbone(model)
    blocks = getattr(vit, "blocks", None)
    if blocks is None or not all(hasattr(b, "attn") for b in blocks):
        raise ValueError("attention rollout needs a ViT backbone")
    captured: list[torch.Tensor] = []
    handles, fused = [], []
    for block in blocks:
        fused.append(block.attn.fused_attn)
        block.attn.fused_attn = False
        handles.append(
            block.attn.attn_drop.register_forward_hook(
                lambda _m, inp, _out: captured.append(inp[0].detach().float())
            )
        )
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
        for block, was in zip(blocks, fused, strict=True):
            block.attn.fused_attn = was
    n_tokens = captured[0].shape[-1]
    eye = torch.eye(n_tokens, device=captured[0].device)
    rollout = eye.expand(len(x), n_tokens, n_tokens)
    for attn in captured:
        a = 0.5 * eye + 0.5 * attn.mean(dim=1)
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout
    prefix = int(getattr(vit, "num_prefix_tokens", 1))
    cls_row = rollout[:, 0, prefix:]
    side = round(cls_row.shape[1] ** 0.5)
    grid = cls_row.reshape(-1, 1, side, side)
    return nn.functional.interpolate(grid, size=x.shape[-2:], mode="bilinear")[:, 0].cpu()
