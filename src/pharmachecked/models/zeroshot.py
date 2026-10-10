"""Zero-shot CLIP with a prompt ensemble (E1 tier 6; Radford et al., 2021).

The text embedding of each class is the normalized mean of its prompts' embeddings. The
counterfeit logit is ``logit_scale * (image . t_counterfeit - image . t_authentic)``, so the
counterfeit probability is the two-class softmax.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

AUTHENTIC_PROMPTS = (
    "a photo of authentic medicine packaging",
    "a photo of genuine medicine packaging",
    "a photo of real pharmaceutical packaging",
    "a photo of legitimate medication packaging",
    "a photo of original brand-name medicine packaging",
)
COUNTERFEIT_PROMPTS = (
    "a photo of counterfeit medicine packaging",
    "a photo of fake medicine packaging",
    "a photo of falsified pharmaceutical packaging",
    "a photo of forged medication packaging",
    "a photo of an imitation of brand-name medicine packaging",
)


class ZeroShotCLIP(nn.Module):
    """Frozen CLIP scoring images against two prompt ensembles; ``forward`` -> logits/embedding."""

    def __init__(
        self,
        clip_model: Any,
        tokenizer: Any,
        authentic: Sequence[str] = AUTHENTIC_PROMPTS,
        counterfeit: Sequence[str] = COUNTERFEIT_PROMPTS,
    ) -> None:
        super().__init__()
        self.visual = clip_model.visual
        with torch.inference_mode():
            classes = []
            for prompts in (authentic, counterfeit):
                text = clip_model.encode_text(tokenizer(list(prompts)))
                text = text / text.norm(dim=-1, keepdim=True)
                mean = text.mean(dim=0)
                classes.append(mean / mean.norm())
        self.register_buffer("text", torch.stack(classes).clone())
        self.register_buffer("scale", clip_model.logit_scale.exp().detach().clone())
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Counterfeit logit (difference of scaled cosine similarities) and image embedding."""
        embedding = self.visual(x)
        normed = embedding / embedding.norm(dim=-1, keepdim=True)
        sims = self.scale * normed @ self.text.T
        return {"logits": sims[:, 1] - sims[:, 0], "embedding": embedding}


def build_zeroshot(cfg: Any) -> ZeroShotCLIP:
    """Build from the CLIP model config (``open_clip_arch``, ``open_clip_pretrained``)."""
    import open_clip

    model = open_clip.create_model(
        cfg.open_clip_arch,
        pretrained=cfg.open_clip_pretrained if cfg.get("pretrained", True) else None,
    )
    model.eval()
    return ZeroShotCLIP(model, open_clip.get_tokenizer(cfg.open_clip_arch))
