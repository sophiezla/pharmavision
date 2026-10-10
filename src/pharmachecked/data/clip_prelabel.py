"""Zero-shot CLIP pre-labels for packaging type (PLAN 1.7; Radford et al., 2021)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from pharmachecked.data.annotations import PACKAGING_TYPES

PROMPT = "a photo of a medicine {}"


def prompts(types: Sequence[str] = PACKAGING_TYPES) -> list[str]:
    """Prompt per packaging type (underscores read as spaces)."""
    return [PROMPT.format(t.replace("_", " ")) for t in types]


@torch.inference_mode()
def prelabel(
    image_ids: Sequence[str],
    images_dir: Path,
    arch: str = "ViT-B-16",
    pretrained: str = "openai",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Return ``image_id, clip_label, clip_conf`` (softmax over the five prompts).

    Args:
        image_ids: Images to label (normalized PNGs in ``images_dir``).
        images_dir: Folder of normalized images.
        arch: open_clip architecture.
        pretrained: open_clip pretrained tag.
        batch_size: Images per forward pass.

    Returns:
        One row per image id, in input order.
    """
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model.eval()
    tokenizer = open_clip.get_tokenizer(arch)
    text = model.encode_text(tokenizer(prompts()))
    text = text / text.norm(dim=-1, keepdim=True)
    scale = model.logit_scale.exp()
    rows = []
    for start in range(0, len(image_ids), batch_size):
        chunk = list(image_ids[start : start + batch_size])
        batch = []
        for image_id in chunk:
            with Image.open(images_dir / f"{image_id}.png") as im:
                batch.append(preprocess(im.convert("RGB")))
        feats = model.encode_image(torch.stack(batch))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        probs = (scale * feats @ text.T).softmax(dim=-1)
        conf, idx = probs.max(dim=-1)
        rows += [
            {"image_id": i, "clip_label": PACKAGING_TYPES[int(k)], "clip_conf": float(c)}
            for i, k, c in zip(chunk, idx, conf, strict=True)
        ]
    return pd.DataFrame(rows)
