"""Resolve, download and instantiate every planned backbone on CPU (PLAN step 0.5).

Writes ``results/tables/backbones.md`` (names, parameter counts, embedding sizes, input
settings) and one ``configs/model/<key>.yaml`` per backbone with the verified exact name.

Usage:
    python scripts/check_backbones.py [--no-configs]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from importlib import metadata

import torch

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.utils.io import atomic_write_text
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("check_backbones")
CHECK_SIZE = 224


@dataclass(frozen=True)
class Backbone:
    """One planned backbone: config key, display name, library and exact pretrained name."""

    key: str
    display: str
    source: str  # "timm" or "open_clip"
    name: str  # timm "arch.tag", or open_clip "arch:pretrained"
    note: str = ""


# Pretrained tags are timm's defaults for each architecture; CLIP uses OpenAI weights (ADR-003).
BACKBONES = (
    Backbone("resnet50", "ResNet-50", "timm", "resnet50.a1_in1k"),
    Backbone(
        "efficientnetv2_s",
        "EfficientNetV2-S",
        "timm",
        "tf_efficientnetv2_s.in21k_ft_in1k",
        "pretrained at 300 px; run at 224 under the protocol",
    ),
    Backbone("convnext_tiny", "ConvNeXt-Tiny", "timm", "convnext_tiny.in12k_ft_in1k"),
    Backbone("vit_b16", "ViT-B/16", "timm", "vit_base_patch16_224.augreg2_in21k_ft_in1k"),
    Backbone("swin_tiny", "Swin-Tiny", "timm", "swin_tiny_patch4_window7_224.ms_in1k"),
    Backbone(
        "dinov2_vits14",
        "DINOv2 ViT-S/14",
        "timm",
        "vit_small_patch14_dinov2.lvd142m",
        "native 518 px; position embeddings interpolated to 224",
    ),
    Backbone(
        "dinov2_vitb14",
        "DINOv2 ViT-B/14",
        "timm",
        "vit_base_patch14_dinov2.lvd142m",
        "native 518 px; position embeddings interpolated to 224",
    ),
    Backbone(
        "clip_vitb16",
        "CLIP ViT-B/16",
        "open_clip",
        "ViT-B-16:openai",
        "timm equivalent of the image tower: vit_base_patch16_clip_224.openai",
    ),
    Backbone("mobilenetv3_large", "MobileNetV3-Large", "timm", "mobilenetv3_large_100.ra_in1k"),
    Backbone("efficientnet_lite0", "EfficientNet-Lite0", "timm", "efficientnet_lite0.ra_in1k"),
)


@dataclass
class Checked:
    """Result of instantiating one backbone."""

    backbone: Backbone
    params_m: float
    embed_dim: int
    input_size: str
    mean: tuple[float, ...]
    std: tuple[float, ...]
    out_shape: tuple[int, ...]


def check_timm(bb: Backbone) -> Checked:
    """Verify a timm name is listed, build it pretrained with no head and run one forward pass."""
    import timm

    if bb.name not in timm.list_models(bb.name, pretrained=True):
        raise LookupError(f"{bb.name} is not a pretrained timm model")
    kwargs = {"img_size": CHECK_SIZE} if "dinov2" in bb.name else {}
    model = timm.create_model(bb.name, pretrained=True, num_classes=0, **kwargs).eval()
    cfg = model.pretrained_cfg
    with torch.inference_mode():
        out = model(torch.zeros(1, 3, CHECK_SIZE, CHECK_SIZE))
    return Checked(
        backbone=bb,
        params_m=sum(p.numel() for p in model.parameters()) / 1e6,
        # Pooled output width; differs from num_features when a head conv follows pooling
        # (MobileNetV3: 960 features -> 1280 after conv_head).
        embed_dim=int(out.shape[-1]),
        input_size="x".join(str(s) for s in cfg["input_size"][1:]),
        mean=tuple(cfg["mean"]),
        std=tuple(cfg["std"]),
        out_shape=tuple(out.shape),
    )


def check_open_clip(bb: Backbone) -> Checked:
    """Verify an open_clip (arch, pretrained) pair is listed, build it and encode one image."""
    import open_clip

    arch, tag = bb.name.split(":")
    if (arch, tag) not in open_clip.list_pretrained():
        raise LookupError(f"{bb.name} is not an open_clip pretrained pair")
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=tag)
    model.eval()
    visual = model.visual
    with torch.inference_mode():
        out = model.encode_image(torch.zeros(1, 3, CHECK_SIZE, CHECK_SIZE))
    preprocess_cfg = open_clip.get_model_preprocess_cfg(model)
    size = preprocess_cfg["size"]
    size = tuple(size) if isinstance(size, list | tuple) else (size, size)
    return Checked(
        backbone=bb,
        params_m=sum(p.numel() for p in visual.parameters()) / 1e6,
        embed_dim=int(out.shape[-1]),
        input_size="x".join(str(s) for s in size),
        mean=tuple(preprocess_cfg["mean"]),
        std=tuple(preprocess_cfg["std"]),
        out_shape=tuple(out.shape),
    )


def model_config(c: Checked) -> str:
    """Return the YAML text of ``configs/model/<key>.yaml`` for a checked backbone."""
    bb = c.backbone
    lines = [
        f"# {bb.display}. Exact name verified by scripts/check_backbones.py "
        "(see results/tables/backbones.md).",
        f"name: {bb.key}",
        f"source: {bb.source}",
    ]
    if bb.source == "timm":
        lines.append(f"timm_name: {bb.name}")
    else:
        arch, tag = bb.name.split(":")
        lines += [f"open_clip_arch: {arch}", f"open_clip_pretrained: {tag}"]
    lines += [
        "pretrained: true",
        f"img_size: {CHECK_SIZE}",
        f"embedding_dim: {c.embed_dim}",
        f"mean: [{', '.join(f'{v:.4f}' for v in c.mean)}]",
        f"std: [{', '.join(f'{v:.4f}' for v in c.std)}]",
    ]
    return "\n".join(lines) + "\n"


def table(rows: list[Checked]) -> str:
    """Return the Markdown table written to ``results/tables/backbones.md``."""
    versions = ", ".join(f"{p} {metadata.version(p)}" for p in ("torch", "timm", "open_clip_torch"))
    out = [
        "# Backbone availability (PLAN 0.5)",
        "",
        f"Generated by `scripts/check_backbones.py`; {versions}.",
        "Each backbone was downloaded, built without a classification head and run on one",
        f"{CHECK_SIZE}x{CHECK_SIZE} input on CPU.",
        "Parameters count the feature extractor only (for CLIP, the image tower).",
        "",
        "| Key | Backbone | Library | Exact name | Params (M) | Embedding dim | "
        "Pretrained input | Output shape | Note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in rows:
        bb = c.backbone
        out.append(
            f"| {bb.key} | {bb.display} | {bb.source} | `{bb.name}` | {c.params_m:.2f} | "
            f"{c.embed_dim} | {c.input_size} | {list(c.out_shape)} | {bb.note} |"
        )
    return "\n".join(out) + "\n"


def main() -> int:
    """Check every backbone; write the table and configs. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-configs", action="store_true", help="do not write model configs")
    args = parser.parse_args()
    configure_logging(logging.INFO)
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    rows: list[Checked] = []
    failures: list[str] = []
    for bb in BACKBONES:
        try:
            checked = check_timm(bb) if bb.source == "timm" else check_open_clip(bb)
        except Exception as exc:  # report every failure, then exit non-zero
            log.error("%s (%s): %s", bb.display, bb.name, exc)
            failures.append(bb.key)
            continue
        log.info("%s resolved: %s, %.2f M params", bb.display, bb.name, checked.params_m)
        rows.append(checked)
    atomic_write_text(REPO_ROOT / "results" / "tables" / "backbones.md", table(rows))
    if not args.no_configs:
        for c in rows:
            atomic_write_text(CONFIG_DIR / "model" / f"{c.backbone.key}.yaml", model_config(c))
    log.info("%d of %d backbones resolved", len(rows), len(BACKBONES))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
