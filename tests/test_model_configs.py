"""Every backbone config names a model that the installed timm / open_clip can load."""

from pathlib import Path

import pytest

from pharmachecked.config import CONFIG_DIR
from pharmachecked.utils.io import read_yaml

BACKBONE_CONFIGS = sorted(
    p
    for p in (CONFIG_DIR / "model").glob("*.yaml")
    if read_yaml(p).get("source") in ("timm", "open_clip")
)
PLANNED = {
    "resnet50",
    "efficientnetv2_s",
    "convnext_tiny",
    "vit_b16",
    "swin_tiny",
    "dinov2_vits14",
    "dinov2_vitb14",
    "clip_vitb16",
    "mobilenetv3_large",
    "efficientnet_lite0",
}


def test_every_planned_backbone_has_a_config() -> None:
    assert {p.stem for p in BACKBONE_CONFIGS} == PLANNED


@pytest.mark.parametrize("path", BACKBONE_CONFIGS, ids=lambda p: p.stem)
def test_backbone_name_resolves(path: Path) -> None:
    cfg = read_yaml(path)
    assert cfg.name == path.stem
    if cfg.source == "timm":
        import timm

        assert timm.list_models(cfg.timm_name, pretrained=True) == [cfg.timm_name]
    elif cfg.source == "open_clip":
        import open_clip

        assert (cfg.open_clip_arch, cfg.open_clip_pretrained) in open_clip.list_pretrained()
    else:
        pytest.fail(f"unknown source {cfg.source!r}")
    assert cfg.embedding_dim > 0
    assert len(cfg.mean) == len(cfg.std) == 3
