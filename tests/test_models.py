"""Every model config builds, runs a 2-image forward pass and returns the required keys/shapes."""

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from pharmachecked.config import CONFIG_DIR
from pharmachecked.models.backbones import BackboneClassifier
from pharmachecked.models.registry import build_model, register, registered
from pharmachecked.utils.io import read_yaml

MODEL_CONFIGS = {
    p.stem: read_yaml(p)
    for p in (CONFIG_DIR / "model").glob("*.yaml")
    if read_yaml(p).get("builder")
}


def _random_weights(name: str) -> DictConfig:
    return OmegaConf.merge(MODEL_CONFIGS[name], {"pretrained": False})


def test_every_registered_builder_is_used_by_a_config() -> None:
    assert set(registered()) == {c.builder for c in MODEL_CONFIGS.values()}


@pytest.mark.parametrize("name", sorted(MODEL_CONFIGS))
def test_forward_contract(name: str) -> None:
    cfg = _random_weights(name)
    model = build_model(cfg).eval()
    size = 64 if cfg.builder == "cnn_small" else int(cfg.img_size)
    with torch.inference_mode():
        out = model(torch.randn(2, 3, size, size))
    assert {"logits", "embedding"} <= set(out)
    assert out["logits"].shape == (2,)
    assert out["embedding"].shape == (2, cfg.embedding_dim)


def test_linear_probe_freezes_backbone_and_keeps_it_in_eval_mode() -> None:
    cfg = OmegaConf.merge(_random_weights("resnet50"), {"mode": "linear_probe"})
    model = build_model(cfg)
    assert isinstance(model, BackboneClassifier)
    model.train()
    assert not model.backbone.training
    assert model.head.training
    trainable = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"head"}


@pytest.mark.parametrize("name", ["vit_b16", "convnext_tiny", "resnet50", "clip_vitb16"])
def test_last_k_trains_only_the_last_blocks(name: str) -> None:
    cfg = OmegaConf.merge(_random_weights(name), {"mode": "last_k", "trainable_blocks": 2})
    model = build_model(cfg)
    n_train = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.backbone.parameters())
    assert 0 < n_train < n_all
    assert all(p.requires_grad for p in model.head.parameters())


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError):
        build_model(OmegaConf.merge(_random_weights("resnet50"), {"mode": "half"}))


def test_unknown_model_rejected() -> None:
    with pytest.raises(KeyError):
        build_model({"name": "nope"})


def test_duplicate_registration_rejected() -> None:
    registered()  # ensure built-ins are imported
    with pytest.raises(KeyError):
        register("cnn_small")(lambda cfg: torch.nn.Identity())


@pytest.mark.slow
@pytest.mark.parametrize(
    "name", sorted(n for n, c in MODEL_CONFIGS.items() if c.get("lora_targets"))
)
def test_lora_trains_only_adapters_and_head(name: str) -> None:
    cfg = OmegaConf.merge(_random_weights(name), {"mode": "lora", "lora_rank": 4})
    model = build_model(cfg)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n or n.startswith("head.") for n in trainable)
    assert any("lora_" in n for n in trainable)
    model.train()
    batchnorms = [
        m for m in model.backbone.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert all(not m.training for m in batchnorms)
    size = int(cfg.img_size)
    out = model(torch.randn(2, 3, size, size))
    out["logits"].sum().backward()
    assert out["logits"].shape == (2,)
