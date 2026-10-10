"""Every registered model runs a 2-image forward pass and returns the required keys and shapes."""

import pytest
import torch

from pharmachecked.config import CONFIG_DIR
from pharmachecked.models.registry import build_model, register, registered
from pharmachecked.utils.io import read_yaml

LOCAL_CONFIGS = {
    p.stem: read_yaml(p)
    for p in (CONFIG_DIR / "model").glob("*.yaml")
    if read_yaml(p).get("source") == "local"
}


def test_every_registered_model_has_a_config() -> None:
    assert set(registered()) <= set(LOCAL_CONFIGS) | {
        c.get("builder") for c in LOCAL_CONFIGS.values()
    }


@pytest.mark.parametrize("name", sorted(LOCAL_CONFIGS))
def test_forward_contract(name: str) -> None:
    cfg = LOCAL_CONFIGS[name]
    model = build_model(cfg).eval()
    with torch.inference_mode():
        out = model(torch.randn(2, 3, 64, 64))
    assert {"logits", "embedding"} <= set(out)
    assert out["logits"].shape == (2,)
    assert out["embedding"].shape == (2, cfg.embedding_dim)


def test_unknown_model_rejected() -> None:
    with pytest.raises(KeyError):
        build_model({"name": "nope"})


def test_duplicate_registration_rejected() -> None:
    registered()  # ensure built-ins are imported
    with pytest.raises(KeyError):
        register("cnn_small")(lambda cfg: torch.nn.Identity())
