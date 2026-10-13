"""Multi-branch model: shapes, MIL attention and gates sum to 1, every ablation flag, DCT."""

import pytest
import torch
from omegaconf import OmegaConf

from pharmachecked.config import CONFIG_DIR
from pharmachecked.models.frequency import dct_matrix, frequency_maps
from pharmachecked.models.fusion import Fusion
from pharmachecked.models.mil import GatedAttentionMIL
from pharmachecked.models.registry import build_model

BASE = OmegaConf.load(CONFIG_DIR / "model" / "multibranch.yaml")


def _model(**flags: object) -> torch.nn.Module:
    cfg = OmegaConf.merge(BASE, {"backbone": {"pretrained": False}}, flags)
    return build_model(cfg).eval()


def _run(model: torch.nn.Module, grid: int = 2) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        return model(torch.randn(2, 3, 224 * grid, 224 * grid))


def test_full_model_contract() -> None:
    out = _run(_model())
    assert out["logits"].shape == (2,)
    assert out["embedding"].shape == (2, 256)
    assert out["mil_attention"].shape == (2, 4)
    assert torch.allclose(out["mil_attention"].sum(dim=1), torch.ones(2))
    assert out["gate_weights"].shape == (2, 3)
    assert torch.allclose(out["gate_weights"].sum(dim=1), torch.ones(2))
    assert out["router_probs"].shape == (2, 4)
    assert torch.allclose(out["router_probs"].sum(dim=1), torch.ones(2))


@pytest.mark.parametrize(
    ("flags", "gates", "keys_absent"),
    [
        ({"use_local": False}, 2, {"mil_attention"}),
        ({"use_frequency": False}, 2, set()),
        ({"use_router": False}, 3, {"router_probs", "router_logits"}),
        ({"fusion": "concat"}, 3, set()),
        ({"shared_local_backbone": False}, 3, set()),
    ],
)
def test_ablations(flags: dict, gates: int, keys_absent: set) -> None:
    model = _model(**flags)
    out = _run(model)
    assert out["logits"].shape == (2,)
    assert out["gate_weights"].shape == (2, gates)
    assert not keys_absent & set(out)
    if flags == {"shared_local_backbone": False}:
        assert model.local_backbone is not None


def test_hard_routing_selects_one_head() -> None:
    out = _run(_model(routing="hard"))
    assert set(out["router_probs"].unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(out["router_probs"].sum(dim=1), torch.ones(2))


@pytest.mark.slow
def test_four_by_four_tiles_at_896() -> None:
    out = _run(_model(tile_grid=4, img_size=896), grid=4)
    assert out["mil_attention"].shape == (2, 16)


def test_mil_and_fusion_alone() -> None:
    bag, attention = GatedAttentionMIL(8)(torch.randn(3, 5, 8))
    assert bag.shape == (3, 8) and torch.allclose(attention.sum(dim=1), torch.ones(3))
    fused, gate = Fusion([8, 4], dim=6)([torch.randn(3, 8), torch.randn(3, 4)])
    assert fused.shape == (3, 6) and torch.allclose(gate.sum(dim=1), torch.ones(3))


def test_dct_is_orthonormal_and_maps_are_finite() -> None:
    d = dct_matrix()
    assert torch.allclose(d @ d.T, torch.eye(8), atol=1e-5)
    flat = torch.ones(1, 64, 64)
    textured = torch.rand(1, 64, 64)
    maps_flat, maps_tex = frequency_maps(flat), frequency_maps(textured)
    assert maps_flat.shape == (1, 2, 128, 128)
    assert torch.isfinite(maps_tex).all() and torch.isfinite(maps_flat).all()
