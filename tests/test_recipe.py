"""E5 recipe pieces: losses, batch mixing, augment switches, SupCon loss, router targets."""

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from pharmachecked.config import CONFIG_DIR
from pharmachecked.data.transforms import build_train_transform
from pharmachecked.models.router import router_targets
from pharmachecked.training.losses import bce_loss, build_loss, focal_loss, mix_batch
from pharmachecked.training.supcon import supcon_loss


def test_focal_with_gamma_zero_is_weighted_bce() -> None:
    logits, y = torch.tensor([2.0, -1.0, 0.5]), torch.tensor([1.0, 0.0, 1.0])
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    expected = (torch.tensor([0.7, 0.3, 0.7]) * plain).mean()
    assert focal_loss(0.0, 0.7)(logits, y) == pytest.approx(float(expected))


def test_focal_downweights_easy_examples() -> None:
    easy, hard = torch.tensor([6.0]), torch.tensor([0.1])
    y = torch.tensor([1.0])
    ratio_focal = focal_loss(2.0, 0.5)(easy, y) / focal_loss(2.0, 0.5)(hard, y)
    ratio_bce = bce_loss()(easy, y) / bce_loss()(hard, y)
    assert ratio_focal < ratio_bce


def test_label_smoothing_targets() -> None:
    logits = torch.tensor([0.0])
    # With s = 0.1 the target 1 becomes 0.95; at logit 0 the loss is log 2 regardless.
    assert bce_loss(0.1)(logits, torch.tensor([1.0])) == pytest.approx(0.6931, abs=1e-4)


def test_build_loss_class_balanced_alpha() -> None:
    cfg = OmegaConf.create({"loss": "focal", "focal_gamma": 0.0})
    loss = build_loss(cfg, [0, 0, 0, 1])  # alpha_pos = share of authentic = 0.75
    logits, y = torch.tensor([0.0]), torch.tensor([1.0])
    assert loss(logits, y) == pytest.approx(0.75 * 0.6931, abs=1e-4)
    with pytest.raises(ValueError):
        build_loss(OmegaConf.create({"loss": "hinge"}), [0, 1])


@pytest.mark.parametrize("mode", ["mixup", "cutmix"])
def test_mix_batch_targets_stay_in_range(mode: str) -> None:
    torch.manual_seed(0)
    images = torch.rand(8, 3, 16, 16)
    y = torch.tensor([0.0, 1.0] * 4)
    mixed, targets = mix_batch(images, y, mode, 1.0)
    assert mixed.shape == images.shape
    assert ((targets >= 0) & (targets <= 1)).all()
    assert not torch.equal(mixed, images)
    same, same_y = mix_batch(images, y, "none", 1.0)
    assert same is images and same_y is y


def test_augment_switches_change_the_pipeline() -> None:
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    base = build_train_transform(tcfg, (0.5,) * 3, (0.5,) * 3, 32)
    full = build_train_transform(
        tcfg, (0.5,) * 3, (0.5,) * 3, 32, {"hflip": True, "color_strong": True, "randaugment": True}
    )
    names = [type(t).__name__ for t in full.transforms]
    assert "RandomHorizontalFlip" in names and "RandAugment" in names
    assert "RandomHorizontalFlip" not in [type(t).__name__ for t in base.transforms]
    assert full(Image.new("RGB", (40, 30))).shape == (3, 32, 32)


def test_supcon_loss_prefers_clustered_features() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    clustered = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    mixed = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]])
    assert supcon_loss(clustered, labels, 0.1) < supcon_loss(mixed, labels, 0.1)


def test_router_targets_ignore_not_a_drug() -> None:
    assert router_targets(["blister", "other", "not_a_drug", "", "both"]).tolist() == [
        0,
        3,
        -1,
        -1,
        2,
    ]
