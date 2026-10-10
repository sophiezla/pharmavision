"""Trainer: learns, resumes exactly after interruption, early-stops, restores the best weights."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from pharmachecked.config import CONFIG_DIR
from pharmachecked.data.datasets import ManifestDataset
from pharmachecked.data.manifest import read_manifest
from pharmachecked.data.splits import load_dev_splits
from pharmachecked.data.toy import ToyData
from pharmachecked.data.transforms import build_eval_transform, build_train_transform
from pharmachecked.models.cnn_small import SmallCNN
from pharmachecked.training import trainer
from pharmachecked.training.optim import param_groups, warmup_cosine
from pharmachecked.utils.seed import seed_everything

TRAIN_CFG = OmegaConf.merge(
    OmegaConf.load(CONFIG_DIR / "train" / "default.yaml"),
    {"batch_size": 8, "eval_batch_size": 16, "lr": 3e-3, "max_epochs": 4, "warmup_epochs": 1},
)
TCFG = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
MEAN, STD = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
CPU = torch.device("cpu")


def _datasets(toy: ToyData) -> tuple[ManifestDataset, ManifestDataset]:
    manifest = read_manifest(toy.manifest)
    splits = load_dev_splits(toy.splits)
    train_ds = ManifestDataset.from_split(
        manifest,
        splits,
        0,
        "train",
        toy.images_dir,
        build_train_transform(TCFG, MEAN, STD, 32),
        True,
    )
    val_ds = ManifestDataset.from_split(
        manifest, splits, 0, "val", toy.images_dir, build_eval_transform(TCFG, MEAN, STD, 32), True
    )
    return train_ds, val_ds


def _fit(toy: ToyData, tmp: Path, cfg=TRAIN_CFG) -> tuple[nn.Module, trainer.FitResult]:
    seed_everything(0)
    model = SmallCNN((8, 16, 16))
    train_ds, val_ds = _datasets(toy)
    result = trainer.fit(model, train_ds, val_ds, cfg, CPU, tmp / "ckpt", tmp / "log.csv", seed=0)
    return model, result


def test_training_reduces_loss(toy: ToyData, tmp_path: Path) -> None:
    _, result = _fit(toy, tmp_path)
    losses = [h["train_loss"] for h in result.history]
    assert losses[-1] < losses[0]
    assert len(pd.read_csv(tmp_path / "log.csv")) == result.epochs_run


def test_best_weights_restored_and_resume_state_removed(toy: ToyData, tmp_path: Path) -> None:
    model, result = _fit(toy, tmp_path)
    best = torch.load(tmp_path / "ckpt" / "best.pt", weights_only=True)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, best[name])
    assert not (tmp_path / "ckpt" / "resume.pt").exists()
    assert result.best_epoch == max(
        range(result.epochs_run), key=lambda e: result.history[e]["val_auroc"]
    )


def test_resume_after_interruption_matches_uninterrupted(
    toy: ToyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full_model, full = _fit(toy, tmp_path / "full")

    real_predict = trainer.predict
    calls = {"n": 0}

    def crash_on_second_validation(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("simulated power cut during epoch 1")
        return real_predict(*args, **kwargs)

    monkeypatch.setattr(trainer, "predict", crash_on_second_validation)
    with pytest.raises(KeyboardInterrupt):
        _fit(toy, tmp_path / "cut")
    assert (tmp_path / "cut" / "ckpt" / "resume.pt").exists()
    monkeypatch.setattr(trainer, "predict", real_predict)
    resumed_model, resumed = _fit(toy, tmp_path / "cut")

    assert [h["train_loss"] for h in resumed.history] == pytest.approx(
        [h["train_loss"] for h in full.history]
    )
    assert resumed.best_epoch == full.best_epoch
    for name, tensor in full_model.state_dict().items():
        assert torch.allclose(tensor, resumed_model.state_dict()[name])


def test_early_stopping(toy: ToyData, tmp_path: Path) -> None:
    cfg = OmegaConf.merge(TRAIN_CFG, {"lr": 0.0, "max_epochs": 10, "patience": 1})
    _, result = _fit(toy, tmp_path, cfg)
    # No learning: val AUROC never improves after epoch 0, so epochs 0-1 run and 2 stops.
    assert result.best_epoch == 0
    assert result.epochs_run == 2


class _WithBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.head = nn.Linear(4, 1)


def test_param_groups_backbone_multiplier_and_no_decay() -> None:
    groups = param_groups(_WithBackbone(), lr=1e-3, backbone_lr_mult=0.1, weight_decay=0.05)
    by_key = {(g["lr"], g["weight_decay"]): sum(p.numel() for p in g["params"]) for g in groups}
    assert by_key == {
        (1e-4, 0.0): 4 + 4 + 4,  # backbone bias + LayerNorm weight and bias
        (1e-4, 0.05): 16,  # backbone weight matrix
        (1e-3, 0.0): 1,  # head bias
        (1e-3, 0.05): 4,  # head weight
    }


def test_warmup_cosine_schedule() -> None:
    assert warmup_cosine(0, 10, 100) == pytest.approx(0.1)
    assert warmup_cosine(9, 10, 100) == pytest.approx(1.0)
    assert warmup_cosine(10, 10, 100) == pytest.approx(1.0)
    assert warmup_cosine(55, 10, 100) == pytest.approx(0.5)
    assert warmup_cosine(100, 10, 100) == pytest.approx(0.0, abs=1e-12)
