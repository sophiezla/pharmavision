"""Rebuild a trained fold model from its run folder (prediction-only passes, Phases 5-8)."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from pharmachecked.config import REPO_ROOT
from pharmachecked.evaluation.branch_outputs import find_checkpoint
from pharmachecked.utils.logging import get_logger

log = get_logger(__name__)


def run_config(run_root: Path, paths: DictConfig, device: str | None = None) -> DictConfig:
    """The run's saved config with this machine's ``paths`` and ``device``."""
    cfg = OmegaConf.load(run_root / "config.yaml")
    cfg.paths = paths
    cfg.device = device
    return cfg


def run_checkpoint(run_root: Path) -> Path | None:
    """The run's weights (``best.pt`` or ``model.joblib``) on this machine, if present.

    Looks under the checkpoints folder recorded in the run's config, then under the repository
    (``find_checkpoint``), so models trained by an earlier kit are found in that kit's folder.
    """
    cfg = OmegaConf.load(run_root / "config.yaml")
    name = "model.joblib" if cfg.task == "handcrafted" else "best.pt"
    ckpt = find_checkpoint(run_root, str(cfg.paths.checkpoints_dir), [REPO_ROOT], name)
    if ckpt is None:
        log.error("no %s for %s", name, run_root.name)
    return ckpt


def load_torch_model(cfg: DictConfig, checkpoint: Path) -> nn.Module:
    """Build ``cfg.model`` without downloading pretrained weights and load ``checkpoint``."""
    from pharmachecked.models.registry import build_model

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    if "pretrained" in model_cfg:
        model_cfg.pretrained = False  # the checkpoint holds every weight
    if "backbone" in model_cfg:
        model_cfg.backbone.pretrained = False
    model = build_model(model_cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()
