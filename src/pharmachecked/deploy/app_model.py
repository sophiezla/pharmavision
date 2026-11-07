"""The app's model: one fold model with the evaluation preprocessing and a Grad-CAM++ map.

Preprocessing is the evaluation transform of the run (``data/transforms.yaml`` geometry at the
model's input size, then normalization), so an uploaded file gives the same logit as the
offline pipeline for the same image (PLAN 8.6 acceptance test).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import nn

from pharmachecked.config import CONFIG_DIR
from pharmachecked.data.transforms import build_eval_geometry, build_tensorize


@dataclass
class AppModel:
    """A fold model plus its evaluation geometry and normalization."""

    model: nn.Module
    geometry: object
    tensorize: object

    @classmethod
    def from_run(cls, run_root: Path, checkpoint: Path, paths: DictConfig) -> AppModel:
        """Rebuild the run's model on the CPU from its saved config and ``checkpoint``."""
        from pharmachecked.models.loading import load_torch_model, run_config

        cfg = run_config(run_root, paths, "cpu")
        return cls.from_model(load_torch_model(cfg, checkpoint), cfg)

    @classmethod
    def from_model(cls, model: nn.Module, cfg: DictConfig) -> AppModel:
        """Wrap an already built model with the preprocessing of ``cfg``."""
        tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
        size = int(cfg.data.get("img_size") or cfg.model.img_size)
        return cls(
            model.eval(),
            build_eval_geometry(tcfg, size),
            build_tensorize(list(cfg.model.mean), list(cfg.model.std)),
        )

    def view(self, image: Image.Image) -> Image.Image:
        """The image at model-input geometry (what the model and the quality checks see)."""
        return self.geometry(image.convert("RGB"))

    def logit(self, view: Image.Image) -> float:
        """Counterfeit logit of an image already at model-input geometry."""
        with torch.no_grad():
            out = self.model(self.tensorize(view).unsqueeze(0))
        return float(out["logits"].float().reshape(-1)[0])

    def heatmap(self, view: Image.Image) -> np.ndarray:
        """Grad-CAM++ map (H, W) in [0, 1] for the predicted class (E12 method)."""
        from pharmachecked.interpret.cams import saliency

        x = self.tensorize(view).unsqueeze(0)
        m = saliency("gradcampp", self.model, x, ["upload"])[0].detach().numpy().astype(float)
        span = m.max() - m.min()
        return (m - m.min()) / span if span > 0 else np.zeros_like(m)
