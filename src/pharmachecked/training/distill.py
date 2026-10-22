"""E16 knowledge distillation task (PLAN 8.1; docs/experiments/E16.md).

Student (MobileNetV3-Large or EfficientNet-Lite0, full fine-tuning) trained with
``loss = (1 - alpha) * BCE(student, y) + alpha * T^2 * KL(teacher_T || student_T)`` where
``p_T = sigmoid(logit / T)`` over the two classes (Hinton et al., 2015), alpha = 0.5,
T in {2, 4}. Teacher: the final model's fold model of the same fold and seed (ADR-027), in eval
mode, seeing the same augmented batch (re-normalized with its own mean/std).
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn

from pharmachecked.evaluation.aggregate import collect_runs
from pharmachecked.models.loading import load_torch_model, run_checkpoint, run_config
from pharmachecked.training.trainer import autocast
from pharmachecked.utils.run_dir import RunDir


def binary_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor, t: float) -> torch.Tensor:
    """Mean KL(teacher || student) of the tempered two-class distributions."""
    pt = torch.sigmoid(teacher_logits / t).clamp(1e-6, 1 - 1e-6)
    log_ps = nn.functional.logsigmoid(student_logits / t)
    log_qs = nn.functional.logsigmoid(-student_logits / t)
    kl = pt * (torch.log(pt) - log_ps) + (1 - pt) * (torch.log(1 - pt) - log_qs)
    return kl.mean()


def teacher_run(cfg: DictConfig) -> Path:
    """The teacher's run folder (``cfg.distill.teacher_*``, same fold and seed).

    Raises:
        FileNotFoundError: If it does not exist.
    """
    d = cfg.distill
    for rec in collect_runs(Path(cfg.paths.runs_dir), str(d.teacher_exp)):
        key = (rec.model, rec.variant, rec.fold, rec.seed)
        if key == (str(d.teacher_model), str(d.teacher_variant), int(cfg.fold), int(cfg.seed)):
            return rec.root
    raise FileNotFoundError(f"no teacher run for fold {cfg.fold} seed {cfg.seed}")


def run_distill(cfg: DictConfig, run: RunDir) -> None:
    """Train one student fold model with the distillation loss."""
    from pharmachecked.training.task import run_training

    t, alpha = float(cfg.distill.temperature), float(cfg.distill.alpha)
    t_root = teacher_run(cfg)
    t_cfg = run_config(t_root, cfg.paths, cfg.get("device"))
    ckpt = run_checkpoint(t_root)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint for teacher {t_root.name}")
    teacher = load_torch_model(t_cfg, ckpt).requires_grad_(False)
    s_mean = torch.tensor(list(cfg.model.mean)).view(1, 3, 1, 1)
    s_std = torch.tensor(list(cfg.model.std)).view(1, 3, 1, 1)
    t_mean = torch.tensor(list(t_cfg.model.mean)).view(1, 3, 1, 1)
    t_std = torch.tensor(list(t_cfg.model.std)).view(1, 3, 1, 1)
    bce = nn.BCEWithLogitsLoss()

    def hooks(model: nn.Module, device: torch.device) -> dict:
        teacher.to(device).eval()
        sm, ss, tm, ts = (v.to(device) for v in (s_mean, s_std, t_mean, t_std))

        def loss_hook(batch: dict, out: dict, y: torch.Tensor) -> torch.Tensor:
            x_t = ((batch["image"] * ss + sm) - tm) / ts
            with torch.no_grad(), autocast(device, bool(cfg.train.amp)):
                z_t = teacher(x_t)["logits"].float()
            z_s = out["logits"].float()
            return (1 - alpha) * bce(z_s, y) + alpha * t * t * binary_kl(z_t, z_s, t)

        return {"loss_hook": loss_hook}

    run_training(cfg, run, hooks=hooks)
