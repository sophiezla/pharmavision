"""Supervised contrastive pretraining (Khosla et al., 2020) for the E5 SupCon arm (PLAN 3.2).

Stage 1 trains the encoder's trainable parameters plus a 2-layer projection head (128-d) with the
SupCon loss on two augmented views per image for a fixed number of epochs (no early stopping).
The caller then freezes the encoder and trains a linear head with the normal recipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pharmachecked.training.optim import warmup_cosine
from pharmachecked.utils.logging import get_logger
from pharmachecked.utils.seed import make_generator, worker_init_fn

log = get_logger(__name__)


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """SupCon loss over L2-normalized ``features`` (N, D) with class ``labels`` (N,).

    For each anchor, positives are the other samples with the same label; the loss is the mean
    over anchors (with at least one positive) of -mean log softmax similarity to its positives.
    """
    z = nn.functional.normalize(features.float(), dim=1)
    sim = z @ z.T / temperature
    n = z.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positives = (labels[:, None] == labels[None, :]) & ~self_mask
    has_pos = positives.any(dim=1)
    pos_log_prob = torch.where(positives, log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    mean_pos = pos_log_prob[has_pos] / positives[has_pos].sum(dim=1)
    return -mean_pos.mean()


class TwoViews(Dataset[dict[str, Any]]):
    """Two independently augmented views of each item of a dataset (same label)."""

    def __init__(self, base: Dataset[dict[str, Any]]) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict[str, Any]:
        first, second = self.base[index], self.base[index]
        return {"view1": first["image"], "view2": second["image"], "label": first["label"]}


def pretrain(
    model: nn.Module,
    train_ds: Dataset[dict[str, Any]],
    cfg: Any,
    device: torch.device,
    out: Path,
    seed: int,
) -> None:
    """Run SupCon on ``model.backbone``; saves its state to ``out`` (reused on resume)."""
    if out.exists():
        model.backbone.load_state_dict(torch.load(out, map_location=device, weights_only=True))
        log.info("SupCon stage already done; loaded %s", out)
        return
    backbone = model.backbone.to(device)
    with torch.no_grad():
        probe = next(iter(DataLoader(train_ds, batch_size=2)))["image"].to(device)
        dim = backbone(probe).shape[-1]
    projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 128)).to(device)
    params = [p for p in backbone.parameters() if p.requires_grad] + list(projection.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(
        TwoViews(train_ds),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=len(train_ds) > cfg.batch_size,  # type: ignore[arg-type]
        num_workers=cfg.num_workers,
        worker_init_fn=worker_init_fn,
        generator=make_generator(seed),
        persistent_workers=cfg.num_workers > 0,  # ADR-018
    )
    epochs = int(cfg.supcon_epochs)
    total = epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, len(loader), total))
    autocast = torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=bool(cfg.amp) and device.type == "cuda"
    )
    for epoch in range(epochs):
        backbone.train()
        losses = []
        for batch in loader:
            views = torch.cat([batch["view1"], batch["view2"]]).to(device)
            labels = torch.cat([batch["label"], batch["label"]]).to(device)
            opt.zero_grad(set_to_none=True)
            with autocast:
                features = projection(backbone(views))
            loss = supcon_loss(features, labels, float(cfg.supcon_temperature))
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()
            losses.append(loss.item())
        log.info("SupCon epoch %d: loss %.4f", epoch, sum(losses) / max(len(losses), 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone.state_dict(), out)


def freeze_encoder(model: nn.Module) -> None:
    """After SupCon: freeze the encoder and keep it in eval mode; only the head trains."""
    model.backbone.requires_grad_(False)
    model.mode = "linear_probe"  # BackboneClassifier.train() keeps a linear-probe backbone in eval
