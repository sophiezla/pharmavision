"""E13 concept-bottleneck training task (PLAN 6.5-6.6; docs/experiments/E13.md).

Modes (Koh et al., 2020), selected by ``cfg.cbm.mode``:

* ``independent``: concept model x -> c trained on the concept loss only; label head c -> y is a
  logistic regression fitted on the *annotated* concepts of the concept-labelled training
  images (unsure / missing values imputed with the training prevalence of that concept);
* ``sequential``: the same concept model; the label head is fitted on the *predicted* concept
  probabilities of every training image;
* ``joint``: end-to-end, loss = BCE(task) + ``cbm.concept_weight`` x masked concept BCE.

Concept stage (independent, sequential): early stopping on the masked concept BCE of the
concept-labelled ``val`` images (patience and epoch cap from ``train``). Joint: the usual
``fit`` (val AUROC, ADR-012 tie-break). The concept loss is masked where a label is missing or
``unsure``. Besides the standard outputs, ``concept_predictions.csv`` holds, per held-out image,
the predicted concept probabilities, the annotations and the task probability after
intervening on the k = 0..7 most uncertain annotated concepts (PLAN 6.6).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader

from pharmachecked.config import CONFIG_DIR
from pharmachecked.data.annotations import CONCEPTS
from pharmachecked.data.datasets import ManifestDataset
from pharmachecked.data.transforms import build_eval_transform
from pharmachecked.models.registry import build_model
from pharmachecked.training.optim import build_optimizer
from pharmachecked.training.trainer import autocast, fit, predict
from pharmachecked.utils.logging import get_logger
from pharmachecked.utils.run_dir import RunDir
from pharmachecked.utils.seed import make_generator, worker_init_fn

log = get_logger(__name__)
MODES = ("independent", "sequential", "joint")
CONCEPT_FILE = "concept_predictions.csv"


def concept_targets(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per image: targets (1 present, 0 absent) and mask (1 where present/absent) over CONCEPTS.

    Images with no definite value for any concept are left out.
    """
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    targets, masks = {}, {}
    for row in table.itertuples(index=False):
        values = [getattr(row, c) for c in CONCEPTS]
        mask = np.array([v in ("present", "absent") for v in values], dtype=np.float32)
        if mask.any():
            targets[row.image_id] = np.array([v == "present" for v in values], dtype=np.float32)
            masks[row.image_id] = mask
    return targets, masks


class ConceptLoss:
    """Masked concept BCE for a batch, looked up by image id."""

    def __init__(self, targets: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> None:
        self.targets, self.masks = targets, masks
        self.zero = np.zeros(len(CONCEPTS), dtype=np.float32)

    def __call__(self, image_ids: list[str], concept_logits: torch.Tensor) -> torch.Tensor:
        t = torch.tensor(np.stack([self.targets.get(i, self.zero) for i in image_ids]))
        m = torch.tensor(np.stack([self.masks.get(i, self.zero) for i in image_ids]))
        t, m = t.to(concept_logits.device), m.to(concept_logits.device)
        per = nn.functional.binary_cross_entropy_with_logits(
            concept_logits.float(), t, reduction="none"
        )
        return (per * m).sum() / m.sum().clamp_min(1.0)


@torch.inference_mode()
def concept_outputs(
    model: nn.Module, ds: object, device: torch.device, cfg_train: DictConfig
) -> tuple[list[str], np.ndarray]:
    """Image ids and predicted concept probabilities ``(N, 7)`` of a dataset."""
    model.eval()
    ids, probs = [], []
    for batch in DataLoader(ds, batch_size=cfg_train.eval_batch_size, num_workers=0):
        with autocast(device, cfg_train.amp):
            out = model(batch["image"].to(device))
        probs.append(out["concept_probs"].float().cpu().numpy())
        ids += list(batch["image_id"])
    return ids, np.concatenate(probs) if probs else np.zeros((0, len(CONCEPTS)))


def train_concepts(
    model: nn.Module,
    train_ds: object,
    val_ds: object,
    loss: ConceptLoss,
    cfg: DictConfig,
    device: torch.device,
    ckpt: Path,
) -> dict[str, float]:
    """Concept stage: minimize the masked concept BCE; keep the best val epoch."""
    t = cfg.train
    model.to(device)
    loader = DataLoader(
        train_ds,
        batch_size=t.batch_size,
        shuffle=True,
        drop_last=len(train_ds) > t.batch_size,
        num_workers=t.num_workers,
        worker_init_fn=worker_init_fn,
        generator=make_generator(int(cfg.seed)),
        persistent_workers=t.num_workers > 0,
    )
    opt, sched = build_optimizer(model, t, len(loader))
    best, best_epoch, start = float("inf"), -1, time.perf_counter()
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(t.max_epochs)):
        if best_epoch >= 0 and epoch - best_epoch > int(t.patience):
            break
        model.train()
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            with autocast(device, t.amp):
                out = model(batch["image"].to(device))
            value = loss(list(batch["image_id"]), out["concept_logits"])
            value.backward()
            if t.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            opt.step()
            sched.step()
        ids, probs = concept_outputs(model, val_ds, device, t)
        logits = torch.logit(torch.tensor(probs).clamp(1e-6, 1 - 1e-6))
        val_loss = float(loss(ids, logits)) if len(ids) else float("nan")
        if val_loss < best or best_epoch < 0:
            best, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), ckpt)
        log.info("concept epoch %d: val concept BCE %.4f", epoch, val_loss)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return {
        "concept_best_epoch": best_epoch,
        "concept_val_bce": best,
        "concept_seconds": time.perf_counter() - start,
    }


def fit_label_head(model: nn.Module, x: np.ndarray, y: np.ndarray, seed: int) -> None:
    """Logistic regression c -> y, copied into ``model.label_head``."""
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(x, y)
    with torch.no_grad():
        model.label_head.weight.copy_(torch.tensor(clf.coef_, dtype=torch.float32))
        model.label_head.bias.copy_(torch.tensor(clf.intercept_, dtype=torch.float32))


def interventions(probs: np.ndarray, annotated: np.ndarray, head: nn.Linear) -> np.ndarray:
    """Task probabilities after replacing the k most uncertain annotated concepts, k = 0..7.

    Args:
        probs: Predicted concept probabilities ``(N, 7)``.
        annotated: 1 / 0 / nan annotations ``(N, 7)``.
        head: The label head.

    Returns:
        ``(N, 8)`` task probabilities; for k beyond the number of annotated concepts of an image,
        every annotated concept is replaced.
    """
    w = head.weight.detach().cpu().numpy()[0]
    b = float(head.bias.detach().cpu().numpy()[0])
    out = np.zeros((len(probs), len(CONCEPTS) + 1))
    for i, (p, a) in enumerate(zip(probs, annotated, strict=True)):
        known = np.flatnonzero(~np.isnan(a))
        order = known[np.argsort(np.abs(p[known] - 0.5), kind="stable")]
        for k in range(len(CONCEPTS) + 1):
            c = p.copy()
            c[order[:k]] = a[order[:k]]
            out[i, k] = 1 / (1 + np.exp(-(c @ w + b)))
    return out


def run_cbm(cfg: DictConfig, run: RunDir) -> None:
    """Train one concept-bottleneck fold model and write the standard outputs + concepts."""
    from pharmachecked.training.task import device_for, fold_datasets, write_outputs

    mode = str(cfg.cbm.mode)
    if mode not in MODES:
        raise ValueError(f"unknown CBM mode {mode!r}")
    targets, masks = concept_targets(Path(cfg.cbm.concepts_file))
    if not targets:
        raise ValueError(f"{cfg.cbm.concepts_file} has no concept labels (PLAN 1.8)")
    loss = ConceptLoss(targets, masks)
    data = fold_datasets(cfg)
    device = device_for(cfg)
    model = build_model(cfg.model)
    ckpt_dir = Path(cfg.paths.checkpoints_dir) / run.run_id
    train_ids = list(data.train.image_ids)
    labelled_train = [i for i in train_ids if i in targets]
    extra: dict[str, object] = {"n_train": len(train_ids), "n_train_concepts": len(labelled_train)}
    start = time.perf_counter()
    if mode == "joint":
        weight = float(cfg.cbm.concept_weight)
        bce = nn.BCEWithLogitsLoss()

        def loss_hook(batch: dict, out: dict, y: torch.Tensor) -> torch.Tensor:
            task = bce(out["logits"].float(), y)
            return task + weight * loss(list(batch["image_id"]), out["concept_logits"])

        result = fit(
            model,
            data.train,
            data.val,
            cfg.train,
            device,
            ckpt_dir,
            run.file("train_log.csv"),
            int(cfg.seed),
            loss_hook=loss_hook,
        )
        extra.update(best_epoch=result.best_epoch, val_auroc_best=result.best_val_auroc)
    else:

        def subset(ds: ManifestDataset, ids: list[str]) -> ManifestDataset:
            manifest = pd.DataFrame(
                {"image_id": list(ds.image_ids), "label": list(ds.labels), "split": "dev"}
            ).join(pd.DataFrame(ds._meta))
            return ManifestDataset(manifest, ids, ds.images_dir, ds.transform, True)

        val_labelled = [i for i in data.val.image_ids if i in targets]
        stats = train_concepts(
            model,
            subset(data.train, labelled_train),
            subset(data.val, val_labelled),
            loss,
            cfg,
            device,
            ckpt_dir / "concepts.pt",
        )
        extra.update(stats)
        y_of = dict(zip(data.train.image_ids, data.train.labels, strict=True))
        if mode == "independent":
            prevalence = np.array(
                [
                    np.mean([targets[i][c] for i in labelled_train if masks[i][c]] or [0.5])
                    for c in range(len(CONCEPTS))
                ]
            )
            x = np.stack([np.where(masks[i] > 0, targets[i], prevalence) for i in labelled_train])
            y = np.array([y_of[i] for i in labelled_train])
        else:
            tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
            eval_train = subset(data.train, train_ids)
            eval_train.transform = build_eval_transform(
                tcfg, list(cfg.model.mean), list(cfg.model.std), int(cfg.model.img_size)
            )
            ids, x = concept_outputs(model, eval_train, device, cfg.train)
            y = np.array([y_of[i] for i in ids])
        fit_label_head(model, x, y, int(cfg.seed))
        torch.save(model.state_dict(), ckpt_dir / "best.pt")
        pd.DataFrame(
            [{"epoch": extra["concept_best_epoch"], "concept_val_bce": extra["concept_val_bce"]}]
        ).to_csv(run.file("train_log.csv"), index=False, lineterminator="\n")
    extra["train_seconds"] = time.perf_counter() - start
    preds = [
        predict(model, ds, device, cfg.train, int(cfg.seed))
        for ds in (data.val, data.heldout, data.external)
    ]
    write_outputs(run, cfg, *preds, checkpoint=ckpt_dir / "best.pt", extra=extra)
    ids, probs = concept_outputs(model, data.heldout, device, cfg.train)
    annotated = np.stack(
        [
            np.where(masks[i] > 0, targets[i], np.nan) if i in targets else np.full(7, np.nan)
            for i in ids
        ]
    )
    task = interventions(probs, annotated, model.label_head)
    y_of = dict(zip(data.heldout.image_ids, data.heldout.labels, strict=True))
    table = pd.DataFrame({"image_id": ids, "y_true": [y_of[i] for i in ids]})
    for c, name in enumerate(CONCEPTS):
        table[f"pred_{name}"] = probs[:, c]
        table[f"true_{name}"] = annotated[:, c]
    for k in range(len(CONCEPTS) + 1):
        table[f"prob_k{k}"] = task[:, k]
    table.to_csv(run.file(CONCEPT_FILE), index=False, lineterminator="\n", float_format="%.6g")
