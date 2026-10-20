"""One-class task (E8, PLAN 4.6): PatchCore or the prototype verifier, fitted on authentic images.

Neither method sees a counterfeit image while fitting: the reference set is the authentic part
of the run's training split. Scores become ``prob`` through a monotone map into [0, 1)
(prototype: cosine distance / 2; PatchCore: score / (1 + score)), so ranking metrics (AUROC,
AUPRC, sensitivity at 95% specificity) are exact while probability metrics (ECE, Brier) are not
meaningful and are not reported for these methods. The threshold is picked on val (Youden), as
for every other task, and the outputs follow CLAUDE.md section 7.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from pharmachecked.config import CONFIG_DIR
from pharmachecked.data.datasets import ManifestDataset
from pharmachecked.models.backbones import make_backbone
from pharmachecked.models.prototype import PrototypeVerifier
from pharmachecked.training.task import (
    _check_threshold_rule,
    _write_single_row_log,
    device_for,
    eval_dataset,
    load_data,
    partitions,
    write_outputs,
)
from pharmachecked.training.trainer import PRED_COLUMNS
from pharmachecked.utils.logging import get_logger
from pharmachecked.utils.run_dir import RunDir

log = get_logger(__name__)


def _meta_rows(batch: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    return [
        {
            "image_id": batch["image_id"][k],
            "fold": int(batch["fold"][k]),
            "seed": seed,
            "y_true": int(batch["label"][k]),
            "packaging_type": batch["packaging_type"][k],
            "source_dataset": batch["source_dataset"][k],
        }
        for k in range(len(batch["image_id"]))
    ]


@torch.inference_mode()
def _embed(
    net: nn.Module, ds: ManifestDataset, device: torch.device, batch_size: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Metadata rows and frozen-backbone embeddings of ``ds``."""
    rows, feats = [], []
    for batch in DataLoader(ds, batch_size=batch_size, num_workers=0):
        feats.append(net(batch["image"].to(device)).float().cpu().numpy())
        rows += _meta_rows(batch, seed)
    width = feats[0].shape[1] if feats else 0
    return pd.DataFrame(rows), np.concatenate(feats) if feats else np.zeros((0, width))


def _predictions(rows: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=list(PRED_COLUMNS))
    p = np.clip(prob.astype(float), 1e-7, 1 - 1e-7)
    out = rows.assign(prob=prob.astype(float), logit=np.log(p / (1 - p)))
    return out[list(PRED_COLUMNS)]


def patchcore_prob(score: np.ndarray) -> np.ndarray:
    """Monotone map of PatchCore anomaly scores (>= 0) into [0, 1).

    ``1 - exp(-score)`` rounds to exactly 1.0 for scores above about 37, which PatchCore's raw
    feature distances exceed, so every image would tie; ``score / (1 + score)`` does not
    saturate in float64 at these magnitudes.
    """
    score = score.astype(float)
    return score / (1.0 + score)


class _PatchCore:
    """anomalib's PatchCore torch model used directly (no Lightning engine).

    Roth et al., 2022, "Towards Total Recall in Industrial Anomaly Detection". anomalib 2.7.0
    API (checked 2026-10-19): in train mode ``forward`` stores patch embeddings,
    ``subsample_embedding(ratio)`` builds the coreset memory bank, and in eval mode ``forward``
    returns an ``InferenceBatch`` whose ``pred_score`` is the image anomaly score.
    """

    def __init__(self, cfg: Any, device: torch.device) -> None:
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        self.model = PatchcoreModel(
            layers=list(cfg.layers),
            backbone=str(cfg.backbone_name),
            pre_trained=bool(cfg.get("pretrained", True)),
            num_neighbors=int(cfg.num_neighbors),
        ).to(device)
        self.device, self.ratio = device, float(cfg.coreset_ratio)

    @torch.inference_mode()
    def fit(self, loader: DataLoader) -> None:
        self.model.train()
        for batch in loader:
            self.model(batch["image"].to(self.device))
        self.model.subsample_embedding(self.ratio)
        self.model.eval()

    @torch.inference_mode()
    def score(
        self, ds: ManifestDataset, batch_size: int, seed: int
    ) -> tuple[pd.DataFrame, np.ndarray]:
        rows, scores = [], []
        for batch in DataLoader(ds, batch_size=batch_size, num_workers=0):
            out = self.model(batch["image"].to(self.device))
            scores.append(out.pred_score.float().reshape(-1).cpu().numpy())
            rows += _meta_rows(batch, seed)
        return pd.DataFrame(rows), np.concatenate(scores) if scores else np.zeros(0)


def run_one_class(cfg: DictConfig, run: RunDir) -> None:
    """Fit ``cfg.model`` (``kind: prototype | patchcore``) on authentic training images."""
    _check_threshold_rule(cfg)
    torch.manual_seed(int(cfg.seed))
    manifest, external, splits, images = load_data(cfg)
    parts = partitions(cfg, splits, external, manifest)
    labels = manifest.set_index("image_id")["label"]
    authentic = [i for i in parts.train if int(labels[i]) == 0]
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    size = int(cfg.model.img_size)
    mean, std = list(cfg.model.mean), list(cfg.model.std)

    def dataset(table: pd.DataFrame, ids: list[str]) -> ManifestDataset:
        return eval_dataset(cfg, tcfg, table, ids, images, mean, std, size)

    device, bs = device_for(cfg), int(cfg.train.eval_batch_size)
    ckpt_dir = Path(cfg.paths.checkpoints_dir) / run.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    targets = ((manifest, parts.val), (manifest, parts.heldout), (external, parts.external))
    if cfg.model.kind == "prototype":
        net, _ = make_backbone(cfg.model.backbone)
        net = net.to(device).eval()
        _, ref = _embed(net, dataset(manifest, authentic), device, bs, int(cfg.seed))
        verifier = PrototypeVerifier(int(cfg.model.k)).fit(ref)
        fit_seconds = time.perf_counter() - start
        np.save(ckpt_dir / "reference.npy", verifier.reference)
        preds = []
        for table, ids in targets:
            rows, emb = _embed(net, dataset(table, ids), device, bs, int(cfg.seed))
            prob = verifier.score(emb) / 2.0 if len(emb) else np.zeros(0)
            preds.append(_predictions(rows, prob))
        checkpoint = ckpt_dir / "reference.npy"
    elif cfg.model.kind == "patchcore":
        pc = _PatchCore(cfg.model, device)
        pc.fit(DataLoader(dataset(manifest, authentic), batch_size=bs, num_workers=0))
        fit_seconds = time.perf_counter() - start
        torch.save(pc.model.memory_bank.cpu(), ckpt_dir / "memory_bank.pt")
        preds = []
        for table, ids in targets:
            rows, score = pc.score(dataset(table, ids), bs, int(cfg.seed))
            preds.append(_predictions(rows, patchcore_prob(score)))
        checkpoint = ckpt_dir / "memory_bank.pt"
    else:
        raise ValueError(f"unknown one-class kind {cfg.model.kind!r}")
    log.info(
        "%s fitted on %d authentic images in %.1f s", cfg.model.name, len(authentic), fit_seconds
    )
    _write_single_row_log(
        run, {"epoch": 0, "fit_seconds": fit_seconds, "n_reference": len(authentic)}
    )
    write_outputs(
        run,
        cfg,
        *preds,
        checkpoint=checkpoint,
        extra={"n_train": len(authentic), "train_seconds": fit_seconds, "device": str(device)},
    )
