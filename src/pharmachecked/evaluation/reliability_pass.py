"""Prediction-only pass for E10/E11 (ADR-026): one fold model, every split it needs.

Each trained fold model re-predicts its own ``train``, ``val``, ``cal`` and held-out images and
both OOD sets, without retraining, and writes ``reliability_predictions.csv`` into its run
folder. For torch models the penultimate embeddings feed two OOD scores fitted on the model's
own training images (Rule 3): Mahalanobis (class means, shared Ledoit-Wolf covariance) and kNN
(k = 50). Embeddings are not written. The handcrafted model has no embedding; its ``maha`` and
``knn`` columns are empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pharmachecked.config import CONFIG_DIR
from pharmachecked.data.ood import OOD_SETS, load_ood_set
from pharmachecked.evaluation.ood import KNN, Mahalanobis
from pharmachecked.utils.logging import get_logger

log = get_logger(__name__)

RELIABILITY_FILE = "reliability_predictions.csv"
DEV_PARTS = ("train", "val", "cal", "heldout")
COLUMNS = (
    "image_id",
    "split",
    "fold",
    "seed",
    "y_true",
    "logit",
    "prob",
    "maha",
    "knn",
    "source_dataset",
)
KNN_K = 50


def embed(
    model: nn.Module, ds: Dataset[dict[str, Any]], device: torch.device, cfg_train: Any
) -> tuple[pd.DataFrame, np.ndarray]:
    """Metadata, logit and probability per image, plus the penultimate embeddings."""
    from pharmachecked.training.trainer import autocast

    model.eval()
    rows, feats = [], []
    loader = DataLoader(ds, batch_size=cfg_train.eval_batch_size, num_workers=0)
    with torch.inference_mode():
        for batch in loader:
            with autocast(device, cfg_train.amp):
                out = model(batch["image"].to(device, non_blocking=True))
            logits = out["logits"].float().cpu()
            feats.append(out["embedding"].float().cpu().numpy())
            for k in range(len(logits)):
                rows.append(
                    {
                        "image_id": batch["image_id"][k],
                        "fold": int(batch["fold"][k]),
                        "y_true": int(batch["label"][k]),
                        "logit": float(logits[k]),
                        "prob": float(torch.sigmoid(logits[k])),
                        "source_dataset": batch["source_dataset"][k],
                    }
                )
    width = feats[0].shape[1] if feats else 0
    return pd.DataFrame(rows), np.concatenate(feats) if feats else np.zeros((0, width))


def _ood_dataset(
    cfg: DictConfig, tcfg: DictConfig, name: str, size: int, mean: list[float], std: list[float]
) -> Any:
    from pharmachecked.data.datasets import ManifestDataset
    from pharmachecked.data.transforms import build_eval_geometry, build_tensorize

    table, images = load_ood_set(Path(cfg.paths.data_dir), name)
    folder = Path(cfg.paths.results_dir) / "scratch" / "eval_cache" / f"ood_{tcfg.eval.mode}_{size}"
    return ManifestDataset(
        table,
        list(table["image_id"]),
        images,
        build_tensorize(mean, std),
        return_meta=True,
        geometry_cache=(build_eval_geometry(tcfg, size), folder),
    )


def _torch_pass(cfg: DictConfig, checkpoint: Path) -> pd.DataFrame:
    from pharmachecked.models.registry import build_model
    from pharmachecked.training.task import device_for, eval_dataset, load_data

    if "pretrained" in cfg.model:
        cfg.model.pretrained = False  # the checkpoint holds every weight
    if "backbone" in cfg.model:
        cfg.model.backbone.pretrained = False
    dev = device_for(cfg)
    model = build_model(cfg.model)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(dev)
    manifest, _, splits, images = load_data(cfg)
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    size = int(cfg.data.get("img_size") or cfg.model.img_size)
    mean, std = list(cfg.model.mean), list(cfg.model.std)
    parts: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for part in DEV_PARTS:
        ids = list(splits.ids(int(cfg.fold), part))
        ds = eval_dataset(cfg, tcfg, manifest, ids, images, mean, std, size)
        parts[part] = embed(model, ds, dev, cfg.train)
    for name in OOD_SETS:
        parts[f"ood_{name}"] = embed(
            model, _ood_dataset(cfg, tcfg, name, size, mean, std), dev, cfg.train
        )
    train_rows, train_x = parts["train"]
    maha = Mahalanobis().fit(train_x, train_rows["y_true"].to_numpy())
    k = min(KNN_K, len(train_x))
    if k < KNN_K:  # only on tiny (toy) training sets; real folds have about 330 images
        log.warning("kNN: %d training embeddings, using k=%d instead of %d", len(train_x), k, KNN_K)
    knn = KNN(k).fit(train_x)
    frames = [
        rows.assign(split=part, maha=maha.score(x), knn=knn.score(x))
        for part, (rows, x) in parts.items()
    ]
    return pd.concat(frames, ignore_index=True)


def _handcrafted_pass(cfg: DictConfig, checkpoint: Path) -> pd.DataFrame:
    import joblib

    from pharmachecked.training.task import handcrafted_features, load_data

    pipeline = joblib.load(checkpoint)
    manifest, _, splits, images = load_data(cfg)
    size = int(cfg.model.img_size)
    scratch = Path(cfg.paths.results_dir) / "scratch" / "features"
    table = manifest.set_index("image_id")
    sources: dict[str, tuple[pd.DataFrame, list[str], Path, Path]] = {}
    for part in DEV_PARTS:
        ids = list(splits.ids(int(cfg.fold), part))
        sources[part] = (table, ids, images, scratch / f"handcrafted_{size}")
    for name in OOD_SETS:
        ood, ood_images = load_ood_set(Path(cfg.paths.data_dir), name)
        ids = list(ood["image_id"])
        sources[f"ood_{name}"] = (
            ood.set_index("image_id"),
            ids,
            ood_images,
            scratch / f"handcrafted_ood_{size}",
        )
    frames = []
    for part, (meta, ids, folder, cache) in sources.items():
        prob = pipeline.predict_proba(handcrafted_features(ids, folder, cache, size))[:, 1]
        p = np.clip(prob, 1e-7, 1 - 1e-7)
        rows = meta.loc[ids]
        frames.append(
            pd.DataFrame(
                {
                    "image_id": ids,
                    "split": part,
                    "fold": rows["fold"].fillna(-1).astype(int).to_numpy(),
                    "y_true": rows["label"].astype(int).to_numpy(),
                    "logit": np.log(p / (1 - p)),
                    "prob": prob,
                    "maha": np.nan,
                    "knn": np.nan,
                    "source_dataset": rows["source_dataset"].to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def reliability_predictions(
    run_root: Path, paths: DictConfig, checkpoint: Path, device: str | None = None
) -> tuple[pd.DataFrame, float]:
    """Run the pass for one fold model and write ``reliability_predictions.csv``.

    Args:
        run_root: The run folder (``config.yaml`` and ``oof_predictions.csv`` are read).
        paths: Paths config of this machine.
        checkpoint: ``best.pt`` (task ``train``) or ``model.joblib`` (task ``handcrafted``).
        device: ``cuda`` / ``cpu``; default: cuda when available.

    Returns:
        The table and the largest absolute held-out probability difference against the run's
        ``oof_predictions.csv`` (a reproduction check).

    Raises:
        ValueError: For other tasks or protocols.
    """
    cfg = OmegaConf.load(run_root / "config.yaml")
    cfg.paths = paths
    cfg.device = device
    if (cfg.data.get("protocol") or "kfold") != "kfold":
        raise ValueError("the reliability pass needs a k-fold run (val/cal/held-out per fold)")
    if cfg.task == "train":
        out = _torch_pass(cfg, checkpoint)
    elif cfg.task == "handcrafted":
        out = _handcrafted_pass(cfg, checkpoint)
    else:
        raise ValueError(f"the reliability pass does not support task {cfg.task!r}")
    out = out.assign(seed=int(cfg.seed))[list(COLUMNS)]
    out.to_csv(run_root / RELIABILITY_FILE, index=False, lineterminator="\n", float_format="%.7g")
    held = out[out["split"] == "heldout"]
    oof = pd.read_csv(run_root / "oof_predictions.csv", dtype={"image_id": str})
    merged = held.merge(oof[["image_id", "prob"]], on="image_id", suffixes=("", "_oof"))
    if len(merged) != len(held) or len(merged) != len(oof):
        raise ValueError("re-predicted held-out images differ from oof_predictions.csv")
    return out, float((merged["prob"] - merged["prob_oof"]).abs().max())
