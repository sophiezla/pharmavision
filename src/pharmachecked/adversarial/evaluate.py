"""E15 attack pass over one trained fold model (PLAN 7.7-7.9; docs/experiments/E15.md).

On the run's held-out counterfeit images (model-input resolution, evaluation geometry) the pass
records the clean probability and the probability after each attack: FGSM and PGD-20 at
eps in {1, 2, 4, 8}/255 (targeted toward authentic), the colour and patch attacks, and the
score-only random search (200 queries). Success (evasion) = probability below the run's
val-selected threshold. Writes ``adversarial_predictions.csv`` into the run folder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from pharmachecked.adversarial.attacks import EPSILONS, Normalized, gradient_attack
from pharmachecked.adversarial.blackbox import random_search
from pharmachecked.adversarial.physical import color_attack, patch_attack
from pharmachecked.config import CONFIG_DIR
from pharmachecked.models.loading import load_torch_model, run_config
from pharmachecked.utils.io import read_json

ADV_FILE = "adversarial_predictions.csv"
BATCH = 16


def unit_images(cfg: DictConfig, ids: list[str]) -> tuple[torch.Tensor, list[np.ndarray]]:
    """Model-input images of development ids: float [0, 1] ``(N, 3, H, W)`` and uint8 arrays."""
    from pharmachecked.training.task import eval_dataset, load_data

    manifest, _, _, images = load_data(cfg)
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    size = int(cfg.data.get("img_size") or cfg.model.img_size)
    ds = eval_dataset(cfg, tcfg, manifest, ids, images, [0.0] * 3, [1.0] * 3, size)
    arrays = [np.asarray(ds._load(i)) for i in ids]
    x = torch.stack([torch.from_numpy(a).permute(2, 0, 1).float() / 255 for a in arrays])
    return x, arrays


def heldout_counterfeits(cfg: DictConfig) -> list[str]:
    """Counterfeit image ids of the run's held-out fold."""
    from pharmachecked.training.task import load_data, partitions

    manifest, external, splits, _ = load_data(cfg)
    held = partitions(cfg, splits, external, manifest).heldout
    labels = manifest.set_index("image_id").loc[held, "label"]
    return [i for i in held if int(labels[i]) == 1]


def attack_images(
    wrapped: Normalized, x: torch.Tensor, arrays: list[np.ndarray], threshold: float, seed: int
) -> pd.DataFrame:
    """Probabilities under every attack for a set of counterfeit images (long table)."""
    device = wrapped.mean.device
    rows = []

    def probs(batch: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(wrapped.logit(batch.to(device))).cpu().numpy()

    for start in range(0, len(x), BATCH):
        xb = x[start : start + BATCH].to(device)
        index = range(start, start + len(xb))
        results: list[tuple[str, float, np.ndarray]] = [("clean", 0.0, probs(xb))]
        for name in ("fgsm", "pgd"):
            for eps in EPSILONS:
                results.append((name, eps, probs(gradient_attack(wrapped, xb, name, eps))))
        results.append(("color", 0.0, probs(color_attack(wrapped, xb))))
        results.append(("patch", 0.0, probs(patch_attack(wrapped, xb, seed=seed + start))))
        for attack, eps, p in results:
            for k, value in zip(index, p, strict=True):
                rows.append(
                    {"k": k, "attack": attack, "eps": eps, "prob": float(value), "queries": 0}
                )
    for k, arr in enumerate(arrays):
        found = random_search(wrapped, arr, threshold, seed=seed * 100_003 + k)
        rows.append(
            {
                "k": k,
                "attack": "blackbox",
                "eps": 0.0,
                "prob": found.best_prob,
                "queries": found.queries,
            }
        )
    return pd.DataFrame(rows)


def adversarial_pass(
    run_root: Path, paths: DictConfig, checkpoint: Path, device: str | None = None
) -> pd.DataFrame:
    """Run every attack on one fold model's held-out counterfeits; write ``ADV_FILE``."""
    from pharmachecked.training.task import device_for

    cfg = run_config(run_root, paths, device)
    dev = device_for(cfg)
    model = load_torch_model(cfg, checkpoint).to(dev)
    model.requires_grad_(False)
    wrapped = Normalized(model, list(cfg.model.mean), list(cfg.model.std)).to(dev).eval()
    threshold = float(read_json(run_root / "metrics.json")["heldout"]["threshold"])
    ids = heldout_counterfeits(cfg)
    x, arrays = unit_images(cfg, ids)
    table = attack_images(wrapped, x, arrays, threshold, seed=int(cfg.seed))
    clean = table[table["attack"] == "clean"].set_index("k")["prob"]
    table = table.assign(
        image_id=[ids[k] for k in table["k"]],
        fold=int(cfg.fold),
        seed=int(cfg.seed),
        threshold=threshold,
        clean_prob=clean.loc[table["k"]].to_numpy(),
    )
    table["detected_clean"] = (table["clean_prob"] >= threshold).astype(int)
    table["evaded"] = (table["prob"] < threshold).astype(int)
    cols = ["image_id", "fold", "seed", "attack", "eps", "prob", "evaded", "queries"]
    out = table[[*cols, "clean_prob", "detected_clean", "threshold"]]
    out.to_csv(run_root / ADV_FILE, index=False, lineterminator="\n", float_format="%.6g")
    return out
