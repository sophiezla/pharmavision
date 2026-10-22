"""Testing with concept activation vectors (Kim et al., 2018; PLAN 6.7, E13).

For one fold model of the final model and a layer L: a concept activation vector (CAV) is the
unit normal of a logistic regression separating the layer activations (flattened) of
concept-positive images (annotated ``present``) from a random set of the same size. The TCAV
score is the fraction of the fold's held-out counterfeit images whose directional derivative of
the counterfeit logit along the CAV is positive. Repeated against ``N_RANDOM`` random negative
sets; the null distribution comes from CAVs between pairs of random sets. Two-sided Welch
t-test of concept vs random scores; Holm correction over concepts x layers within the fold
model. Positive and random images come from the fold's training partition only; a concept needs
at least ``MIN_POSITIVES`` annotated positives there (otherwise it is reported as skipped).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from torch import nn

from pharmachecked.config import CONFIG_DIR
from pharmachecked.data.annotations import CONCEPTS
from pharmachecked.evaluation.stats import holm
from pharmachecked.interpret.cams import unwrap_backbone
from pharmachecked.models.loading import load_torch_model, run_config

TCAV_FILE = "tcav.csv"
N_RANDOM = 20
MIN_POSITIVES = 10
MAX_POSITIVES = 50
LAYERS = ("blocks.3", "blocks.4", "blocks.5")  # last three stages of EfficientNetV2-S


def _layer(model: nn.Module, name: str) -> nn.Module:
    module: nn.Module = unwrap_backbone(model)
    for part in name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def activations(model: nn.Module, layer: nn.Module, x: torch.Tensor, batch: int = 16) -> np.ndarray:
    """Flattened activations of ``layer`` for inputs ``x``."""
    store: list[torch.Tensor] = []
    handle = layer.register_forward_hook(lambda _m, _i, out: store.append(out.detach().float()))
    try:
        with torch.no_grad():
            for s in range(0, len(x), batch):
                model(x[s : s + batch])
    finally:
        handle.remove()
    return torch.cat(store).flatten(1).cpu().numpy()


def gradients(model: nn.Module, layer: nn.Module, x: torch.Tensor, batch: int = 8) -> np.ndarray:
    """d(counterfeit logit)/d(activation of ``layer``), flattened, per input."""
    grads: list[np.ndarray] = []
    for s in range(0, len(x), batch):
        held: list[torch.Tensor] = []

        def keep(_m: nn.Module, _i: object, out: torch.Tensor, held: list = held) -> None:
            out.retain_grad()
            held.append(out)

        handle = layer.register_forward_hook(keep)
        try:
            inp = x[s : s + batch].clone().requires_grad_(True)
            model(inp)["logits"].float().sum().backward()
        finally:
            handle.remove()
        grads.append(held[0].grad.detach().float().flatten(1).cpu().numpy())
    return np.concatenate(grads)


def cav(pos: np.ndarray, neg: np.ndarray, seed: int) -> np.ndarray:
    """Unit CAV pointing toward ``pos``."""
    x = np.concatenate([pos, neg])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(x, y)
    w = clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-12)


def tcav_score(grads: np.ndarray, vector: np.ndarray) -> float:
    """Fraction of positive directional derivatives."""
    return float((grads @ vector > 0).mean())


def tcav_pass(
    run_root: Path,
    paths: DictConfig,
    checkpoint: Path,
    concepts_file: Path,
    device: str | None = None,
) -> pd.DataFrame:
    """TCAV for one fold model; writes ``tcav.csv`` into the run folder."""
    from pharmachecked.training.cbm import concept_targets
    from pharmachecked.training.task import device_for, eval_dataset, load_data

    cfg = run_config(run_root, paths, device)
    dev = device_for(cfg)
    model = load_torch_model(cfg, checkpoint).to(dev)
    model.requires_grad_(False)
    manifest, _, splits, images = load_data(cfg)
    fold, seed = int(cfg.fold), int(cfg.seed)
    train = sorted(splits.ids(fold, "train"))
    held = splits.ids(fold, "heldout")
    labels = manifest.set_index("image_id")["label"]
    held_cf = sorted(i for i in held if labels[i] == 1)
    targets, masks = concept_targets(concepts_file)
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    size = int(cfg.data.get("img_size") or cfg.model.img_size)

    def inputs(ids: list[str]) -> torch.Tensor:
        ds = eval_dataset(
            cfg, tcfg, manifest, ids, images, list(cfg.model.mean), list(cfg.model.std), size
        )
        return torch.stack([ds[k]["image"] for k in range(len(ds))]).to(dev)

    rng = np.random.default_rng(seed)
    positives = {
        c: [i for i in train if i in targets and masks[i][k] and targets[i][k]][:MAX_POSITIVES]
        for k, c in enumerate(CONCEPTS)
    }
    n_set = max([len(v) for v in positives.values() if len(v) >= MIN_POSITIVES] or [MIN_POSITIVES])
    n_set = min(n_set, len(train) // 2)
    random_sets = [
        sorted(rng.choice(train, size=n_set, replace=False)) for _ in range(N_RANDOM + 1)
    ]
    x_cf = inputs(held_cf)
    rows = []
    for layer_name in LAYERS:
        layer = _layer(model, layer_name)
        g = gradients(model, layer, x_cf)
        rand_acts = [activations(model, layer, inputs(list(s))) for s in random_sets]
        null = [tcav_score(g, cav(rand_acts[r], rand_acts[r + 1], r)) for r in range(N_RANDOM)]
        for concept, pos_ids in positives.items():
            base = {"fold": fold, "seed": seed, "layer": layer_name, "concept": concept}
            if len(pos_ids) < MIN_POSITIVES:
                rows.append({**base, "n_positive": len(pos_ids), "status": "skipped"})
                continue
            pos = activations(model, layer, inputs(pos_ids))
            scores = [tcav_score(g, cav(pos, rand_acts[r][: len(pos)], r)) for r in range(N_RANDOM)]
            p = float(ttest_ind(scores, null, equal_var=False).pvalue)
            rows.append(
                {
                    **base,
                    "n_positive": len(pos_ids),
                    "status": "tested",
                    "tcav_mean": float(np.mean(scores)),
                    "tcav_sd": float(np.std(scores, ddof=1)),
                    "random_mean": float(np.mean(null)),
                    "random_sd": float(np.std(null, ddof=1)),
                    "p": p,
                    "n_heldout_counterfeit": len(held_cf),
                }
            )
    table = pd.DataFrame(rows)
    tested = table["status"] == "tested"
    if tested.any():
        reject, p_holm = holm(list(table.loc[tested, "p"]))
        table.loc[tested, "p_holm"] = p_holm
        table.loc[tested, "significant"] = [int(v) for v in reject]
    table.to_csv(run_root / TCAV_FILE, index=False, lineterminator="\n", float_format="%.6g")
    return table
