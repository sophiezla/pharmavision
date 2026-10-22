"""E15 defences and arms race as training tasks (PLAN 7.10-7.11; docs/experiments/E15.md).

``adv_train``: PGD adversarial training (eps 4/255, 7 steps of eps/4, half of every batch
adversarial), optionally on top of the E14 synthetic images (``cfg.synthetic``, the "both"
defence).

``race``: one round r of the arms race for one (fold, seed) lineage. The previous detector
(round r - 1; round 0 = the final model's run of the same fold and seed) is attacked by the
score-only random search on every counterfeit of the run's *training* partition; each evasive
image found (model-input resolution) joins the lineage's pool, labelled counterfeit, inheriting
its source's group and fold. The new detector is trained from the pretrained weights on the
training partition plus the evasive images of rounds 1..r. Afterwards a fresh search on the
held-out counterfeits measures the evasion rate (``race.json``, ``race_heldout.csv``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image

from pharmachecked.adversarial.attacks import Normalized, adversarial_batch_hook
from pharmachecked.adversarial.blackbox import random_search
from pharmachecked.data.normalize import image_id_from_bytes
from pharmachecked.evaluation.aggregate import collect_runs
from pharmachecked.models.loading import load_torch_model, run_checkpoint
from pharmachecked.utils.io import read_json, write_json
from pharmachecked.utils.logging import get_logger
from pharmachecked.utils.run_dir import RunDir, slug

log = get_logger(__name__)
RACE_FILE = "race.json"
EVASIVE_COLUMNS = (
    "image_id",
    "source_image_id",
    "label",
    "split",
    "group_id",
    "fold",
    "source_dataset",
    "packaging_type",
)


def run_adv_train(cfg: DictConfig, run: RunDir) -> None:
    """PGD adversarial training (``cfg.adversarial``: eps, steps, fraction)."""
    from pharmachecked.training.task import run_training

    a = cfg.adversarial

    def hooks(model: torch.nn.Module, device: torch.device) -> dict:
        hook = adversarial_batch_hook(
            list(cfg.model.mean), list(cfg.model.std), float(a.eps), int(a.steps), float(a.fraction)
        )
        return {"batch_hook": hook}

    run_training(cfg, run, hooks=hooks)


def previous_run(cfg: DictConfig, runs_dir: Path) -> Path:
    """The detector the round attacks: round r - 1 of the lineage, or the final model's run.

    Raises:
        FileNotFoundError: If that run does not exist (yet).
    """
    r = int(cfg.race.round)
    fold, seed = int(cfg.fold), int(cfg.seed)
    if r == 1:
        exp, model, variant = (
            str(cfg.race.base_exp),
            str(cfg.race.base_model),
            str(cfg.race.base_variant),
        )
    else:
        exp, model, variant = str(cfg.exp), slug(cfg.model.name), f"round{r - 1}"
    for rec in collect_runs(runs_dir, exp):
        if (rec.model, rec.variant, rec.fold, rec.seed) == (model, variant, fold, seed):
            return rec.root
    raise FileNotFoundError(f"race round {r}: no {exp}/{model}/{variant} run for f{fold} s{seed}")


def lineage_dir(cfg: DictConfig) -> Path:
    """Evasive-image pool of this (fold, seed) lineage."""
    root = Path(cfg.paths.toy_dir) if cfg.data.dataset == "toy" else Path(cfg.paths.data_dir)
    return root / "processed" / "adversarial" / f"{cfg.exp}_f{cfg.fold}_s{cfg.seed}"


def evasive_images(
    pool: Path, rounds: int, train_ids: list[str]
) -> list[tuple[pd.DataFrame, list[str], Path]]:
    """Evasive images of rounds 1..``rounds`` whose source is in ``train_ids`` (Rule 3)."""
    out = []
    for k in range(1, rounds + 1):
        folder = pool / f"round{k}"
        t = pd.read_csv(
            folder / "evasive.csv",
            dtype={"image_id": str, "group_id": str, "source_image_id": str},
        )
        keep = t[t["source_image_id"].isin(set(train_ids))]
        out.append((t, list(keep["image_id"]), folder / "images"))
    return out


def _search(
    model_run: Path, cfg: DictConfig, ids: list[str], seed: int
) -> tuple[list[np.ndarray | None], pd.DataFrame]:
    """Random search against ``model_run``'s detector on ``ids`` (model-input images)."""
    from pharmachecked.adversarial.evaluate import unit_images
    from pharmachecked.models.loading import run_config
    from pharmachecked.training.task import device_for

    mcfg = run_config(model_run, cfg.paths, cfg.get("device"))
    ckpt = run_checkpoint(model_run)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint for {model_run.name}")
    dev = device_for(mcfg)
    model = load_torch_model(mcfg, ckpt).to(dev)
    wrapped = Normalized(model, list(mcfg.model.mean), list(mcfg.model.std)).to(dev).eval()
    threshold = float(read_json(model_run / "metrics.json")["heldout"]["threshold"])
    _, arrays = unit_images(mcfg, ids)
    found, rows = [], []
    for k, (image_id, arr) in enumerate(zip(ids, arrays, strict=True)):
        with torch.no_grad():
            x = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255
            p_clean = float(torch.sigmoid(wrapped.logit(x)))
        result = random_search(wrapped, arr, threshold, seed=seed * 100_003 + k)
        found.append(result.best_image if result.success else None)
        rows.append(
            {
                "image_id": image_id,
                "clean_prob": p_clean,
                "detected_clean": int(p_clean >= threshold),
                "evaded": int(result.success),
                "queries": result.queries,
                "best_prob": result.best_prob,
                "kind": result.kind,
            }
        )
    return found, pd.DataFrame(rows)


def run_race(cfg: DictConfig, run: RunDir) -> None:
    """One arms-race round (see the module docstring)."""
    from pharmachecked.training.task import load_data, partitions, run_training

    r = int(cfg.race.round)
    runs_dir = Path(cfg.paths.runs_dir)
    prev = previous_run(cfg, runs_dir)
    manifest, external, splits, _ = load_data(cfg)
    parts = partitions(cfg, splits, external, manifest)
    table = manifest.set_index("image_id")
    train_cf = [i for i in parts.train if int(table.loc[i, "label"]) == 1]
    pool = lineage_dir(cfg)
    round_dir = pool / f"round{r}"
    manifest_path = round_dir / "evasive.csv"
    if not manifest_path.is_file():  # resumable: the search of this round runs once
        found, searched = _search(prev, cfg, train_cf, seed=1000 * r + int(cfg.seed))
        (round_dir / "images").mkdir(parents=True, exist_ok=True)
        rows = []
        detected = searched["detected_clean"].to_numpy()
        for source, image, hit in zip(train_cf, found, detected, strict=True):
            if image is None or not hit:  # evasive = detected when clean, missed after the edit
                continue
            path_tmp = round_dir / "images" / "tmp.png"
            Image.fromarray(image).save(path_tmp)
            image_id = image_id_from_bytes(path_tmp.read_bytes())
            path_tmp.replace(round_dir / "images" / f"{image_id}.png")
            src = table.loc[source]
            rows.append(
                {
                    "image_id": image_id,
                    "source_image_id": source,
                    "label": 1,
                    "split": "synthetic",
                    "group_id": src["group_id"],
                    "fold": src["fold"],
                    "source_dataset": f"race_round{r}",
                    "packaging_type": src["packaging_type"],
                }
            )
        pd.DataFrame(rows, columns=list(EVASIVE_COLUMNS)).to_csv(
            manifest_path, index=False, lineterminator="\n"
        )

    run_training(cfg, run, extra=lambda p, _m: evasive_images(pool, r, list(p.train)))
    held = parts.heldout
    held_cf = [i for i in held if int(table.loc[i, "label"]) == 1]
    _, report = _search(run.root, cfg, held_cf, seed=7000 + 1000 * r + int(cfg.seed))
    report.to_csv(run.file("race_heldout.csv"), index=False, lineterminator="\n")
    detected = report[report["detected_clean"] == 1]
    sizes = [len(pd.read_csv(pool / f"round{k}" / "evasive.csv")) for k in range(1, r + 1)]
    write_json(
        run.file(RACE_FILE),
        {
            "round": r,
            "previous_run": prev.name,
            "train_counterfeits_attacked": len(train_cf),
            "evasive_added_per_round": sizes,
            "heldout_counterfeits": len(report),
            "detected_clean": len(detected),
            "evasion_rate": float(detected["evaded"].mean()) if len(detected) else float("nan"),
            "mean_queries_evaded": float(detected.loc[detected["evaded"] == 1, "queries"].mean())
            if int(detected["evaded"].sum())
            else float("nan"),
        },
    )
