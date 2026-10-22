"""E12 saliency pass over one trained fold model (PLAN 6.1-6.3; docs/experiments/E12.md).

For the run's held-out fold: deletion / insertion AUC of every method on every image
(``faithfulness.csv``). Optionally (fold 0, seed 0): cascading-randomization Spearman curves on
the first 50 held-out images in image-id order (``randomization.csv``) and the maps of four
example images (``saliency_examples.npz``: the first two counterfeit and first two authentic
held-out images in image-id order). Optionally, localization on the E14 synthetic counterfeits
whose source is in the run's held-out fold (``localization.csv``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.transforms import v2

from pharmachecked.config import CONFIG_DIR
from pharmachecked.data.datasets import ManifestDataset
from pharmachecked.data.transforms import build_eval_geometry, build_tensorize
from pharmachecked.evaluation.faithfulness import (
    curve_auc,
    perturbation_curves,
    pointing_hit,
    randomization_curve,
    top_iou,
)
from pharmachecked.interpret.cams import saliency
from pharmachecked.models.loading import load_torch_model, run_config

FAITH_FILE = "faithfulness.csv"
RANDOM_FILE = "randomization.csv"
EXAMPLES_FILE = "saliency_examples.npz"
LOCAL_FILE = "localization.csv"
N_RANDOMIZATION = 50
BATCH = 8


def _batches(ds: ManifestDataset) -> list[tuple[torch.Tensor, list[str], list[int]]]:
    out = []
    for start in range(0, len(ds), BATCH):
        items = [ds[k] for k in range(start, min(start + BATCH, len(ds)))]
        out.append(
            (
                torch.stack([it["image"] for it in items]),
                [it["image_id"] for it in items],
                [int(it["label"]) for it in items],
            )
        )
    return out


def mask_geometry(tcfg: DictConfig, size: int) -> v2.Compose:
    """The evaluation geometry with nearest-neighbour resampling, for binary masks."""
    nearest = OmegaConf.merge(tcfg, {"interpolation": "nearest"})
    return build_eval_geometry(nearest, size)


def e12_pass(
    run_root: Path,
    paths: DictConfig,
    checkpoint: Path,
    methods: tuple[str, ...],
    randomization: bool = False,
    examples: bool = False,
    localization: bool = False,
    device: str | None = None,
) -> None:
    """Run the requested parts of the pass for one fold model."""
    from pharmachecked.training.task import device_for, eval_dataset, load_data, partitions

    cfg = run_config(run_root, paths, device)
    dev = device_for(cfg)
    model = load_torch_model(cfg, checkpoint).to(dev)
    model.requires_grad_(False)
    manifest, external, splits, images = load_data(cfg)
    held = sorted(partitions(cfg, splits, external, manifest).heldout)
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    size = int(cfg.data.get("img_size") or cfg.model.img_size)
    mean, std = list(cfg.model.mean), list(cfg.model.std)
    ds = eval_dataset(cfg, tcfg, manifest, held, images, mean, std, size)
    fold, seed = int(cfg.fold), int(cfg.seed)

    rows = []
    for x, ids, labels in _batches(ds):
        x = x.to(dev)
        for method in methods:
            maps = saliency(method, model, x, ids)
            for k, image_id in enumerate(ids):
                deletion, insertion = perturbation_curves(model, x[k], maps[k].to(dev))
                rows.append(
                    {
                        "image_id": image_id,
                        "fold": fold,
                        "seed": seed,
                        "y_true": labels[k],
                        "method": method,
                        "deletion_auc": curve_auc(deletion),
                        "insertion_auc": curve_auc(insertion),
                    }
                )
    pd.DataFrame(rows).to_csv(
        run_root / FAITH_FILE, index=False, lineterminator="\n", float_format="%.6g"
    )

    if randomization:
        sub = eval_dataset(cfg, tcfg, manifest, held[:N_RANDOMIZATION], images, mean, std, size)
        x = torch.stack([sub[k]["image"] for k in range(len(sub))]).to(dev)
        ids = list(sub.image_ids)
        out = []
        for method in methods:
            if method == "random":
                continue  # independent of the weights by construction
            curve = randomization_curve(
                model,
                x,
                lambda m, inp, method=method: torch.cat(
                    [
                        saliency(method, m, inp[s : s + BATCH], ids[s : s + BATCH])
                        for s in range(0, len(inp), BATCH)
                    ]
                ),
            )
            out += [{"method": method, "n_images": len(ids), **row} for row in curve]
        pd.DataFrame(out).to_csv(
            run_root / RANDOM_FILE, index=False, lineterminator="\n", float_format="%.6g"
        )

    if examples:
        labels = manifest.set_index("image_id").loc[held, "label"]
        chosen = [i for i in held if labels[i] == 1][:2] + [i for i in held if labels[i] == 0][:2]
        sub = eval_dataset(cfg, tcfg, manifest, chosen, images, mean, std, size)
        x = torch.stack([sub[k]["image"] for k in range(len(sub))]).to(dev)
        arrays = {m: saliency(m, model, x, chosen).numpy().astype(np.float16) for m in methods}
        np.savez_compressed(run_root / EXAMPLES_FILE, image_ids=np.array(chosen), **arrays)

    if localization:
        from pharmachecked.synthesis.generate import synthetic_store

        store = synthetic_store(cfg)
        table = store.read()
        table = table[table["source_image_id"].isin(set(held))].sort_values("image_id")
        syn = ManifestDataset(
            table,
            list(table["image_id"]),
            store.images,
            v2.Compose([build_eval_geometry(tcfg, size), build_tensorize(mean, std)]),
            return_meta=True,
        )
        geom = mask_geometry(tcfg, size)
        meta = table.set_index("image_id")
        out = []
        for x, ids, _ in _batches(syn):
            x = x.to(dev)
            with torch.no_grad():
                probs = torch.sigmoid(model(x)["logits"].float()).cpu().numpy()
            masks = []
            for image_id in ids:
                with Image.open(store.masks / f"{image_id}.png") as im:
                    masks.append(torch.from_numpy(np.asarray(geom(im.convert("L"))) > 127))
            for method in methods:
                maps = saliency(method, model, x, ids)
                for k, image_id in enumerate(ids):
                    row = meta.loc[image_id]
                    out.append(
                        {
                            "image_id": image_id,
                            "source_image_id": row["source_image_id"],
                            "generator": row["generator"],
                            "edit_type": row["edit_type"],
                            "fold": fold,
                            "seed": seed,
                            "method": method,
                            "prob": float(probs[k]),
                            "mask_fraction": float(masks[k].float().mean()),
                            "hit": int(pointing_hit(maps[k], masks[k])),
                            "iou": top_iou(maps[k], masks[k]),
                        }
                    )
        pd.DataFrame(out).to_csv(
            run_root / LOCAL_FILE, index=False, lineterminator="\n", float_format="%.6g"
        )
