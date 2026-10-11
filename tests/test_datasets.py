"""Datasets never yield images from another partition or the locked test set; transforms behave."""

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from pharmachecked.config import CONFIG_DIR
from pharmachecked.data.datasets import ManifestDataset
from pharmachecked.data.manifest import read_manifest
from pharmachecked.data.splits import PARTS, load_dev_splits
from pharmachecked.data.toy import ToyData
from pharmachecked.data.transforms import (
    PadToSquare,
    build_eval_transform,
    build_train_transform,
)

TCFG = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def test_partition_datasets_yield_only_their_ids(toy: ToyData) -> None:
    manifest = read_manifest(toy.manifest)
    splits = load_dev_splits(toy.splits)
    for fold in range(5):
        for part in PARTS:
            ds = ManifestDataset.from_split(manifest, splits, fold, part, toy.images_dir)
            allowed = set(splits.ids(fold, part))
            seen = {ds[i]["image_id"] for i in range(len(ds))}
            assert seen == allowed
            others = set().union(*(splits.ids(fold, p) for p in PARTS if p != part))
            assert not seen & others


def test_locked_test_ids_are_refused(toy: ToyData) -> None:
    manifest = read_manifest(toy.manifest)
    test_ids = json.loads(toy.splits.read_text())["locked_test"]
    assert test_ids
    with pytest.raises(PermissionError):
        ManifestDataset(manifest, test_ids[:1], toy.images_dir)


def test_unknown_id_refused(toy: ToyData) -> None:
    with pytest.raises(KeyError):
        ManifestDataset(read_manifest(toy.manifest), ["0" * 16], toy.images_dir)


def test_external_dataset(toy: ToyData) -> None:
    ext = read_manifest(toy.external_manifest)
    ds = ManifestDataset.external(ext, toy.images_dir, sources=["toy_ext_cf"], return_meta=True)
    assert len(ds) == 8
    item = ds[0]
    assert item["label"] == 1
    assert item["source_dataset"] == "toy_ext_cf"
    assert item["fold"] == -1


def test_items_and_batches(toy: ToyData) -> None:
    manifest = read_manifest(toy.manifest)
    splits = load_dev_splits(toy.splits)
    tf = build_train_transform(TCFG, MEAN, STD, size=32)
    ds = ManifestDataset.from_split(manifest, splits, 0, "train", toy.images_dir, tf, True)
    batch = next(iter(DataLoader(ds, batch_size=4)))
    assert batch["image"].shape == (4, 3, 32, 32)
    assert batch["image"].dtype == torch.float32
    assert batch["label"].dtype == torch.int64
    assert len(batch["image_id"]) == 4
    raw = ManifestDataset.from_split(manifest, splits, 0, "train", toy.images_dir)[0]["image"]
    assert isinstance(raw, Image.Image)


@pytest.mark.parametrize("mode", ["crop", "pad"])
def test_eval_transform_is_deterministic_and_sized(mode: str) -> None:
    cfg = OmegaConf.merge(TCFG, {"eval": {"mode": mode}})
    tf = build_eval_transform(cfg, MEAN, STD)
    im = Image.new("RGB", (300, 180), (10, 200, 30))
    a, b = tf(im), tf(im)
    assert a.shape == (3, 224, 224)
    assert torch.equal(a, b)


def test_pad_to_square_keeps_whole_image() -> None:
    im = Image.new("RGB", (40, 10), (0, 0, 0))
    out = PadToSquare((255, 255, 255))(im)
    assert out.size == (40, 40)
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((20, 20)) == (0, 0, 0)


def test_train_transform_never_flips() -> None:
    """A left-right asymmetric image never comes out mirrored (no flip in the recipe)."""
    cfg = OmegaConf.merge(TCFG, {"train": {"rrc_scale": [1.0, 1.0], "rotation_degrees": 0}})
    tf = build_train_transform(cfg, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), size=32)
    im = Image.new("RGB", (32, 32), (0, 0, 0))
    for x in range(16):
        for y in range(32):
            im.putpixel((x, y), (255, 255, 255))  # left half white
    torch.manual_seed(0)
    for _ in range(20):
        out = tf(im)
        assert out[:, :, :8].mean() > out[:, :, -8:].mean()


def test_eval_mode_validated() -> None:
    with pytest.raises(ValueError):
        build_eval_transform(OmegaConf.merge(TCFG, {"eval": {"mode": "zoom"}}), MEAN, STD)


def test_geometry_cache_gives_identical_tensors(toy: ToyData, tmp_path: Path) -> None:
    from pharmachecked.data.transforms import build_eval_geometry, build_tensorize

    manifest = read_manifest(toy.manifest)
    splits = load_dev_splits(toy.splits)
    ids = list(splits.ids(0, "val"))
    plain = ManifestDataset(manifest, ids, toy.images_dir, build_eval_transform(TCFG, MEAN, STD))
    cache = tmp_path / "cache"
    cached = ManifestDataset(
        manifest,
        ids,
        toy.images_dir,
        build_tensorize(MEAN, STD),
        geometry_cache=(build_eval_geometry(TCFG), cache),
    )
    for k in range(len(ids)):
        first = cached[k]["image"]  # computes and stores
        assert torch.equal(first, plain[k]["image"])
        assert torch.equal(cached[k]["image"], first)  # served from the cache
    assert len(list(cache.glob("*.npy"))) == len(ids)
