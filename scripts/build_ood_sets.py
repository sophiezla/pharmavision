"""Download, sample and normalize the OOD reference sets (PLAN 1.9).

far:  DTD textures (torchvision download), stratified across the 47 texture classes, and
      Open Images V7 validation scene photos (human-verified scene labels, CC BY 2.0 only,
      none verified to show medicine or packaging), spread over scene labels.
near: packaged products ("Packages" classes) of the Grocery Store Dataset (git clone).

Images are normalized like the main pipeline into data/ood/{far,near}/images/{image_id}.png
(git-ignored); data/ood/{far,near}/manifest.csv (committed) records source, class, original file,
licence and URL. results/tables/ood_sets.md reports counts and the nearest pHash distance to the
development pool.

Usage:
    python scripts/build_ood_sets.py
"""

from __future__ import annotations

import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.data.dedup import pairwise_hamming, phash_hex, to_uint64
from pharmachecked.data.manifest import read_manifest
from pharmachecked.data.normalize import normalize_file
from pharmachecked.utils.io import atomic_write_text, sha256_file
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("build_ood_sets")
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
OOD = REPO_ROOT / "data" / "ood"
CLASS_DESCRIPTIONS = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"


def round_robin(by_class: dict[str, list[Path]], n: int, seed: int) -> list[tuple[str, Path]]:
    """Sample ``n`` files spreading evenly over classes (deterministic for a seed)."""
    rng = random.Random(seed)
    pools = {c: rng.sample(sorted(fs), len(fs)) for c, fs in sorted(by_class.items())}
    picked: list[tuple[str, Path]] = []
    while len(picked) < n and any(pools.values()):
        for c in sorted(pools):
            if pools[c] and len(picked) < n:
                picked.append((c, pools[c].pop()))
    return picked


def dtd_files(download_dir: Path) -> dict[str, list[Path]]:
    """Download DTD through torchvision; return image files per texture class."""
    from torchvision.datasets import DTD

    DTD(root=str(download_dir), split="train", download=True)
    images = download_dir / "dtd" / "dtd" / "images"
    return {
        d.name: [p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXT]
        for d in sorted(images.iterdir())
        if d.is_dir()
    }


def grocery_files(download_dir: Path, url: str) -> dict[str, list[Path]]:
    """Clone the Grocery Store Dataset; return unique Packages images per fine class."""
    repo = download_dir / "GroceryStoreDataset"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", url, str(repo)], check=True)
    by_class: dict[str, list[Path]] = defaultdict(list)
    seen: set[str] = set()
    for path in sorted((repo / "dataset").rglob("*")):
        parts = path.parts
        if path.suffix.lower() not in IMAGE_EXT or "Packages" not in parts:
            continue
        digest = sha256_file(path)
        if digest in seen:
            continue
        seen.add(digest)
        fine = parts[parts.index("Packages") + 1]
        by_class[fine].append(path)
    return dict(by_class)


def _fetch(url: str, dest: Path, attempts: int = 5) -> Path:
    """Download ``url`` to ``dest`` once (cached), retrying dropped connections with backoff."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            break
        except (OSError, urllib.error.URLError) as exc:
            if attempt == attempts:
                raise
            log.warning("download failed (%s), retry %d/%d: %s", exc, attempt, attempts - 1, url)
            time.sleep(2**attempt)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def openimages_candidates(src: DictConfig, download_dir: Path) -> pd.DataFrame:
    """Validation images with a verified scene label, no excluded label, CC BY 2.0, no rotation.

    Returns one row per (image, scene label) with author and licence metadata.
    """
    root = download_dir / "openimages"
    meta = pd.read_csv(
        _fetch(src.metadata, root / Path(src.metadata).name), dtype=str, keep_default_na=False
    )
    labels = pd.read_csv(_fetch(src.labels, root / Path(src.labels).name), dtype=str)
    classes = pd.read_csv(
        _fetch(CLASS_DESCRIPTIONS, root / Path(CLASS_DESCRIPTIONS).name), dtype=str
    )
    name_of = dict(zip(classes["LabelName"], classes["DisplayName"], strict=True))
    labels["name"] = labels["LabelName"].map(name_of)
    positive = labels[labels["Confidence"] == "1"]
    excluded = set(positive.loc[positive["name"].isin(list(src.exclude_labels)), "ImageID"])
    scenes = positive[positive["name"].isin(list(src.scene_labels))]
    scenes = scenes[~scenes["ImageID"].isin(excluded)]
    ok_meta = meta[
        (meta["License"] == src.accepted_licence) & meta["Rotation"].isin(["", "0", "0.0"])
    ]
    return scenes.merge(ok_meta, on="ImageID")[
        ["ImageID", "name", "Author", "License", "OriginalLandingURL"]
    ]


def openimages_files(src: DictConfig, download_dir: Path, seed: int) -> list[dict[str, object]]:
    """Sample ``src.n`` scene images round-robin over scene labels and download them."""
    cand = openimages_candidates(src, download_dir)
    by_label: dict[str, list[str]] = defaultdict(list)
    for image_id, name in zip(cand["ImageID"], cand["name"], strict=True):
        by_label[name].append(image_id)
    rng = random.Random(seed)
    pools = {k: rng.sample(sorted(set(v)), len(set(v))) for k, v in sorted(by_label.items())}
    chosen: dict[str, str] = {}
    while len(chosen) < int(src.n) and any(pools.values()):
        for name in sorted(pools):
            while pools[name] and len(chosen) < int(src.n):
                image_id = pools[name].pop()
                if image_id not in chosen:
                    chosen[image_id] = name
                    break
    info = cand.drop_duplicates("ImageID").set_index("ImageID")
    out = []
    for image_id, name in chosen.items():
        path = _fetch(
            src.image_url.format(image_id=image_id),
            download_dir / "openimages" / "images" / f"{image_id}.jpg",
        )
        out.append(
            {
                "class": name,
                "path": path,
                "author": info.loc[image_id, "Author"],
                "licence": info.loc[image_id, "License"],
                "url": info.loc[image_id, "OriginalLandingURL"],
            }
        )
    log.info("openimages_scenes: %d candidate images, %d sampled", cand.ImageID.nunique(), len(out))
    return out


def main() -> int:
    """Build both sets; returns a process exit code."""
    configure_logging()
    cfg = OmegaConf.load(CONFIG_DIR / "data" / "ood.yaml")
    download_dir = REPO_ROOT / cfg.download_dir
    download_dir.mkdir(parents=True, exist_ok=True)
    finders = {
        "dtd": lambda: dtd_files(download_dir),
        "grocery_store_packages": lambda: grocery_files(
            download_dir, cfg.sources.grocery_store_packages.url
        ),
    }
    rows = []
    for name, src in cfg.sources.items():
        if name == "openimages_scenes":
            picked = openimages_files(src, download_dir, int(cfg.seed))
        else:
            by_class = finders[name]()
            picked = [
                {"class": c, "path": p, "author": "", "licence": src.licence, "url": src.url}
                for c, p in round_robin(by_class, int(src.n), int(cfg.seed))
            ]
            log.info(
                "%s: %d classes, %d images available, %d sampled",
                name,
                len(by_class),
                sum(map(len, by_class.values())),
                len(picked),
            )
        out_dir = OOD / src.set / "images"
        for pick in picked:
            path = Path(str(pick["path"]))
            image_id, w, h = normalize_file(path, out_dir)
            rows.append(
                {
                    "image_id": image_id,
                    "set": src.set,
                    "source_dataset": name,
                    "source_class": pick["class"],
                    "original_file": path.relative_to(download_dir).as_posix(),
                    "width": w,
                    "height": h,
                    "author": pick["author"],
                    "licence": pick["licence"],
                    "url": pick["url"],
                }
            )
    df = pd.DataFrame(rows).drop_duplicates("image_id")
    for ood_set, part in df.groupby("set"):
        (OOD / ood_set).mkdir(parents=True, exist_ok=True)
        part.sort_values("image_id").to_csv(
            OOD / ood_set / "manifest.csv", index=False, lineterminator="\n"
        )

    dev = read_manifest(REPO_ROOT / "data" / "processed" / "manifest_v1.csv")
    dev_hash = to_uint64(list(dev["phash"]))
    lines = [
        "# OOD reference sets (PLAN 1.9)",
        "",
        "Generated by `scripts/build_ood_sets.py` (`configs/data/ood.yaml`).",
        "Images are local only; identical files sampled twice are kept once.",
        "",
        "| Set | Source | Classes | Images | Smallest pHash distance to a development image |",
        "|---|---|---|---|---|",
    ]
    for (ood_set, name), part in df.groupby(["set", "source_dataset"]):
        hashes = [phash_hex(OOD / ood_set / "images" / f"{i}.png") for i in part["image_id"]]
        d = pairwise_hamming(to_uint64(hashes), dev_hash)
        lines.append(
            f"| {ood_set} | {name} | {part.source_class.nunique()} | {len(part)} | {int(d.min())} |"
        )
    atomic_write_text(REPO_ROOT / "results" / "tables" / "ood_sets.md", "\n".join(lines) + "\n")
    log.info("wrote manifests and results/tables/ood_sets.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
