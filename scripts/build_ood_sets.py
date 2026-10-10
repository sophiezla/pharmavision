"""Download, sample and normalize the OOD reference sets (PLAN 1.9).

far:  DTD textures (torchvision download), stratified across the 47 texture classes.
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
from collections import defaultdict
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.data.dedup import pairwise_hamming, phash_hex, to_uint64
from pharmachecked.data.manifest import read_manifest
from pharmachecked.data.normalize import normalize_file
from pharmachecked.utils.io import atomic_write_text, sha256_file
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("build_ood_sets")
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
OOD = REPO_ROOT / "data" / "ood"


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
        by_class = finders[name]()
        available = sum(map(len, by_class.values()))
        picked = round_robin(by_class, int(src.n), int(cfg.seed))
        log.info(
            "%s: %d classes, %d images available, %d sampled",
            name,
            len(by_class),
            available,
            len(picked),
        )
        out_dir = OOD / src.set / "images"
        for cls, path in picked:
            image_id, w, h = normalize_file(path, out_dir)
            rows.append(
                {
                    "image_id": image_id,
                    "set": src.set,
                    "source_dataset": name,
                    "source_class": cls,
                    "original_file": path.relative_to(download_dir).as_posix(),
                    "width": w,
                    "height": h,
                    "licence": src.licence,
                    "url": src.url,
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
