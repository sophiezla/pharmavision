"""Normalize raw images and build the v1 manifests with near-duplicate groups (PLAN 1.2-1.4).

Outputs:
    data/processed/images/{image_id}.png      normalized images (git-ignored)
    data/processed/manifest_v1.csv            development pool (split/fold filled by 1.5)
    data/processed/manifest_external_v1.csv   external evaluation sets
    data/processed/manifest_v1_rawpaths.csv   every raw file behind each image id
    results/tables/dedup_summary.md           distance histogram, groups per threshold,
                                              development-vs-external near duplicates
    results/figures/dedup_check.png           contact sheet for choosing the threshold

Usage:
    python scripts/build_manifest.py [--workers 8]
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.data.dedup import group_ids, pairwise_hamming, phash_hex, to_uint64
from pharmachecked.data.manifest import collapse_duplicates, empty_partition_columns, write_manifest
from pharmachecked.data.normalize import normalize_file
from pharmachecked.data.sources import list_raw_files, load_specs
from pharmachecked.utils.io import atomic_write_text
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("build_manifest")
PROCESSED = REPO_ROOT / "data" / "processed"
IMAGES = PROCESSED / "images"
LABEL = {0: "auth", 1: "cf"}


def _normalize(raw_path: str) -> tuple[str, int, int]:
    return normalize_file(REPO_ROOT / raw_path, IMAGES)


def _phash(args: tuple[str, int]) -> str:
    image_id, hash_size = args
    return phash_hex(IMAGES / f"{image_id}.png", hash_size)


def build_frame(rows: list[dict[str, object]], workers: int, hash_size: int) -> pd.DataFrame:
    """Normalize every unique image, add sizes and pHashes, return a manifest frame."""
    df = pd.DataFrame(rows)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_normalize, df["raw_path"], chunksize=8))
        for (image_id, _, _), expected in zip(results, df["image_id"], strict=True):
            assert image_id == expected
        df["width"] = [r[1] for r in results]
        df["height"] = [r[2] for r in results]
        df["phash"] = list(pool.map(_phash, [(i, hash_size) for i in df["image_id"]], chunksize=16))
    return empty_partition_columns(df)


def group_table(df: pd.DataFrame, thresholds: list[int]) -> list[str]:
    """Markdown rows: group statistics of the development pool at each candidate threshold."""
    lines = [
        "| Threshold | Groups | Multi-image groups | Images in multi-image groups | "
        "Largest group | Mixed-label groups |",
        "|---|---|---|---|---|---|",
    ]
    for t in thresholds:
        g = pd.Series(group_ids(list(df["image_id"]), list(df["phash"]), t))
        sizes = g.value_counts()
        multi = sizes[sizes > 1]
        mixed = df.groupby(g.values)["label"].nunique().gt(1).sum()
        lines.append(
            f"| {t} | {len(sizes)} | {len(multi)} | {int(multi.sum())} | {int(sizes.max())} | "
            f"{int(mixed)} |"
        )
    return lines


def distance_histogram(dev: pd.DataFrame, max_d: int) -> list[str]:
    """Markdown rows: number of development pairs at each distance, all and cross-label."""
    h = to_uint64(list(dev["phash"]))
    d = pairwise_hamming(h, h)
    iu = np.triu_indices(len(dev), k=1)
    dist = d[iu]
    labels = dev["label"].to_numpy()
    cross = labels[iu[0]] != labels[iu[1]]
    lines = ["| Distance | Pairs | Cross-label pairs |", "|---|---|---|"]
    for k in range(max_d + 1):
        lines.append(f"| {k} | {int((dist == k).sum())} | {int(((dist == k) & cross).sum())} |")
    return lines


def external_near(dev: pd.DataFrame, ext: pd.DataFrame, threshold: int) -> list[str]:
    """Markdown rows: development-vs-external pairs within the threshold, per external source."""
    lines = [
        f"| External source | Images | Images within distance {threshold} of a development "
        "image | Smallest distance |",
        "|---|---|---|---|",
    ]
    hd = to_uint64(list(dev["phash"]))
    for source, part in ext.groupby("source_dataset"):
        d = pairwise_hamming(to_uint64(list(part["phash"])), hd)
        lines.append(
            f"| {source} | {len(part)} | {int((d.min(axis=1) <= threshold).sum())} | "
            f"{int(d.min())} |"
        )
    return lines


def contact_sheet(dev: pd.DataFrame, threshold: int, n_pairs: int, out: Path) -> None:
    """Save the closest cross-group and farthest within-group pairs side by side."""
    ids = list(dev["image_id"])
    d = pairwise_hamming(to_uint64(list(dev["phash"])), to_uint64(list(dev["phash"])))
    groups = np.array(dev["group_id"])
    labels = dev["label"].to_numpy()
    iu = np.triu_indices(len(ids), k=1)
    same = groups[iu[0]] == groups[iu[1]]
    dist = d[iu]
    cross_order = np.argsort(np.where(same, 10**6, dist), kind="stable")[:n_pairs]
    cross_order = cross_order[~same[cross_order]]
    within_order = np.argsort(np.where(same, -dist, 10**6), kind="stable")[:n_pairs]
    within_order = within_order[same[within_order]]
    sections = [
        (f"Closest pairs in DIFFERENT groups (threshold {threshold})", cross_order),
        (f"Farthest pairs in the SAME group (threshold {threshold})", within_order),
    ]
    cols = 5  # pairs per row
    rows_total = sum(int(np.ceil(len(o) / cols)) for _, o in sections) + len(sections)
    fig = plt.figure(figsize=(cols * 2.4, rows_total * 1.35), dpi=90)
    grid = fig.add_gridspec(rows_total, cols * 2, hspace=0.45, wspace=0.05)
    row = 0
    for title, order in sections:
        ax = fig.add_subplot(grid[row, :])
        ax.axis("off")
        ax.text(0, 0.2, f"{title}: {len(order)} pairs", fontsize=11, weight="bold")
        row += 1
        for k, idx in enumerate(order):
            r, c = row + k // cols, (k % cols) * 2
            i, j = iu[0][idx], iu[1][idx]
            for off, m in ((0, i), (1, j)):
                ax = fig.add_subplot(grid[r, c + off])
                with Image.open(IMAGES / f"{ids[m]}.png") as im:
                    im.thumbnail((96, 96))
                    ax.imshow(np.asarray(im))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(LABEL[int(labels[m])], fontsize=6, labelpad=1)
                if off == 0:
                    ax.set_title(f"d={int(dist[idx])}", fontsize=7, loc="left", pad=2)
        row += int(np.ceil(len(order) / cols))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Run 1.2-1.4 and write all outputs; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    configure_logging()
    src_cfg = OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / "data" / "sources.yaml"))
    dd = OmegaConf.load(CONFIG_DIR / "data" / "dedup.yaml")
    assert isinstance(src_cfg, dict)
    specs = load_specs(src_cfg, REPO_ROOT)
    extensions = tuple(e.lower() for e in src_cfg["image_extensions"])
    raw = [
        f
        for f in list_raw_files(REPO_ROOT / "data" / "raw", specs, extensions)
        if f.role != "excluded" and f.included
    ]
    collapsed = collapse_duplicates(raw, REPO_ROOT)
    log.info("%d raw files -> %d unique images", len(raw), len(collapsed.rows))
    df = build_frame(collapsed.rows, args.workers, dd.hash_size)
    role = {s.name: s.role for s in specs}
    is_dev = df["source_dataset"].map(role) == "development"
    dev, ext = df[is_dev].copy(), df[~is_dev].copy()
    t = int(dd.phash_threshold)
    dev["group_id"] = group_ids(list(dev["image_id"]), list(dev["phash"]), t)
    ext["group_id"] = group_ids(list(ext["image_id"]), list(ext["phash"]), t)
    ext["split"] = "external"
    write_manifest(dev, PROCESSED / "manifest_v1.csv")
    write_manifest(ext, PROCESSED / "manifest_external_v1.csv")
    pd.DataFrame(collapsed.raw_paths, columns=["sha256", "raw_path"]).assign(
        image_id=lambda x: x["sha256"].str[:16]
    )[["image_id", "raw_path"]].to_csv(
        PROCESSED / "manifest_v1_rawpaths.csv", index=False, lineterminator="\n"
    )

    lines = [
        "# Near-duplicate analysis (PLAN 1.4)",
        "",
        f"Generated by `scripts/build_manifest.py`. 64-bit pHash (hash size {dd.hash_size}) of the",
        "normalized images; groups are connected components of pairs within the threshold.",
        f"Configured threshold: {t} (`configs/data/dedup.yaml`).",
        "",
        f"Development pool: {len(dev)} unique images; external sets: {len(ext)} unique images; "
        f"identical-byte groups dropped for conflicting labels: {len(collapsed.conflicts)}.",
        "",
        "## Development pool: pairs per pHash distance",
        "",
        *distance_histogram(dev, max(dd.report_thresholds)),
        "",
        "## Development pool: groups per threshold",
        "",
        *group_table(dev, list(dd.report_thresholds)),
        "",
        "## External images near a development image",
        "",
        *external_near(dev, ext, t),
        "",
    ]
    atomic_write_text(REPO_ROOT / "results" / "tables" / "dedup_summary.md", "\n".join(lines))
    contact_sheet(
        dev, t, int(dd.contact_sheet_pairs), REPO_ROOT / "results" / "figures" / "dedup_check.png"
    )
    log.info("wrote manifests, dedup_summary.md and dedup_check.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
