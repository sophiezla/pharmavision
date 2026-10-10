"""Bundle the processed images and manifests for the GPU PC, or verify and unpack such a bundle.

Only images from sources marked portable in ``configs/data/sources.yaml`` are packed; Split E
never is. The zip holds ``images/{image_id}.png``, the manifest CSVs and ``DATA_MANIFEST.json``
(SHA-256 of every file), and unpacking checks every hash before writing.

Usage:
    python scripts/pack_data.py [--out bundles/data_v1.zip]
    python scripts/pack_data.py --unpack bundles/data_v1.zip     (on the GPU PC)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

from omegaconf import OmegaConf

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.data.manifest import read_manifest
from pharmachecked.data.sources import load_specs
from pharmachecked.utils.io import sha256_file
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("pack_data")
PROCESSED = REPO_ROOT / "data" / "processed"
MANIFESTS = ("manifest_v1.csv", "manifest_external_v1.csv")
INDEX = "DATA_MANIFEST.json"


class DataBundleError(RuntimeError):
    """Raised for non-portable content, bad paths or hash mismatches."""


def select_images(manifests: list[Path], portable: dict[str, bool]) -> list[str]:
    """Return image ids of every manifest row whose source is portable."""
    ids: list[str] = []
    for path in manifests:
        df = read_manifest(path)
        unknown = set(df["source_dataset"]) - set(portable)
        if unknown:
            raise DataBundleError(f"sources missing from sources.yaml: {sorted(unknown)}")
        ids += df.loc[df["source_dataset"].map(portable), "image_id"].tolist()
    return sorted(set(ids))


def pack(images_dir: Path, manifests: list[Path], portable: dict[str, bool], out: Path) -> int:
    """Write the data bundle; returns the number of images packed.

    Manifest rows of non-portable sources are removed from the packed manifests, so the
    bundle never references images it does not contain.
    """
    if out.exists():
        raise DataBundleError(f"{out} already exists")
    ids = select_images(manifests, portable)
    files = {f"images/{i}.png": images_dir / f"{i}.png" for i in ids}
    out.parent.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_STORED) as zf:
        for name, src in sorted(files.items()):
            zf.write(src, name)
            index[name] = sha256_file(src)
        for path in manifests:
            df = read_manifest(path)
            text = df[df["source_dataset"].map(portable)].to_csv(index=False, lineterminator="\n")
            data = text.encode("utf-8")
            zf.writestr(f"manifests/{path.name}", data)
            index[f"manifests/{path.name}"] = hashlib.sha256(data).hexdigest()
        zf.writestr(INDEX, json.dumps(index, indent=2, sort_keys=True) + "\n")
    return len(ids)


def unpack(bundle: Path, processed_dir: Path) -> int:
    """Verify every hash, then extract images and manifests into ``processed_dir``."""
    with zipfile.ZipFile(bundle) as zf:
        index = json.loads(zf.read(INDEX))
        if set(index) != set(zf.namelist()) - {INDEX}:
            raise DataBundleError("bundle contents do not match its index")
        for name, digest in index.items():
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.parts[0]
                not in (
                    "images",
                    "manifests",
                )
            ):
                raise DataBundleError(f"unsafe path in bundle: {name!r}")
            if hashlib.sha256(zf.read(name)).hexdigest() != digest:
                raise DataBundleError(f"hash mismatch: {name}")
        for name in index:
            path = PurePosixPath(name)
            target = processed_dir / (
                path.name if path.parts[0] == "manifests" else Path("images", path.name)
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
    return sum(1 for n in index if n.startswith("images/"))


def main() -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "bundles" / "data_v1.zip")
    parser.add_argument("--unpack", type=Path, default=None)
    args = parser.parse_args()
    configure_logging()
    try:
        if args.unpack:
            n = unpack(args.unpack, PROCESSED)
            log.info("verified and unpacked %d images into %s", n, PROCESSED)
            return 0
        cfg = OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / "data" / "sources.yaml"))
        assert isinstance(cfg, dict)
        portable = {s.name: s.portable for s in load_specs(cfg, REPO_ROOT)}
        n = pack(PROCESSED / "images", [PROCESSED / m for m in MANIFESTS], portable, args.out)
    except DataBundleError as exc:
        log.error("%s", exc)
        return 1
    log.info("wrote %s with %d images", args.out, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
