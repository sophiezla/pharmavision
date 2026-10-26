"""Verify and install a MODELS bundle from the Phases 6-8 kit (see scripts/pack_models.py).

Every file's SHA-256 is checked before anything is written. Weights go to ``checkpoints/``
(git-ignored), the synthetic review sample to ``data/processed/synthetic_qc/`` (git-ignored).
Existing files are never overwritten with different content.

Usage:
    python scripts/unpack_models.py MODELS_<name>_<stamp>.zip [--verify-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

from pharmachecked.config import REPO_ROOT
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("unpack_models")
TARGETS = {
    "checkpoints": REPO_ROOT / "checkpoints",
    "synthetic_qc": REPO_ROOT / "data" / "processed" / "synthetic_qc",
}


def target(name: str) -> Path:
    """Destination of an archive member.

    Raises:
        ValueError: On unsafe or unexpected paths.
    """
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"unsafe path {name!r}")
    if path.parts[0] not in TARGETS or len(path.parts) < 2:
        raise ValueError(f"unexpected path {name!r}")
    return TARGETS[path.parts[0]].joinpath(*path.parts[1:])


def main() -> int:
    """Verify, then install; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    configure_logging()
    with zipfile.ZipFile(args.bundle) as zf:
        files = json.loads(zf.read("MANIFEST.json"))["files"]
        if set(files) != set(zf.namelist()) - {"MANIFEST.json"}:
            log.error("manifest does not match the archive")
            return 1
        for name, meta in files.items():
            target(name)
            if hashlib.sha256(zf.read(name)).hexdigest() != meta["sha256"]:
                log.error("hash mismatch: %s", name)
                return 1
        conflicts = [
            n
            for n, meta in files.items()
            if target(n).exists()
            and hashlib.sha256(target(n).read_bytes()).hexdigest() != meta["sha256"]
        ]
        if conflicts:
            log.error("would overwrite different files: %s", conflicts[:5])
            return 1
        if args.verify_only:
            log.info("bundle OK: %d files", len(files))
            return 0
        for name in files:
            out = target(name)
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(name))
    log.info("installed %d files from %s", len(files), args.bundle.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
