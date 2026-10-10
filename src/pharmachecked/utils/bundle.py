"""Results bundles: move run folders from the GPU PC to the main repository.

A bundle is a zip holding run folders (``runs/<run_id>/...``), optional result tables
(``tables/...``) and ``MANIFEST.json`` with the size and SHA-256 of every file. Weights are never
included. Unpacking verifies every hash before anything is written.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pharmachecked.utils.io import sha256_file
from pharmachecked.utils.logging import get_logger
from pharmachecked.utils.run_dir import parse_run_id

log = get_logger(__name__)

MANIFEST = "MANIFEST.json"
WEIGHT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".h5"})
TOP_LEVEL = ("runs", "tables")


class BundleError(RuntimeError):
    """Raised when a bundle is malformed, tampered with, or would overwrite different data."""


@dataclass(frozen=True)
class PackResult:
    """Summary of a packed bundle."""

    path: Path
    n_runs: int
    n_files: int
    skipped_weights: tuple[str, ...]


def _is_weight(path: Path) -> bool:
    return path.suffix.lower() in WEIGHT_SUFFIXES


def pack(
    run_dirs: list[Path], table_files: list[Path], out: Path, created_by: str = ""
) -> PackResult:
    """Write run folders and table files into a bundle zip with a hash manifest.

    Args:
        run_dirs: Run folders (each named by a valid run id).
        table_files: Extra files placed under ``tables/`` (e.g. benchmark tables).
        out: Destination zip; must not exist.
        created_by: Free text stored in the manifest (e.g. host name).

    Returns:
        A :class:`PackResult`.

    Raises:
        BundleError: If a folder is not a run folder or ``out`` exists.
    """
    if out.exists():
        raise BundleError(f"{out} already exists")
    entries: dict[str, Path] = {}
    skipped: list[str] = []
    for run in run_dirs:
        try:
            parse_run_id(run.name)
        except ValueError as exc:
            raise BundleError(f"not a run folder: {run}") from exc
        for file in sorted(p for p in run.rglob("*") if p.is_file()):
            arcname = f"runs/{run.name}/{file.relative_to(run).as_posix()}"
            if _is_weight(file):
                skipped.append(arcname)
                continue
            entries[arcname] = file
    for table in table_files:
        entries[f"tables/{table.name}"] = table
    manifest = {
        "created_by": created_by,
        "files": {
            name: {"sha256": sha256_file(src), "size": src.stat().st_size}
            for name, src in sorted(entries.items())
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, src in sorted(entries.items()):
            zf.write(src, name)
        zf.writestr(MANIFEST, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for name in skipped:
        log.warning("not packed (weights): %s", name)
    return PackResult(out, len(run_dirs), len(entries), tuple(skipped))


def _safe_name(name: str) -> PurePosixPath:
    """Reject absolute paths, ``..`` components and anything outside runs/ or tables/."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise BundleError(f"unsafe path in bundle: {name!r}")
    if len(path.parts) < 2 or path.parts[0] not in TOP_LEVEL:
        raise BundleError(f"unexpected path in bundle: {name!r}")
    if path.parts[0] == "runs":
        if len(path.parts) < 3:
            raise BundleError(f"file outside a run folder: {name!r}")
        try:
            parse_run_id(path.parts[1])
        except ValueError as exc:
            raise BundleError(f"not a run folder: {path.parts[1]!r}") from exc
    return path


def verify(bundle: Path) -> dict[str, dict[str, int | str]]:
    """Check a bundle's manifest against its contents.

    Args:
        bundle: Zip written by :func:`pack`.

    Returns:
        The manifest's ``files`` mapping.

    Raises:
        BundleError: On a missing manifest, unsafe or unexpected paths, files missing from
            either side, or a size or hash mismatch.
    """
    with zipfile.ZipFile(bundle) as zf:
        names = set(zf.namelist()) - {MANIFEST}
        if MANIFEST not in zf.namelist():
            raise BundleError("bundle has no MANIFEST.json")
        files = json.loads(zf.read(MANIFEST))["files"]
        if set(files) != names:
            missing = sorted(set(files) - names)
            extra = sorted(names - set(files))
            raise BundleError(f"manifest mismatch; missing {missing}, unlisted {extra}")
        for name, meta in files.items():
            _safe_name(name)
            data = zf.read(name)
            if len(data) != meta["size"] or hashlib.sha256(data).hexdigest() != meta["sha256"]:
                raise BundleError(f"hash mismatch: {name}")
    return files


def unpack(bundle: Path, results_dir: Path) -> list[str]:
    """Verify a bundle, then install its run folders and tables under ``results_dir``.

    Existing files are left alone when byte-identical; a differing existing file aborts
    the whole unpack before anything is written.

    Args:
        bundle: Zip written by :func:`pack`.
        results_dir: The repository's ``results/`` folder.

    Returns:
        Names of the run folders contained in the bundle.

    Raises:
        BundleError: If verification fails or a file would be overwritten with other content.
    """
    files = verify(bundle)
    conflicts = []
    for name, meta in files.items():
        target = results_dir / _safe_name(name)
        if target.exists() and sha256_file(target) != meta["sha256"]:
            conflicts.append(name)
    if conflicts:
        raise BundleError(f"would overwrite different existing files: {conflicts}")
    with zipfile.ZipFile(bundle) as zf:
        for name in files:
            target = results_dir / _safe_name(name)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
    runs = sorted({PurePosixPath(n).parts[1] for n in files if n.startswith("runs/")})
    log.info("installed %d run folders from %s", len(runs), bundle.name)
    return runs
