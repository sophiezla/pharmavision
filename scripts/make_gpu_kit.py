"""Build a no-command GPU kit (ADR-011): unzip, double-click 1_SETUP.bat, double-click 2_RUN.bat.

The kit holds a source snapshot of HEAD (``git archive``) with ``SNAPSHOT.json`` (commit + file
SHA-256s), the processed development images minus the locked test set, all external images
(Split E included: the kit is PRIVATE and stays on Sophie's machines), ``uv.exe`` (verified
against its published SHA-256), the two .bat files, README_FIRST.txt and KIT.json naming the job
list. The working tree must be clean.

Usage:
    python scripts/make_gpu_kit.py --jobs jobs/e01_gpu_timing.txt --name e01-timing
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.data.manifest import read_manifest
from pharmachecked.jobs import read_jobs
from pharmachecked.utils.env import SNAPSHOT_FILE, git_info
from pharmachecked.utils.logging import configure_logging, get_logger

log = get_logger("make_gpu_kit")
KIT_SRC = REPO_ROOT / "kit"
TOOLS_CACHE = REPO_ROOT / "bundles" / "_tools"
UV_ASSET = "uv-x86_64-pc-windows-msvc.zip"


def kit_image_ids(dev: list[dict[str, str]], external: list[dict[str, str]]) -> list[str]:
    """Development images outside the locked test set, plus every external image."""
    keep = [r["image_id"] for r in dev if r["split"] == "dev"]
    keep += [r["image_id"] for r in external if r["split"] == "external"]
    return sorted(set(keep))


def source_snapshot(commit: str) -> dict[str, bytes]:
    """Every tracked file of ``commit`` as ``{posix path: bytes}`` (via git archive)."""
    tar_bytes = subprocess.run(
        ["git", "archive", "--format=tar", commit], cwd=REPO_ROOT, check=True, capture_output=True
    ).stdout
    files = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        for member in tar.getmembers():
            if member.isfile():
                extracted = tar.extractfile(member)
                assert extracted is not None
                files[member.name] = extracted.read()
    return files


def snapshot_json(commit: str, files: dict[str, bytes]) -> bytes:
    """SNAPSHOT.json content: commit, creation time and the SHA-256 of every source file."""
    return (
        json.dumps(
            {
                "commit": commit,
                "created": datetime.now().isoformat(timespec="seconds"),
                "files": {p: hashlib.sha256(b).hexdigest() for p, b in sorted(files.items())},
            },
            indent=1,
        )
        + "\n"
    ).encode()


def fetch_uv(version: str) -> bytes:
    """uv.exe for Windows x86_64, downloaded once and checked against the release SHA-256."""
    TOOLS_CACHE.mkdir(parents=True, exist_ok=True)
    archive = TOOLS_CACHE / f"uv-{version}-{UV_ASSET}"
    base = f"https://github.com/astral-sh/uv/releases/download/{version}/{UV_ASSET}"
    if not archive.exists():
        with urllib.request.urlopen(base, timeout=120) as resp:
            archive.write_bytes(resp.read())
    with urllib.request.urlopen(base + ".sha256", timeout=60) as resp:
        expected = resp.read().decode().split()[0].lower()
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        archive.unlink()
        raise RuntimeError(f"uv download hash mismatch ({actual} != {expected})")
    with zipfile.ZipFile(archive) as zf:
        return zf.read("uv.exe")


def crlf(text: str) -> bytes:
    """Windows line endings (cmd.exe misreads labels in LF-only batch files)."""
    return text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")


def main() -> int:
    """Build the kit zip; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--estimate", default="see the experiment page")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="laptop dry run: CPU torch, no images (toy-data jobs), CPU allowed by the GPU check",
    )
    args = parser.parse_args()
    configure_logging()
    if git_info(REPO_ROOT)["git_dirty"]:
        log.error("working tree has uncommitted changes; commit first")
        return 1
    kcfg = OmegaConf.load(CONFIG_DIR / "kit.yaml")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    job_rel = args.jobs.resolve().relative_to(REPO_ROOT).as_posix()
    files = source_snapshot(commit)
    if job_rel not in files:
        log.error("%s is not committed", job_rel)
        return 1
    n_jobs = len(read_jobs(args.jobs))

    processed = REPO_ROOT / "data" / "processed"
    dev = read_manifest(processed / "manifest_v1.csv").to_dict("records")
    ext = read_manifest(processed / "manifest_external_v1.csv").to_dict("records")
    ids = [] if args.selftest else kit_image_ids(dev, ext)
    image_bytes = sum((processed / "images" / f"{i}.png").stat().st_size for i in ids)
    disk_gb = max(60, round(image_bytes / 1e9 * 2 + int(kcfg.disk_gb_per_job * n_jobs)))

    prefix = "SELFTEST_kit" if args.selftest else "PRIVATE_gpu_kit"
    out = REPO_ROOT / "bundles" / f"{prefix}_{args.name}_{commit[:7]}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    readme = (KIT_SRC / "README_FIRST.txt").read_text(encoding="utf-8")
    for key, value in {
        "__NAME__": args.name,
        "__COMMIT__": commit[:7],
        "__DISK__": str(disk_gb),
        "__JOBS__": str(n_jobs),
        "__JOBFILE__": job_rel,
        "__ESTIMATE__": args.estimate,
    }.items():
        readme = readme.replace(key, value)
    setup = (KIT_SRC / "1_SETUP.bat").read_text(encoding="utf-8")
    index = "https://download.pytorch.org/whl/cpu" if args.selftest else str(kcfg.torch_index)
    setup = setup.replace("__TORCH_INDEX__", index)
    kit_json = {
        "name": args.name,
        "commit": commit,
        "job_file": job_rel,
        "allow_cpu": bool(args.selftest),
        "bench": not args.selftest,
    }
    with zipfile.ZipFile(out, "w") as zf:
        for path, data in sorted(files.items()):
            zf.writestr(f"pharmavision/{path}", data, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr(f"pharmavision/{SNAPSHOT_FILE}", snapshot_json(commit, files))
        zf.writestr("pharmavision/KIT.json", json.dumps(kit_json, indent=1) + "\n")
        for image_id in ids:  # PNGs are already compressed
            zf.write(
                processed / "images" / f"{image_id}.png",
                f"pharmavision/data/processed/images/{image_id}.png",
            )
        zf.writestr(
            "tools/uv.exe", fetch_uv(str(kcfg.uv_version)), compress_type=zipfile.ZIP_DEFLATED
        )
        zf.writestr("1_SETUP.bat", crlf(setup))
        zf.writestr("2_RUN.bat", crlf((KIT_SRC / "2_RUN.bat").read_text(encoding="utf-8")))
        zf.writestr("README_FIRST.txt", crlf(readme))
    log.info(
        "wrote %s: %d source files, %d images (%.2f GB), %d jobs from %s",
        out.relative_to(REPO_ROOT),
        len(files),
        len(ids),
        image_bytes / 1e9,
        n_jobs,
        job_rel,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
