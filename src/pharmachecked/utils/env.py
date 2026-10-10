"""Capture the software and hardware environment of a run (written to ``env.json``)."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

# Distributions whose versions are recorded in every env.json.
TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "timm",
    "open_clip_torch",
    "numpy",
    "pandas",
    "scikit-learn",
    "scikit-image",
    "xgboost",
    "hydra-core",
    "omegaconf",
    "optuna",
    "statsmodels",
    "torchmetrics",
    "peft",
    "cleanlab",
    "mlflow",
    "Pillow",
    "imagehash",
)


def _git(args: list[str], cwd: Path) -> str | None:
    """Run a git command and return stripped stdout, or None if git or the repo is unavailable."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_info(repo: Path) -> dict[str, Any]:
    """Return the commit hash and dirty flag of the repository containing ``repo``.

    A tree is dirty when tracked files have uncommitted changes; untracked files (for example
    new run folders) do not count.

    Args:
        repo: Any path inside the repository.

    Returns:
        ``{"git_commit": str | None, "git_dirty": bool | None}``; None when not a git repo.
    """
    commit = _git(["rev-parse", "HEAD"], repo)
    if commit is None:
        return {"git_commit": None, "git_dirty": None}
    status = _git(["status", "--porcelain", "--untracked-files=no"], repo)
    return {"git_commit": commit, "git_dirty": bool(status)}


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Return installed versions of the given distributions (None if not installed).

    Args:
        names: Distribution names.

    Returns:
        Mapping from name to version string or None.
    """
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def device_info() -> dict[str, Any]:
    """Return GPU name and CUDA version as seen by torch (None on CPU-only machines).

    Returns:
        ``{"cuda_available", "cuda_version", "gpu_name", "gpu_count"}``.
    """
    import torch

    available = torch.cuda.is_available()
    return {
        "cuda_available": available,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "gpu_count": torch.cuda.device_count() if available else 0,
    }


def capture_env(repo: Path) -> dict[str, Any]:
    """Collect everything ``env.json`` records.

    Args:
        repo: Path inside the repository whose commit is recorded.

    Returns:
        Dict with git commit and dirty flag, Python version, platform, processor, package
        versions and device information.
    """
    return {
        **git_info(repo),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "packages": package_versions(),
        **device_info(),
    }
