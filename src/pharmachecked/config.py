"""Locations of the repository and its Hydra configs, and the custom OmegaConf resolvers."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

# src/pharmachecked/config.py -> repository root (the package is installed in editable mode).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def register_resolvers() -> None:
    """Register ``${pc_repo_root:}`` (repository root as a POSIX path); idempotent."""
    if not OmegaConf.has_resolver("pc_repo_root"):
        OmegaConf.register_new_resolver("pc_repo_root", lambda: REPO_ROOT.as_posix())


register_resolvers()
