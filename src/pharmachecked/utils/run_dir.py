"""Run identifiers and run folders (layout in CLAUDE.md section 7)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from pharmachecked.utils.io import write_json, write_yaml

RUN_ID_PATTERN = re.compile(
    r"^(?P<exp>[a-z0-9-]+)_(?P<model>[a-z0-9-]+)_(?P<variant>[a-z0-9-]+)"
    r"_f(?P<fold>\d+)_s(?P<seed>\d+)_(?P<stamp>\d{8}-\d{6})(?P<dirty>_dirty)?$"
)
_SLUG = re.compile(r"[^a-z0-9-]+")

# Files every run folder holds when the run finishes (notes.md is optional).
CREATED_FILES = ("config.yaml", "env.json")
COMPLETE_FILES = (
    "config.yaml",
    "env.json",
    "train_log.csv",
    "metrics.json",
    "oof_predictions.csv",
    "external_predictions.csv",
    "checkpoint_path.txt",
)


class DirtyTreeError(RuntimeError):
    """Raised when a run other than fold 0 / seed 0 is started from an uncommitted tree."""


def slug(text: str) -> str:
    """Lower-case ``text`` and replace anything outside ``[a-z0-9-]`` with ``-``.

    Underscores separate run-id fields, so they are not allowed inside a field.

    Args:
        text: Free text such as a model name.

    Returns:
        The slug; never empty.
    """
    out = _SLUG.sub("-", text.lower()).strip("-")
    if not out:
        raise ValueError(f"cannot build a run-id field from {text!r}")
    return out


def make_run_id(
    exp: str,
    model: str,
    variant: str,
    fold: int,
    seed: int,
    now: datetime,
    dirty: bool = False,
) -> str:
    """Build ``{exp}_{model}_{variant}_f{fold}_s{seed}_{YYYYMMDD-HHMMSS}`` (+ ``_dirty``).

    Args:
        exp: Experiment name, e.g. ``e01``.
        model: Model name.
        variant: Arm or variant name.
        fold: Outer fold index.
        seed: Seed.
        now: Timestamp of the run start.
        dirty: Whether the git tree was dirty; tags the id.

    Returns:
        The run id.
    """
    stamp = now.strftime("%Y%m%d-%H%M%S")
    run_id = f"{slug(exp)}_{slug(model)}_{slug(variant)}_f{fold}_s{seed}_{stamp}"
    return run_id + ("_dirty" if dirty else "")


def parse_run_id(run_id: str) -> dict[str, str | int | bool]:
    """Split a run id into its fields.

    Args:
        run_id: A run id built by :func:`make_run_id`.

    Returns:
        Dict with ``exp, model, variant, fold, seed, stamp, dirty``.

    Raises:
        ValueError: If the id does not match the format.
    """
    match = RUN_ID_PATTERN.match(run_id)
    if match is None:
        raise ValueError(f"not a run id: {run_id!r}")
    fields = match.groupdict()
    return {
        "exp": fields["exp"],
        "model": fields["model"],
        "variant": fields["variant"],
        "fold": int(fields["fold"]),
        "seed": int(fields["seed"]),
        "stamp": fields["stamp"],
        "dirty": fields["dirty"] is not None,
    }


@dataclass(frozen=True)
class RunDir:
    """Paths of one run folder."""

    root: Path
    run_id: str

    def file(self, name: str) -> Path:
        """Return the path of an artifact inside the run folder."""
        return self.root / name

    @property
    def dirty(self) -> bool:
        """Whether the run was started from a dirty tree."""
        return bool(parse_run_id(self.run_id)["dirty"])


def create_run_dir(
    runs_root: Path,
    cfg: DictConfig | dict[str, Any],
    env: dict[str, Any],
    exp: str,
    model: str,
    variant: str,
    fold: int,
    seed: int,
    now: datetime | None = None,
) -> RunDir:
    """Create a run folder and write ``config.yaml`` and ``env.json`` into it.

    Args:
        runs_root: Parent folder, normally ``results/runs``.
        cfg: Fully resolved config.
        env: Output of :func:`pharmachecked.utils.env.capture_env`.
        exp: Experiment name.
        model: Model name.
        variant: Arm or variant name.
        fold: Outer fold index.
        seed: Seed.
        now: Start time (defaults to the current local time).

    Returns:
        The created :class:`RunDir`.

    Raises:
        DirtyTreeError: If the tree is dirty and the run is not fold 0 / seed 0.
        FileExistsError: If the folder already exists.
    """
    dirty = bool(env.get("git_dirty"))
    if dirty and (fold, seed) != (0, 0):
        raise DirtyTreeError(
            "uncommitted changes: dirty runs are allowed only for fold=0 seed=0 debugging"
        )
    run_id = make_run_id(exp, model, variant, fold, seed, now or datetime.now(), dirty)
    root = runs_root / run_id
    root.mkdir(parents=True, exist_ok=False)
    write_yaml(root / "config.yaml", cfg)
    write_json(root / "env.json", env)
    return RunDir(root=root, run_id=run_id)


def find_resumable(
    runs_root: Path,
    cfg: DictConfig | dict[str, Any],
    exp: str,
    model: str,
    variant: str,
    fold: int,
    seed: int,
) -> RunDir | None:
    """Return the newest incomplete run folder with these fields and an identical config.

    "Identical" means the saved ``config.yaml`` equals this config's resolved YAML, so a run is
    only ever continued with exactly the settings it started with.
    """
    if not runs_root.is_dir():
        return None
    prefix = f"{slug(exp)}_{slug(model)}_{slug(variant)}_f{fold}_s{seed}_"
    conf = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    wanted = OmegaConf.to_yaml(conf, resolve=True)
    for folder in sorted(runs_root.glob(prefix + "*"), reverse=True):
        saved = folder / "config.yaml"
        if (
            folder.is_dir()
            and saved.is_file()
            and missing_files(folder)
            and saved.read_text(encoding="utf-8") == wanted
        ):
            return RunDir(root=folder, run_id=folder.name)
    return None


def missing_files(run: Path, required: tuple[str, ...] = COMPLETE_FILES) -> list[str]:
    """List the required artifacts a run folder lacks.

    Args:
        run: Run folder.
        required: File names to check.

    Returns:
        Names of missing files (empty when complete).
    """
    return [name for name in required if not (run / name).is_file()]
