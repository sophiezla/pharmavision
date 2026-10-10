"""Small file I/O helpers: JSON, YAML, hashing, atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to ``path`` via a temporary file and rename, so readers never see half a file.

    Args:
        path: Destination file; parent directories are created.
        text: Content, written as UTF-8 with ``\\n`` line endings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` as indented, key-sorted JSON.

    Args:
        path: Destination file.
        data: JSON-serializable object.
    """
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    """Read a JSON file.

    Args:
        path: File to read.

    Returns:
        The decoded object.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, cfg: DictConfig | dict[str, Any]) -> None:
    """Write a config (DictConfig or plain dict) as YAML, resolving interpolations.

    Args:
        path: Destination file.
        cfg: Config to save.
    """
    conf = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    atomic_write_text(path, OmegaConf.to_yaml(conf, resolve=True))


def read_yaml(path: Path) -> DictConfig:
    """Read a YAML file into a DictConfig.

    Args:
        path: File to read.

    Returns:
        The loaded config.
    """
    conf = OmegaConf.load(path)
    if not isinstance(conf, DictConfig):
        raise TypeError(f"{path} does not contain a mapping")
    return conf


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file.

    Args:
        path: File to hash.
        chunk_size: Read size in bytes.

    Returns:
        64-character lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
