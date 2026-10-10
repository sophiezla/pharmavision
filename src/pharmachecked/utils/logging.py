"""Logging configuration; ``src/`` uses ``logging`` and never ``print``."""

from __future__ import annotations

import logging
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int | str = logging.INFO, log_file: Path | None = None) -> None:
    """Configure the root logger with a console handler and an optional file handler.

    Calling it again replaces the handlers, so it is safe to call once per run.

    Args:
        level: Logging level for the root logger.
        log_file: If given, also write the log to this file (parent directories are created).
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(FORMAT, DATEFMT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (``get_logger(__name__)``).

    Args:
        name: Logger name, normally the module ``__name__``.

    Returns:
        The named logger.
    """
    return logging.getLogger(name)
