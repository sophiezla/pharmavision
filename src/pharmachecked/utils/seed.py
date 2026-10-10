"""Seeding for Python, NumPy, torch and DataLoader workers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed every random number generator the pipeline uses.

    Args:
        seed: Seed for Python ``random``, NumPy, torch (CPU and all CUDA devices).
        deterministic: If True, also force deterministic cuDNN and torch algorithms
            (used for evaluation; slower for training).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        # Required by some CUDA ops when deterministic algorithms are enforced.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)


def worker_init_fn(worker_id: int) -> None:
    """Seed NumPy and ``random`` inside a DataLoader worker.

    torch already gives each worker the seed ``base_seed + worker_id``; this propagates it to
    the other libraries so augmentations drawn with them are reproducible.

    Args:
        worker_id: Index of the worker (unused; the per-worker seed already includes it).
    """
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a torch generator for a DataLoader, so shuffling depends only on ``seed``.

    Args:
        seed: Seed for the generator.

    Returns:
        A seeded ``torch.Generator``.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
