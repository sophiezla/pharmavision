"""Seeding makes torch, NumPy, random and DataLoader shuffling/augmentation reproducible."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pharmachecked.utils.seed import make_generator, seed_everything, worker_init_fn


def _draw() -> tuple[torch.Tensor, np.ndarray, float]:
    return torch.randn(4, 3), np.random.rand(5), random.random()


def test_same_seed_gives_identical_draws() -> None:
    seed_everything(123)
    first = _draw()
    seed_everything(123)
    second = _draw()
    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_different_seed_gives_different_draws() -> None:
    seed_everything(1)
    first = _draw()
    seed_everything(2)
    second = _draw()
    assert not torch.equal(first[0], second[0])


def test_model_init_is_reproducible() -> None:
    seed_everything(7)
    a = torch.nn.Linear(16, 4).weight.detach().clone()
    seed_everything(7)
    b = torch.nn.Linear(16, 4).weight.detach().clone()
    assert torch.equal(a, b)


class _NumpyNoise(Dataset[torch.Tensor]):
    """Each item uses NumPy randomness, as a NumPy-based augmentation would."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([index, np.random.rand(), random.random()])


def _epoch(seed: int, workers: int) -> torch.Tensor:
    seed_everything(seed)
    loader = DataLoader(
        _NumpyNoise(),
        batch_size=4,
        shuffle=True,
        num_workers=workers,
        worker_init_fn=worker_init_fn,
        generator=make_generator(seed),
    )
    return torch.cat(list(loader))


def test_dataloader_reproducible_in_process() -> None:
    assert torch.equal(_epoch(3, workers=0), _epoch(3, workers=0))


def test_dataloader_reproducible_with_workers() -> None:
    first = _epoch(3, workers=2)
    second = _epoch(3, workers=2)
    assert torch.equal(first, second)
    # Workers must not share NumPy state: their random columns differ.
    assert len(set(first[:, 1].tolist())) == len(first)


def test_deterministic_flag_sets_cudnn() -> None:
    seed_everything(0, deterministic=True)
    assert torch.backends.cudnn.deterministic
    seed_everything(0, deterministic=False)
    assert not torch.backends.cudnn.deterministic
