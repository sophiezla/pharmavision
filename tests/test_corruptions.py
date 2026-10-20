"""E9 corruptions: severity 0 is the identity, severity is monotonic, randomness is per image."""

import itertools

import numpy as np
import pytest
from PIL import Image

from pharmachecked.data.corruptions import CORRUPTIONS, corrupt, load_corruption_config

CFG = load_corruption_config()


def _image(seed: int = 0) -> Image.Image:
    """A 224 px test card: colour gradients, a checkerboard and fine random texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:224, 0:224] / 223.0
    checker = ((np.mgrid[0:224, 0:224] // 16).sum(axis=0) % 2).astype(float)
    array = np.stack([0.2 + 0.6 * xx, 0.3 + 0.4 * checker, 0.7 - 0.5 * yy], axis=-1)
    array += 0.08 * rng.standard_normal(array.shape)
    return Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))


def _arr(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float64) / 255.0


def _laplacian_var(image: Image.Image) -> float:
    g = _arr(image).mean(axis=2)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def test_config_lists_every_corruption_with_five_levels() -> None:
    assert set(CFG.severities) == set(CORRUPTIONS)
    assert all(len(levels) == 5 for levels in CFG.severities.values())
    assert CFG.quality_auroc == 0.85


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_severity_zero_is_identity(name: str) -> None:
    image = _image()
    assert np.array_equal(_arr(corrupt(image, "img", name, 0, CFG)), _arr(image))


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_distance_from_clean_grows_with_severity(name: str) -> None:
    image = _image()
    dist = [
        np.abs(_arr(corrupt(image, "img", name, s, CFG)) - _arr(image)).mean() for s in range(6)
    ]
    assert dist[0] == 0
    assert all(b > a for a, b in itertools.pairwise(dist)), dist


@pytest.mark.parametrize(
    ("name", "statistic", "direction"),
    [
        ("gaussian_blur", _laplacian_var, -1),
        ("gaussian_noise", _laplacian_var, 1),
        ("brightness", lambda im: _arr(im).mean(), -1),
        ("contrast", lambda im: _arr(im).std(), -1),
        ("glare", lambda im: _arr(im).mean(), 1),
    ],
)
def test_specific_statistic_is_monotonic(name: str, statistic, direction: int) -> None:
    # Motion blur is covered by the distance test only: on the periodic checkerboard a long
    # directional kernel can raise the Laplacian variance.
    image = _image()
    values = [statistic(corrupt(image, "img", name, s, CFG)) for s in range(6)]
    steps = [direction * (b - a) for a, b in itertools.pairwise(values)]
    assert all(step > 0 for step in steps), values


@pytest.mark.parametrize("name", ["gaussian_noise", "motion_blur", "perspective", "glare"])
def test_randomness_depends_on_image_id_only(name: str) -> None:
    image = _image()
    a = _arr(corrupt(image, "img-a", name, 3, CFG))
    assert np.array_equal(a, _arr(corrupt(image, "img-a", name, 3, CFG)))
    assert not np.array_equal(a, _arr(corrupt(image, "img-b", name, 3, CFG)))


def test_rejects_unknown_corruption_and_severity() -> None:
    with pytest.raises(KeyError):
        corrupt(_image(), "img", "fog", 1, CFG)
    with pytest.raises(ValueError):
        corrupt(_image(), "img", "jpeg", 6, CFG)
