"""E14 synthetic counterfeits: procedural and stub-inpainting edits, OCR cache (PLAN 7.1-7.3)."""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from pharmachecked.synthesis.generate import edit_seed, training_images
from pharmachecked.synthesis.inpaint import Inpainter, crop_around
from pharmachecked.synthesis.ocr import OcrReader, TextBox, cached_boxes
from pharmachecked.synthesis.procedural import EDIT_TYPES, EDITS, TEXT_EDITS, random_edit

BOXES = [TextBox(20, 20, 140, 44, "Para 500mg", 0.9)]


def _image() -> Image.Image:
    im = Image.new("RGB", (200, 160), (240, 240, 240))
    draw = ImageDraw.Draw(im)
    draw.rectangle((100, 80, 180, 140), fill=(200, 30, 30))  # saturated "logo"
    draw.text((24, 24), "Para 500mg", fill=(10, 10, 10))
    return im


@pytest.mark.parametrize("name", EDIT_TYPES)
def test_edit_changes_only_masked_pixels(name: str) -> None:
    im = _image()
    edit = EDITS[name](im, BOXES, np.random.default_rng(0))
    assert edit is not None and edit.edit_type == name
    assert edit.image.size == im.size and edit.mask.shape == (im.height, im.width)
    changed = (np.asarray(edit.image) != np.asarray(im)).any(axis=2)
    assert edit.mask.any() and not changed[~edit.mask].any()
    assert 0 <= edit.strength <= 1


def test_text_edits_need_text() -> None:
    for name in TEXT_EDITS:
        assert EDITS[name](_image(), [], np.random.default_rng(0)) is None
    no_digit = [TextBox(20, 20, 140, 44, "Para", 0.9)]
    assert EDITS["dose_digit_change"](_image(), no_digit, np.random.default_rng(0)) is None


def test_random_edit_is_deterministic_and_falls_back_to_region_edits() -> None:
    a = random_edit(_image(), [], np.random.default_rng(5))
    b = random_edit(_image(), [], np.random.default_rng(5))
    assert a.edit_type == b.edit_type and a.edit_type not in TEXT_EDITS
    assert np.array_equal(np.asarray(a.image), np.asarray(b.image))
    assert edit_seed("procedural", "x", 0, 42) == edit_seed("procedural", "x", 0, 42)
    assert edit_seed("procedural", "x", 0, 42) != edit_seed("procedural", "x", 1, 42)


def test_stub_inpainting_is_local() -> None:
    cfg = OmegaConf.load(Path(__file__).parents[1] / "configs" / "synthesis" / "e14.yaml")
    cfg.inpaint.backend = "stub"
    im = _image()
    edit, prompt = Inpainter(cfg.inpaint).edit(im, BOXES, np.random.default_rng(0))
    changed = (np.asarray(edit.image) != np.asarray(im)).any(axis=2)
    assert prompt in cfg.inpaint.prompts and edit.mask.any()
    assert not changed[~edit.mask].any()
    left, top, right, bottom = crop_around(im, (20, 20, 140, 44))
    assert right - left == bottom - top <= min(im.size) and left >= 0 and top >= 0


def test_ocr_cache_round_trip(tmp_path: Path) -> None:
    reader = OcrReader("stub")
    first = cached_boxes(reader, "abc", _image(), tmp_path)
    assert (tmp_path / "abc.json").is_file()
    assert cached_boxes(reader, "abc", _image(), tmp_path) == first


def test_training_images_respect_the_partition() -> None:
    import pandas as pd

    table = pd.DataFrame(
        {
            "image_id": [f"s{k}" for k in range(6)],
            "source_image_id": ["a", "a", "b", "b", "c", "c"],
            "generator": ["procedural"] * 4 + ["inpaint"] * 2,
        }
    )
    assert training_images(table, ["a", "c"], "procedural", 10, 0) == ["s0", "s1"]
    assert len(training_images(table, ["a", "b"], "procedural", 3, 0)) == 3
    assert training_images(table, ["a", "b"], "procedural", 3, 0) == training_images(
        table, ["a", "b"], "procedural", 3, 0
    )
