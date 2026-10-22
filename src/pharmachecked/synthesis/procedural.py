"""Procedural counterfeit edits on authentic images (PLAN 7.2, E14).

Each edit takes a full-resolution RGB image, its OCR text boxes and a seeded generator and
returns an :class:`Edit` (edited image, binary mask of changed pixels, edit type, strength in
[0, 1]) or None when it does not apply (e.g. no text box with a digit). Edit types:

* ``font_swap``: one text line repainted in another font (background = median colour of the
  box border, ink = the box pixels farthest from it);
* ``char_substitution``: one character replaced by a look-alike (O/0, I/1, S/5, B/8, ...);
* ``dose_digit_change``: one digit replaced by another digit;
* ``local_hue_shift``: hue rotated by 10-30 degrees in a region;
* ``local_blur_or_rehalftone``: Gaussian blur (sigma 1.5-4 px) or an ordered-dither halftone in
  a region;
* ``logo_warp``: a sinusoidal warp (2-6% of the region size) inside a region;
* ``local_jpeg``: the region re-encoded as JPEG at quality 5-20.

Regions: the largest saturated blob outside text boxes ("logo"), else a random box with sides
15-30% of the image. Character cells are estimated as equal slices of the text box.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

from pharmachecked.synthesis.ocr import TextBox

EDIT_TYPES = (
    "font_swap",
    "char_substitution",
    "dose_digit_change",
    "local_hue_shift",
    "local_blur_or_rehalftone",
    "logo_warp",
    "local_jpeg",
)
TEXT_EDITS = EDIT_TYPES[:3]
FONTS = ("arial.ttf", "times.ttf", "cour.ttf", "verdana.ttf", "georgia.ttf", "tahoma.ttf")
LOOKALIKES = {
    "O": "0", "0": "O", "I": "1", "1": "I", "l": "1", "S": "5", "5": "S", "B": "8", "8": "B",
    "Z": "2", "2": "Z", "G": "6", "6": "G", "o": "0", "e": "c", "c": "e", "a": "o", "n": "h",
    "m": "n", "u": "v", "E": "F", "F": "E", "C": "G", "D": "O",
}  # fmt: skip


@dataclass
class Edit:
    """Result of one edit."""

    image: Image.Image
    mask: np.ndarray  # bool (H, W)
    edit_type: str
    strength: float


Box = tuple[int, int, int, int]


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """A TrueType font by file name, or Pillow's scalable default when it is not installed."""
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _usable(boxes: Sequence[TextBox], min_side: int = 8) -> list[TextBox]:
    return [b for b in boxes if b.x1 - b.x0 >= min_side and b.y1 - b.y0 >= min_side]


def _colours(arr: np.ndarray, box: Box) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(background, ink): median of the box border; median of the 15% farthest pixels."""
    x0, y0, x1, y1 = box
    patch = arr[y0:y1, x0:x1].reshape(-1, 3).astype(float)
    border = np.concatenate(
        [arr[y0, x0:x1], arr[y1 - 1, x0:x1], arr[y0:y1, x0], arr[y0:y1, x1 - 1]]
    ).astype(float)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(patch - bg, axis=1)
    ink = np.median(patch[dist >= np.quantile(dist, 0.85)], axis=0)
    return tuple(int(v) for v in bg), tuple(int(v) for v in ink)


def _repaint(image: Image.Image, box: Box, text: str, font_name: str) -> np.ndarray:
    """Fill ``box`` with its background colour and draw ``text`` fitted to it; returns pixels."""
    arr = np.asarray(image).copy()
    bg, ink = _colours(arr, box)
    x0, y0, x1, y1 = box
    canvas = Image.new("RGB", (x1 - x0, y1 - y0), bg)
    draw = ImageDraw.Draw(canvas)
    size = max(6, int((y1 - y0) * 0.9))
    f = font(font_name, size)
    while size > 6 and draw.textlength(text, font=f) > (x1 - x0):
        size -= 1
        f = font(font_name, size)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=f)
    pos = ((x1 - x0 - (right - left)) / 2 - left, (y1 - y0 - (bottom - top)) / 2 - top)
    draw.text(pos, text, fill=ink, font=f)
    arr[y0:y1, x0:x1] = np.asarray(canvas)
    return arr


def _mask(shape: tuple[int, int], box: Box) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return mask


def _box_of(b: TextBox) -> Box:
    return (b.x0, b.y0, b.x1, b.y1)


def font_swap(
    image: Image.Image, boxes: Sequence[TextBox], rng: np.random.Generator
) -> Edit | None:
    """Repaint one text line in a randomly chosen font."""
    lines = [b for b in _usable(boxes) if len(b.text.strip()) >= 2]
    if not lines:
        return None
    b = lines[int(rng.integers(len(lines)))]
    arr = _repaint(image, _box_of(b), b.text.strip(), FONTS[int(rng.integers(1, len(FONTS)))])
    area = b.area / (image.width * image.height)
    return Edit(Image.fromarray(arr), _mask(arr.shape[:2], _box_of(b)), "font_swap", min(1, area))


def _cell_edit(
    image: Image.Image,
    boxes: Sequence[TextBox],
    rng: np.random.Generator,
    pick: Callable[[str], str | None],
    edit_type: str,
) -> Edit | None:
    """Replace one character (chosen by ``pick``) inside its estimated character cell."""
    candidates = []
    for b in _usable(boxes):
        for k, ch in enumerate(b.text):
            if pick(ch) is not None:
                candidates.append((b, k, ch))
    if not candidates:
        return None
    b, k, ch = candidates[int(rng.integers(len(candidates)))]
    new = pick(ch)
    assert new is not None
    if edit_type == "dose_digit_change":
        options = [d for d in "0123456789" if d != ch]
        new = options[int(rng.integers(len(options)))]
    width = (b.x1 - b.x0) / max(1, len(b.text))
    cell = (int(b.x0 + k * width), b.y0, int(np.ceil(b.x0 + (k + 1) * width)), b.y1)
    if cell[2] - cell[0] < 3:
        return None
    arr = _repaint(image, cell, new, FONTS[0])
    return Edit(Image.fromarray(arr), _mask(arr.shape[:2], cell), edit_type, 1.0 / len(b.text))


def char_substitution(
    image: Image.Image, boxes: Sequence[TextBox], rng: np.random.Generator
) -> Edit | None:
    """Swap one character for a look-alike."""
    return _cell_edit(image, boxes, rng, LOOKALIKES.get, "char_substitution")


def dose_digit_change(
    image: Image.Image, boxes: Sequence[TextBox], rng: np.random.Generator
) -> Edit | None:
    """Change one digit."""
    return _cell_edit(image, boxes, rng, lambda c: c if c.isdigit() else None, "dose_digit_change")


def pick_region(image: Image.Image, boxes: Sequence[TextBox], rng: np.random.Generator) -> Box:
    """The largest saturated blob outside text boxes (1-25% of the image), else a random box."""
    w, h = image.size
    scale = 256 / max(w, h)
    small = image.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    sat = np.asarray(small.convert("HSV"))[..., 1] / 255.0
    blob = sat > 0.35
    for b in boxes:
        blob[
            int(b.y0 * scale) : int(b.y1 * scale) + 1, int(b.x0 * scale) : int(b.x1 * scale) + 1
        ] = False
    labels, n = ndimage.label(blob)
    if n:
        sizes = ndimage.sum(blob, labels, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        frac = sizes[k - 1] / blob.size
        if 0.01 <= frac <= 0.25:
            ys, xs = np.nonzero(labels == k)
            x0, x1 = xs.min() / scale, (xs.max() + 1) / scale
            y0, y1 = ys.min() / scale, (ys.max() + 1) / scale
            px, py = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
            return (
                int(max(0, x0 - px)),
                int(max(0, y0 - py)),
                int(min(w, x1 + px)),
                int(min(h, y1 + py)),
            )
    bw = int(w * rng.uniform(0.15, 0.30))
    bh = int(h * rng.uniform(0.15, 0.30))
    x0 = int(rng.integers(0, max(1, w - bw)))
    y0 = int(rng.integers(0, max(1, h - bh)))
    return (x0, y0, x0 + max(bw, 2), y0 + max(bh, 2))


def _region_edit(
    image: Image.Image,
    box: Box,
    transform: Callable[[Image.Image], Image.Image],
    edit_type: str,
    strength: float,
) -> Edit:
    x0, y0, x1, y1 = box
    out = image.copy()
    out.paste(transform(image.crop(box)).convert("RGB").resize((x1 - x0, y1 - y0)), (x0, y0))
    return Edit(out, _mask((image.height, image.width), box), edit_type, strength)


def local_hue_shift(
    image: Image.Image, boxes: Sequence[TextBox], rng: np.random.Generator
) -> Edit | None:
    """Rotate the hue of a region by 10-30 degrees (random direction)."""
    degrees = float(rng.uniform(10, 30)) * (1 if rng.random() < 0.5 else -1)

    def shift(patch: Image.Image) -> Image.Image:
        hsv = np.asarray(patch.convert("HSV")).copy()
        hsv[..., 0] = ((hsv[..., 0].astype(int) + round(degrees / 360 * 255)) % 256).astype(
            np.uint8
        )
        return Image.fromarray(hsv, "HSV")

    return _region_edit(
        image, pick_region(image, boxes, rng), shift, "local_hue_shift", abs(degrees) / 30
    )


def halftone(patch: Image.Image, cell: int = 4) -> Image.Image:
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0
    arr = np.asarray(patch).astype(float) / 255.0
    h, w = arr.shape[:2]
    tile = np.kron(bayer, np.ones((cell // 4 or 1, cell // 4 or 1)))
    reps = (h // tile.shape[0] + 1, w // tile.shape[1] + 1)
    thresh = np.tile(tile, reps)[:h, :w, None]
    return Image.fromarray(((arr > thresh) * 255).astype(np.uint8))


def local_blur_or_rehalftone(
    image: Image.Image, boxes: Sequence[TextBox], rng: np.random.Generator
) -> Edit | None:
    """Blur a region (sigma 1.5-4 px) or replace it with an ordered-dither halftone."""
    box = pick_region(image, boxes, rng)
    if rng.random() < 0.5:
        sigma = float(rng.uniform(1.5, 4.0))
        return _region_edit(
            image,
            box,
            lambda p: p.filter(ImageFilter.GaussianBlur(sigma)),
            "local_blur_or_rehalftone",
            sigma / 4,
        )
    return _region_edit(image, box, halftone, "local_blur_or_rehalftone", 1.0)


def logo_warp(
    image: Image.Image, boxes: Sequence[TextBox], rng: np.random.Generator
) -> Edit | None:
    """Sinusoidal displacement (amplitude 2-6% of the region) inside a region."""
    amp = float(rng.uniform(0.02, 0.06))
    phase = float(rng.uniform(0, 2 * np.pi))

    def warp(patch: Image.Image) -> Image.Image:
        arr = np.asarray(patch).astype(float)
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        dx = amp * w * np.sin(2 * np.pi * yy / max(h, 1) + phase)
        dy = amp * h * np.sin(2 * np.pi * xx / max(w, 1) + phase)
        coords = [yy + dy, xx + dx]
        out = np.stack(
            [
                ndimage.map_coordinates(arr[..., c], coords, order=1, mode="nearest")
                for c in range(3)
            ],
            axis=-1,
        )
        return Image.fromarray(out.clip(0, 255).astype(np.uint8))

    return _region_edit(image, pick_region(image, boxes, rng), warp, "logo_warp", amp / 0.06)


def local_jpeg(
    image: Image.Image, boxes: Sequence[TextBox], rng: np.random.Generator
) -> Edit | None:
    """Re-encode a region as JPEG at quality 5-20."""
    quality = int(rng.integers(5, 21))

    def jpeg(patch: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        patch.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    return _region_edit(
        image, pick_region(image, boxes, rng), jpeg, "local_jpeg", (20 - quality) / 15
    )


EDITS: dict[str, Callable[[Image.Image, Sequence[TextBox], np.random.Generator], Edit | None]] = {
    "font_swap": font_swap,
    "char_substitution": char_substitution,
    "dose_digit_change": dose_digit_change,
    "local_hue_shift": local_hue_shift,
    "local_blur_or_rehalftone": local_blur_or_rehalftone,
    "logo_warp": logo_warp,
    "local_jpeg": local_jpeg,
}


def random_edit(
    image: Image.Image,
    boxes: Sequence[TextBox],
    rng: np.random.Generator,
    types: Sequence[str] = EDIT_TYPES,
) -> Edit:
    """Apply one edit type drawn uniformly from those that apply to this image.

    Region edits always apply, so a result is always returned.
    """
    order = list(types)
    rng.shuffle(order)
    for name in order:
        edit = EDITS[name](image, boxes, rng)
        if edit is not None and edit.mask.any():
            return edit
    raise ValueError("no edit applies (pass at least one region edit type)")
