"""Label-blind image normalization: EXIF orientation, sRGB, RGB on white, lossless PNG.

Every image used anywhere in the pipeline is read from ``data/processed/images/{image_id}.png``,
where ``image_id`` is the first 16 hex characters of the SHA-256 of the original file bytes.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, ImageCms, ImageOps

from pharmachecked.utils.logging import get_logger

log = get_logger(__name__)
ID_LENGTH = 16
WHITE = (255, 255, 255)
_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


def image_id_from_bytes(data: bytes) -> str:
    """Return the image id: the first 16 hex characters of SHA-256 of the original bytes."""
    return hashlib.sha256(data).hexdigest()[:ID_LENGTH]


def _to_srgb(im: Image.Image) -> Image.Image:
    """Convert an image with an embedded ICC profile to sRGB; return it unchanged otherwise."""
    icc = im.info.get("icc_profile")
    if not icc or im.mode not in ("RGB", "RGBA", "CMYK", "L"):
        return im
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        out_mode = "RGBA" if im.mode == "RGBA" else "RGB"
        converted = ImageCms.profileToProfile(im, src, _SRGB, outputMode=out_mode)
    except (ImageCms.PyCMSError, OSError, ValueError) as exc:
        log.warning("ICC conversion failed (%s); using pixel values as is", exc)
        return im
    assert converted is not None
    return converted


def to_rgb(im: Image.Image) -> Image.Image:
    """Return an RGB copy: EXIF orientation applied, ICC converted to sRGB, alpha on white.

    Handles RGB, RGBA, L, LA, P (with or without transparency), CMYK, 1, I and I;16 inputs.

    Args:
        im: Opened image (any mode).

    Returns:
        A new RGB image with no alpha channel.
    """
    im = ImageOps.exif_transpose(im)
    im = _to_srgb(im)
    if im.mode == "P":
        im = im.convert("RGBA")
    elif im.mode in ("I", "I;16", "I;16B", "I;16L"):
        # 16-bit grey: scale to 8 bit before conversion (PIL would clip instead).
        im = im.point(lambda v: v / 256).convert("L")
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (*WHITE, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return im.convert("RGB")


def normalize_file(src: Path, out_dir: Path) -> tuple[str, int, int]:
    """Normalize one raw file to ``out_dir/{image_id}.png`` (skipped if it already exists).

    Args:
        src: Raw image file (never modified).
        out_dir: Destination folder.

    Returns:
        ``(image_id, width, height)`` of the normalized image.
    """
    data = src.read_bytes()
    image_id = image_id_from_bytes(data)
    dst = out_dir / f"{image_id}.png"
    if dst.exists():
        with Image.open(dst) as done:
            return image_id, done.width, done.height
    with Image.open(io.BytesIO(data)) as im:
        rgb = to_rgb(im)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".png.tmp")
    rgb.save(tmp, format="PNG")
    tmp.replace(dst)
    return image_id, rgb.width, rgb.height
