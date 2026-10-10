"""Normalization: RGB PNG for every input mode, EXIF orientation applied, alpha on white."""

import hashlib
import io
from pathlib import Path

import pytest
from PIL import Image, ImageCms

from pharmachecked.data.normalize import image_id_from_bytes, normalize_file, to_rgb


def _save(im: Image.Image, path: Path, fmt: str, **kwargs: object) -> Path:
    im.save(path, format=fmt, **kwargs)
    return path


@pytest.mark.parametrize(
    ("mode", "fmt", "color"),
    [
        ("RGB", "PNG", (10, 20, 30)),
        ("RGBA", "PNG", (10, 20, 30, 255)),
        ("L", "PNG", 128),
        ("LA", "PNG", (128, 255)),
        ("P", "PNG", 3),
        ("CMYK", "JPEG", (0, 0, 0, 0)),
        ("I;16", "PNG", 65535),
        ("1", "PNG", 1),
    ],
)
def test_every_mode_becomes_rgb_png(tmp_path: Path, mode: str, fmt: str, color: object) -> None:
    src = _save(Image.new(mode, (12, 8), color), tmp_path / f"in.{fmt.lower()}", fmt)
    image_id, w, h = normalize_file(src, tmp_path / "out")
    out = tmp_path / "out" / f"{image_id}.png"
    with Image.open(out) as im:
        assert im.mode == "RGB"
        assert im.format == "PNG"
        assert (im.width, im.height) == (w, h) == (12, 8)


def test_transparent_pixels_composited_on_white() -> None:
    im = Image.new("RGBA", (2, 1), (255, 0, 0, 255))
    im.putpixel((1, 0), (0, 0, 255, 0))
    out = to_rgb(im)
    assert out.getpixel((0, 0)) == (255, 0, 0)
    assert out.getpixel((1, 0)) == (255, 255, 255)


def test_palette_transparency_composited_on_white() -> None:
    im = Image.new("P", (2, 1), 0)
    im.putpalette([0, 0, 0, 200, 0, 0] + [0] * 762)
    im.putpixel((1, 0), 1)
    im.info["transparency"] = 0
    out = to_rgb(im)
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((1, 0)) == (200, 0, 0)


def test_sixteen_bit_grey_is_scaled_not_clipped() -> None:
    out = to_rgb(Image.new("I;16", (1, 1), 32768))
    assert out.getpixel((0, 0)) == (128, 128, 128)


def test_exif_orientation_applied(tmp_path: Path) -> None:
    im = Image.new("RGB", (40, 20), (0, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    src = _save(im, tmp_path / "rot.jpg", "JPEG", exif=exif.tobytes())
    _, w, h = normalize_file(src, tmp_path / "out")
    assert (w, h) == (20, 40)


def test_icc_profile_converted_to_srgb() -> None:
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    im = Image.new("RGB", (1, 1), (50, 100, 150))
    im.info["icc_profile"] = srgb.tobytes()
    assert to_rgb(im).getpixel((0, 0)) == pytest.approx((50, 100, 150), abs=1)


def test_image_id_is_sha256_prefix_of_original_bytes(tmp_path: Path) -> None:
    src = _save(Image.new("RGB", (3, 3)), tmp_path / "x.png", "PNG")
    data = src.read_bytes()
    assert image_id_from_bytes(data) == hashlib.sha256(data).hexdigest()[:16]
    image_id, _, _ = normalize_file(src, tmp_path / "out")
    assert image_id == image_id_from_bytes(data)


def test_identical_bytes_share_one_output_and_raw_is_untouched(tmp_path: Path) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (5, 5), (1, 2, 3)).save(buf, format="PNG")
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(buf.getvalue())
    b.write_bytes(buf.getvalue())
    before = a.read_bytes()
    assert normalize_file(a, tmp_path / "out")[0] == normalize_file(b, tmp_path / "out")[0]
    assert len(list((tmp_path / "out").iterdir())) == 1
    assert a.read_bytes() == before
