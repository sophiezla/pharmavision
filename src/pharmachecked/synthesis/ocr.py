"""Text boxes per image with EasyOCR, cached as JSON (PLAN 7.1, E14).

OCR is label-blind and unfitted, so it may run on any development image. Images are resized to
``OCR_SIDE`` px on the long side (most development photos are small, about 200-300 px, and
EasyOCR finds more text after upscaling); boxes are scaled back to the full resolution and
stored as axis-aligned ``[x0, y0, x1, y1]`` with the recognized text and confidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pharmachecked.utils.io import atomic_write_text

OCR_SIDE = 1024
MIN_CONFIDENCE = 0.2


@dataclass(frozen=True)
class TextBox:
    """One recognized text line."""

    x0: int
    y0: int
    x1: int
    y1: int
    text: str
    conf: float

    @property
    def area(self) -> int:
        """Box area in pixels."""
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)


class OcrReader:
    """EasyOCR (English) or a stub that returns one fixed box (laptop self-test)."""

    def __init__(self, backend: str = "easyocr", gpu: bool | None = None) -> None:
        self.backend = backend
        self._reader: Any = None
        if backend == "easyocr":
            import easyocr
            import torch

            use_gpu = torch.cuda.is_available() if gpu is None else gpu
            self._reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        elif backend != "stub":
            raise ValueError(f"unknown OCR backend {backend!r}")

    def read(self, image: Image.Image) -> list[TextBox]:
        """Text boxes of ``image`` at full resolution."""
        if self.backend == "stub":
            w, h = image.size
            return [TextBox(w // 8, h // 8, w // 2, h // 4, "AB12", 0.9)]
        scale = OCR_SIDE / max(image.size)
        resized = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
        boxes = []
        for points, text, conf in self._reader.readtext(np.asarray(resized)):
            xs = [p[0] / scale for p in points]
            ys = [p[1] / scale for p in points]
            if conf < MIN_CONFIDENCE or not str(text).strip():
                continue
            boxes.append(
                TextBox(
                    int(max(0, min(xs))),
                    int(max(0, min(ys))),
                    int(min(image.width, max(xs))),
                    int(min(image.height, max(ys))),
                    str(text),
                    float(conf),
                )
            )
        return boxes


def cached_boxes(
    reader: OcrReader, image_id: str, image: Image.Image, cache: Path
) -> list[TextBox]:
    """Boxes of one image, from ``cache/<image_id>.json`` when present (written otherwise)."""
    path = cache / f"{image_id}.json"
    if path.is_file():
        return [TextBox(**b) for b in json.loads(path.read_text(encoding="utf-8"))]
    boxes = reader.read(image)
    cache.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps([asdict(b) for b in boxes]) + "\n")
    return boxes
