"""Stable Diffusion inpainting edits on authentic images (PLAN 7.3, E14).

Model: ``stable-diffusion-v1-5/stable-diffusion-inpainting`` at a pinned revision
(CreativeML OpenRAIL-M; the planned SD 2 inpainting repository is no longer public, checked
2026-10-21), fp16 weights, safety checker off (packaging photos; a false positive would black
out the image). The mask is one OCR text line (expanded by 10%) or, without text, the logo /
random region of ``procedural.pick_region``. A square crop around the mask (at least 2x the mask
and 256 px, within the image) is resized to 512 px, inpainted with a prompt from the fixed list
in ``configs/synthesis/e14.yaml``, resized back, and only the masked pixels are pasted into the
full-resolution image, so the mask is exactly the changed area.

The ``stub`` backend (laptop self-test) replaces SD with OpenCV Telea inpainting plus a colour
shift; it exercises the plumbing and is never used for results.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from pharmachecked.synthesis.ocr import TextBox
from pharmachecked.synthesis.procedural import Box, Edit, pick_region

SD_SIZE = 512


def inpaint_box(image: Image.Image, boxes: Sequence[TextBox], rng: np.random.Generator) -> Box:
    """A text line (expanded by 10%), or a region when there is no usable text."""
    lines = [b for b in boxes if b.x1 - b.x0 >= 8 and b.y1 - b.y0 >= 8]
    if not lines:
        return pick_region(image, boxes, rng)
    b = lines[int(rng.integers(len(lines)))]
    px, py = 0.1 * (b.x1 - b.x0), 0.1 * (b.y1 - b.y0)
    return (
        int(max(0, b.x0 - px)),
        int(max(0, b.y0 - py)),
        int(min(image.width, b.x1 + px)),
        int(min(image.height, b.y1 + py)),
    )


def crop_around(image: Image.Image, box: Box) -> Box:
    """A square crop containing ``box``: side max(2 x box side, 256), clipped to the image."""
    x0, y0, x1, y1 = box
    side = int(min(max(2 * (x1 - x0), 2 * (y1 - y0), 256), image.width, image.height))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    left = int(np.clip(cx - side // 2, 0, image.width - side))
    top = int(np.clip(cy - side // 2, 0, image.height - side))
    return (left, top, left + side, top + side)


class Inpainter:
    """SD inpainting (``sd``) or the self-test stub (``stub``)."""

    def __init__(self, cfg: Any, device: str | None = None) -> None:
        self.cfg = cfg
        self.backend = str(cfg.backend)
        self._pipe: Any = None
        if self.backend == "sd":
            import torch
            from diffusers import StableDiffusionInpaintPipeline

            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if dev == "cuda" else torch.float32
            self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
                str(cfg.model_id),
                revision=str(cfg.revision),
                variant="fp16",
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(dev)
            self._pipe.set_progress_bar_config(disable=True)
            self._device = dev
        elif self.backend != "stub":
            raise ValueError(f"unknown inpainting backend {self.backend!r}")

    def _generate(
        self, crop: Image.Image, mask: Image.Image, prompt: str, strength: float, seed: int
    ) -> Image.Image:
        if self.backend == "stub":
            import cv2

            arr = np.asarray(crop).copy()
            m = (np.asarray(mask) > 127).astype(np.uint8) * 255
            out = cv2.inpaint(arr, m, 3, cv2.INPAINT_TELEA)
            out[m > 0] = np.clip(out[m > 0].astype(int) + np.array([25, -15, 10]), 0, 255)
            return Image.fromarray(out.astype(np.uint8))
        import torch

        generator = torch.Generator(device=self._device).manual_seed(seed)
        result = self._pipe(
            prompt=prompt,
            image=crop,
            mask_image=mask,
            height=SD_SIZE,
            width=SD_SIZE,
            strength=strength,
            num_inference_steps=int(self.cfg.steps),
            guidance_scale=float(self.cfg.guidance_scale),
            generator=generator,
        )
        return result.images[0]

    def edit(
        self, image: Image.Image, boxes: Sequence[TextBox], rng: np.random.Generator
    ) -> tuple[Edit, str]:
        """One inpainting edit; returns the edit and the prompt used."""
        box = inpaint_box(image, boxes, rng)
        crop_box = crop_around(image, box)
        side = crop_box[2] - crop_box[0]
        full_mask = np.zeros((image.height, image.width), dtype=bool)
        full_mask[box[1] : box[3], box[0] : box[2]] = True
        crop_mask = full_mask[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
        prompts = list(self.cfg.prompts)
        prompt = prompts[int(rng.integers(len(prompts)))]
        lo, hi = (float(v) for v in self.cfg.strength_range)
        strength = float(rng.uniform(lo, hi))
        seed = int(rng.integers(2**31 - 1))
        crop = image.crop(crop_box).resize((SD_SIZE, SD_SIZE), Image.Resampling.LANCZOS)
        mask_img = Image.fromarray((crop_mask * 255).astype(np.uint8)).resize(
            (SD_SIZE, SD_SIZE), Image.Resampling.NEAREST
        )
        generated = self._generate(crop, mask_img, prompt, strength, seed)
        back = np.asarray(generated.convert("RGB").resize((side, side), Image.Resampling.LANCZOS))
        out = np.asarray(image).copy()
        region = out[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
        region[crop_mask] = back[crop_mask]
        return Edit(Image.fromarray(out), full_mask, "inpaint", strength), prompt
