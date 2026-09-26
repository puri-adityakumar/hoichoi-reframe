"""Render a crop box to the target output size (plain Lanczos resize; the box
aspect always equals the target ratio, so no fill/extension is needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

JPEG_QUALITY = 90


def render_crop(img: Image.Image, box: list[int], out_w: int, out_h: int, out_path: str | Path) -> Path:
    x, y, w, h = box
    crop = img.crop((x, y, x + w, y + h))
    crop = crop.resize((out_w, out_h), Image.LANCZOS)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".png":
        crop.save(out)
    else:
        crop.convert("RGB").save(out, "JPEG", quality=JPEG_QUALITY)
    return out


def to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2_bgr_to_rgb(img_bgr))


def cv2_bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
