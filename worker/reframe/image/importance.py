"""Importance map: faces (MediaPipe) + saliency (OpenCV spectral residual), with
the watermark band zeroed before faces are painted (faces win). Band suppression
is conditional - see band_is_actionable.

FACE DETECTION IS TILED, NOT WHOLE-FRAME. The BlazeFace short-range head has a
fixed input size, so it only proposes a face that occupies roughly >=13-15% of
the frame. On a master shot that condition is never met, and the whole-frame
pass returns ZERO faces. Measured on assets/given/input_image.png (4000x4000,
two faces covering 4.1% and 4.2% of the frame):
  * whole frame at 4000/3200/2400/1600/1200/1000/800/600 px -> 0 faces, at
    min_face_detection_confidence 0.4/0.2/0.05/0.01 -> still 0 faces.
  * 2x upscale of the 1600 copy -> 0 faces (upscaling adds no information).
  * tiles cut from the 1600 analysis copy, 20% overlap:
      5x5 (step 320, tile 384) -> 2 faces, 0 false positives   <- used here
      4x4 (step 400, tile 480) -> 1-2 faces, 13.8% fill: already on the cliff
      3x3 (tile 533)           -> 1 face, tiles too big
      sliding 300px tiles      -> 2 faces + 2 FALSE positives (the carved ceiling
                                  motif at y~0.076, the blood pool at y~0.625)
  * holding one face at a constant ~180px while the frame grows, detection dies
    exactly between 1300px (13.8% fill, 1 face) and 1400px (12.9% fill, 0 faces).
The constraint is therefore a FILL FRACTION, never a pixel count, and no
resolution or confidence knob can recover it. Do not "optimise" the tile size
back down: 320px sits comfortably above the cliff, 300px falls off it into
hallucinated faces. Do not raise it either - 480px is already marginal.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

# crop-window geometry (zoom levels, aspect math) lives in crops.py; importing it
# here is safe because crops.py has no import-time dependency on this module.
from .crops import MIN_ZOOM, can_avoid_band, iou

P4_ROOT = Path(__file__).resolve().parents[3]
FACE_MODEL = P4_ROOT / "worker" / "models" / "face_landmarker.task"

ANALYSIS_LONG_EDGE = 1600
FACE_WEIGHT = 1.0
SALIENCY_WEIGHT = 0.5
FEATHER_PX = 31
WATERMARK_BAND = (0.40, 0.60)  # normalized y range of the burned-in watermark

TILE_STEP = 320  # analysis-px stride between tile origins
TILE_SCALE = 1.2  # tile edge = TILE_STEP * TILE_SCALE, i.e. ~20% overlap
TILE_PX = int(TILE_STEP * TILE_SCALE)  # 384
MIN_TILE_PX = 300  # below this, tiles start hallucinating faces (see docstring)
BOX_PAD = 0.15  # face box padding, as a fraction of the landmark box
MERGE_IOU = 0.3  # above this, two boxes are the same face seen twice


def importance_map(
    img_bgr: np.ndarray,
    ratios: Sequence[str] | None = None,
) -> tuple[np.ndarray, float]:
    """Return (map, scale) where map is (h, w) float 0-1 at the analysis scale
    and scale = analysis_width / original_width.

    `ratios` are the target aspect ratios this map will be searched for; the
    watermark band is zeroed only if every one of them can be steered clear of
    it (see band_is_actionable). None keeps the legacy always-zero behaviour.
    """
    h, w = img_bgr.shape[:2]
    scale = ANALYSIS_LONG_EDGE / max(h, w)
    aw, ah = round(w * scale), round(h * scale)
    small = cv2.resize(img_bgr, (aw, ah), interpolation=cv2.INTER_AREA)

    sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, sal_map = sal.computeSaliency(small.astype(np.float32) / 255.0)
    sal_map = np.asarray(sal_map, dtype=np.float32) if ok else np.zeros((ah, aw), np.float32)
    if sal_map.max() > 0:
        sal_map = sal_map / sal_map.max()
    sal_map = cv2.GaussianBlur(sal_map, (0, 0), 3)

    importance = sal_map * SALIENCY_WEIGHT

    # Zero the watermark band FIRST so burned-in text never attracts crops -
    # but only where suppressing it can actually steer a crop (see below).
    if _suppresses_band(w, h, ratios):
        y0, y1 = int(WATERMARK_BAND[0] * ah), int(WATERMARK_BAND[1] * ah)
        importance[y0:y1, :] = 0.0

    # Then paint faces (faces painted after win over the band).
    face_mask = np.zeros((ah, aw), np.float32)
    for x, y, fw, fh in _face_boxes(small):
        x1, y1 = min(aw, x + fw), min(ah, y + fh)
        face_mask[max(0, y):y1, max(0, x):x1] = FACE_WEIGHT
    face_mask = cv2.GaussianBlur(face_mask, (FEATHER_PX, FEATHER_PX), 0)
    face_mask = np.clip(face_mask / max(face_mask.max(), 1e-6), 0.0, 1.0)
    importance = np.maximum(importance, face_mask)

    return np.clip(importance, 0.0, 1.0), scale


def band_is_actionable(img_w: int, img_h: int, ratio: str) -> bool:
    """True when a crop of `ratio` can be placed entirely outside the watermark
    band at the zoom levels the search actually uses.

    Measured on the 1:1 master: 16:9 CAN (its smallest window is 1125px tall
    against 1600px of band-free height), while 9:16, 4:5 and 1:1 CANNOT - their
    smallest window is 2000px tall, so every candidate at every zoom intersects
    y 0.40-0.60. Zeroing the band for those ratios is a uniform dead zone: it
    cannot steer anything, it only deflates all scores equally, and it deletes
    the saliency of exactly the region the faces occupy (face B at y 0.466-0.507
    sits entirely inside it).
    """
    return can_avoid_band(img_w, img_h, ratio, WATERMARK_BAND, MIN_ZOOM)


def _suppresses_band(img_w: int, img_h: int, ratios: Sequence[str] | None) -> bool:
    """Zero the band only if EVERY requested ratio can be steered away from it.
    The map is shared by all ratios, so as soon as one consumer is forced to
    overlap the band, suppression stops steering anything for that consumer and
    only costs saliency. None (no ratio given) keeps the legacy behaviour."""
    if not ratios:
        return True
    return all(band_is_actionable(img_w, img_h, r) for r in ratios)


def _tile_origins(length: int, tile: int) -> list[int]:
    """Tile origins along one axis, the last one flush with the far edge so
    the grid still covers the whole image (see the module docstring)."""
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile + 1, TILE_STEP))
    if origins[-1] + tile < length:
        origins.append(length - tile)
    return origins


def _face_boxes(img_bgr: np.ndarray, min_score: float = 0.4) -> list[tuple[int, int, int, int]]:
    """Face boxes (x, y, w, h) in the pixel space of img_bgr.

    Whole-frame pass PLUS a tile grid, deduped. The tile grid is what makes small
    faces detectable at all; the whole-frame pass is the cheap fast path when
    faces are large and the only one that runs for images smaller than a tile.
    One landmarker serves every pass - it is the single most expensive object
    here, so building it per tile would dominate the runtime.
    """
    h, w = img_bgr.shape[:2]
    opts = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=6,
        min_face_detection_confidence=min_score,
    )
    boxes: list[tuple[int, int, int, int]] = []
    tile = min(TILE_PX, h, w)
    with vision.FaceLandmarker.create_from_options(opts) as landmarker:
        boxes += _detect_faces(landmarker, img_bgr, 0, 0)
        if tile >= MIN_TILE_PX:
            for y0 in _tile_origins(h, tile):
                for x0 in _tile_origins(w, tile):
                    if (x0, y0) == (0, 0) and tile == w and tile == h:
                        continue  # single tile: already covered above
                    boxes += _detect_faces(landmarker, img_bgr[y0:y0 + tile, x0:x0 + tile],
                                           x0, y0)
    return _merge_boxes(boxes, w, h)


def _detect_faces(landmarker: vision.FaceLandmarker, tile_bgr: np.ndarray,
                  off_x: int, off_y: int) -> list[tuple[int, int, int, int]]:
    """Boxes for one tile, translated into the parent image's pixel space."""
    th, tw = tile_bgr.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_img)
    boxes: list[tuple[int, int, int, int]] = []
    for det in result.face_landmarks:
        xs = [lm.x * tw for lm in det]
        ys = [lm.y * th for lm in det]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        pad_x, pad_y = (x1 - x0) * BOX_PAD, (y1 - y0) * BOX_PAD
        x0, x1 = max(0, int(x0 - pad_x)), min(tw, int(x1 + pad_x))
        y0, y1 = max(0, int(y0 - pad_y)), min(th, int(y1 + pad_y))
        if x1 > x0 and y1 > y0:
            boxes.append((x0 + off_x, y0 + off_y, x1 - x0, y1 - y0))
    return boxes


def _merge_boxes(boxes: list[tuple[int, int, int, int]], w: int, h: int
                 ) -> list[tuple[int, int, int, int]]:
    """Clamp to the image, then drop duplicates: a box whose centre falls inside
    a larger kept box, or overlapping one by more than MERGE_IOU, is the same
    face caught in a neighbouring tile. Larger boxes win."""
    kept: list[tuple[int, int, int, int]] = []
    for x, y, bw, bh in sorted(boxes, key=lambda b: -b[2] * b[3]):
        bw, bh = min(bw, w), min(bh, h)
        if bw <= 0 or bh <= 0:
            continue
        x, y = min(max(0, x), w - bw), min(max(0, y), h - bh)
        cx, cy = x + bw / 2, y + bh / 2
        cand = (x, y, bw, bh)
        if any((kx <= cx <= kx + kw and ky <= cy <= ky + kh) or iou(cand, k) > MERGE_IOU
               for k in kept for kx, ky, kw, kh in [k]):
            continue
        kept.append((x, y, bw, bh))
    return kept
