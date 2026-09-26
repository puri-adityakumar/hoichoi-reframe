"""Importance map: faces (MediaPipe) + saliency (OpenCV spectral residual),
with the watermark band zeroed before faces are painted (faces win)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

P4_ROOT = Path(__file__).resolve().parents[3]
FACE_MODEL = P4_ROOT / "worker" / "models" / "face_landmarker.task"

ANALYSIS_LONG_EDGE = 1600
FACE_WEIGHT = 1.0
SALIENCY_WEIGHT = 0.5
FEATHER_PX = 31
WATERMARK_BAND = (0.40, 0.60)  # normalized y range of the burned-in watermark


def importance_map(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (map, scale) where map is (h, w) float 0-1 at the analysis scale
    and scale = analysis_width / original_width."""
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

    # Zero the watermark band FIRST so burned-in text never attracts crops.
    y0, y1 = int(WATERMARK_BAND[0] * ah), int(WATERMARK_BAND[1] * ah)
    importance[y0:y1, :] = 0.0

    # Then paint faces (faces painted after win over the band).
    face_mask = np.zeros((ah, aw), np.float32)
    for x, y, fw, fh in _face_boxes(small):
        face_mask[max(0, y):y + fh, max(0, x):x + fw] = FACE_WEIGHT
    face_mask = cv2.GaussianBlur(face_mask, (FEATHER_PX, FEATHER_PX), 0)
    face_mask = np.clip(face_mask / max(face_mask.max(), 1e-6), 0.0, 1.0)
    importance = np.maximum(importance, face_mask)

    return np.clip(importance, 0.0, 1.0), scale


def _face_boxes(img_bgr: np.ndarray, min_score: float = 0.4) -> list[tuple[int, int, int, int]]:
    """Face bounding boxes at the given image scale."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    opts = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=6,
        min_face_detection_confidence=min_score,
    )
    boxes: list[tuple[int, int, int, int]] = []
    with vision.FaceLandmarker.create_from_options(opts) as landmarker:
        result = landmarker.detect(mp_img)
    for det in result.face_landmarks:
        xs = [lm.x * w for lm in det]
        ys = [lm.y * h for lm in det]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        pad_x, pad_y = (x1 - x0) * 0.15, (y1 - y0) * 0.15
        boxes.append((
            max(0, int(x0 - pad_x)),
            max(0, int(y0 - pad_y)),
            min(w, int(x1 + pad_x)) - max(0, int(x0 - pad_x)),
            min(h, int(y1 + pad_y)) - max(0, int(y0 - pad_y)),
        ))
    return boxes
