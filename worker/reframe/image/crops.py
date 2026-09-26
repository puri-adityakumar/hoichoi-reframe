"""Candidate crop search: slide windows at several zoom levels, score by
importance kept + face-containment bonus, keep top-k distinct."""
from __future__ import annotations

import numpy as np

ZOOM_FRACTIONS = [0.95, 0.8, 0.65, 0.5]  # of the largest window that fits
FACE_BONUS = 0.3
IOU_DISTINCT = 0.9
TARGET_CANDIDATES = 200  # aim ~150-250 per ratio


def best_crop_candidates(
    importance: np.ndarray,
    img_w: int,
    img_h: int,
    ratio: str,
    k: int = 3,
    face_mask: np.ndarray | None = None,
) -> list[dict]:
    """Return top-k candidates: {"score", "box": [x, y, w, h] on full-res}."""
    rw, rh = (int(v) for v in ratio.split(":"))
    target_ar = rw / rh
    img_ar = img_w / img_h

    # Largest window with the target aspect that fits inside the image.
    if target_ar > img_ar:  # target wider than image -> width-constrained
        max_w, max_h = img_w, round(img_w / target_ar)
    else:
        max_h, max_w = img_h, round(img_h * target_ar)

    candidates: list[dict] = []
    for frac in ZOOM_FRACTIONS:
        cw = max_w * frac
        ch = max_h * frac
        if cw < 8 or ch < 8:
            continue
        # stride chosen so total candidates across zooms lands ~150-250
        per_zoom = max(1, TARGET_CANDIDATES // len(ZOOM_FRACTIONS))
        max_dx = img_w - cw
        max_dy = img_h - ch
        step_x = max(1.0, (max_dx / max(1, per_zoom - 1)) if max_dx > 0 else 1.0)
        step_y = max(1.0, (max_dy / max(1, per_zoom - 1)) if max_dy > 0 else 1.0)
        ys = np.arange(0, max_dy + 0.5, step_y) if max_dy > 0 else np.array([0.0])
        xs = np.arange(0, max_dx + 0.5, step_x) if max_dx > 0 else np.array([0.0])
        for yf in ys:
            for xf in xs:
                x, y = int(round(xf)), int(round(yf))
                bw, bh = int(round(cw)), int(round(ch))
                bx, by = min(x, img_w - bw), min(y, img_h - bh)
                score = _score(importance, face_mask, bx, by, bw, bh, img_h)
                candidates.append({"score": float(score), "box": [bx, by, bw, bh]})

    candidates.sort(key=lambda c: -c["score"])

    kept: list[dict] = []
    for cand in candidates:
        if all(_iou(cand["box"], k2["box"]) < IOU_DISTINCT for k2 in kept):
            kept.append(cand)
        if len(kept) >= k:
            break
    return kept


def _score(
    importance: np.ndarray,
    face_mask: np.ndarray | None,
    x: int,
    y: int,
    w: int,
    h: int,
    img_h: int,
) -> float:
    win = importance[y:y + h, x:x + w]
    mean_imp = float(win.mean()) if win.size else 0.0
    if face_mask is None:
        return mean_imp
    fwin = face_mask[y:y + h, x:x + w]
    face_total = float(face_mask.sum())
    if face_total <= 0:
        return mean_imp
    inside = float(fwin.sum()) / face_total  # fraction of face area fully inside
    return mean_imp + FACE_BONUS * inside


def _iou(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
