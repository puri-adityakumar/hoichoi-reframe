"""Candidate crop search: slide windows at several zoom levels, rank by
(faces fully retained, importance kept), keep top-k distinct.

Two invariants this module exists to hold. Both were violated by the run that
shipped an empty temple pillar as the 9:16 crop, and neither is visible in the
output files - every one of the 5 validations passed.

1. Coordinate space. Candidate boxes are FULL-RESOLUTION; the importance map and
   the face mask are at the analysis scale (ANALYSIS_LONG_EDGE, see
   importance.py). Every slice converts the box into the array's own pixel space
   first, because numpy clamps silently: a 9:16 window at x=1600 sliced a
   1600-wide map to an EMPTY array, scored exactly 0.0, and made every window
   that could contain a face score worse than the sunlit floor. Only 31% of the
   true crop area was ever scored.

2. Faces are a HARD constraint, ranked ahead of saliency, not a bonus inside it.
   Candidates sort by (faces_kept desc, score desc) so a face-preserving crop
   always beats an equally-sized face-free one no matter how bright the latter
   is. A soft bonus cannot do this: with zero faces detected the term vanishes
   and nothing anywhere rejected a crop that excluded every face.
"""
from __future__ import annotations

import cv2
import numpy as np

ZOOM_FRACTIONS = [0.95, 0.8, 0.65, 0.5]  # of the largest window that fits
MIN_ZOOM = min(ZOOM_FRACTIONS)
FACE_BONUS = 0.3  # soft tie-break term only; the hard constraint is faces_kept
FACE_KEEP = 0.98  # fraction of a face that must be inside to count as kept
FACE_LEVEL = 0.5  # mask level at which a face blob is considered present
IOU_DISTINCT = 0.9
TARGET_CANDIDATES = 200  # sample positions per axis per zoom, not a final count


class BoxSums:
    """Sum of any axis-aligned box in a 2-D array, via an integral image.

    The search evaluates ~10k windows per ratio. Summing 1600x1600 slices
    directly - and re-summing the whole face mask per candidate - copies
    gigabytes and took 7.5s per ratio; with the integral image each window is
    O(1) and the same arithmetic costs milliseconds. One implementation, so the
    score cannot drift between the search and a single-window inspection.
    """

    __slots__ = ("ii", "shape", "total")

    def __init__(self, arr: np.ndarray) -> None:
        self.ii = cv2.integral(arr.astype(np.float64, copy=False), sdepth=cv2.CV_64F)
        self.shape = arr.shape
        self.total = float(arr.sum())

    def __getitem__(self, box: tuple[int, int, int, int]) -> float:
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return 0.0
        ii = self.ii
        return float(ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x])


def best_crop_candidates(
    importance: np.ndarray,
    img_w: int,
    img_h: int,
    ratio: str,
    k: int = 3,
    face_mask: np.ndarray | None = None,
) -> dict:
    """Return {"candidates": [...], "faces_found": int, "faces_kept": int}.

    Each candidate is {"score", "box": [x, y, w, h] in FULL-RES coords,
    "faces_kept": int}, ranked by (faces_kept, score) descending.

    `faces_found` is how many faces the mask holds, `faces_kept` how many the
    best returned candidate fully retains. faces_kept == 0 with faces_found > 0
    means no candidate could hold a face whole - the caller should say so rather
    than ship the crop as if it did.

    The analysis scale is derived from the map's own shape against
    (img_w, img_h), so the map must be the analysis-scale map from
    importance_map() and the box coordinates full-res. Passing a full-res map
    works too (the derived scale is then 1.0); passing mismatched dimensions
    does not, because the conversion is only as good as the two agree.
    """
    max_w, max_h = largest_window(img_w, img_h, ratio)
    face_boxes = _face_blobs(face_mask) if face_mask is not None else []
    imp_sums = BoxSums(importance)
    face_sums = None if face_mask is None else BoxSums(face_mask)

    candidates: list[dict] = []
    for frac in ZOOM_FRACTIONS:
        cw = max_w * frac
        ch = max_h * frac
        if cw < 8 or ch < 8:
            continue
        bw, bh = int(round(cw)), int(round(ch))
        # ~50 positions per axis per zoom; the distinctness filter below then
        # reduces ~10k windows per ratio to the k best distinct ones
        per_zoom = max(1, TARGET_CANDIDATES // len(ZOOM_FRACTIONS))
        max_dx = img_w - bw
        max_dy = img_h - bh
        step_x = max(1.0, (max_dx / max(1, per_zoom - 1)) if max_dx > 0 else 1.0)
        step_y = max(1.0, (max_dy / max(1, per_zoom - 1)) if max_dy > 0 else 1.0)
        ys = np.arange(0, max_dy + 0.5, step_y) if max_dy > 0 else np.array([0.0])
        xs = np.arange(0, max_dx + 0.5, step_x) if max_dx > 0 else np.array([0.0])
        for yf in ys:
            for xf in xs:
                bx = min(int(round(xf)), img_w - bw)
                by = min(int(round(yf)), img_h - bh)
                mx, my, mw, mh = _map_box(bx, by, bw, bh, face_sums.shape
                                          if face_sums else imp_sums.shape,
                                          img_w, img_h)
                candidates.append({
                    "score": _score(imp_sums, face_sums, bx, by, bw, bh, img_w, img_h),
                    "box": [bx, by, bw, bh],
                    "faces_kept": _faces_kept(face_boxes, mx, my, mw, mh),
                })

    # Lexicographic: faces first, saliency only to break ties.
    candidates.sort(key=lambda c: (-c["faces_kept"], -c["score"]))

    kept: list[dict] = []
    for cand in candidates:
        if all(iou(cand["box"], k2["box"]) < IOU_DISTINCT for k2 in kept):
            kept.append(cand)
        if len(kept) >= k:
            break
    return {
        "candidates": kept,
        "faces_found": len(face_boxes),
        "faces_kept": max((c["faces_kept"] for c in kept), default=0),
    }


def largest_window(img_w: int, img_h: int, ratio: str) -> tuple[int, int]:
    """Largest (w, h) window with the target aspect that fits inside the image."""
    rw, rh = (int(v) for v in ratio.split(":"))
    target_ar = rw / rh
    if target_ar > img_w / img_h:  # target wider than image -> width-constrained
        return img_w, round(img_w / target_ar)
    return round(img_h * target_ar), img_h


def band_free_rows(img_h: int, band: tuple[float, float]) -> int:
    """Tallest run of rows that avoids the normalized band, in pixels."""
    return max(int(band[0] * img_h), int((1.0 - band[1]) * img_h))


def can_avoid_band(img_w: int, img_h: int, ratio: str, band: tuple[float, float],
                   min_zoom: float = MIN_ZOOM) -> bool:
    """Can a crop of this ratio be placed entirely outside the watermark band?

    Only if the search reaches a window short enough to fit in the tallest
    band-free run. A window taller than that must overlap the band at every
    position, so suppressing the band there steers nothing - it can only deflate
    every score equally while deleting the saliency of the very region the faces
    occupy. See importance.band_is_actionable for the measured cases.
    """
    _, max_h = largest_window(img_w, img_h, ratio)
    return int(max_h * min_zoom) <= band_free_rows(img_h, band)


def _map_box(x: int, y: int, w: int, h: int, shape: tuple[int, ...],
             img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """FULL-RES box (x, y, w, h) -> the same window in an array's pixel space.

    Clamped to the array, never empty, never wrapped. This is the conversion
    whose absence made every candidate with x >= analysis width score 0.0.
    """
    ah, aw = shape[0], shape[1]
    x0 = min(max(int(round(x * aw / max(1, img_w))), 0), aw - 1)
    y0 = min(max(int(round(y * ah / max(1, img_h))), 0), ah - 1)
    x1 = min(max(int(round((x + w) * aw / max(1, img_w))), x0 + 1), aw)
    y1 = min(max(int(round((y + h) * ah / max(1, img_h))), y0 + 1), ah)
    return x0, y0, x1 - x0, y1 - y0


def _score(
    imp_sums: BoxSums,
    face_sums: BoxSums | None,
    x: int,
    y: int,
    w: int,
    h: int,
    img_w: int,
    img_h: int,
) -> float:
    """Importance kept in the window, plus a soft fraction-of-face-area term.
    A face-free window and a face-holding one of the SAME size must not tie;
    the hard guarantee lives in best_crop_candidates' sort key."""
    box = _map_box(x, y, w, h, imp_sums.shape, img_w, img_h)
    mean_imp = imp_sums[box] / (box[2] * box[3])
    if face_sums is None or face_sums.total <= 0:
        return mean_imp
    fbox = _map_box(x, y, w, h, face_sums.shape, img_w, img_h)
    return mean_imp + FACE_BONUS * face_sums[fbox] / face_sums.total


def _face_blobs(face_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """One box per connected face blob in the mask, in the mask's pixel space.
    Blob size is what the crop then has to retain, so a caller that wants whole
    faces (not just their bright cores) should pass the mask thresholded high
    enough to exclude the feathered halo - two faces whose halos touch would
    otherwise arrive here as a single blob."""
    binary = (face_mask >= FACE_LEVEL).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    return [(int(stats[i][0]), int(stats[i][1]), int(stats[i][2]), int(stats[i][3]))
            for i in range(1, count) if stats[i][4] > 0]


def _faces_kept(face_boxes: list[tuple[int, int, int, int]],
                x: int, y: int, w: int, h: int) -> int:
    """How many faces have >= FACE_KEEP of their area inside a window given in
    the FACE MASK's pixel space (the same space face_boxes lives in)."""
    kept = 0
    for fx, fy, fw, fh in face_boxes:
        ix = min(x + w, fx + fw) - max(x, fx)
        iy = min(y + h, fy + fh) - max(y, fy)
        if ix > 0 and iy > 0 and ix * iy >= FACE_KEEP * fw * fh:
            kept += 1
    return kept


def iou(a: tuple[int, int, int, int] | list[int],
        b: tuple[int, int, int, int] | list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
