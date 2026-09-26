"""Tests for the image smart-crop path.

The synthetic fixtures below are the point of this file. The run that shipped an
empty temple pillar as the 9:16 crop passed all five validations because the only
tests were shape/range assertions on a real image, and the candidate search was
never called with a face_mask at all - so the face branch of the scorer had zero
coverage in the repo. Nothing here depends on assets/given (git-ignored, absent on
CI); the tests that do use the real master are skipped when it is missing.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from worker.reframe.image import importance as imp_mod
from worker.reframe.image.crops import (
    ZOOM_FRACTIONS,
    BoxSums,
    _face_blobs,
    _map_box,
    _score,
    band_free_rows,
    best_crop_candidates,
    can_avoid_band,
    largest_window,
)
from worker.reframe.image.importance import (
    _merge_boxes,
    _tile_origins,
    band_is_actionable,
    importance_map,
)

P4_ROOT = Path(__file__).resolve().parents[2]
MASTER = P4_ROOT / "assets" / "given" / "input_image.png"

# Analysis-space geometry for the synthetic fixtures: a 100x75 map (NOT square, so
# a silently ignored aspect conversion cannot pass by luck) standing in for a
# 300x400 full-res image, i.e. a uniform 0.25 scale on both axes.
MAP_H, MAP_W = 100, 75
IMG_W, IMG_H = 300, 400
FACE_BOX = (50, 70, 20, 20)  # analysis space -> (200, 280, 80, 80) full-res
SAL_BOX = (0, 50, 30, 50)  # lit floor, analysis space, overlapping no face
SAL_LEVEL = 0.1  # so a face-free window scores 0.1 here, under FACE_BONUS: the soft
# term alone can lift a face window above it once the coordinates are right
SCALE = IMG_W / MAP_W  # full-res px per map px
FACE_FULL = tuple(int(v * SCALE) for v in FACE_BOX)  # (200, 280, 80, 80)


def _fixtures() -> tuple[np.ndarray, np.ndarray]:
    """(importance, face_mask) in analysis space. Saliency sits bottom-left, the
    face middle-right, so only the face term can make a face-holding window beat
    the lit one."""
    importance = np.zeros((MAP_H, MAP_W), np.float32)
    sx, sy, sw, sh = SAL_BOX
    importance[sy:sy + sh, sx:sx + sw] = SAL_LEVEL
    mask = np.zeros((MAP_H, MAP_W), np.float32)
    fx, fy, fw, fh = FACE_BOX
    mask[fy:fy + fh, fx:fx + fw] = 1.0
    return importance, mask


def _holds(box: list[int], other_map: tuple[int, int, int, int], tol: int = 1,
          shape: tuple[int, int] = (MAP_H, MAP_W),
          img: tuple[int, int] = (IMG_W, IMG_H)) -> bool:
    """Does a FULL-RES candidate box contain a map-space (x, y, w, h)?

    Compared in the analysis space the decision is actually made in, with `tol`
    map pixels of slack: the map quantises a full-res box onto its own grid, so a
    crop one analysis pixel short of a face edge is still a crop holding the face
    at every resolution this engine can see.
    """
    mx, my, mw, mh = _map_box(box[0], box[1], box[2], box[3], shape, img[0], img[1])
    x, y, w, h = other_map
    return (mx - tol <= x and my - tol <= y
            and mx + mw + tol >= x + w and my + mh + tol >= y + h)


def _holds_full(box: list[int], other_full: tuple[int, int, int, int], tol: int) -> bool:
    """Same containment question asked directly in full-res coordinates, i.e.
    without going through the conversion under test."""
    x, y, w, h = other_full
    return (box[0] - tol <= x and box[1] - tol <= y
            and box[0] + box[2] + tol >= x + w and box[1] + box[3] + tol >= y + h)


# ------------------------------------------------------- coordinate space ----

def test_score_converts_full_res_box_into_map_space() -> None:
    """Equal-sized windows, one holding the face and one holding only lit floor.
    Slicing with full-res indices into a smaller map clamps to an empty array,
    which scores 0.0 and inverts this comparison."""
    importance, mask = _fixtures()
    imp_sums, face_sums = BoxSums(importance), BoxSums(mask)
    # 9:16 at the 0.5 zoom (112x200), right edge and bottom edge flush with the
    # face's own, so the only thing that can lift it above the floor window is
    # the face term - and only if the box is converted into map space first.
    cw, ch = 112, 200
    fx, fy, fw, fh = FACE_FULL
    with_face = _score(imp_sums, face_sums, fx + fw - cw, fy + fh - ch, cw, ch,
                       IMG_W, IMG_H)
    without = _score(imp_sums, face_sums, 0, IMG_H - ch, cw, ch, IMG_W, IMG_H)
    assert without > 0.0, "fixture sanity: the floor window must be lit"
    assert with_face > without, (
        f"face window scored {with_face}, floor window {without}")


def test_candidates_916_prefers_the_face_window_over_saliency() -> None:
    """End-to-end version of the same regression: the top candidate must hold
    the face even though its window is dark."""
    importance, mask = _fixtures()
    res = best_crop_candidates(importance, IMG_W, IMG_H, "9:16", k=3, face_mask=mask)
    top = res["candidates"][0]
    assert _holds(top["box"], FACE_BOX), top
    assert _holds_full(top["box"], FACE_FULL, tol=int(SCALE)), top
    assert res["faces_found"] == 1
    assert res["faces_kept"] == 1


def test_map_box_never_returns_an_empty_window() -> None:
    """A full-res box starting past the analysis map must still slice a non-empty
    window; this is the case that scored exactly 0.0 for x >= 1600."""
    for x in (0, 149, 1600, 3999):
        mx, my, mw, mh = _map_box(x, 0, 2250, 4000, (1600, 1600), 4000, 4000)
        assert mw > 0 and mh > 0
        assert 0 <= mx < 1600 and 0 <= my < 1600
        assert mx + mw <= 1600 and my + mh <= 1600


# --------------------------------------------------------- face influence ----

def test_face_mask_changes_the_selected_box() -> None:
    """Same map, same ratio: adding the face mask must move the answer, and the
    answer must contain the face."""
    importance, mask = _fixtures()
    blind = best_crop_candidates(importance, IMG_W, IMG_H, "9:16", k=3)
    aware = best_crop_candidates(importance, IMG_W, IMG_H, "9:16", k=3, face_mask=mask)
    assert blind["candidates"][0]["box"] != aware["candidates"][0]["box"]
    assert not _holds(blind["candidates"][0]["box"], FACE_BOX)
    assert _holds(aware["candidates"][0]["box"], FACE_BOX)
    assert _holds_full(aware["candidates"][0]["box"], FACE_FULL, tol=int(SCALE))


def test_hard_constraint_beats_brightness() -> None:
    """The lit floor is made maximally bright, so the best face-free window wins
    on raw score - and must still lose to the face window. Only the lexicographic
    key can produce that; no additive term can."""
    importance, mask = _fixtures()
    sx, sy, sw, sh = SAL_BOX
    importance[sy:sy + sh, sx:sx + sw] = 1.0
    blind = best_crop_candidates(importance, IMG_W, IMG_H, "9:16", k=3)
    assert not _holds(blind["candidates"][0]["box"], FACE_BOX), "fixture sanity"
    res = best_crop_candidates(importance, IMG_W, IMG_H, "9:16", k=3, face_mask=mask)
    top = res["candidates"][0]
    assert _holds(top["box"], FACE_BOX), res
    assert top["score"] < blind["candidates"][0]["score"], (
        "the face window must lose on raw score and win anyway")


# ------------------------------------------------------------ hard signal ----

def test_two_far_apart_faces_are_both_kept() -> None:
    importance = np.zeros((MAP_H, MAP_W), np.float32)
    mask = np.zeros((MAP_H, MAP_W), np.float32)
    mask[20:40, 5:25] = 1.0  # far left
    mask[20:40, 50:70] = 1.0  # far right
    res = best_crop_candidates(importance, IMG_W, IMG_H, "1:1", k=3, face_mask=mask)
    assert res["faces_found"] == 2
    assert res["faces_kept"] == 2
    top = res["candidates"][0]
    assert _holds(top["box"], (5, 20, 20, 20))
    assert _holds(top["box"], (50, 20, 20, 20))
    assert top["faces_kept"] == 2


def test_no_faces_reports_zero_instead_of_pretending() -> None:
    """The case that shipped: nothing detected, nothing said. It must return
    usable candidates AND say faces_found == faces_kept == 0."""
    importance, _ = _fixtures()
    empty = np.zeros((MAP_H, MAP_W), np.float32)
    res = best_crop_candidates(importance, IMG_W, IMG_H, "9:16", k=3, face_mask=empty)
    assert res["faces_found"] == 0
    assert res["faces_kept"] == 0
    assert len(res["candidates"]) == 3
    assert all(c["faces_kept"] == 0 for c in res["candidates"])

    same = best_crop_candidates(importance, IMG_W, IMG_H, "9:16", k=3)
    assert same["faces_found"] == 0 and same["faces_kept"] == 0
    assert same["candidates"][0]["box"] == res["candidates"][0]["box"]


def test_faces_that_cannot_fit_report_what_was_kept() -> None:
    """Two faces further apart than the largest window: no candidate can hold
    both. The best partial answer is still returned and faces_kept says 1 of 2."""
    importance = np.zeros((MAP_H, MAP_W), np.float32)
    mask = np.zeros((MAP_H, MAP_W), np.float32)
    mask[0:5, 0:5] = 1.0
    mask[95:100, 70:75] = 1.0
    res = best_crop_candidates(importance, IMG_W, IMG_H, "1:1", k=3, face_mask=mask)
    assert res["faces_found"] == 2
    assert res["faces_kept"] == 1
    assert res["candidates"]
    assert all(c["faces_kept"] < 2 for c in res["candidates"])

    # a blob bigger than any window is retained by nothing: honest zero
    mask = np.ones((MAP_H, MAP_W), np.float32)
    res = best_crop_candidates(importance, IMG_W, IMG_H, "1:1", k=3, face_mask=mask)
    assert res["faces_found"] == 1
    assert res["faces_kept"] == 0
    assert res["candidates"]


def test_faces_kept_is_reported_per_candidate_and_ordered() -> None:
    importance, mask = _fixtures()
    res = best_crop_candidates(importance, IMG_W, IMG_H, "16:9", k=3, face_mask=mask)
    kept = [c["faces_kept"] for c in res["candidates"]]
    assert kept == sorted(kept, reverse=True)
    assert all(isinstance(c["faces_kept"], int) for c in res["candidates"])


# ------------------------------------------------------- watermark band ------

def test_band_is_actionable_matches_the_measured_cases() -> None:
    """1:1 master, 4000px: only 16:9 can be steered clear of y 0.40-0.60."""
    assert band_is_actionable(4000, 4000, "16:9") is True
    assert band_is_actionable(4000, 4000, "9:16") is False
    assert band_is_actionable(4000, 4000, "4:5") is False
    assert band_is_actionable(4000, 4000, "1:1") is False


def test_band_is_actionable_for_other_sources_and_ratios() -> None:
    # a 16:9 crop of a 16:9 master is the whole frame: nowhere to hide
    assert band_is_actionable(1920, 1080, "16:9") is False
    # wide ratios from a square source also clear the band
    assert band_is_actionable(4000, 4000, "16:10") is True
    assert band_is_actionable(4000, 4000, "21:9") is True
    # a short band on a wide master lets 4:5 fit above it
    assert band_is_actionable(4000, 2000, "4:5") is False
    assert can_avoid_band(4000, 2000, "4:5", (0.30, 0.45)) is True
    # the geometry helpers agree with the hand calculation
    assert band_free_rows(4000, (0.40, 0.60)) == 1600
    assert largest_window(4000, 4000, "9:16") == (2250, 4000)
    assert largest_window(4000, 4000, "16:9") == (4000, 2250)


def test_band_suppression_is_skipped_when_any_ratio_cannot_avoid_it() -> None:
    assert imp_mod._suppresses_band(4000, 4000, ["16:9"]) is True
    assert imp_mod._suppresses_band(4000, 4000, ["9:16", "4:5", "1:1"]) is False
    assert imp_mod._suppresses_band(4000, 4000, ["16:9", "9:16"]) is False
    assert imp_mod._suppresses_band(4000, 4000, None) is True  # legacy default


# -------------------------------------------------- face detection plumbing --

def test_tile_grid_covers_the_analysis_frame_at_the_measured_size() -> None:
    """5x5 over 1600 at step 320 / tile 384, last origin flush with the edge."""
    origins = _tile_origins(1600, imp_mod.TILE_PX)
    assert imp_mod.TILE_PX == 384 and imp_mod.TILE_STEP == 320
    assert origins == [0, 320, 640, 960, 1216]
    assert len(origins) * len(origins) == 25
    assert origins[0] == 0 and origins[-1] + imp_mod.TILE_PX == 1600
    # any face-sized region of the frame lands in at least one full tile
    for y in range(0, 1600, 37):
        for x in range(0, 1600, 37):
            assert any(x0 <= x < x0 + imp_mod.TILE_PX for x0 in origins)
            assert any(y0 <= y < y0 + imp_mod.TILE_PX for y0 in origins)


def test_merge_boxes_dedupes_overlapping_tiles_and_clamps() -> None:
    same_face = [(650, 604, 54, 87), (648, 603, 88, 90)]  # measured duplicates
    assert _merge_boxes(same_face, 1600, 1600) == [(648, 603, 88, 90)]
    # a second, distinct face survives
    assert len(_merge_boxes(same_face + [(838, 735, 77, 84)], 1600, 1600)) == 2
    # boxes are clamped into the image, not left hanging outside it
    x, y, w, h = _merge_boxes([(1590, 1590, 40, 40)], 1600, 1600)[0]
    assert 0 <= x and 0 <= y and x + w <= 1600 and y + h <= 1600


# ------------------------------------------------------------- real master ---

@pytest.mark.skipif(not MASTER.exists(), reason="master image not present")
def test_importance_map_shape_and_range() -> None:
    """Shape, dtype and range - plus the strengthening: an all-zero map passes
    every one of those, so require real signal in it too."""
    img = cv2.imread(str(MASTER))
    assert img is not None
    imp, scale = importance_map(img)
    assert imp.ndim == 2
    assert imp.dtype == np.float32
    assert imp.min() >= 0.0 and imp.max() <= 1.0
    assert 0.0 < scale <= 1.0
    assert abs(imp.shape[0] / imp.shape[1] - img.shape[0] / img.shape[1]) < 0.01
    assert imp.max() > 0.2, "importance map is empty or nearly empty"
    assert float((imp > 0.5).mean()) > 0.001, "map has no salient pixels at all"


@pytest.mark.skipif(not MASTER.exists(), reason="master image not present")
def test_importance_map_finds_both_faces_on_the_master() -> None:
    """The bug this file exists for: the whole-frame pass finds ZERO faces on
    this image at any scale or confidence, because both faces cover ~4% of the
    frame and BlazeFace short-range needs ~13-15%."""
    img = cv2.imread(str(MASTER))
    assert img is not None
    h, w = img.shape[:2]
    scale = imp_mod.ANALYSIS_LONG_EDGE / max(h, w)
    small = cv2.resize(img, (round(w * scale), round(h * scale)),
                       interpolation=cv2.INTER_AREA)
    boxes = imp_mod._face_boxes(small)
    assert len(boxes) == 2, f"expected the two women, got {boxes}"
    # measured normalized positions of the padded boxes
    for bx, by, _, _ in boxes:
        nx, ny = bx / small.shape[1], by / small.shape[0]
        assert min(abs(nx - 0.405) + abs(ny - 0.377),
                   abs(nx - 0.524) + abs(ny - 0.459)) < 0.06, (nx, ny)


@pytest.mark.skipif(not MASTER.exists(), reason="master image not present")
def test_candidates_916() -> None:
    img = cv2.imread(str(MASTER))
    imp, _ = importance_map(img, ratios=["16:9", "1:1", "9:16", "4:5"])
    face_mask = (imp >= 0.85).astype(np.float32)
    h, w = img.shape[:2]
    res = best_crop_candidates(imp, w, h, "9:16", k=3, face_mask=face_mask)
    assert res["faces_found"] == 2, "master has two faces"
    assert res["faces_kept"] == 2, f"9:16 dropped a face: {res}"
    for c in res["candidates"]:
        x, y, bw, bh = c["box"]
        assert 0 <= x and 0 <= y and x + bw <= w and y + bh <= h
        assert abs((bw / bh) - 9 / 16) < 0.01
        assert 0.0 <= c["score"] <= 1.3

    top = res["candidates"][0]
    for blob in _face_blobs(face_mask):
        assert _holds(top["box"], blob, tol=1, shape=imp.shape, img=(w, h)), top
    # the shipped run scored 0.084 for this ratio with every face-bearing window
    # sliced to an empty array; coordinates fixed lands near 0.40
    assert top["score"] > 0.2, top


@pytest.mark.parametrize("ratio", ["16:9", "1:1", "9:16", "4:5"])
def test_crop_boxes_match_the_target_ratio(ratio: str) -> None:
    """render_crop resizes the box to the platform size with no letterboxing, so
    any aspect drift is a real distortion. Exhaustive over the four ratios and
    all four zooms: the two 0.95-zoom windows round unevenly and are the worst
    case at 0.024% (half a pixel across a 1920px platform), which is why
    render.py needs no fit/pad path. The rounding here mirrors the two lines in
    best_crop_candidates; the shortlist below checks the real path too.
    """
    rw, rh = (int(v) for v in ratio.split(":"))
    max_w, max_h = largest_window(4000, 4000, ratio)
    for frac in ZOOM_FRACTIONS:
        bw, bh = int(round(max_w * frac)), int(round(max_h * frac))
        drift = abs((bw / bh) / (rw / rh) - 1.0)
        assert drift < 0.001, (ratio, frac, bw, bh, drift)

    importance, _ = _fixtures()
    for c in best_crop_candidates(importance, 4000, 4000, ratio, k=3)["candidates"]:
        drift = abs((c["box"][2] / c["box"][3]) / (rw / rh) - 1.0)
        assert drift < 0.001, (ratio, c["box"], drift)


# -------------------------------------------------------------------- vlm ----

def test_vlm_rejection_is_not_coerced(monkeypatch: pytest.MonkeyPatch) -> None:
    """choice -1 must survive pick_best so the caller can act on it."""
    from worker.reframe import vlm

    monkeypatch.setattr(vlm, "ask_vlm",
                        lambda imgs, prompt: ('{"choice": -1, "reason": "no people"}', "gmi"))
    pick, provider = vlm.pick_best([cv2.UMat(0)], "ctx")
    assert pick["choice"] == -1
    assert pick["rejected"] is True
    assert provider == "gmi"

    monkeypatch.setattr(vlm, "ask_vlm",
                        lambda imgs, prompt: ('{"choice": 0, "reason": "ok"}', "gmi"))
    pick, _ = vlm.pick_best([cv2.UMat(0)], "ctx")
    assert pick["choice"] == 0 and pick["rejected"] is False


def test_vlm_prompt_never_offers_least_cropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """'feels least cropped' is satisfiable by an empty, well-exposed subject -
    it is the criterion that picked the empty pillar."""
    from worker.reframe import vlm

    seen: list[str] = []

    def capture(imgs, prompt):
        seen.append(prompt)
        return '{"choice": 0, "reason": "ok"}', "gmi"

    monkeypatch.setattr(vlm, "ask_vlm", capture)
    vlm.pick_best([cv2.UMat(0)] * 3, "ctx")
    prompt = seen[0]
    assert "least cropped" not in prompt
    assert "FACES are the largest" in prompt
    assert "already filtered" in prompt
    assert "-1 if NONE of" in prompt

    seen.clear()
    vlm.pick_best([cv2.UMat(0)] * 3, "ctx", subjects_preserved=False)
    assert "already filtered" not in seen[0]
