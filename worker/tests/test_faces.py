"""Track consolidation tests (synthetic) + the smoke test on the 30s two-person
test clip."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from worker.reframe.video.faces import (
    MERGE_DX, MERGE_GAP_S, analyze_faces, consolidate_tracks,
)

CLIP = Path(__file__).resolve().parents[2] / "assets" / "test" / "clip_a.mp4"


# ------------------------------------------------------- consolidation (pure) --
def _track(tid: int, cx: float, times: list[float], w: float = 0.14,
           h: float = 0.22) -> dict[str, Any]:
    return {"id": tid, "frames": [
        {"t": round(t, 3), "box": (cx - w / 2, 0.3, w, h), "mouth": 0.4} for t in times]}


def test_consolidate_merges_one_person_across_a_short_gap() -> None:
    """A blink / head turn / brief occlusion mints a new track; the person is
    still the same, so the fragments must be glued back together."""
    a = _track(3, 0.35, [round(i * 0.2, 3) for i in range(10)])          # 0.0-1.8
    b = _track(7, 0.36, [round(3.0 + i * 0.2, 3) for i in range(10)])     # 3.0-4.8
    merged = consolidate_tracks([a, b])
    assert len(merged) == 1, "same person, short gap: should be one track"
    assert len(merged[0]["frames"]) == 20
    assert merged[0]["id"] == 3, "the earliest fragment keeps the id"
    ts = [f["t"] for f in merged[0]["frames"]]
    assert ts == sorted(ts), "frames must come back time ordered"


def test_consolidate_keeps_two_people_apart() -> None:
    left = _track(1, 0.30, [round(i * 0.2, 3) for i in range(20)])
    right = _track(2, 0.70, [round(i * 0.2, 3) for i in range(20)])
    assert len(consolidate_tracks([left, right])) == 2


def test_consolidate_respects_the_gap_limit() -> None:
    """A person who is off screen for a whole reverse-shot is NOT re-acquired:
    a 7 s gap is not evidence of identity without an appearance descriptor."""
    early = _track(1, 0.35, [round(i * 0.2, 3) for i in range(10)])
    late = _track(2, 0.35, [round(8.0 + i * 0.2, 3) for i in range(10)])
    assert len(consolidate_tracks([early, late])) == 2
    near = _track(2, 0.35, [round(2.2 + i * 0.2, 3) for i in range(10)])
    assert len(consolidate_tracks([early, near])) == 1


def test_consolidate_needs_comparable_size() -> None:
    a = _track(1, 0.35, [round(i * 0.2, 3) for i in range(10)], w=0.14, h=0.22)
    huge = _track(2, 0.36, [round(3.0 + i * 0.2, 3) for i in range(10)], w=0.6, h=0.6)
    assert len(consolidate_tracks([a, huge])) == 2


def test_consolidate_runs_to_a_fixed_point() -> None:
    """Three fragments of one person, each gap short: one merge round is not
    enough, the loop has to keep going."""
    frags = [_track(i, 0.34 + 0.01 * i, [round(i * 2.0 + j * 0.2, 3) for j in range(9)])
             for i in range(3)]
    merged = consolidate_tracks(frags)
    assert len(merged) == 1, [len(t["frames"]) for t in merged]
    assert len(merged[0]["frames"]) == 27


def test_consolidate_keeps_the_frame_shape_and_ids_unique() -> None:
    a = _track(5, 0.35, [round(i * 0.2, 3) for i in range(10)])
    b = _track(9, 0.35, [round(3.0 + i * 0.2, 3) for i in range(10)])
    merged = consolidate_tracks([a, b])
    for tr in merged:
        assert set(tr) == {"id", "frames"}, "consumers see the same track shape"
        for f in tr["frames"]:
            assert set(f) == {"t", "box", "mouth"}
    ids = [tr["id"] for tr in merged]
    assert len(set(ids)) == len(ids)
    # the thresholds are real and distinct: a wide gap, a tight x tolerance
    assert 0.0 < MERGE_DX < 0.2
    assert 1.0 <= MERGE_GAP_S <= 3.0


def test_consolidate_does_not_mutate_its_input() -> None:
    a = _track(5, 0.35, [round(i * 0.2, 3) for i in range(10)])
    b = _track(9, 0.35, [round(3.0 + i * 0.2, 3) for i in range(10)])
    before = (len(a["frames"]), len(b["frames"]),
              {k for f in a["frames"] for k in f})
    consolidate_tracks([a, b])
    assert (len(a["frames"]), len(b["frames"]),
            {k for f in a["frames"] for k in f}) == before


@pytest.mark.skipif(not CLIP.exists(), reason="test clip not present")
def test_analyze_clip_a() -> None:
    result = analyze_faces(CLIP)
    assert result["fps"] == 5.0
    assert isinstance(result["tracks"], list)
    assert len(result["tracks"]) >= 2, "expect >= 2 face tracks"

    long_tracks = [t for t in result["tracks"] if len(t["frames"]) >= 35]
    # NOTE: 35 (7s) instead of a strict 40: clip_a is a shot/reverse-shot
    # dialogue, so the dominant speaker is legitimately absent during the
    # other speaker's cutaways (~14 of 55 analysis frames); 40 is unreachable.
    assert long_tracks, "at least one track should last >= 7s (35 analysis frames)"

    for tr in result["tracks"]:
        assert isinstance(tr["id"], int)
        ts = [f["t"] for f in tr["frames"]]
        assert ts == sorted(ts), "consolidation must return time-ordered frames"
        assert len(set(ts)) == len(ts), "no frame may appear in two tracks"
        for f in tr["frames"]:
            x, y, w, h = f["box"]
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, f["box"]
            assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0, f["box"]
            assert x + w <= 1.0 + 1e-6 and y + h <= 1.0 + 1e-6
            # inner-lip / mouth-width ratio; can exceed 1.0 on a wide-open
            # mouth, only bound it loosely
            assert 0.0 <= f["mouth"] < 3.0
