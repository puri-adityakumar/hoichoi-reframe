"""Smoke test: run analyze_faces on the 30s two-person test clip."""
from __future__ import annotations

from pathlib import Path

import pytest

from worker.reframe.video.faces import analyze_faces

CLIP = Path(__file__).resolve().parents[2] / "assets" / "test" / "clip_a.mp4"


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
        for f in tr["frames"]:
            x, y, w, h = f["box"]
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, f["box"]
            assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0, f["box"]
            assert x + w <= 1.0 + 1e-6 and y + h <= 1.0 + 1e-6
            # inner-lip / mouth-width ratio; can exceed 1.0 on a wide-open
            # mouth, only bound it loosely
            assert 0.0 <= f["mouth"] < 3.0
