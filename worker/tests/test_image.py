"""Smoke tests for the image smart-crop path (uses the real local master)."""
from __future__ import annotations

import numpy as np
import pytest
import cv2

from worker.reframe.image.importance import importance_map
from worker.reframe.image.crops import best_crop_candidates

from pathlib import Path

P4_ROOT = Path(__file__).resolve().parents[2]
MASTER = P4_ROOT / "assets" / "given" / "input_image.png"


@pytest.mark.skipif(not MASTER.exists(), reason="master image not present")
def test_importance_map_shape_and_range() -> None:
    img = cv2.imread(str(MASTER))
    assert img is not None
    imp, scale = importance_map(img)
    assert imp.ndim == 2
    assert imp.dtype == np.float32
    assert imp.min() >= 0.0 and imp.max() <= 1.0
    assert 0.0 < scale <= 1.0
    assert abs(imp.shape[0] / imp.shape[1] - img.shape[0] / img.shape[1]) < 0.01


@pytest.mark.skipif(not MASTER.exists(), reason="master image not present")
def test_candidates_916() -> None:
    img = cv2.imread(str(MASTER))
    imp, _ = importance_map(img)
    h, w = img.shape[:2]
    cands = best_crop_candidates(imp, w, h, "9:16", k=3)
    assert len(cands) >= 1
    for c in cands:
        x, y, bw, bh = c["box"]
        assert 0 <= x and 0 <= y and x + bw <= w and y + bh <= h
        assert abs((bw / bh) - 9 / 16) < 0.01
        assert 0.0 <= c["score"] <= 1.3
