"""Tests for the validator: one fully passing output, one wrong-ratio failure."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from worker.reframe.validate import validate_all

P4_ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((P4_ROOT / "spec" / "spec.json").read_text())
PLATFORMS = SPEC["platforms"]


def _make_video(path: Path, w: int, h: int, secs: float = 1.0, fps: int = 25) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}",
         "-t", str(secs), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def test_passing_video_output(tmp_path: Path) -> None:
    vid = _make_video(tmp_path / "reel.mp4", 1080, 1920, secs=1.0)
    rows = validate_all(SPEC, [{"platform": "instagram_reel", "path": vid}])
    by_rule = {r["rule"]: r for r in rows}
    for rule in ("ratio", "resolution", "duration_sec", "fps", "codec", "container", "size_mb"):
        assert rule in by_rule, f"missing rule {rule}"
        assert by_rule[rule]["passed"], f"{rule}: {by_rule[rule]}"


def test_failing_ratio_output(tmp_path: Path) -> None:
    # 16:9 video checked against the 9:16 reel spec -> ratio + resolution fail
    vid = _make_video(tmp_path / "wrong.mp4", 1280, 720, secs=1.0)
    rows = validate_all(SPEC, [{"platform": "instagram_reel", "path": vid}])
    by_rule = {r["rule"]: r for r in rows}
    assert not by_rule["ratio"]["passed"]
    assert not by_rule["resolution"]["passed"]
    assert by_rule["duration_sec"]["passed"]  # 1s <= 90s, honest per-rule reporting


def test_failing_image_size(tmp_path: Path) -> None:
    from PIL import Image
    img = tmp_path / "story_image.jpg"
    Image.new("RGB", (1080, 1920), "red").save(img, "JPEG", quality=100)
    # pad the JPEG with trailing bytes to blow past max_size_mb
    with open(img, "ab") as fh:
        fh.write(b"\0" * (9 * 1024 * 1024))
    rows = validate_all(SPEC, [{"platform": "story_image", "path": img}])
    by_rule = {r["rule"]: r for r in rows}
    assert not by_rule["size_mb"]["passed"]
    assert by_rule["ratio"]["passed"]
