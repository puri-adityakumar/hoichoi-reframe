"""Preview invariants: preview_key must always be an image the web can render.

Regression cover for the bug where video outputs stored a downscaled mp4 as
`outputs.preview_key`; the web renders that in an <img>, so the thumbnail came
out as a broken-image icon.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from worker.reframe.preview import (
    assert_image_preview, is_image_preview, make_image_preview, make_video_poster,
)

PP = "outputs/adb804fe/instagram_reel.preview"


def _make_video(path: Path, w: int, h: int, secs: float = 4.0, fps: int = 25) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}",
         "-t", str(secs), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.mark.parametrize("key,ok", [
    (f"{PP}.jpg", True),
    (f"{PP}.jpeg", True),
    (f"{PP}.png", True),
    (f"{PP}.webp", True),
    (f"{PP}.JPG", True),
    (f"{PP}.mp4", False),      # the bug: video in an <img>
    (f"{PP}.webm", False),
    (f"{PP}.mov", False),
    (f"{PP}.mp4.txt", False),  # extension is what the browser sniffs on
    (PP, False),               # no extension at all
])
def test_is_image_preview(key: str, ok: bool) -> None:
    assert is_image_preview(key) is ok


def test_assert_image_preview_rejects_video() -> None:
    assert_image_preview(f"{PP}.jpg")
    with pytest.raises(ValueError, match="must be an image"):
        assert_image_preview(f"{PP}.mp4")


def test_video_poster_is_a_decodable_image(tmp_path: Path) -> None:
    vid = _make_video(tmp_path / "reel.mp4", 1080, 1920)
    poster = make_video_poster(vid, tmp_path / "reel.preview.jpg", duration=4.0)

    assert poster.suffix == ".jpg"
    assert is_image_preview(poster.name)
    with Image.open(poster) as im:
        im.load()  # raises if ffmpeg wrote something undecodable
        assert im.format == "JPEG"
        # long edge capped at VIDEO_POSTER_EDGE, aspect ratio preserved
        assert max(im.size) == 480
        assert im.size[1] > im.size[0]  # 9:16 stays vertical


def test_image_preview_is_a_decodable_image(tmp_path: Path) -> None:
    src = tmp_path / "still.png"
    Image.new("RGB", (1920, 1080), (10, 20, 30)).save(src)
    out = make_image_preview(src, tmp_path / "still.preview.jpg")
    with Image.open(out) as im:
        im.load()
        assert im.format == "JPEG"
        assert max(im.size) == 240


def test_poster_falls_back_to_probing_duration(tmp_path: Path) -> None:
    vid = _make_video(tmp_path / "clip.mp4", 640, 360, secs=3.0)
    poster = make_video_poster(vid, tmp_path / "clip.preview.jpg")  # no duration hint
    with Image.open(poster) as im:
        im.load()
        assert im.format == "JPEG"
