"""Tests for the tracking debug video and the raw VLM call log. All synthetic
and offline: a 2 s two-square video stands in for the proxy, the VLM HTTP call
is monkeypatched."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from worker.reframe.video.annotate import SPEAKER_COLORS, annotate_tracks
from worker.reframe.video.faces import ANALYSIS_FPS

# ------------------------------------------------------------ synthetic video --
FPS, W, H, DUR = 25, 320, 240, 2.0
BLUE, RED = (255, 0, 0), (0, 0, 255)  # BGR squares in the source


def _make_video(path: Path) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened()
    n = int(FPS * DUR)
    for i in range(n):
        t = i / FPS
        frame = np.full((H, W, 3), 40, dtype=np.uint8)
        # square A drifts right from x=40, square B drifts left from x=240
        ax = int(40 + 50 * t)
        bx = int(240 - 50 * t)
        frame[100:140, ax:ax + 40] = BLUE
        frame[100:140, bx:bx + 40] = RED
        vw.write(frame)
    vw.release()


def _square_track(tid: int, x0: float, dx: float) -> dict[str, Any]:
    """Normalized track frames on the 0.2 s analysis grid, box 40x40 px."""
    frames = []
    for i in range(int(DUR / 0.2) + 1):
        t = round(i * 0.2, 3)
        x_px = x0 + dx * t
        frames.append({"t": t, "box": (x_px / W, 100 / H, 40 / W, 40 / H),
                       "mouth": 0.5})
    return {"id": tid, "frames": frames}


def _fused() -> dict[str, Any]:
    return {"speaker_track": {"0": 0, "1": 1},
            "speaker_confidence": {"0": 0.83, "1": 0.71},
            "method": "sarvam+mouth"}


def _ffprobe(path: Path) -> dict[str, Any]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        check=True, capture_output=True, text=True).stdout
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    return {"duration": float(data["format"]["duration"]),
            "codec": v["codec_name"]}


def _color_pixels(frame: np.ndarray, color: tuple[int, int, int],
                  region: tuple[int, int, int, int], tol: int = 60) -> int:
    """Count pixels in region within tol of the BGR color."""
    x0, y0, x1, y1 = region
    crop = frame[y0:y1, x0:x1].astype(int)
    dist = np.abs(crop - np.array(color)).max(axis=2)
    return int((dist < tol).sum())


# ---------------------------------------------------------------- tests --
def test_annotate_draws_boxes_and_labels(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _make_video(src)
    analysis = {"video": str(src), "probe": {"duration": DUR, "fps": FPS,
                                             "width": W, "height": H},
                "shots": [], "tracks": [_square_track(0, 40, +50),
                                        _square_track(1, 240, -50)]}
    out = tmp_path / "debug_tracking.mp4"
    annotate_tracks(src, analysis, _fused(), out)

    assert out.exists() and out.stat().st_size > 0
    pr = _ffprobe(out)
    assert pr["codec"] == "h264"
    assert abs(pr["duration"] - DUR) < 0.5

    # mid frame (t=1.0 -> frame 5 at the 5 fps analysis grid)
    cap = cv2.VideoCapture(str(out))
    fps_out = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(1.0 * fps_out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None

    teal, amber = SPEAKER_COLORS[0], SPEAKER_COLORS[1]
    # track 0 at t=1.0: x_px = 40+50 = 90; track 1: 240-50 = 190 (y 100-140)
    # box edge pixels near the expected top edge, per-speaker color
    assert _color_pixels(frame, teal, (75, 95, 120, 145)) > 50, \
        "track 0 (Speaker A) box not drawn at its expected location"
    assert _color_pixels(frame, amber, (175, 95, 220, 145)) > 50, \
        "track 1 (Speaker B) box not drawn at its expected location"
    # label backing rect sits ABOVE the box top edge
    assert _color_pixels(frame, teal, (70, 60, 160, 99)) > 100, \
        "Speaker A label backing not found above its box"
    assert _color_pixels(frame, amber, (170, 60, 260, 99)) > 100, \
        "Speaker B label backing not found above its box"


def test_annotate_unmatched_track_is_gray(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _make_video(src)
    analysis = {"video": str(src), "probe": {"duration": DUR, "fps": FPS,
                                             "width": W, "height": H},
                "shots": [], "tracks": [_square_track(7, 40, 0)]}
    out = tmp_path / "debug_tracking.mp4"
    annotate_tracks(src, analysis, {"speaker_track": {}, "method": "mouth-only"}, out)
    assert out.exists()
    cap = cv2.VideoCapture(str(out))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ok, frame = cap.read()
    cap.release()
    assert ok
    # 160,160,160 gray box near the (static) square, and no teal anywhere dense
    assert _color_pixels(frame, (160, 160, 160), (20, 90, 100, 150)) > 50
    teal, _ = SPEAKER_COLORS[0], SPEAKER_COLORS[1]
    assert _color_pixels(frame, teal, (0, 0, W, H)) < 50


# ------------------------------------------------------------- vlm log --
def test_vlm_log_capture(monkeypatch: Any) -> None:
    from worker.reframe import vlm

    class FakeResp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {
                "content": '{"choice": 0, "reason": "ok"}'}}]}

    seen: dict[str, Any] = {}

    def fake_post(url: str, **kw: Any) -> FakeResp:
        seen["url"], seen["payload"] = url, kw["json"]
        return FakeResp()

    monkeypatch.setattr(vlm.httpx, "post", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("GMI_API_KEY", raising=False)

    pick, provider = vlm.pick_best([Image.new("RGB", (64, 64))], "pick one")
    assert pick["choice"] == 0 and provider == "openrouter"

    log = vlm.collect_vlm_log()
    assert len(log) == 1
    e = log[0]
    assert e["provider"] == "openrouter"
    assert e["model"] == vlm.OPENROUTER_MODEL
    assert "pick one" in e["prompt"]
    assert '"choice": 0' in e["reply"]
    assert e["choice"] == 0
    assert isinstance(e["latency_s"], float) and e["latency_s"] >= 0.0
    assert seen["payload"]["messages"][0]["content"][0]["text"].startswith("pick one")

    assert vlm.collect_vlm_log() == [], "collect must drain the buffer"
