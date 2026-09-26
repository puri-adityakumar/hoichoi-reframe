"""Single-pass vertical render: sendcmd drives crop x per frame (verified
experimentally: crop x followed a command file that jumped 0 -> 1312), then
crop -> scale to 1080x1920, libx264 + aac in one ffmpeg invocation.

The crop x arithmetic lives in video/camera.py (crop_left_px): path entries are
`cx` = FRACTION OF AVAILABLE TRAVEL in [0, 1], and worker/reframe/
validate_speaker.py scores the same window via the same helper. Nothing here
re-derives the conversion."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .camera import crop_left_px, crop_size

__all__ = ["OUT_W", "OUT_H", "OUT_FPS", "crop_size", "crop_x_sequence", "render_vertical"]

OUT_W, OUT_H = 1080, 1920
OUT_FPS = 25.0


def crop_x_sequence(path: list[dict[str, Any]], src_w: int, crop_w: int) -> list[int]:
    """Crop left edge in pixels per path entry -- exactly what ffmpeg is told.

    Split out of render_vertical so the render and the validator can be checked
    against the same numbers without running ffmpeg.
    """
    return [crop_left_px(p["cx"], src_w, crop_w) for p in path]


def render_vertical(video_path: str | Path, path: list[dict[str, Any]],
                    ratio: str, out_path: str | Path,
                    src_w: int = 1920, src_h: int = 1080,
                    out_w: int = OUT_W, out_h: int = OUT_H) -> Path:
    """Render one pass: per-frame crop x from the camera path."""
    src = Path(video_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cw, ch = crop_size(src_w, src_h, ratio)

    with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False) as fh:
        for i, x in enumerate(crop_x_sequence(path, src_w, cw)):
            fh.write(f"{i / OUT_FPS:.3f} crop x {x};\n")
        cmd_file = fh.name

    try:
        vf = (
            f"sendcmd=f={cmd_file},"
            f"crop={cw}:{ch}:x=0:y=0,"
            f"scale={out_w}:{out_h},format=yuv420p"
        )
        cmd = [
            "ffmpeg", "-v", "error", "-y", "-i", str(src),
            "-vf", vf, "-r", str(OUT_FPS),
            "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
            "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "128k",
            str(out),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg render failed: {res.stderr[-800:]}")
    finally:
        Path(cmd_file).unlink(missing_ok=True)
    return out
