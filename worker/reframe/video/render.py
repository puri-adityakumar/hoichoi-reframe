"""Single-pass vertical render: sendcmd drives crop x per frame (verified
experimentally: crop x followed a command file that jumped 0 -> 1312), then
crop -> scale to 1080x1920, libx264 + aac in one ffmpeg invocation."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

OUT_W, OUT_H = 1080, 1920
OUT_FPS = 25.0


def crop_size(src_w: int, src_h: int, ratio: str = "9:16") -> tuple[int, int]:
    if ratio != "9:16":
        pass  # generalized below: any ratio wider than the source height allows
    if (src_w, src_h) == (1920, 1080) and ratio == "9:16":
        return 608, 1080
    rw, rh = (int(v) for v in ratio.split(":"))
    w, h = round(src_h * rw / rh), src_h
    if w > src_w:  # target wider than source allows -> height-constrained
        w, h = src_w, round(src_w * rh / rw)
    w, h = w - w % 2, h - h % 2
    return w, h


def render_vertical(video_path: str | Path, path: list[dict[str, float]],
                    ratio: str, out_path: str | Path,
                    src_w: int = 1920, src_h: int = 1080,
                    out_w: int = OUT_W, out_h: int = OUT_H) -> Path:
    """Render one pass: per-frame crop x from the camera path."""
    src = Path(video_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cw, ch = crop_size(src_w, src_h, ratio)

    with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False) as fh:
        for i, p in enumerate(path):
            x_f = p["cx"] * (src_w - cw)
            x = max(0, min(src_w - cw, round(x_f)))
            x -= x % 2  # even x for yuv420p-safe chroma
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
