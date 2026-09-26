"""ffprobe wrapper: basic stream metadata for a video file."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TypedDict


class ProbeResult(TypedDict):
    width: int
    height: int
    fps: float
    duration: float
    codec: str
    has_audio: bool


def _rational(s: str | None) -> float:
    if not s:
        return 0.0
    if "/" in s:
        num, _, den = s.partition("/")
        return float(num) / float(den) if float(den) else 0.0
    return float(s)


def probe(video_path: str | Path) -> ProbeResult:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_streams", "-show_format",
            str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    a = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    if v is None:
        raise ValueError(f"no video stream in {path}")
    fps = _rational(v.get("avg_frame_rate")) or _rational(v.get("r_frame_rate")) or 0.0
    duration = float(data.get("format", {}).get("duration") or v.get("duration") or 0.0)
    return {
        "width": int(v.get("width", 0)),
        "height": int(v.get("height", 0)),
        "fps": round(fps, 4),
        "duration": round(duration, 3),
        "codec": str(v.get("codec_name", "")),
        "has_audio": a,
    }
