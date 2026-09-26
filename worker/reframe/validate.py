"""Honest validator: every output is checked against spec/spec.json.
Returns one row per (output, rule) with expected/actual/pass — failures stay visible."""
from __future__ import annotations

import subprocess
import json
from pathlib import Path
from typing import Any

from PIL import Image

ASPECT_TOL = 0.01  # ±1%
RES_TOL = 0.01
MB = 1024 * 1024


def _probe_video(path: Path) -> dict[str, Any]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    v = next(s for s in data["streams"] if s.get("codec_type") == "video")
    fps = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0"
    if "/" in fps:
        num, _, den = fps.partition("/")
        fps_v = float(num) / float(den) if float(den) else 0.0
    else:
        fps_v = float(fps)
    return {
        "width": int(v["width"]), "height": int(v["height"]),
        "fps": round(fps_v, 3), "codec": v.get("codec_name", ""),
        "duration": float(data.get("format", {}).get("duration") or 0.0),
        "container": path.suffix.lstrip(".").lower(),
        "size_bytes": path.stat().st_size,
    }


def _probe_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as im:
        return {
            "width": im.width, "height": im.height,
            "codec": (im.format or "").lower(),  # "jpeg" / "png"
            "container": path.suffix.lstrip(".").lower(),
            "size_bytes": path.stat().st_size,
        }


def validate_all(spec: dict, outputs: list[dict]) -> list[dict[str, Any]]:
    """outputs: [{"platform": <spec key>, "path": <Path|str>, ...}].
    Returns rows [{output, rule, expected, actual, passed}]."""
    platforms = spec["platforms"] if "platforms" in spec else spec
    rows: list[dict[str, Any]] = []
    for out in outputs:
        p = platforms[out["platform"]]
        path = Path(out["path"])
        name = out.get("name") or path.name
        if p["kind"] == "video":
            rows += _validate_one(name, p, _probe_video(path), video=True)
        else:
            rows += _validate_one(name, p, _probe_image(path), video=False)
    return rows


def _validate_one(name: str, p: dict[str, Any], a: dict[str, Any], video: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def row(rule: str, expected: Any, actual: Any, passed: bool) -> None:
        rows.append({"output": name, "rule": rule, "expected": str(expected),
                     "actual": str(actual), "passed": bool(passed)})

    sw, sh = p["width"], p["height"]
    spec_ar = sw / sh
    act_ar = a["width"] / a["height"]
    row("ratio", p["ratio"],
        f"{a['width']}x{a['height']} ar={act_ar:.4f}",
        abs(act_ar - spec_ar) / spec_ar <= ASPECT_TOL)
    row("resolution", f"{sw}x{sh} (±1%)",
        f"{a['width']}x{a['height']}",
        abs(a["width"] - sw) / sw <= RES_TOL and abs(a["height"] - sh) / sh <= RES_TOL)
    row("size_mb", f"<= {p['max_size_mb']}",
        round(a["size_bytes"] / MB, 2), a["size_bytes"] <= p["max_size_mb"] * MB)
    if video:
        row("duration_sec", f"<= {p['max_duration_sec']}", a["duration"],
            a["duration"] <= p["max_duration_sec"] + 0.05)
        row("fps", f">= {p['min_fps']}", a["fps"], a["fps"] >= p["min_fps"] - 0.01)
        row("codec", p["codec"], a["codec"], a["codec"] == p["codec"])
        row("container", p["container"], a["container"], a["container"] == p["container"])
    else:
        row("codec", p["codec"], a["codec"], p["codec"] in a["codec"])
        row("container", p["container"], a["container"], a["container"] == p["container"])
    return rows
