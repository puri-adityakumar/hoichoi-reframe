"""Tracking debug video: draw each face track's box + speaker label on the
480p analysis proxy and re-encode as a small QA clip. Raw artifact only -- it
never enters the outputs table, so it gets no validations and never ships to
the library. Cheap on purpose: 5 fps decode/draw, single ffmpeg pass."""
from __future__ import annotations

import subprocess
from bisect import bisect_left
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..probe import probe

ANALYSIS_FPS = 5.0
MATCH_TOL = 0.15  # s; nearest analysis frame, no interpolation
THICKNESS = 2  # px at 480p
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICK = 1
# two hues that read on light AND dark footage; unmatched tracks gray (BGR)
SPEAKER_COLORS = [(0, 160, 160), (0, 150, 230)]  # teal, amber
UNMATCHED_COLOR = (160, 160, 160)
LABEL_TEXT = (25, 25, 25)  # dark text on the solid color backing


def _ffmpeg(args: list[str]) -> None:
    res = subprocess.run(["ffmpeg", "-v", "error", "-y", *args],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr[-800:]}")


def make_analysis_proxy(video: Path, dst: Path) -> Path:
    """480p (long edge) H.264 proxy WITH audio, for cheap annotation decode.

    The face analysis downsizes in memory (faces.py) so no proxy file exists in
    the pipeline yet; this one is only for the debug video."""
    vf = "scale='if(gt(iw,ih),480,-2)':'if(gt(iw,ih),-2,480)'"
    _ffmpeg(["-i", str(video), "-vf", vf, "-c:v", "libx264", "-crf", "27",
             "-preset", "veryfast", "-c:a", "aac", "-movflags", "+faststart",
             str(dst)])
    return dst


def speaker_meta(fused: dict[str, Any]) -> dict[int, tuple[int, Any, tuple[int, int, int]]]:
    """track id -> (speaker index, confidence or None, BGR color).

    Speaker ids sort to a stable order so speaker 0 is always teal, speaker 1
    always amber, regardless of which face track they bound to."""
    sp_track = fused.get("speaker_track") or {}
    conf = fused.get("speaker_confidence") or {}
    order = {sp: i for i, sp in enumerate(sorted(sp_track, key=str))}
    return {
        int(tid): (order[sp], conf.get(sp), SPEAKER_COLORS[order[sp] % len(SPEAKER_COLORS)])
        for sp, tid in sp_track.items()
    }


def _label(meta: tuple[int, Any, tuple[int, int, int]]) -> str:
    idx, confidence, _ = meta
    letter = chr(ord("A") + idx)
    if confidence is None:
        return f"Speaker {letter}"
    return f"Speaker {letter} - {float(confidence):.2f}"


def _nearest_box(track: dict[str, Any], t: float) -> tuple[float, float, float, float] | None:
    """Box of the nearest analysis frame within MATCH_TOL (no interpolation)."""
    ts = [f["t"] for f in track["frames"]]
    i = bisect_left(ts, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(ts):
            d = abs(ts[j] - t)
            if best is None or d < best[0]:
                best = (d, track["frames"][j]["box"])
    if best is None or best[0] > MATCH_TOL:
        return None
    return tuple(best[1])  # type: ignore[return-value]


def _draw(frame: np.ndarray, boxes: list[tuple[tuple[float, float, float, float],
                                               tuple[int, int, int], str]]) -> None:
    h, w = frame.shape[:2]
    for (x, y, bw, bh), color, label in boxes:
        p1 = (int(round(x * w)), int(round(y * h)))
        p2 = (int(round((x + bw) * w)), int(round((y + bh) * h)))
        cv2.rectangle(frame, p1, p2, color, THICKNESS, cv2.LINE_AA)
        (tw, th), base = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
        pad = 3
        # above the box by default, inside the frame when it would clip
        ly = p1[1] - th - base - 2 * pad
        if ly < 0:
            ly = p1[1] + 2
        lx = max(0, min(w - tw - 2 * pad, p1[0]))
        cv2.rectangle(frame, (lx, ly), (lx + tw + 2 * pad, ly + th + base + 2 * pad),
                      color, -1)
        cv2.putText(frame, label, (lx + pad, ly + pad + th), FONT, FONT_SCALE,
                    LABEL_TEXT, FONT_THICK, cv2.LINE_AA)


def annotate_tracks(proxy_video_path: str | Path, analysis: dict[str, Any],
                    fused: dict[str, Any], out_path: Path) -> Path:
    """Draw per-track boxes + 'Speaker A - 0.83' labels at the analysis fps,
    re-encode H.264 (+faststart), muxing the proxy's audio when present."""
    src = Path(proxy_video_path)
    p = probe(src)
    meta = speaker_meta(fused)
    step = max(1, round(p["fps"] / ANALYSIS_FPS))
    out_fps = p["fps"] / step  # the analysis fps, on the proxy's frame grid

    cmd: list[str] = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{p['width']}x{p['height']}", "-r", f"{out_fps:g}", "-i", "-",
    ]
    if p["has_audio"]:
        cmd += ["-i", str(src), "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest",
            str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(str(src))
    idx = 0
    drawn = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step != 0:
                idx += 1
                continue
            ok, frame = cap.retrieve()
            idx += 1
            if not ok or frame is None:
                break
            t = (idx - 1) / p["fps"]
            boxes = []
            for tr in analysis["tracks"]:
                box = _nearest_box(tr, t)
                if box is None:
                    continue
                m = meta.get(int(tr["id"]))
                color = m[2] if m else UNMATCHED_COLOR
                label = _label(m) if m else ""
                boxes.append((box, color, label))
                drawn += 1
            if boxes:
                _draw(frame, boxes)
            proc.stdin.write(frame.tobytes())
    finally:
        cap.release()
        proc.stdin.close()
        err = proc.stderr.read().decode() if proc.stderr else ""
        if proc.stderr:
            proc.stderr.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg annotate failed: {err[-800:]}")
    return out_path
