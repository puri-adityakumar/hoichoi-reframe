#!/usr/bin/env python3
"""CLI: speaker-tracked 9:16 reel for one clip.

Usage: .venv/bin/python scripts/render_speaker_reel.py <clip_name>
  e.g. clip_c -> assets/test/clip_c.mp4, work/analysis/clip_c.json,
  work/sarvam/clip_c.json, work/render/clip_c_vertical.mp4 + clip_c_meta.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

P4 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P4))

from worker.reframe import probe, validate_speaker  # noqa: E402
from worker.reframe.video.camera import camera_path  # noqa: E402
from worker.reframe.video.render import crop_size, render_vertical  # noqa: E402
from worker.reframe.video.speaker import assign_speakers  # noqa: E402

STILL_W, STILL_H = 270, 480


def _merge_segments(segments: list[dict]) -> list[dict]:
    out = []
    for s in sorted(segments, key=lambda s: s.get("start_time_seconds") or 0.0):
        st, en = s.get("start_time_seconds"), s.get("end_time_seconds")
        if st is None or en is None:
            continue
        sp = str(s.get("speaker_id"))
        if out and out[-1]["speaker"] == sp and st - out[-1]["end"] <= 0.05:
            out[-1]["end"] = max(out[-1]["end"], en)
        else:
            out.append({"speaker": sp, "start": float(st), "end": float(en)})
    return out


def still_times(timeline: list[dict], segments: list[dict], duration: float) -> list[float]:
    """3 s, the two biggest speaker switches, and each segment end."""
    times = [3.0] if duration > 3.0 else [duration / 2]
    prev = timeline[0]["active_track"]
    switches: list[float] = []
    for f in timeline[1:]:
        if f["active_track"] != prev:
            switches.append(f["t"])
            prev = f["active_track"]
    switches.sort(key=lambda t: -t)
    times += sorted(switches)[:2]
    times += [min(s["end"], duration - 0.1) for s in segments]
    return times


def extract_stills(video: Path, times: list[float], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("still_*.jpg"):
        old.unlink()
    written = []
    for t in sorted(set(round(t, 2) for t in times)):
        out = out_dir / f"still_{int(t)}s.jpg"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", str(video),
            "-frames:v", "1", "-vf", f"scale={STILL_W}:{STILL_H}", str(out),
        ], check=True, capture_output=True)
        written.append(str(out.relative_to(P4)))
    return written


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    name = sys.argv[1]
    t0 = time.monotonic()

    video = P4 / f"assets/test/{name}.mp4"
    analysis_p = P4 / f"work/analysis/{name}.json"
    diar_p = P4 / f"work/sarvam/{name}.json"
    out_video = P4 / f"work/render/{name}_vertical.mp4"
    out_meta = P4 / f"work/render/{name}_meta.json"

    analysis = json.loads(analysis_p.read_text())
    diarization = json.loads(diar_p.read_text()) if diar_p.exists() else None
    if diarization and diarization.get("error"):
        diarization = None

    p = analysis["probe"]
    print(f"[1/4] fusing speakers ({'sarvam+mouth' if diarization else 'mouth-only'}) ...")
    fused = assign_speakers(analysis, diarization)
    n_active = sum(1 for f in fused["timeline"] if f["active_track"] is not None)
    print(f"      method={fused['method']} active frames {n_active}/{len(fused['timeline'])}")

    print("[2/4] camera path ...")
    path = camera_path(analysis, fused, "9:16", p["width"], p["height"], out_fps=25)

    print("[3/4] rendering (single-pass sendcmd) ...")
    render_vertical(video, path, "9:16", out_video)

    print("[4/4] validating ...")
    cw, _ = crop_size(p["width"], p["height"])
    score = validate_speaker.speaker_on_screen(
        analysis, diarization, path, fused["timeline"], crop_w=cw / p["width"])

    segments = _merge_segments(diarization["segments"]) if diarization and diarization.get("segments") else []
    stills = extract_stills(video, still_times(fused["timeline"], segments, p["duration"]),
                            P4 / "work/render/stills")

    meta = {
        "clip": name,
        "video": str(video.relative_to(P4)),
        "output": str(out_video.relative_to(P4)),
        "method": fused["method"],
        "speaker_track": fused["speaker_track"],
        "duration_s": p["duration"],
        "size_mb": round(out_video.stat().st_size / (1024 * 1024), 2),
        "validation": score,
        "stills": stills,
        "wall_time_s": round(time.monotonic() - t0, 1),
    }
    out_meta.write_text(json.dumps(meta, indent=1))

    print(json.dumps(meta, indent=1))
    print(f"PASS={score['pass']} overall={score['overall']} confident={score['overall_confident']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
