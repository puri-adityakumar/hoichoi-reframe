"""Validator: speaker-on-screen % — fraction of frames per diarized segment
where the active speaker's face box intersects the rendered crop window."""
from __future__ import annotations

from typing import Any

CROP_W = 608 / 1920  # normalized crop width (9:16 from 1920x1080)
CONFIDENT = 0.5
TARGET = 0.85
SAMPLE_DT = 0.2  # 5 fps analysis grid


def _boxes_at(analysis: dict[str, Any], tid: int, t: float, tol: float = 0.21):
    tr = next((tr for tr in analysis["tracks"] if tr["id"] == tid), None)
    if not tr:
        return []
    return [f["box"] for f in tr["frames"] if abs(f["t"] - t) <= tol]


def _intersects(box: tuple[float, float, float, float], x0: float, x1: float) -> bool:
    bx, bw = box[0], box[2]
    return (bx + bw) > x0 + 0.01 and bx < x1 - 0.01


def _merge_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def speaker_on_screen(analysis: dict[str, Any], diarization: dict[str, Any] | None,
                      path: list[dict[str, float]], tl: list[dict[str, Any]],
                      crop_w: float = CROP_W) -> dict[str, Any]:
    """Per-segment and overall fraction of frames with the speaker in the crop.

    `tl` is the fused timeline (from speaker.assign_speakers). Confident
    scoring counts only frames whose fusion confidence >= CONFIDENT.
    """
    def cx_at(t: float) -> float:
        best = min(path, key=lambda p: abs(p["t"] - t))
        return best["cx"] if abs(best["t"] - t) <= 0.5 / 25 else path[-1]["cx"]

    def active_at(t: float) -> tuple[int | None, float]:
        best = min(tl, key=lambda f: abs(f["t"] - t))
        if abs(best["t"] - t) > SAMPLE_DT * 0.75:
            return None, 0.0
        return best["active_track"], best["confidence"]

    segments = _merge_segments(diarization["segments"]) if diarization and diarization.get("segments") else []

    per_seg, all_frames, all_hits, conf_frames, conf_hits = [], 0, 0, 0, 0
    for seg in segments:
        frames = hits = cframes = chits = 0
        t = seg["start"]
        while t <= seg["end"] + 1e-6:
            tid, conf = active_at(t)
            if tid is not None:
                frames += 1
                all_frames += 1
                cx = cx_at(t)
                x0, x1 = cx - crop_w / 2, cx + crop_w / 2
                hit = any(_intersects(b, x0, x1) for b in _boxes_at(analysis, tid, t))
                if hit:
                    hits += 1
                    all_hits += 1
                if conf >= CONFIDENT:
                    cframes += 1
                    conf_frames += 1
                    if hit:
                        chits += 1
                        conf_hits += 1
            t += SAMPLE_DT
        per_seg.append({
            "speaker": seg["speaker"],
            "start": round(seg["start"], 2), "end": round(seg["end"], 2),
            "score": round(hits / frames, 3) if frames else None,
            "score_confident": round(chits / cframes, 3) if cframes else None,
            "frames": frames,
        })

    overall = round(all_hits / all_frames, 3) if all_frames else 0.0
    conf_overall = round(conf_hits / conf_frames, 3) if conf_frames else None
    return {
        "per_segment": per_seg,
        "overall": overall,
        "overall_confident": conf_overall,
        "target": TARGET,
        "pass": conf_overall >= TARGET if conf_overall is not None else False,
        "diarized": bool(segments),
    }
