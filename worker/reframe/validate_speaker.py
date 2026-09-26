"""Validator: speaker-on-screen % — the fraction of diarized speech frames where
the active speaker's face is meaningfully inside the crop window that was
actually rendered.

Two things this used to get wrong, both measured:

1. It scored `[cx - crop_w/2, cx + crop_w/2]` while the render emitted
   `[cx*(src_w-cw), cx*(src_w-cw)+cw]`, so the headline number described a window
   that was never rendered: mean divergence 60.7 px on clip_c, 58.1 px on
   clip_a, 208 px worst case. The window now comes from the shared helpers in
   video/camera.py, the same ones video/render.py uses to emit crop x.
2. `if tid is not None:` excluded untracked frames from BOTH the numerator and
   the denominator, so the headline was computed only on the third of speech
   time where the system was already confident: 19 of 35 diarized segments on the
   190 s master came back as `frames: 0` / `score: null`. Coverage (was anyone
   tracked at all?) is now reported separately from hit rate (were they in the
   crop?), so `overall` can no longer be read as "we followed the speaker 92% of
   the time" when the honest number is 92% of 33%.

Backwards compatible: `overall`, `overall_confident`, `target`, `pass`,
`diarized` and `per_segment[...]["frames"]` keep their old meaning. New keys are
documented in the return statement below.
"""
from __future__ import annotations

from typing import Any

from .video.camera import CROP_W, crop_window_norm
from .video.speaker import _merge_segments, _resolve_overlaps

CONFIDENT = 0.5
TARGET = 0.85
SAMPLE_DT = 0.2  # 5 fps analysis grid
# A face counts as on screen only if at least this fraction of its horizontal
# extent is inside the window. The old pure-overlap test counted a 2 px sliver at
# the window edge as a full hit: 16 of 228 hits on the 190 s master were slivers.
# 0.5 also subsumes "the face centre is inside the window", which is the property
# we actually care about for a vertical crop.
MIN_FACE_IN_WINDOW = 0.5
# A hit rate only means something if a speaker was actually tracked for a real
# share of the speech; `pass_with_coverage` is the honest single boolean.
COVERAGE_FLOOR = 0.5


def _boxes_at(analysis: dict[str, Any], tid: int, t: float, tol: float = 0.21):
    tr = next((tr for tr in analysis["tracks"] if tr["id"] == tid), None)
    if not tr:
        return []
    return [f["box"] for f in tr["frames"] if abs(f["t"] - t) <= tol]


def _face_in_window(box: tuple[float, float, float, float], x0: float, x1: float,
                    min_frac: float = MIN_FACE_IN_WINDOW) -> bool:
    """True if >= min_frac of the face's width lies inside [x0, x1]."""
    bx, bw = box[0], box[2]
    overlap = max(0.0, min(bx + bw, x1) - max(bx, x0))
    return overlap >= min_frac * bw


def speaker_on_screen(analysis: dict[str, Any], diarization: dict[str, Any] | None,
                      path: list[dict[str, Any]], tl: list[dict[str, Any]],
                      crop_w: float = CROP_W) -> dict[str, Any]:
    """Per-segment and overall speaker-on-screen stats.

    `tl` is the fused timeline (from speaker.assign_speakers). `crop_w` is the
    normalized crop width; the window is derived from the path's cx with the
    shared render helper, so this scores exactly what ffmpeg was told to render.

    Returned dict:
      per_segment     list of {speaker, start, end, score, score_confident,
                        frames, coverage, score_all, speech_frames}
      overall         hit rate over frames where a speaker WAS tracked (legacy)
      overall_confident  same, restricted to fusion confidence >= CONFIDENT
      coverage        fraction of diarized speech frames where a speaker was
                      tracked at all -- the denominator `overall` does not show
      overall_all     hits / ALL diarized speech frames = overall * coverage,
                      the number to quote (legacy keys never covered this)
      speech_frames / tracked_frames / untracked_frames   the counts behind it
      min_face_in_window  the sliver threshold in force
      pass_with_coverage  `pass` AND coverage >= COVERAGE_FLOOR: the boolean to
                      quote, because `pass` alone can be true on the third of
                      the speech where we happened to be confident
      target, pass, diarized  unchanged
    """
    src_w = int((analysis.get("probe") or {}).get("width") or 1920)
    crop_w_px = max(1, int(round(crop_w * src_w)))

    def window_at(t: float) -> tuple[float, float]:
        best = min(path, key=lambda p: abs(p["t"] - t))
        cx = best["cx"] if abs(best["t"] - t) <= 0.5 / 25 else path[-1]["cx"]
        return crop_window_norm(cx, src_w, crop_w_px)

    def active_at(t: float) -> tuple[int | None, float]:
        best = min(tl, key=lambda f: abs(f["t"] - t))
        if abs(best["t"] - t) > SAMPLE_DT * 0.75:
            return None, 0.0
        return best["active_track"], best["confidence"]

    # overlap resolution is shared with the fusion, so a contested span is
    # scored once, for the speaker that was actually credited with it
    segments = _resolve_overlaps(_merge_segments(diarization["segments"]))[0] \
        if diarization and diarization.get("segments") else []

    per_seg: list[dict[str, Any]] = []
    speech = tracked = hits = conf_frames = conf_hits = 0
    for seg in segments:
        n_seg = hit_seg = f_seg = hit_conf_seg = f_conf_seg = 0
        t = seg["start"]
        while t <= seg["end"] + 1e-6:
            speech += 1
            n_seg += 1
            tid, conf = active_at(t)
            if tid is not None:
                tracked += 1
                f_seg += 1
                x0, x1 = window_at(t)
                hit = any(_face_in_window(b, x0, x1) for b in _boxes_at(analysis, tid, t))
                if hit:
                    hits += 1
                    hit_seg += 1
                    if conf >= CONFIDENT:
                        conf_hits += 1
                        hit_conf_seg += 1
                if conf >= CONFIDENT:
                    conf_frames += 1
                    f_conf_seg += 1
            t += SAMPLE_DT
        per_seg.append({
            "speaker": seg["speaker"],
            "start": round(seg["start"], 2), "end": round(seg["end"], 2),
            "score": round(hit_seg / f_seg, 3) if f_seg else None,
            "score_confident": (round(hit_conf_seg / f_conf_seg, 3)
                                if f_conf_seg else None),
            "frames": f_seg,
            # NEW: was anyone tracked for this segment at all, and the hit rate
            # if untracked frames count as misses
            "coverage": round(f_seg / n_seg, 3) if n_seg else 0.0,
            "score_all": round(hit_seg / n_seg, 3) if n_seg else 0.0,
            "speech_frames": n_seg,
        })

    overall = round(hits / tracked, 3) if tracked else 0.0
    conf_overall = round(conf_hits / conf_frames, 3) if conf_frames else None
    coverage = round(tracked / speech, 3) if speech else 0.0
    passed = conf_overall >= TARGET if conf_overall is not None else False
    return {
        "per_segment": per_seg,
        "overall": overall,
        "overall_confident": conf_overall,
        "target": TARGET,
        "pass": passed,
        "diarized": bool(segments),
        # --- added: coverage reported separately from hit rate -------------
        "coverage": coverage,
        "overall_all": round(hits / speech, 3) if speech else 0.0,
        "speech_frames": speech,
        "tracked_frames": tracked,
        "untracked_frames": speech - tracked,
        "min_face_in_window": MIN_FACE_IN_WINDOW,
        "pass_with_coverage": bool(passed and coverage >= COVERAGE_FLOOR),
    }
