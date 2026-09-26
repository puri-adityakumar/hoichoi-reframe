"""Speaker fusion: match Sarvam diarization speakers to face tracks via
mouth-energy response, with a mouth-only fallback when no diarization exists."""
from __future__ import annotations

import math
from typing import Any

ANALYSIS_DT = 0.2  # 5 fps analysis grid
SEG_EPS = 0.05  # merge tolerance for diarized segments
SMOOTH_WIN = 3  # frames for mouth signal smoothing (0.6 s)
MOUTH_THRESHOLD = 0.35  # normalized, above smoothed track baseline
MIN_SEGMENT_S = 0.7  # an active stretch must last this long to count
MIN_TRACK_FRAMES = 8  # ~1.6 s; ignore shorter (degenerate) tracks for speaker maps


def _track_grid(track: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Per-frame time and mouth values of a track, regridded by nearest sample."""
    frames = track["frames"]
    return [f["t"] for f in frames], [float(f["mouth"]) for f in frames]


def _sample(tracks_t: list[float], tracks_v: list[float], t: float) -> float | None:
    """Nearest-sample value at time t if within one analysis frame."""
    import bisect

    i = bisect.bisect_left(tracks_t, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(tracks_t):
            d = abs(tracks_t[j] - t)
            if best is None or d < best[0]:
                best = (d, tracks_v[j])
    if best and best[0] <= ANALYSIS_DT * 1.5:
        return best[1]
    return None


def _merge_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for s in sorted(segments, key=lambda s: s.get("start_time_seconds") or 0.0):
        st, en = s.get("start_time_seconds"), s.get("end_time_seconds")
        if st is None or en is None:
            continue
        sp = str(s.get("speaker_id"))
        if out and out[-1]["speaker"] == sp and st - out[-1]["end"] <= SEG_EPS:
            out[-1]["end"] = max(out[-1]["end"], en)
        else:
            out.append({"speaker": sp, "start": float(st), "end": float(en)})
    return out


def _track_energy(track: dict[str, Any], spans: list[tuple[float, float]]) -> tuple[float, int]:
    """Mean smoothed mouth signal of the track inside spans, and sample count."""
    ts, vs = _track_grid(track)
    # smooth (moving average)
    sm: list[float] = []
    for i in range(len(vs)):
        lo, hi = max(0, i - SMOOTH_WIN // 2), min(len(vs), i + SMOOTH_WIN // 2 + 1)
        sm.append(sum(vs[lo:hi]) / (hi - lo))
    vals = [sm[i] for i, t in enumerate(ts) if any(a <= t < b for a, b in spans)]
    if not vals:
        return 0.0, 0
    return sum(vals) / len(vals), len(vals)


def assign_speakers(analysis: dict[str, Any], diarization: dict[str, Any] | None) -> dict[str, Any]:
    """Fuse diarization + mouth energy into a per-frame active-track timeline."""
    tracks = analysis["tracks"]
    duration = analysis["probe"]["duration"]
    grid = [round(i * ANALYSIS_DT, 3) for i in range(int(duration / ANALYSIS_DT) + 1)]

    if diarization and diarization.get("segments"):
        return _assign_sarvam(analysis, diarization, grid)
    return _assign_mouth_only(tracks, grid)


def _assign_sarvam(analysis: dict[str, Any], diarization: dict[str, Any],
                   grid: list[float]) -> dict[str, Any]:
    tracks = [tr for tr in analysis["tracks"] if len(tr["frames"]) >= MIN_TRACK_FRAMES]
    segments = _merge_segments(diarization["segments"])
    speakers = sorted({s["speaker"] for s in segments})
    all_speech = [(s["start"], s["end"]) for s in segments]

    # per-track baseline from silence (grid time outside speech) and inside-speech
    # energy per (track, speaker)
    response: dict[tuple[int, str], float] = {}
    for tr in tracks:
        base, n_base = _track_energy(tr, [(0.0, grid[-1] + 1)])
        silent, n_silent = _track_energy(tr, [
            (a, b) for a, b in zip([0.0] + [e for _, e in all_speech],
                                   [s for s, _ in all_speech] + [grid[-1] + 1])
            if b > a
        ])
        baseline = silent if n_silent >= 3 else base * 0.6
        for sp in speakers:
            spans = [(s["start"], s["end"]) for s in segments if s["speaker"] == sp]
            energy, _ = _track_energy(tr, spans)
            response[(tr["id"], sp)] = max(0.0, energy - baseline)

    # normalize responses per speaker (max response = 1.0)
    norm: dict[tuple[int, str], float] = {}
    for sp in speakers:
        mx = max((response[(tr["id"], sp)] for tr in tracks), default=0.0)
        for tr in tracks:
            norm[(tr["id"], sp)] = response[(tr["id"], sp)] / mx if mx > 1e-6 else 0.0

    # speaker -> tracks ranked by normalized response; a near-zero-response
    # track (listening face) is never eligible
    speaker_rank: dict[str, list[int]] = {}
    for sp in speakers:
        ranked = sorted((tr for tr in tracks if norm[(tr["id"], sp)] > 1e-3),
                        key=lambda tr: -norm[(tr["id"], sp)])
        if ranked:
            speaker_rank[sp] = [tr["id"] for tr in ranked]
    speaker_track = {sp: ids[0] for sp, ids in speaker_rank.items()}

    # per-frame: active speaker's best visible track; confidence = margin
    frames_by_id = {tr["id"]: (_track_grid(tr)[0], _track_grid(tr)[1]) for tr in tracks}

    def visible(tid: int, t: float) -> bool:
        ts, _ = frames_by_id[tid]
        return any(abs(u - t) <= ANALYSIS_DT * 1.5 for u in ts)

    timeline = []
    for t in grid:
        seg = next((s for s in segments if s["start"] <= t < s["end"]), None)
        tid: int | None = None
        conf = 0.0
        if seg and seg["speaker"] in speaker_rank:
            ranked = speaker_rank[seg["speaker"]]
            for cand in ranked:  # first visible track of the active speaker
                if visible(cand, t):
                    tid = cand
                    break
            top = norm[(ranked[0], seg["speaker"])]
            second = norm[(ranked[1], seg["speaker"])] if len(ranked) > 1 else 0.0
            # margin between best and second-best track response, tanh-squashed
            conf = math.tanh(4.0 * max(0.0, top - second))
        timeline.append({"t": t, "active_track": tid, "confidence": round(conf, 3)})

    timeline = _smooth(timeline)
    return {"timeline": timeline, "speaker_track": speaker_track, "method": "sarvam+mouth"}


def _assign_mouth_only(tracks: list[dict[str, Any]], grid: list[float]) -> dict[str, Any]:
    """Fallback: argmax smoothed mouth signal above its own baseline."""
    tracks = [tr for tr in tracks if len(tr["frames"]) >= MIN_TRACK_FRAMES]
    frames_by_id = {tr["id"]: _track_grid(tr) for tr in tracks}
    timeline = []
    for t in grid:
        best_tid, best_val, second = None, 0.0, 0.0
        for tid, (ts, _) in frames_by_id.items():
            v = _sample(ts, frames_by_id[tid][1], t)
            if v is None:
                continue
            if v > best_val:
                second = best_val
                best_val, best_tid = v, tid
            elif v > second:
                second = v
        conf = max(0.0, min(1.0, best_val - second))
        timeline.append({"t": t, "active_track": best_tid if best_val > MOUTH_THRESHOLD else None,
                         "confidence": round(conf, 3)})
    timeline = _smooth(timeline)
    return {"timeline": timeline, "speaker_track": {}, "method": "mouth-only"}


def _smooth(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Suppress blips: an active stretch shorter than MIN_SEGMENT_S is dropped."""
    changed = True
    while changed:
        changed = False
        runs: list[tuple[int, int, int | None]] = []
        for i, f in enumerate(timeline):
            if runs and runs[-1][2] == f["active_track"]:
                runs[-1] = (runs[-1][0], i, f["active_track"])
            else:
                runs.append((i, i, f["active_track"]))
        for a, b, tid in runs:
            if tid is not None and (b - a + 1) * ANALYSIS_DT < MIN_SEGMENT_S:
                for i in range(a, b + 1):
                    timeline[i]["active_track"] = None
                changed = True
                break
    return timeline
