"""Speaker fusion: match Sarvam diarization speakers to face tracks via
mouth-energy CONTRAST, with a mouth-only fallback when no diarization exists.

WHY CONTRAST, NOT "ENERGY ABOVE A BASELINE"
-------------------------------------------
The old binding took a bare argmax of normalized mouth response, where the
baseline was taken from diarized silence and, when there was little silence,
degraded to `base * 0.6` -- a self-referential whole-track mean. That handed a
positive response to every speaker for any generically mouthy track, so on
clip_a and clip_c BOTH diarized speakers were mapped onto the same physical
person (`speaker_track = {'0': 9, '1': 3}` with cx 0.55 / 0.64; `{'0': 8,
'1': 4}` with cx ~0.33-0.36). A track is now scored by how much MORE it moves
its mouth while speaker S talks than while the other speakers talk, which
disqualifies a generically mouthy track on its own, and speakers are bound
one-to-one so two speakers cannot land on one face. When the evidence is
genuinely ambiguous the binding reports low confidence and video/camera.py
takes its WIDE rung instead of confidently framing one person for both.

Track length matters too: measured, speaker 0 spoke for 42.8 s but was bound to
a track that existed for only 8 frames (1.4 s), because MIN_TRACK_FRAMES was the
only length filter. The score is now damped by how much of that speaker's
speech a track was actually on screen for (PRESENCE_FLOOR), which makes the
argmax duration-aware without discarding a speaker who is legitimately off
screen during a reverse-shot.
"""
from __future__ import annotations

import bisect
import math
from typing import Any

ANALYSIS_DT = 0.2  # 5 fps analysis grid
SEG_EPS = 0.05  # merge tolerance for diarized segments
SMOOTH_WIN = 3  # frames for mouth signal smoothing (0.6 s)
MOUTH_THRESHOLD = 0.35  # normalized, above smoothed track baseline (mouth-only)
MIN_SEGMENT_S = 0.7  # an active stretch must last this long to count
MIN_TRACK_FRAMES = 8  # ~1.6 s; ignore shorter (degenerate) tracks for speaker maps
PRESENCE_FLOOR = 0.35  # weight floor for a track that is visible for only a
# sliver of a speaker's speech. Presence is a *damping* factor on the contrast,
# not a hard gate: in a shot/reverse-shot the speaker legitimately leaves the
# screen for most of their own turn, and a hard 0.5 gate emptied the candidate
# lists entirely (measured: speaker 0 on clip_c is on screen for 0.29 of their
# 25.2 s of speech). The floor is what makes the argmax duration-aware: measured,
# speaker 0 spoke 42.8 s on the master but its bare argmax track existed for only
# 8 frames (1.4 s), i.e. presence 0.06, and damping cuts its score to 0.39x.
SECOND_CHOICE_PENALTY = 0.5  # binding a speaker to its second-choice track
# (because the better match belongs to another speaker) is a genuinely ambiguous
# clip: report this much confidence so the camera takes the WIDE rung
FALLBACK_TRACK_PENALTY = 0.6  # the speaker's top-ranked face is off screen, so
# this frame is framed on a fallback track: say so in the confidence


def _smoothed(track: dict[str, Any]) -> tuple[list[float], list[float]]:
    """(frame times, moving-averaged mouth values) of a track."""
    frames = track["frames"]
    ts = [f["t"] for f in frames]
    vs = [float(f["mouth"]) for f in frames]
    sm: list[float] = []
    for i in range(len(vs)):
        lo, hi = max(0, i - SMOOTH_WIN // 2), min(len(vs), i + SMOOTH_WIN // 2 + 1)
        sm.append(sum(vs[lo:hi]) / (hi - lo))
    return ts, sm


def _sample(tracks_t: list[float], tracks_v: list[float], t: float) -> float | None:
    """Nearest-sample value at time t if within one analysis frame."""
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


def _resolve_overlaps(segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    """Make diarized segments a non-overlapping partition, longest turn first.

    Sarvam's speaker labels overlap: measured 6 overlapping pairs on the 190 s
    master, up to 0.86 s, and `next(...)` = first match silently awarded one
    speaker's speech to the other. Here the longest segment claims its span and
    every later segment gets that span subtracted, so contested time goes to the
    turn that is most likely right instead of to whoever happened to be listed
    first. Returns the partition and the seconds dropped as overlap.
    """
    claimed: list[tuple[float, float]] = []
    out: list[dict[str, Any]] = []
    dropped = 0.0
    for seg in sorted(segments, key=lambda s: (-(s["end"] - s["start"]), s["start"])):
        free = [(max(seg["start"], a), min(seg["end"], b))
                for a, b in _subtract(seg["start"], seg["end"], claimed)]
        claimed.append((seg["start"], seg["end"]))
        for a, b in free:
            if b - a > 1e-3:
                out.append({"speaker": seg["speaker"], "start": a, "end": b})
        dropped += (seg["end"] - seg["start"]) - sum(b - a for a, b in free)
    return sorted(out, key=lambda s: s["start"]), round(dropped, 3)


def _subtract(start: float, end: float,
              claimed: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """[start, end) minus every claimed span."""
    pieces = [(start, end)]
    for a, b in claimed:
        nxt = []
        for p0, p1 in pieces:
            if b <= p0 or a >= p1:
                nxt.append((p0, p1))
                continue
            if p0 < a:
                nxt.append((p0, min(a, p1)))
            if p1 > b:
                nxt.append((max(b, p0), p1))
        pieces = nxt
    return pieces


def _track_energy(ts: list[float], vs: list[float],
                  spans: list[tuple[float, float]]) -> tuple[float, int]:
    """Mean smoothed mouth signal of the track inside spans, and sample count."""
    vals = [vs[i] for i, t in enumerate(ts) if any(a <= t < b for a, b in spans)]
    if not vals:
        return 0.0, 0
    return sum(vals) / len(vals), len(vals)


def _presence(ts: list[float], spans: list[tuple[float, float]]) -> float:
    """Share of the span grid at which the track has a sample (0 if no spans)."""
    total = 0
    for a, b in spans:
        total += max(0, int(round((b - a) / ANALYSIS_DT)))
    if total == 0:
        return 0.0
    hits = sum(1 for a, b in spans for t in ts if a <= t < b)
    return min(1.0, hits / total)


def audio_offset(analysis: dict[str, Any]) -> float:
    """Seconds to add to a diarization timestamp to reach video time.

    Diarization runs on a WAV extracted with `ffmpeg -i <video> -ac 1 -ar 16000`,
    so its t=0 is the audio stream's start. probe.py does not record stream
    start_time, so today this is always 0.0 -- measured offset on all four test
    assets is 0.000 s -- and a master with a non-zero audio start offset would
    silently shift every diarization timestamp. Reading the field here means
    that when probe.py starts recording it (no change needed on this side) the
    fusion corrects itself; until then the value is the documented 0.0.
    """
    probe = analysis.get("probe") or {}
    return float(probe.get("audio_start") or 0.0) - float(probe.get("video_start") or 0.0)


def assign_speakers(analysis: dict[str, Any], diarization: dict[str, Any] | None) -> dict[str, Any]:
    """Fuse diarization + mouth energy into a per-frame active-track timeline.

    Returned dict keys (main.py reads `timeline`, `speaker_track`, `method`):
      timeline          per analysis frame {t, active_track, confidence}
      speaker_track     diarized speaker id -> face track id (one-to-one)
      speaker_confidence  same keys -> 0..1 confidence in that binding
      ambiguous         True when two speakers wanted the same person
      overlap_dropped_s  seconds of overlapping diarization time resolved away
      method            "sarvam+mouth" | "mouth-only"
    """
    tracks = analysis["tracks"]
    duration = analysis["probe"]["duration"]
    grid = [round(i * ANALYSIS_DT, 3) for i in range(int(duration / ANALYSIS_DT) + 1)]

    if diarization and diarization.get("segments"):
        return _assign_sarvam(analysis, diarization, grid)
    return _assign_mouth_only(tracks, grid)


def _assign_sarvam(analysis: dict[str, Any], diarization: dict[str, Any],
                   grid: list[float]) -> dict[str, Any]:
    tracks = [tr for tr in analysis["tracks"] if len(tr["frames"]) >= MIN_TRACK_FRAMES]
    offset = audio_offset(analysis)
    raw = _merge_segments(diarization["segments"])
    for s in raw:
        s["start"] += offset
        s["end"] += offset
    segments, dropped = _resolve_overlaps(raw)
    speakers = sorted({s["speaker"] for s in segments})
    if not tracks or not segments:
        return _assign_mouth_only(analysis["tracks"], grid)

    spans_by_speaker = {sp: [(s["start"], s["end"]) for s in segments if s["speaker"] == sp]
                        for sp in speakers}
    other_spans = {sp: [(s["start"], s["end"]) for s in segments if s["speaker"] != sp]
                   for sp in speakers}
    grids = {tr["id"]: _smoothed(tr) for tr in tracks}

    # score per (track, speaker): mouth-energy CONTRAST, damped by how much of
    # that speaker's speech the track was actually on screen for. A track with no
    # samples inside the spans scores 0, so a face that is only ever visible
    # while the other person talks can never be bound to this speaker.
    score: dict[tuple[int, str], float] = {}
    for tr in tracks:
        ts, vs = grids[tr["id"]]
        for sp in speakers:
            own, _ = _track_energy(ts, vs, spans_by_speaker[sp])
            other, _ = _track_energy(ts, vs, other_spans[sp])
            pres = _presence(ts, spans_by_speaker[sp])
            damp = PRESENCE_FLOOR + (1.0 - PRESENCE_FLOOR) * pres
            score[(tr["id"], sp)] = max(0.0, (own - other) * damp)

    # normalize per speaker (best track = 1.0)
    norm: dict[tuple[int, str], float] = {}
    for sp in speakers:
        mx = max((score[(tr["id"], sp)] for tr in tracks), default=0.0)
        for tr in tracks:
            norm[(tr["id"], sp)] = (score[(tr["id"], sp)] / mx) if mx > 1e-6 else 0.0

    # rank each speaker's plausible tracks (never a near-zero-response listener)
    rank: dict[str, list[int]] = {
        sp: [tr["id"] for tr in sorted(
            (tr for tr in tracks if norm[(tr["id"], sp)] > 1e-3),
            key=lambda tr: (-norm[(tr["id"], sp)], -len(tr["frames"])))]
        for sp in speakers
    }

    # one-to-one assignment: best (speaker, track) pair first. This is what stops
    # {'0': 9, '1': 3} on clip_a, where both speakers landed on the right-hand
    # person because the mouthy track won the argmax twice.
    pairs = sorted(((-norm[(tid, sp)], sp, tid) for sp in speakers for tid in rank[sp]))
    speaker_track: dict[str, int] = {}
    taken: set[int] = set()
    for _, sp, tid in pairs:
        if sp in speaker_track or tid in taken:
            continue
        speaker_track[sp] = tid
        taken.add(tid)

    # confidence in each binding: how much better the winner is than the
    # runner-up for the SAME speaker (a small margin means we do not really know)
    margin: dict[str, float] = {}
    speaker_conf: dict[str, float] = {}
    for sp in speakers:
        ranked = rank[sp]
        if not ranked or sp not in speaker_track:
            margin[sp] = speaker_conf[sp] = 0.0
            continue
        top = norm[(ranked[0], sp)]
        second = norm[(ranked[1], sp)] if len(ranked) > 1 else 0.0
        m = math.tanh(4.0 * max(0.0, top - second))
        margin[sp] = m
        # a second-choice track means the better match went to another speaker:
        # genuinely ambiguous, so say so in the confidence
        speaker_conf[sp] = round(
            m if speaker_track[sp] == ranked[0] else m * SECOND_CHOICE_PENALTY, 3)
    # true when some speaker could not be bound to their own best person
    ambiguous = bool(speakers) and not all(
        sp in speaker_track and rank[sp] and speaker_track[sp] == rank[sp][0]
        for sp in speakers)

    # per-frame: active speaker's best VISIBLE track, confidence = fusion margin
    # capped by the binding confidence, so an ambiguous binding cannot
    # confidently lock the camera onto one person (video/camera.py then takes
    # its WIDE rung instead).
    def visible(tid: int, t: float) -> bool:
        ts, _ = grids[tid]
        return any(abs(u - t) <= ANALYSIS_DT * 1.5 for u in ts)

    timeline = []
    for t in grid:
        seg = next((s for s in segments if s["start"] <= t < s["end"]), None)
        tid: int | None = None
        conf = 0.0
        if seg and seg["speaker"] in speaker_track:
            sp = seg["speaker"]
            ranked = rank[sp]
            for cand in ranked:  # first visible track of the active speaker
                if visible(cand, t):
                    tid = cand
                    break
            top = norm[(ranked[0], sp)]
            second = norm[(ranked[1], sp)] if len(ranked) > 1 else 0.0
            margin[sp] = math.tanh(4.0 * max(0.0, top - second))
            # the top-ranked face is off screen, so we are framing a fallback
            fallback = 1.0 if tid is None or tid == ranked[0] else FALLBACK_TRACK_PENALTY
            conf = min(margin[sp], speaker_conf[sp]) * fallback
        timeline.append({"t": t, "active_track": tid, "confidence": round(conf, 3)})

    timeline = _smooth(timeline)
    return {
        "timeline": timeline,
        "speaker_track": speaker_track,
        "speaker_confidence": speaker_conf,
        "ambiguous": ambiguous,
        "overlap_dropped_s": dropped,
        "method": "sarvam+mouth",
    }


def _assign_mouth_only(tracks: list[dict[str, Any]], grid: list[float]) -> dict[str, Any]:
    """Fallback: argmax smoothed mouth signal above its own baseline."""
    tracks = [tr for tr in tracks if len(tr["frames"]) >= MIN_TRACK_FRAMES]
    frames_by_id = {tr["id"]: _smoothed(tr) for tr in tracks}
    timeline = []
    for t in grid:
        best_tid, best_val, second = None, 0.0, 0.0
        for tid, (ts, vs) in frames_by_id.items():
            v = _sample(ts, vs, t)
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
    return {"timeline": timeline, "speaker_track": {}, "speaker_confidence": {},
            "ambiguous": False, "overlap_dropped_s": 0.0, "method": "mouth-only"}


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
