"""Camera / speaker-fusion tests. All synthetic: no ffmpeg, no test clips.

These guard the two disqualifiers directly -- "vertical video reframed once at
frame 0 and held static" and "face detection bolted on but not actually
influencing the crop box" -- plus the four measured defects behind them:

  1. the no-active-speaker fallback is a strategy ladder (TRACK/WIDE/HOLD)
  2. cx has ONE meaning, shared by camera / render / validator
  3. the One Euro filter is live (step in -> ramp out, fast)
  4. two diarized speakers must map to two DIFFERENT people
"""
from __future__ import annotations

from typing import Any

import pytest

from worker.reframe import validate_speaker
from worker.reframe.video import camera as cam
from worker.reframe.video.camera import (
    DEAD_ZONE, HOLD, TRACK, WIDE, camera_path, crop_left_px, crop_window_px,
    face_center_to_cx, max_step,
)
from worker.reframe.video.render import crop_size, crop_x_sequence
from worker.reframe.video.speaker import (
    _resolve_overlaps, assign_speakers, audio_offset,
)

SRC_W, SRC_H = 1920, 1080
CROP_PX = crop_size(SRC_W, SRC_H, "9:16")[0]  # 608
FPS = 25
DT = 0.2  # analysis grid


# ----------------------------------------------------------------- fixtures --
def track(tid: int, cx: float, times: list[float], mouth: Any = 0.2,
          w: float = 0.14, h: float = 0.22) -> dict[str, Any]:
    """A face track centred at cx, one box per analysis frame."""
    frames = []
    for t in times:
        m = mouth(t) if callable(mouth) else mouth
        frames.append({"t": round(t, 3), "box": (cx - w / 2, 0.30, w, h), "mouth": round(m, 4)})
    return {"id": tid, "frames": frames}


def grid(duration: float, start: float = 0.0, step: float = DT) -> list[float]:
    n = int((duration - start) / step)
    return [round(start + i * step, 3) for i in range(n + 1)]


def analysis(tracks: list[dict[str, Any]], duration: float,
             shots: list[float] | None = None) -> dict[str, Any]:
    cuts = shots if shots is not None else [0.0]
    return {
        "video": "synthetic",
        "probe": {"width": SRC_W, "height": SRC_H, "fps": 25.0,
                  "duration": duration, "codec": "h264", "has_audio": True},
        "shots": [{"start": s, "end": duration} for s in cuts],
        "tracks": tracks,
    }


def diarization(turns: list[tuple[str, float, float]]) -> dict[str, Any]:
    return {"segments": [{"speaker_id": sp, "start_time_seconds": st,
                          "end_time_seconds": en} for sp, st, en in turns]}


def mean_cx(tr: dict[str, Any]) -> float:
    return sum(f["box"][0] + f["box"][2] / 2 for f in tr["frames"]) / len(tr["frames"])


def speaking_mouth(turns: list[tuple[str, float, float]], speaker: str,
                   hot: float = 0.95, cold: float = 0.15) -> Any:
    """Mouth signal of the person who owns `speaker`: open only on their turns."""
    spans = [(st, en) for sp, st, en in turns if sp == speaker]

    def f(t: float) -> float:
        return hot if any(a <= t < b for a, b in spans) else cold

    return f


TURNS = [("0", 0.5, 2.5), ("1", 2.5, 4.5), ("0", 4.5, 6.5), ("1", 6.5, 8.0)]
LEFT, RIGHT = 0.30, 0.70


def two_person(duration: float = 8.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two people in one shot at x=0.30 and x=0.70, alternating turns."""
    ts = grid(duration)
    a = track(11, LEFT, ts, speaking_mouth(TURNS, "0"))
    b = track(12, RIGHT, ts, speaking_mouth(TURNS, "1"))
    return analysis([a, b], duration), diarization(TURNS)


def unreliable_two_person() -> tuple[dict[str, Any], dict[str, Any]]:
    """The measured clip_a / clip_c situation, rebuilt deterministically.

    Person A (x=0.30) is turned away and barely moves their mouth, so their own
    signal is weak. Person B (x=0.70) is fragmented into two tracks: B1 is up to
    4.0 s and holds their mouth open through EVERYONE's speech (a reactor --
    plausible looking, but carrying no information about who is talking), B2
    takes over from 4.0 s and is clearly mouthy only on speaker "1" turns.

    The old binding (mouth energy above a silence baseline, else `base * 0.6`,
    then a bare argmax per speaker) scored the reactor highest for speaker 0 and
    B2 highest for speaker 1 -- `{'0': 13, '1': 14}`, i.e. two diarized speakers
    on one physical person, the measured clip_a / clip_c failure.
    """
    ts = grid(8.0)
    a = track(11, LEFT, ts, speaking_mouth(TURNS, "0", hot=0.30, cold=0.10))
    b1 = track(13, RIGHT, [t for t in ts if t <= 4.0],
               lambda t: 0.10 if t < 0.5 else 0.70)
    b2 = track(14, RIGHT, [t for t in ts if t >= 4.0], speaking_mouth(TURNS, "1"))
    return analysis([a, b1, b2], 8.0), diarization(TURNS)


def hand_timeline(duration: float, switch_at: float, conf: float = 1.0,
                  ids: tuple[int, int] = (11, 12)) -> dict[str, Any]:
    """A fused timeline with one instantaneous speaker change, no shot cuts."""
    return {"timeline": [
        {"t": t, "active_track": ids[0] if t < switch_at else ids[1],
         "confidence": conf} for t in grid(duration)]}


def deltas(path: list[dict[str, Any]]) -> list[float]:
    return [b["cx"] - a["cx"] for a, b in zip(path, path[1:])]


def t_at(path: list[dict[str, Any]], t: float) -> float:
    return min(path, key=lambda p: abs(p["t"] - t))["cx"]


# ------------------------------------------------- DEFECT 4: two people, two --
def test_two_speakers_map_to_two_different_people() -> None:
    """THE test for the disqualifier: a two-person alternating clip must bind
    the two diarized speakers to two different faces, and the crop must follow
    each of them in turn."""
    an, diar = two_person()
    fused = assign_speakers(an, diar)

    assert set(fused["speaker_track"]) == {"0", "1"}, fused["speaker_track"]
    assert len(set(fused["speaker_track"].values())) == 2, (
        f"both speakers bound to one person: {fused['speaker_track']}")
    by_id = {tr["id"]: mean_cx(tr) for tr in an["tracks"]}
    cx0 = by_id[fused["speaker_track"]["0"]]
    cx1 = by_id[fused["speaker_track"]["1"]]
    assert abs(cx0 - cx1) > 0.2, f"speakers bound to the same side: {cx0} {cx1}"
    # the binding is not flagged ambiguous, so the camera is allowed to lock on
    assert not fused["ambiguous"], fused

    path = camera_path(an, fused, "9:16", SRC_W, SRC_H, out_fps=FPS)
    assert any(e["strategy"] == TRACK for e in path), "no frame followed a speaker"
    # and the rendered window really visits each person's side of the frame
    px = [crop_window_px(e["cx"], SRC_W, CROP_PX) for e in path]
    while_tracking = [x for e, (x, _) in zip(path, px) if e["strategy"] == TRACK]
    left_px = LEFT * SRC_W
    right_px = RIGHT * SRC_W
    assert min(while_tracking) + CROP_PX / 2 < 0.5 * SRC_W + 40, "never framed the left person"
    assert max(while_tracking) + CROP_PX / 2 > right_px - 40, "never framed the right person"


def test_binding_survives_a_reactor_and_a_fragmented_person() -> None:
    """The measured clip_a / clip_c failure: two diarized speakers must not end
    up on one physical person because a reactor's open mouth out-scored the real
    speaker and the right-hand person is split into two tracks."""
    an, diar = unreliable_two_person()
    fused = assign_speakers(an, diar)

    by_id = {tr["id"]: mean_cx(tr) for tr in an["tracks"]}
    assert len(set(fused["speaker_track"].values())) == 2, (
        f"both speakers on one person: {fused['speaker_track']}")
    cxs = [by_id[t] for t in fused["speaker_track"].values()]
    assert abs(cxs[0] - cxs[1]) > 0.2, cxs
    # and the person who is actually mouthy on speaker 0's turn owns speaker 0
    assert by_id[fused["speaker_track"]["0"]] == pytest.approx(LEFT, abs=1e-6)
    assert by_id[fused["speaker_track"]["1"]] == pytest.approx(RIGHT, abs=1e-6)


# ------------------------------------------------- DEFECT 1/3: not static ---
def test_camera_is_not_static() -> None:
    """Regression for "reframed once at frame 0 and held static"."""
    an, diar = two_person()
    fused = assign_speakers(an, diar)
    path = camera_path(an, fused, "9:16", SRC_W, SRC_H, out_fps=FPS)

    cxs = [e["cx"] for e in path]
    assert len(set(cxs)) >= 10, f"only {len(set(cxs))} distinct crop positions"
    travel = sum(abs(d) for d in deltas(path))
    assert travel >= 0.3, f"total travel {travel:.3f} of available travel"
    # in pixels, on the real render: it must actually pan a long way
    px_travel = sum(abs(d) * (SRC_W - CROP_PX) for d in deltas(path))
    assert px_travel >= 200, f"only {px_travel:.0f} px of pan"


def test_camera_converges_on_the_speaker_it_follows() -> None:
    """Not just moving: it has to arrive. The old velocity clamp needed 12.6 s to
    cross the frame, so in a 8 s clip it was still hundreds of px short."""
    an, diar = two_person()
    path = camera_path(an, assign_speakers(an, diar), "9:16", SRC_W, SRC_H, out_fps=FPS)
    for face, turn_end in ((LEFT, 2.4), (RIGHT, 4.4), (LEFT, 6.4), (RIGHT, 7.9)):
        settled = [e for e in path if turn_end - 0.6 <= e["t"] <= turn_end]
        assert settled, face
        x0, x1 = crop_window_px(settled[-1]["cx"], SRC_W, CROP_PX)
        assert abs((x0 + x1) / 2 - face * SRC_W) <= 20, (
            f"at t={turn_end} the window centre is "
            f"{(x0 + x1) / 2 - face * SRC_W:+.0f} px off the speaker")


# ------------------------------------------------------------- DEFECT 3 -----
def test_one_euro_is_live_step_ramps_and_is_fast() -> None:
    """A step in the desired cx must come out as a RAMP, and much faster than
    the old 12.6 s constant-velocity traverse."""
    an, _ = two_person()
    fused = hand_timeline(8.0, switch_at=2.0)
    path = camera_path(an, fused, "9:16", SRC_W, SRC_H, out_fps=FPS)

    start = t_at(path, 1.8)
    target = face_center_to_cx(RIGHT, SRC_W, CROP_PX)
    total = abs(target - start)
    assert total > 0.3, "fixture should demand a real pan"

    after = [e for e in path if e["t"] > 1.9]
    steps = [abs(b["cx"] - a["cx"]) for a, b in zip(after, after[1:])]
    assert steps[0] < 0.4 * total, f"first step {steps[0]:.3f} of {total:.3f} is a jump"
    moving = [s for s in steps if s > 1e-4]
    assert len(moving) >= 5, f"pan took {len(moving)} frames: not a ramp"
    # reaches the target well inside the old 12.6 s traverse
    reached = next(e["t"] for e in after if abs(e["cx"] - target) <= 0.02 * total)
    assert reached - 1.9 < 3.0, f"took {reached - 1.9:.2f}s to arrive"


def test_filter_respects_velocity_clamp_as_a_ceiling() -> None:
    """The clamp is a safety net: no frame may exceed it, ever."""
    an, _ = two_person()
    fused = hand_timeline(8.0, switch_at=2.0)
    ceiling = max_step(FPS)
    # a whole-clip sequence of cut-free pans, worst case for the limiter
    path = camera_path(an, fused, "9:16", SRC_W, SRC_H, out_fps=FPS)
    for d in deltas(path):
        assert abs(d) <= ceiling + 1e-6, f"frame step {d:.4f} over the {ceiling:.4f} ceiling"


def test_one_euro_filter_seeds_without_passthrough() -> None:
    """A reset must not emit the raw sample (the old first-sample passthrough
    turned every shot cut into an unfiltered jump)."""
    f = cam._OneEuro()
    f.seed(0.5)
    assert f(0.9, 1 / FPS) < 0.9, "first output after a seed is the raw input"
    assert f(0.9, 1 / FPS) > 0.5, "first output after a seed went backwards"
    f.seed(0.2)
    assert f(0.2, 1 / FPS) == pytest.approx(0.2), "a seeded filter is a no-op on its seed"


def test_shot_cut_jumps_instead_of_panning() -> None:
    """At a cut the crop jumps to the new shot's subject; without the cut the
    same speaker change is rate-limited into a ramp."""
    an, _ = two_person()
    smooth = camera_path(an, hand_timeline(8.0, switch_at=2.0), "9:16",
                         SRC_W, SRC_H, out_fps=FPS)
    ramped = abs(t_at(smooth, 2.0) - t_at(smooth, 1.96))
    assert ramped <= max_step(FPS) + 1e-6, "cut-free pan should be rate limited"

    an["shots"] = [{"start": 0.0, "end": 2.0}, {"start": 2.0, "end": 8.0}]
    path = camera_path(an, hand_timeline(8.0, switch_at=2.0), "9:16",
                       SRC_W, SRC_H, out_fps=FPS)
    at_cut = abs(t_at(path, 2.0) - t_at(path, 1.96))
    assert at_cut > 4 * ramped, f"only moved {at_cut:.3f} at the cut: panned across it"
    assert at_cut > max_step(FPS), "a cut must be allowed to exceed the velocity clamp"
    after = [abs(path[i + 1]["cx"] - path[i]["cx"]) for i in range(len(path) - 1)
             if 2.0 < path[i]["t"] <= 2.12]
    assert max(after) < 1e-3, f"ramped across the cut: {after}"


# ------------------------------------------------------ DEFECT 1: the ladder --
def test_strategy_ladder_hold_wide_track() -> None:
    # (a) no face at t=0: HOLD, but seeded from the first real evidence -- NOT a
    # centre crop. The old code emitted cx == 0.5 for 2.80 s on clip_c.
    late = track(3, 0.42, grid(5.0, start=1.0), 0.8)
    an = analysis([late], 5.0)
    fused = {"timeline": [{"t": t, "active_track": None, "confidence": 0.0}
                          for t in grid(5.0)]}
    path = camera_path(an, fused, "9:16", SRC_W, SRC_H, out_fps=FPS)
    assert path[0]["strategy"] == HOLD
    assert path[0]["cx"] == pytest.approx(face_center_to_cx(0.42, SRC_W, CROP_PX), abs=1e-3)
    assert path[0]["cx"] != pytest.approx(0.5, abs=1e-3), "seeded to the frame centre"

    # (b) faces visible, nobody speaking -> WIDE, framed on the group, no panning
    both = analysis([track(4, 0.25, grid(5.0), 0.3), track(5, 0.75, grid(5.0), 0.3)], 5.0)
    wide = camera_path(both, fused, "9:16", SRC_W, SRC_H, out_fps=FPS)
    assert {e["strategy"] for e in wide} == {WIDE}, {e["strategy"] for e in wide}
    group_mid = face_center_to_cx(0.5, SRC_W, CROP_PX)
    assert wide[-1]["cx"] == pytest.approx(group_mid, abs=1e-3)
    assert len(set(e["cx"] for e in wide)) <= 3, "panned between the faces while wide"

    # (c) a confident speaker -> TRACK, centred on that person
    tr_fused = {"timeline": [{"t": t, "active_track": 4, "confidence": 0.99}
                             for t in grid(5.0)]}
    trk = camera_path(both, tr_fused, "9:16", SRC_W, SRC_H, out_fps=FPS)
    assert trk[-1]["strategy"] == TRACK
    assert trk[-1]["cx"] == pytest.approx(face_center_to_cx(0.25, SRC_W, CROP_PX), abs=2e-3)

    # (d) an active_track we are not confident about is NOT a track decision
    shy = {"timeline": [{"t": t, "active_track": 4, "confidence": 0.0}
                        for t in grid(5.0)]}
    assert {e["strategy"] for e in camera_path(both, shy, "9:16", SRC_W, SRC_H, out_fps=FPS)} \
        == {WIDE}


def test_hold_in_the_middle_does_not_snap_to_centre() -> None:
    """Faces vanish mid-clip: keep the last crop, do not re-centre."""
    ts = grid(6.0)
    early = track(7, 0.30, [t for t in ts if t <= 2.6], 0.8)
    an = analysis([early], 6.0)
    fused = {"timeline": [
        {"t": t, "active_track": 7 if t <= 2.6 else None, "confidence": 0.95}
        for t in ts]}
    path = camera_path(an, fused, "9:16", SRC_W, SRC_H, out_fps=FPS)
    held = face_center_to_cx(0.30, SRC_W, CROP_PX)
    assert path[-1]["strategy"] == HOLD
    assert path[-1]["cx"] == pytest.approx(held, abs=DEAD_ZONE + 1e-3)
    assert path[-1]["cx"] != pytest.approx(0.5, abs=0.05)


# --------------------------------------------------------- DEFECT 2: one cx --
def test_render_window_matches_the_shared_helper() -> None:
    """Round trip: the window the validator scores IS the window render.py
    emits, and it is centred on the face the camera is tracking."""
    an, diar = two_person()
    fused = assign_speakers(an, diar)
    path = camera_path(an, fused, "9:16", SRC_W, SRC_H, out_fps=FPS)

    xs = crop_x_sequence(path, SRC_W, CROP_PX)
    assert len(xs) == len(path)
    for e, x in zip(path, xs):
        assert x == crop_left_px(e["cx"], SRC_W, CROP_PX)
        assert 0 <= x <= SRC_W - CROP_PX, "out of bounds"
        assert x % 2 == 0, "odd x is not yuv420p-safe"
    # the semantic check the two-convention code failed: for a settled TRACK
    # frame the rendered window is centred on the tracked face, not on cx
    tracked = [e for e in path if e["strategy"] == TRACK and e["t"] > 7.0]
    assert tracked
    face_px = RIGHT * SRC_W
    x0, x1 = crop_window_px(tracked[-1]["cx"], SRC_W, CROP_PX)
    assert abs((x0 + x1) / 2 - face_px) <= 2, "rendered window is not on the face"


def test_face_center_to_cx_round_trip() -> None:
    """cx -> rendered window is centred on the face, everywhere the window fits
    (a face within half a crop width of an edge saturates at the edge)."""
    for face in (0.20, 0.30, 0.5, 0.66, 0.80):
        cx = face_center_to_cx(face, SRC_W, CROP_PX)
        x0, x1 = crop_window_px(cx, SRC_W, CROP_PX)
        assert abs((x0 + x1) / 2 / SRC_W - face) <= 2.0 / SRC_W, face
    # a face hard against the edge saturates instead of going out of bounds
    assert face_center_to_cx(0.0, SRC_W, CROP_PX) == 0.0
    assert face_center_to_cx(1.0, SRC_W, CROP_PX) == 1.0
    assert face_center_to_cx(0.05, SRC_W, CROP_PX) == 0.0
    assert crop_left_px(-5.0, SRC_W, CROP_PX) == 0
    assert crop_left_px(5.0, SRC_W, CROP_PX) <= SRC_W - CROP_PX
    # full travel is reachable: cx=0 and cx=1 are the two extreme windows
    assert crop_left_px(0.0, SRC_W, CROP_PX) == 0
    assert crop_left_px(1.0, SRC_W, CROP_PX) == SRC_W - CROP_PX


def test_ambiguous_binding_reports_low_confidence_and_goes_wide() -> None:
    """One face for two diarized speakers: we must NOT confidently frame it for
    both. Speaker 0 keeps the track, speaker 1 gets no track, and the camera
    falls back to the WIDE rung for that stretch."""
    ts = grid(6.0)
    an = analysis([track(21, 0.40, ts, speaking_mouth([("0", 0.5, 3.0)], "0"))], 6.0)
    fused = assign_speakers(an, diarization([("0", 0.5, 3.0), ("1", 3.0, 5.5)]))

    assert fused["speaker_track"] == {"0": 21}, fused["speaker_track"]
    assert fused["speaker_confidence"]["1"] == 0.0
    assert fused["ambiguous"] is True

    conf = {f["t"]: f["confidence"] for f in fused["timeline"]}
    assert conf[1.0] > 0.25, "speaker 0's own turn is still confident"
    assert conf[4.0] == 0.0, "speaker 1 has no person: zero confidence"

    path = camera_path(an, fused, "9:16", SRC_W, SRC_H, out_fps=FPS)
    tracked = [e for e in path if 1.5 <= e["t"] <= 2.5]
    assert {e["strategy"] for e in tracked} == {TRACK}
    during_one = [e for e in path if 3.6 <= e["t"] <= 5.2]
    assert {e["strategy"] for e in during_one} == {WIDE}, "should not lock a face"
    assert during_one[-1]["cx"] == pytest.approx(
        face_center_to_cx(0.40, SRC_W, CROP_PX), abs=1e-3)


def test_overlapping_diarization_is_resolved_once() -> None:
    """Sarvam overlaps speakers (measured: 6 pairs, up to 0.86 s). The contested
    span goes to the longer turn, is counted once, and is reported."""
    segs = [{"speaker": "0", "start": 0.0, "end": 3.0},
            {"speaker": "1", "start": 2.5, "end": 5.0}]
    part, dropped = _resolve_overlaps(segs)
    assert dropped == pytest.approx(0.5)
    spans = sorted((s["start"], s["end"]) for s in part)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start >= end - 1e-9, f"still overlapping: {spans}"
    owner = next(s["speaker"] for s in part if s["start"] < 2.7 < s["end"])
    assert owner == "0", "the longer turn keeps the contested half second"
    total = sum(e - s for s, e in spans)
    assert total == pytest.approx(5.0)


def test_audio_offset_shifts_diarization_when_probe_reports_a_stream_start() -> None:
    """probe.py does not record start_time today (measured offset 0.000 s on all
    four assets), so this is 0.0 unless a future probe reports one."""
    ts = grid(6.0)
    an = analysis([track(31, 0.40, ts, speaking_mouth([("0", 0.5, 3.0)], "0"))], 6.0)
    diar = diarization([("0", 0.5, 3.0)])
    plain = assign_speakers(an, diar)
    assert audio_offset(an) == 0.0

    an["probe"].update({"audio_start": 0.6, "video_start": 0.0})
    assert audio_offset(an) == pytest.approx(0.6)
    shifted = assign_speakers(an, diar)
    assert audio_offset(an) == pytest.approx(0.6)
    plain_active = {f["t"] for f in plain["timeline"] if f["active_track"] is not None}
    moved_active = {f["t"] for f in shifted["timeline"] if f["active_track"] is not None}
    assert moved_active == {round(t + 0.6, 3) for t in plain_active}


# ------------------------------------------- validator: coverage vs hit rate --
def test_validator_reports_coverage_separately_from_hit_rate() -> None:
    an, diar = two_person(4.0)
    ts = grid(4.0)
    # only the first half of the speech has a tracked active speaker
    fused = {"timeline": [{"t": t, "active_track": 11 if t <= 1.9 else None,
                           "confidence": 0.95} for t in ts]}
    path = camera_path(an, fused, "9:16", SRC_W, SRC_H, out_fps=FPS)
    score = validate_speaker.speaker_on_screen(
        an, diar, path, fused["timeline"], crop_w=CROP_PX / SRC_W)

    assert score["speech_frames"] > 0
    assert score["untracked_frames"] > 0, "untracked frames vanished from the denominator"
    assert score["tracked_frames"] + score["untracked_frames"] == score["speech_frames"]
    assert 0.0 < score["coverage"] < 1.0, score["coverage"]
    assert score["overall_all"] == pytest.approx(score["overall"] * score["coverage"], abs=0.02)
    assert score["overall_all"] < score["overall"], "hit rate is hiding the misses"
    # the zero-coverage segment is still reported, with its real length
    untracked = [s for s in score["per_segment"] if s["frames"] == 0]
    assert untracked and all(s["coverage"] == 0.0 for s in untracked)
    assert all(s["score"] is None for s in untracked)
    # legacy keys main.py reads are still present
    for key in ("overall", "overall_confident", "target", "pass", "diarized", "per_segment"):
        assert key in score, key
    # and the honest boolean cannot be true while a third of the speech is dark
    assert score["pass"] is True
    assert score["pass_with_coverage"] is False, "hit rate without coverage is not a pass"


def test_validator_ignores_a_face_sliver_at_the_window_edge() -> None:
    ts = grid(2.0)
    an = analysis([track(9, 0.30, ts, 0.8)], 2.0)
    fused = {"timeline": [{"t": t, "active_track": 9, "confidence": 0.95} for t in ts]}
    diar = diarization([("0", 0.0, 2.0)])

    def parked(left_px: float) -> list[dict[str, Any]]:
        # cx is a fraction of TRAVEL: this is the crop window the render would emit
        cx = left_px / (SRC_W - CROP_PX)
        return [{"t": round(i / FPS, 3), "cx": cx, "strategy": WIDE}
                for i in range(int(2.0 * FPS) + 1)]

    # park the window so only the last 3% of the face width is inside it
    face_left, face_w = 0.30 * SRC_W - 0.07 * SRC_W, 0.14 * SRC_W
    score = validate_speaker.speaker_on_screen(
        an, diar, parked(face_left + face_w * 0.97), fused["timeline"],
        crop_w=CROP_PX / SRC_W)
    assert score["tracked_frames"] == score["speech_frames"]
    assert score["overall"] == 0.0, "a sliver of a face counted as on-screen"
    assert score["min_face_in_window"] == validate_speaker.MIN_FACE_IN_WINDOW

    # and a window that really holds the face does count
    centred = face_center_to_cx(0.30, SRC_W, CROP_PX)
    score = validate_speaker.speaker_on_screen(
        an, diar, parked(centred * (SRC_W - CROP_PX)), fused["timeline"],
        crop_w=CROP_PX / SRC_W)
    assert score["overall"] == 1.0
