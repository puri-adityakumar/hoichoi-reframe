"""Face analysis: MediaPipe FaceLandmarker on a downscaled 5 fps stream,
linked into tracks by IoU, then consolidated (see consolidate_tracks: IoU
linking alone shattered two people into 11 and 8 fragments on the 190 s master).
Analysis-only brightening (output is untouched)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, TypedDict

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

P4_ROOT = Path(__file__).resolve().parents[3]
FACE_MODEL = P4_ROOT / "worker" / "models" / "face_landmarker.task"

ANALYSIS_FPS = 5.0
LONG_EDGE = 480
DARK_LUMA = 60
IOU_MATCH = 0.3
MAX_GAP_FRAMES = 2  # 0.4 s at 5 fps
CROSS_CUT_FRAMES = 15  # allow re-link across a shot cut within ~3 s
CROSS_CUT_CENTER = 0.18  # normalized center distance to bridge across a cut
NUM_FACES = 6

# ---- track consolidation (see consolidate_tracks) ------------------------
MERGE_GAP_S = 2.0  # same person may vanish for a blink / head turn / a short
# occlusion; 10 analysis frames, ~3x the 0.6 s window IoU linking allows
MERGE_DX = 0.08  # normalized centre-x distance (~3% of the frame, ~61 px)
MERGE_AREA = 0.55  # 0.55-1.8x box-area ratio

# MediaPipe FaceMesh landmark indices
_LIP_INNER_TOP, _LIP_INNER_BOTTOM = 13, 14
_LIP_CORNER_L, _LIP_CORNER_R = 61, 291


class FrameRecord(TypedDict):
    t: float
    box: tuple[float, float, float, float]  # x, y, w, h normalized on full frame
    mouth: float


class Track(TypedDict):
    id: int
    frames: list[FrameRecord]


def _brighten(frame_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on L channel of LAB. Analysis stream only."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


def _expand(box: tuple[float, float, float, float], pad: float) -> tuple[float, float, float, float]:
    """Pad a box by `pad` fraction on each side (clipped to 0-1)."""
    x, y, w, h = box
    return (max(0.0, x - w * pad), max(0.0, y - h * pad),
            min(1.0, x + w * (1 + pad)) - max(0.0, x - w * pad),
            min(1.0, y + h * (1 + pad)) - max(0.0, y - h * pad))


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------ consolidation --
def _summary(track: dict[str, Any]) -> tuple[float, float, float, float]:
    """(centre-x, area, t_start, t_end) of a track, meaned over its frames."""
    frames = track["frames"]
    cx = sum(f["box"][0] + f["box"][2] / 2 for f in frames) / len(frames)
    area = sum(f["box"][2] * f["box"][3] for f in frames) / len(frames)
    return cx, area, frames[0]["t"], frames[-1]["t"]


def _mergeable(a: tuple[float, float, float, float],
               b: tuple[float, float, float, float]) -> bool:
    """True if two tracks are plausibly the SAME person.

    IoU linking only ever matches consecutive analysis frames, so a face that
    blinks, turns away or is hidden by a shot cut mints a brand new track and the
    abandoned one is never re-acquired. Measured on the 190 s master: 461
    detections across 43 tracks, and clustering the 19 speaker-eligible tracks by
    mean face centre-x gave exactly two people shattered into 11 and 8
    fragments. Consequences downstream: speaker 0 spoke for 42.8 s but got bound
    to a track that existed for 8 frames (1.4 s), and on clip_a/clip_c both
    diarized speakers were mapped onto the SAME physical person. Do not revert
    this pass without re-measuring the fragment count.

    The three tests, all required:
      * time: non-overlapping (or touching) with a gap <= MERGE_GAP_S
      * position: mean centre-x within MERGE_DX in normalized frame coords
      * scale: comparable mean box area (ratio within MERGE_AREA .. 1/MERGE_AREA)
    There is no appearance descriptor in this pipeline, so MERGE_GAP_S is
    deliberately short: a fragment separated by a full reverse-shot (measured
    2.4-7.6 s of absence on the test clips) is NOT merged, because a 7 s gap is
    not evidence of identity and merging on it would happily glue two people who
    happen to sit at similar x. Those long-gap fragments are handled downstream
    instead, by duration-aware speaker binding (see video/speaker.py).
    """
    a_cx, a_area, a_t0, a_t1 = a
    b_cx, b_area, b_t0, b_t1 = b
    gap = max(a_t0, b_t0) - min(a_t1, b_t1)  # negative when they interleave
    if gap > MERGE_GAP_S:
        return False
    if abs(a_cx - b_cx) > MERGE_DX:
        return False
    ratio = a_area / max(1e-6, b_area)
    return MERGE_AREA <= ratio <= 1.0 / MERGE_AREA


def consolidate_tracks(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Iteratively merge tracks that are plausibly the same person.

    Pure function on the track dicts: the per-frame {t, box, mouth} shape is
    untouched, so nothing downstream changes, and the input is not mutated (the
    frames are copied, because the merge appends to them). Merge rounds run to a
    fixed point (a merge changes the summary, which can enable a further merge)
    and the earliest fragment's id wins, so ids stay stable and unique.
    """
    groups: list[dict[str, Any]] = [
        {"id": tr["id"], "frames": [dict(f) for f in tr["frames"]]} for tr in tracks]

    def rekey(g: dict[str, Any]) -> None:
        fr = sorted(g["frames"], key=lambda f: f["t"])
        ids = [f.get("tid", g["id"]) for f in fr]
        g["frames"] = [{"t": f["t"], "box": tuple(f["box"]), "mouth": f["mouth"]} for f in fr]
        g["id"] = min(ids)

    changed = True
    while changed:
        changed = False
        summaries = [_summary(g) for g in groups]
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if not _mergeable(summaries[i], summaries[j]):
                    continue
                keep_i, drop_i = sorted((i, j), key=lambda k: -len(groups[k]["frames"]))
                keep, dropped = groups[keep_i], groups[drop_i]
                for f in dropped["frames"]:
                    f["tid"] = f.get("tid", dropped["id"])
                keep["frames"].extend(dropped["frames"])
                groups.pop(drop_i)
                rekey(keep)
                summaries = [_summary(g) for g in groups]
                changed = True
                break
            if changed:
                break
    return [{"id": g["id"], "frames": g["frames"]} for g in groups]


def _landmark_pts(lm: Any) -> np.ndarray:
    return np.array([[p.x, p.y] for p in lm], dtype=np.float64)


def _mouth_signal(pts: np.ndarray) -> float:
    inner = float(np.linalg.norm(pts[_LIP_INNER_TOP] - pts[_LIP_INNER_BOTTOM]))
    width = float(np.linalg.norm(pts[_LIP_CORNER_L] - pts[_LIP_CORNER_R]))
    return inner / width if width > 1e-6 else 0.0


def analyze_faces(video_path: str | Path, model_path: Path | None = None) -> dict[str, Any]:
    """Decode at ANALYSIS_FPS, detect up to 6 faces/frame, link into IoU tracks."""
    path = Path(video_path)
    model = Path(model_path) if model_path else FACE_MODEL
    if not model.exists():
        raise FileNotFoundError(f"face model missing: {model}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(src_fps / ANALYSIS_FPS))
    frame_step_t = step / src_fps

    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=NUM_FACES,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )

    # open tracks: track_id -> (last_frame_idx, box)
    open_tracks: dict[int, tuple[int, tuple[float, float, float, float]]] = {}
    tracks: dict[int, list[FrameRecord]] = {}
    next_id = 0
    brightened = 0

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step != 0:
                idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                break
            t = idx / src_fps
            idx += 1

            scale = LONG_EDGE / max(frame.shape[:2])
            analysis = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if float(analysis.mean()) < DARK_LUMA:
                analysis = _brighten(analysis)
                brightened += 1

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(analysis, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_img, timestamp_ms=int(t * 1000))

            detections: list[tuple[tuple[float, float, float, float], float, np.ndarray]] = []
            for det in result.face_landmarks:
                pts = _landmark_pts(det)
                xs, ys = pts[:, 0], pts[:, 1]
                # normalized on the full frame; landmarks can spill slightly
                # past frame bounds, so clamp to 0-1
                x0, y0 = max(0.0, float(xs.min())), max(0.0, float(ys.min()))
                x1, y1 = min(1.0, float(xs.max())), min(1.0, float(ys.max()))
                box = (x0, y0, x1 - x0, y1 - y0)
                detections.append((box, _mouth_signal(pts), pts))

            # match detections to open tracks: IoU on expanded boxes for
            # consecutive frames; conservative center-proximity re-link when a
            # track was interrupted by a shot cut (shot/reverse-shot dialogue)
            candidates: list[tuple[float, int, int]] = []
            for tid, (last_idx, last_box) in open_tracks.items():
                gap_decode = idx - 1 - last_idx
                near = gap_decode <= step * (MAX_GAP_FRAMES + 1)
                across_cut = step * (MAX_GAP_FRAMES + 1) < gap_decode <= step * (CROSS_CUT_FRAMES + 1)
                if not (near or across_cut):
                    continue
                for di, (box, _, _) in enumerate(detections):
                    if near:
                        score = _iou(_expand(last_box, 0.4), _expand(box, 0.4))
                        if score > IOU_MATCH:
                            candidates.append((score, tid, di))
                    else:
                        c1, c2 = _center(last_box), _center(box)
                        dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                        size_ratio = (last_box[2] * last_box[3]) / max(1e-6, box[2] * box[3])
                        if dist < CROSS_CUT_CENTER and 0.5 < size_ratio < 2.0:
                            candidates.append((1.0 - dist, tid, di))
            matched_det: set[int] = set()
            for _, tid, di in sorted(candidates, reverse=True):
                if di in matched_det:
                    continue
                box, mouth, _ = detections[di]
                tracks[tid].append({"t": round(t, 3), "box": box, "mouth": round(mouth, 4)})
                open_tracks[tid] = (idx - 1, box)
                matched_det.add(di)

            for di, (box, mouth, _) in enumerate(detections):
                if di in matched_det:
                    continue
                tid = next_id
                next_id += 1
                tracks[tid] = [{"t": round(t, 3), "box": box, "mouth": round(mouth, 4)}]
                open_tracks[tid] = (idx - 1, box)
    cap.release()

    merged = consolidate_tracks([
        {"id": tid, "frames": tracks[tid]} for tid in sorted(tracks, key=lambda k: -len(tracks[k]))
    ])
    return {
        "fps": ANALYSIS_FPS,
        "brightened_frames": brightened,
        # longest first, as before; consolidation only removes fragments
        "tracks": sorted(merged, key=lambda tr: -len(tr["frames"])),
    }
