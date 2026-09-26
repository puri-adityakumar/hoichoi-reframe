"""Face analysis: MediaPipe FaceLandmarker on a downscaled 5 fps stream,
linked into tracks by IoU. Analysis-only brightening (output is untouched)."""
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

    return {
        "fps": ANALYSIS_FPS,
        "brightened_frames": brightened,
        "tracks": [
            {"id": tid, "frames": tracks[tid]}
            for tid in sorted(tracks, key=lambda k: -len(tracks[k]))
        ],
    }
