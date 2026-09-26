#!/usr/bin/env python3
"""CLI: .venv/bin/python scripts/analyze_clip.py <video> <out.json>
Writes merged analysis JSON + a debug contact sheet PNG next to it."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.reframe import probe  # noqa: E402
from worker.reframe.video.faces import ANALYSIS_FPS, analyze_faces  # noqa: E402
from worker.reframe.video.shots import detect_shots  # noqa: E402


def contact_sheet(video: Path, out_png: Path, tracks: list[dict], cols: int = 6,
                  cell_w: int = 320, max_frames_per_track: int = 6) -> None:
    """Grid of sampled analysis frames with track boxes + ids drawn."""
    color = [(60, 60, 240), (60, 220, 60), (240, 120, 60), (220, 60, 220),
             (60, 220, 220), (240, 240, 60)]
    # sample one time point per track (spread over its life) + some extra times
    times: list[tuple[float, dict]] = []
    for tr in tracks:
        fr = tr["frames"]
        for f in fr[:: max(1, len(fr) // max_frames_per_track)][:max_frames_per_track]:
            times.append((f["t"], f))
    if not times:
        print("no faces found; empty sheet")
        Path(out_png).write_bytes(b"")
        return

    import numpy as np

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cell_w * 2)
    cells: list[np.ndarray] = []
    for t, f in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = cell_w / w
        frame = cv2.resize(frame, (cell_w, int(h * scale)))
        tid = next(tr["id"] for tr in tracks if any(f_ is f for f_ in tr["frames"]))
        c = color[tid % len(color)]
        x, y, bw, bh = f["box"]
        ih, iw = frame.shape[:2]
        p1 = (int(x * iw), int(y * ih))
        p2 = (int((x + bw) * iw), int((y + bh) * ih))
        cv2.rectangle(frame, p1, p2, c, 2)
        cv2.putText(frame, f"id{tid} t={t:.1f}s", (p1[0], max(12, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        cells.append(frame)
    cap.release()
    if not cells:
        Path(out_png).write_bytes(b"")
        return
    cell_h = cells[0].shape[0]
    for c in cells:
        c[:] = cv2.resize(c, (cell_w, cell_h))
    rows = math.ceil(len(cells) / cols)
    sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        sheet[r * cell_h:(r + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = c
    cv2.imwrite(str(out_png), sheet)
    print(f"contact sheet: {out_png}")


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    video, out_json = Path(sys.argv[1]), Path(sys.argv[2])
    out_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"probing {video} ...")
    p = probe.probe(video)
    print(f"shots ...")
    shots = detect_shots(video)
    print(f"faces (analysis fps={ANALYSIS_FPS}) ...")
    faces = analyze_faces(video)

    result = {"video": str(video), "probe": p, "shots": shots, "tracks": faces["tracks"]}
    out_json.write_text(json.dumps(result, indent=1))
    print(f"wrote {out_json}: {len(shots)} shots, {len(faces['tracks'])} tracks")

    contact_sheet(video, out_json.with_suffix(".png"), faces["tracks"])


if __name__ == "__main__":
    main()
