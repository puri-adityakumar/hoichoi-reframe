"""Shot detection with PySceneDetect on a downscaled analysis stream."""
from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from scenedetect import ContentDetector, SceneManager, open_video


class Shot(TypedDict):
    start: float
    end: float


def detect_shots(video_path: str | Path, threshold: float = 27.0) -> list[dict[str, Any]]:
    """Return shot boundaries in seconds: [{start, end}]."""
    video = open_video(str(video_path))
    down = 480 / max(video.frame_size)
    factor = max(1, round(1 / down)) if down < 1.0 else 1
    sm = SceneManager()
    sm.auto_downscale = False
    sm.downscale = factor
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video, show_progress=False)
    return [
        {"start": round(s[0].get_seconds(), 3), "end": round(s[1].get_seconds(), 3)}
        for s in sm.get_scene_list()
    ]
