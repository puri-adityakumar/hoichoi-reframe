"""Preview stills for the web thumbnail grid.

Invariant: every ``outputs.preview_key`` is an IMAGE. The web renders it in an
``<img src>``, and a browser cannot decode a video there — an mp4 preview_key
renders as a broken-image icon. So a video output gets a single JPEG poster
frame, never a downscaled mp4.
"""
from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

from PIL import Image

# Poster edge for video outputs; still-image masters get a smaller thumb.
VIDEO_POSTER_EDGE = 480
IMAGE_PREVIEW_EDGE = 240

# Extensions the web is allowed to put in an <img>. Anything else is a bug.
IMAGE_PREVIEW_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"})

# Poster timestamp as a fraction of the clip, nudged off the first frame so we
# never land on a fade-in. Clamped so short clips still resolve.
POSTER_AT_FRACTION = 0.15


def preview_ext(key: str) -> str:
    return PurePosixPath(key).suffix.lower()


def is_image_preview(key: str) -> bool:
    return preview_ext(key) in IMAGE_PREVIEW_EXTS


def assert_image_preview(key: str) -> None:
    """Raise unless `key` names an image. Call before writing preview_key."""
    if not is_image_preview(key):
        raise ValueError(
            f"preview_key must be an image ({sorted(IMAGE_PREVIEW_EXTS)}), got {key!r}"
        )


def _ffmpeg(args: list[str]) -> None:
    res = subprocess.run(["ffmpeg", "-v", "error", "-y", *args],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr[-800:]}")


def make_image_preview(src: Path, dst: Path) -> Path:
    im = Image.open(src).convert("RGB")
    im.thumbnail((IMAGE_PREVIEW_EDGE, IMAGE_PREVIEW_EDGE), Image.LANCZOS)
    im.save(dst, "JPEG", quality=70)
    return dst


def make_video_poster(src: Path, dst: Path, duration: float | None = None) -> Path:
    """Write one representative frame of `src` to `dst` as a JPEG poster."""
    if duration is None:
        from . import probe as probe_mod
        duration = probe_mod.probe(src)["duration"]
    at = duration * POSTER_AT_FRACTION
    if duration > 0:
        at = min(max(at, 0.0), max(0.0, duration - 0.2))

    vf = (f"scale='if(gt(iw,ih),{VIDEO_POSTER_EDGE},-2)':"
          f"'if(gt(iw,ih),-2,{VIDEO_POSTER_EDGE})'")
    _ffmpeg(["-ss", f"{at:.3f}", "-i", str(src), "-frames:v", "1",
             "-vf", vf, "-q:v", "4", str(dst)])

    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"video poster empty for {src}")
    return dst
