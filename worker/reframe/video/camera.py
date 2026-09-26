"""Camera path: smooth normalized center-x of the crop window, following the
active speaker. One Euro filter + velocity clamp + dead zone; never pans
across a shot cut (state resets, jump to the new desired cx)."""
from __future__ import annotations

import math
from typing import Any

CROP_W = 608 / 1920  # normalized crop width for 9:16 from 16:9
ONE_EURO_FREQ = 25.0
MIN_CUTOFF = 0.8
BETA = 1.5
D_CUTOFF = 1.0
DEAD_ZONE = 0.02


class _LowPass:
    def __init__(self) -> None:
        self.y = None
        self.ready = False

    def __call__(self, x: float, alpha: float) -> float:
        self.y = x if not self.ready else alpha * x + (1 - alpha) * self.y
        self.ready = True
        return self.y


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _OneEuro:
    def __init__(self, freq: float = ONE_EURO_FREQ, min_cutoff: float = MIN_CUTOFF,
                 beta: float = BETA, d_cutoff: float = D_CUTOFF) -> None:
        self.freq, self.min_cutoff, self.beta, self.d_cutoff = freq, min_cutoff, beta, d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._prev = 0.0
        self._first = True

    def __call__(self, x: float, dt: float) -> float:
        if self._first:
            self._prev = x
            self._first = False
            self._x(x, 1.0)
            self._dx(0.0, 1.0)
            return x
        dx = (x - self._prev) / dt
        edx = self._dx(dx, _alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        out = self._x(x, _alpha(cutoff, dt))
        self._prev = x
        return out

    def reset(self) -> None:
        self._x = _LowPass()
        self._dx = _LowPass()
        self._first = True


def _track_at(tracks_by_id: dict[int, dict[str, Any]], tid: int, t: float,
              tol: float = 0.21) -> tuple[float, float, float, float] | None:
    """Nearest track box at time t (within tol) or None."""
    frames = tracks_by_id[tid]["frames"]
    best, best_d = None, tol
    for f in frames:
        d = abs(f["t"] - t)
        if d <= best_d:
            best, best_d = f, d
    return best["box"] if best else None


def _desired(analysis: dict[str, Any], timeline: dict[str, Any], t: float,
             crop_w: float) -> float | None:
    tracks_by_id = {tr["id"]: tr for tr in analysis["tracks"]}
    cur = next((f for f in timeline if abs(f["t"] - t) < 1e-6), None)
    if cur is None:
        cur = min(timeline, key=lambda f: abs(f["t"] - t))
    tid = cur["active_track"]
    if tid is not None:
        box = _track_at(tracks_by_id, tid, t)
        if box:
            return box[0] + box[2] / 2
    # fallback: no active face visible
    visible = [f["box"] for tr in analysis["tracks"] for f in tr["frames"]
               if abs(f["t"] - t) <= 0.21]
    if not visible:
        return None
    # if the two widest faces fit inside one crop window, center between them
    by_size = sorted(visible, key=lambda b: -b[2] * b[3])[:2]
    two_c = [(b[0] + b[2] / 2, b) for b in by_size]
    if len(two_c) == 2:
        lo = min(c for c, _ in two_c)
        hi = max(c for c, _ in two_c)
        span_lo = min(b[0] for _, b in two_c)
        span_hi = max(b[0] + b[2] for _, b in two_c)
        if span_hi - span_lo <= crop_w:
            return (lo + hi) / 2
    centers = [b[0] + b[2] / 2 for b in visible]
    return sum(centers) / len(centers)


def camera_path(analysis: dict[str, Any], timeline: dict[str, Any], target_ratio: str,
                src_w: int, src_h: int, out_fps: float = 25.0) -> list[dict[str, float]]:
    """Normalized center-x per output frame, smoothed and clamped to [0,1]."""
    rw, rh = (int(v) for v in target_ratio.split(":"))
    # normalized crop-window width for the target ratio from this source
    crop_w = CROP_W if (target_ratio == "9:16" and src_w == 1920) else min(
        1.0, (src_h * rw / rh) / src_w)
    max_speed_n = 0.25 * crop_w  # per second; crosses frame in ~2 s
    dead = DEAD_ZONE * (1.0 if crop_w <= CROP_W else crop_w / CROP_W)

    shots = [s["start"] for s in analysis["shots"]]
    tl = timeline["timeline"] if "timeline" in timeline else timeline
    filt = _OneEuro()
    dt = 1.0 / out_fps
    cut_tol = 0.5 / out_fps

    duration = analysis["probe"]["duration"]
    out: list[dict[str, float]] = []
    cur_cx = 0.5

    n = int(duration * out_fps)
    for i in range(n + 1):
        t = round(i / out_fps, 3)

        desired = _desired(analysis, tl, t, crop_w)
        if desired is None:
            desired = cur_cx

        # shot cut: reset filter, jump to the new desired cx (no pan across cut)
        if i == 0 or any(abs(t - s) <= cut_tol for s in shots):
            filt.reset()
            cur_cx = desired
            out.append({"t": t, "cx": round(_clamp(cur_cx, crop_w), 4)})
            continue

        # dead zone + rate limiting
        err = desired - cur_cx
        if abs(err) > dead:
            step = filt(err, dt)  # smoothed approach toward the target
            max_step = max_speed_n * dt
            step = max(-max_step, min(max_step, step))
            cur_cx += step
        out.append({"t": t, "cx": round(_clamp(cur_cx, crop_w), 4)})
    return out


def _clamp(cx: float, crop_w: float) -> float:
    """Keep the crop window inside [0,1]."""
    half = crop_w / 2
    return max(half, min(1.0 - half, cx))
