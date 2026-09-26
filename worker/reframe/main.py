"""Creative Reformatting Engine CLI.

Usage: .venv/bin/python -m reframe <input_path> --title "T" [--outdir DIR] [--publish]

Image master: importance map -> 3 candidates per image ratio -> VLM pick (4 calls)
-> render -> validations.
Video master: probe -> 5 fps analysis -> Sarvam batch diarization (cached, ONE
submission) -> speaker fusion -> camera path + sendcmd render per target ratio
-> 16:9 stream copy -> VLM still pick (1 call) -> validations + speaker-on-screen.
Writes <outdir>/manifest.json; --publish uploads to Neon (S3 + Postgres).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from . import db, probe as probe_mod, storage, validate, validate_speaker
from .config import load_config
from .decisions import DecisionLog
from .image.crops import best_crop_candidates
from .image.importance import importance_map
from .image.render import render_crop
from .video.camera import camera_path
from .video.faces import analyze_faces
from .video.render import crop_size, render_vertical
from .video.shots import detect_shots
from .video.speaker import assign_speakers
from .vlm import pick_best

P4_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = P4_ROOT / "spec" / "spec.json"
SARVAM_DIR = P4_ROOT / "work" / "sarvam"

# image platforms by ratio: 16:9, 1:1, 9:16, 4:5 (4 VLM calls total)
IMAGE_PLATFORMS = ["youtube_thumbnail", "square_image", "story_image", "feed_image"]
# video target ratios: 9:16 crop-tracked, 4:5 crop-tracked, 1:1 crop-tracked,
# 16:9 = stream copy (master is already 16:9); still = 16:9 thumbnail
CROP_PLATFORMS = ["instagram_reel", "instagram_feed", "youtube_square"]

VIDEO_PREVIEW_EDGE = 480
IMAGE_PREVIEW_EDGE = 240


# ---------------------------------------------------------------- helpers ----

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


def make_video_preview(src: Path, dst: Path) -> Path:
    vf = (f"scale='if(gt(iw,ih),{VIDEO_PREVIEW_EDGE},-2)':"
          f"'if(gt(iw,ih),-2,{VIDEO_PREVIEW_EDGE})'")
    _ffmpeg(["-i", str(src), "-vf", vf, "-c:v", "libx264", "-crf", "28",
             "-preset", "veryfast", "-an", "-movflags", "+faststart", str(dst)])
    return dst


def _fallback_spans(timeline: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Time spans where no active speaker track was engaged."""
    spans: list[tuple[float, float]] = []
    for f in timeline:
        if f["active_track"] is None:
            if spans and f["t"] - spans[-1][1] <= 0.21:
                spans[-1] = (spans[-1][0], f["t"])
            else:
                spans.append((f["t"], f["t"]))
    return spans


# ------------------------------------------------------------- image path ----

def run_image_master(img_path: Path, outdir: Path, platforms: dict[str, Any],
                     log: DecisionLog, timings: dict[str, float]) -> list[dict[str, Any]]:
    t0 = time.monotonic()
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"cannot read image {img_path}")
    img_h, img_w = img_bgr.shape[:2]
    log.add("watermark_mask",
            "zeroed importance band y=0.40-0.60 before painting faces",
            "burned-in watermark must not attract crops; faces painted after win")

    imp, _ = importance_map(img_bgr)
    face_mask = (imp > 0.85).astype(np.float32)
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    timings["importance_map"] = round(time.monotonic() - t0, 2)

    outputs: list[dict[str, Any]] = []
    for platform in IMAGE_PLATFORMS:
        t1 = time.monotonic()
        p = platforms[platform]
        cands = best_crop_candidates(imp, img_w, img_h, p["ratio"], k=3,
                                     face_mask=face_mask)
        crops = [img.crop((c["box"][0], c["box"][1],
                           c["box"][0] + c["box"][2], c["box"][1] + c["box"][3]))
                 for c in cands]
        context = (f"Target platform: {p['label']} ({p['ratio']}, "
                   f"{p['width']}x{p['height']}). Pick the crop keeping the most "
                   "important subjects (faces/people) fully visible and well framed.")
        pick, provider = pick_best(crops, context)
        best = cands[pick["choice"]]
        log.add(f"image_crop:{platform}",
                f"candidate {pick['choice']} of {len(cands)}",
                f"{pick['reason']} (provider={provider}, scores="
                f"{[round(c['score'], 3) for c in cands]})")
        out_path = render_crop(img, best["box"], p["width"], p["height"],
                               outdir / f"{platform}.jpg")
        prev = make_image_preview(out_path, outdir / f"{platform}.preview.jpg")
        outputs.append({
            "platform": platform, "ratio": p["ratio"], "kind": "crop",
            "path": out_path, "preview_path": prev,
            "size": out_path.stat().st_size, "speaker_pct": None,
        })
        timings[f"image:{platform}"] = round(time.monotonic() - t1, 2)
    return outputs


# ------------------------------------------------------------- video path ----

def diarize_cached(video: Path, log: DecisionLog, timings: dict[str, float]) -> Optional[dict[str, Any]]:
    """ONE Sarvam batch submission for the whole video, cached by sha1 of the
    first 2 MB. Max 2 attempts incl. retry; falls back to mouth-only."""
    t0 = time.monotonic()
    SARVAM_DIR.mkdir(parents=True, exist_ok=True)
    with open(video, "rb") as fh:
        head = fh.read(2 * 1024 * 1024)
    key = hashlib.sha1(head).hexdigest()[:16]
    cache = SARVAM_DIR / f"{key}.json"
    if cache.exists():
        log.add("diarization", f"reused cache {cache.name}",
                "sha1(first 2 MB) matched a previous batch result")
        timings["diarization"] = round(time.monotonic() - t0, 2)
        return json.loads(cache.read_text())

    wav = SARVAM_DIR / f"{key}_mono.wav"
    if not wav.exists():
        _ffmpeg(["-i", str(video), "-ac", "1", "-ar", "16000", str(wav)])

    from .sarvam import JobParameters, SarvamBatchClient, SarvamError
    payload: Optional[dict[str, Any]] = None
    err: Optional[str] = None
    for attempt in (1, 2):  # max 2 submissions total
        try:
            with SarvamBatchClient() as client:
                job_id = client.submit_batch(
                    wav, JobParameters(model="saaras:v3", mode="transcribe",
                                       with_timestamps=True, with_diarization=True,
                                       num_speakers=2))
                status = client.wait_until_complete(job_id, poll_interval=5.0,
                                                    timeout=600.0)
                result = client.fetch_result(job_id)
            segments = [s.model_dump() for s in result.segments]
            payload = {"job_id": job_id, "segments": segments}
            cache.write_text(json.dumps(payload, ensure_ascii=False))
            log.add("diarization", f"batch job {job_id}: {len(segments)} segments",
                    "one batch submission for the whole video (saaras:v3, "
                    "with_diarization, num_speakers=2)")
            break
        except (SarvamError, Exception) as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            print(f"diarization attempt {attempt} failed: {err}", file=sys.stderr)
    if payload is None:
        log.add("diarization", "mouth-only fallback",
                f"Sarvam failed after 2 attempts ({err}); fusion uses mouth motion only")
        timings["diarization"] = round(time.monotonic() - t0, 2)
        return None
    timings["diarization"] = round(time.monotonic() - t0, 2)
    return payload


def _nearest_face_centroid(analysis: dict[str, Any], t: float,
                           tol: float = 0.3) -> Optional[tuple[float, float]]:
    """Normalized centroid of the largest face box on the nearest analysis frame."""
    best, best_area, best_dt = None, -1.0, tol
    for tr in analysis["tracks"]:
        for f in tr["frames"]:
            d = abs(f["t"] - t)
            if d <= best_dt and f["box"][2] * f["box"][3] > best_area:
                best, best_area, best_dt = f["box"], f["box"][2] * f["box"][3], d
    if best is None:
        return None
    x, y, w, h = best
    return (x + w / 2, y + h / 2)


def make_still(video: Path, outdir: Path, p: dict[str, Any],
               platforms: dict[str, Any], analysis: dict[str, Any],
               log: DecisionLog, timings: dict[str, float]) -> dict[str, Any]:
    t0 = time.monotonic()
    sp = platforms["youtube_thumbnail"]
    duration = p["duration"]
    times = [duration * (i + 0.5) / 8 for i in range(8)]

    cap = cv2.VideoCapture(str(video))
    frames: list[np.ndarray] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        frames.append(frame if ok and frame is not None else None)
    cap.release()
    pairs = [(t, f) for t, f in zip(times, frames) if f is not None]
    images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for _, f in pairs]

    pick, provider = pick_best(images, (
        "Pick the best YouTube thumbnail still: expressive faces, clear subject, "
        "no motion blur, strong composition."))
    t_best, frame_best = pairs[pick["choice"]]
    log.add("still_pick", f"frame at t={t_best:.1f}s (candidate {pick['choice']} of 8)",
            f"{pick['reason']} (provider={provider})")

    centroid = _nearest_face_centroid(analysis, t_best)
    sw, sh = p["width"], p["height"]
    cw, ch = sp["width"], sp["height"]  # 1280x720 window on the full-res frame
    if centroid:
        cx, cy = centroid[0] * sw, centroid[1] * sh
    else:
        cx, cy = sw / 2, sh / 2
    x = int(max(0, min(sw - cw, cx - cw / 2)))
    y = int(max(0, min(sh - ch, cy - ch / 2)))
    crop = Image.fromarray(cv2.cvtColor(frame_best, cv2.COLOR_BGR2RGB)).crop(
        (x, y, x + cw, y + ch))
    crop.save(outdir / "youtube_thumbnail.jpg", "JPEG", quality=90)
    out_path = outdir / "youtube_thumbnail.jpg"
    prev = make_image_preview(out_path, outdir / "youtube_thumbnail.preview.jpg")
    timings["still"] = round(time.monotonic() - t0, 2)
    return {
        "platform": "youtube_thumbnail", "ratio": sp["ratio"], "kind": "still",
        "path": out_path, "preview_path": prev,
        "size": out_path.stat().st_size, "speaker_pct": None,
    }


def run_video_master(video: Path, outdir: Path, platforms: dict[str, Any],
                     log: DecisionLog, timings: dict[str, float]) -> list[dict[str, Any]]:
    t0 = time.monotonic()
    p = probe_mod.probe(video)
    timings["probe"] = round(time.monotonic() - t0, 2)

    t0 = time.monotonic()
    shots = detect_shots(video)
    faces = analyze_faces(video)
    analysis = {"video": str(video), "probe": p, "shots": shots,
                "tracks": faces["tracks"]}
    timings["analysis"] = round(time.monotonic() - t0, 2)
    if faces.get("brightened_frames"):
        log.add("dark_scene_brighten",
                f"{faces['brightened_frames']} analysis frames CLAHE-brightened",
                "mean luma below threshold; analysis stream only, outputs untouched")
    log.add("shots", f"{len(shots)} shot cuts detected",
            "PySceneDetect content detector on downscaled stream")

    t0 = time.monotonic()
    diar = diarize_cached(video, log, timings)
    fused = assign_speakers(analysis, diar)
    timings["speaker_fusion"] = round(time.monotonic() - t0, 2)
    log.add("speaker_map", f"method={fused['method']}, speakers->tracks={fused['speaker_track']}",
            "Sarvam diarization matched to face tracks via mouth-energy response"
            if fused["method"] == "sarvam+mouth"
            else "mouth-motion argmax fallback (no usable diarization)")

    outputs: list[dict[str, Any]] = []
    reel_meta: dict[str, Any] = {"path": None, "crop_w_n": None}

    for platform in CROP_PLATFORMS:
        sp = platforms[platform]
        t0 = time.monotonic()
        path = camera_path(analysis, fused, sp["ratio"], p["width"], p["height"],
                           out_fps=25)
        cw, ch = crop_size(p["width"], p["height"], sp["ratio"])
        spans = _fallback_spans(fused["timeline"])
        n_fb = sum(1 for f in fused["timeline"] if f["active_track"] is None)
        log.add(f"camera:{platform}",
                f"crop {cw}x{ch}, OneEuro(min_cutoff=0.8, beta=1.5, d_cutoff=1.0), "
                f"dead_zone=0.02, vel_clamp=0.25*crop_w/s; fallback frames={n_fb}",
                "smoothed active-speaker follow; fallback (hold/wide) engaged at "
                f"t={[round(s[0], 1) for s in spans[:8]]}..."
                if spans else "active speaker present for the whole timeline")
        out_path = outdir / f"{platform}.mp4"
        render_vertical(video, path, sp["ratio"], out_path,
                        src_w=p["width"], src_h=p["height"],
                        out_w=sp["width"], out_h=sp["height"])
        timings[f"render:{platform}"] = round(time.monotonic() - t0, 2)
        kind = "reel" if sp["ratio"] == "9:16" else "crop"
        prev = make_video_preview(out_path, outdir / f"{platform}.preview.mp4")
        outputs.append({
            "platform": platform, "ratio": sp["ratio"], "kind": kind,
            "path": out_path, "preview_path": prev,
            "size": out_path.stat().st_size, "speaker_pct": None,
        })
        if platform == "instagram_reel":
            reel_meta = {"path": out_path, "crop_w_n": cw / p["width"], "cam_path": path}

    # 16:9: master is already 16:9 -> stream copy, no crop
    t0 = time.monotonic()
    sp = platforms["youtube_landscape"]
    out169 = outdir / "youtube_landscape.mp4"
    _ffmpeg(["-i", str(video), "-c", "copy", str(out169)])
    timings["copy:16:9"] = round(time.monotonic() - t0, 2)
    prev = make_video_preview(out169, outdir / "youtube_landscape.preview.mp4")
    outputs.append({
        "platform": "youtube_landscape", "ratio": sp["ratio"], "kind": "copy",
        "path": out169, "preview_path": prev,
        "size": out169.stat().st_size, "speaker_pct": None,
    })

    outputs.append(make_still(video, outdir, p, platforms, analysis, log, timings))

    # speaker-on-screen % for the 9:16 reel
    if reel_meta.get("path") is not None:
        score = validate_speaker.speaker_on_screen(
            analysis, diar, reel_meta["cam_path"],
            fused["timeline"], crop_w=reel_meta["crop_w_n"])
        pct = (score.get("overall_confident") if score.get("overall_confident") is not None
               else score["overall"])
        for o in outputs:
            if o["platform"] == "instagram_reel":
                o["speaker_pct"] = pct
                o["speaker_detail"] = score
        log.add("speaker_on_screen", f"{round(pct * 100, 1)}% of frames",
                "active speaker's face inside the 9:16 crop window "
                "(confident frames only)", confidence=pct)
    return outputs


# ---------------------------------------------------------------- publish ----

def publish(input_path: Path, kind: str, title: str, outputs: list[dict[str, Any]],
            rows: list[dict[str, Any]], log: DecisionLog) -> int:
    load_config()
    if kind == "video":
        m = probe_mod.probe(input_path)
        w, h, dur, fps = m["width"], m["height"], m["duration"], m["fps"]
    else:
        with Image.open(input_path) as im:
            w, h = im.size
        dur = fps = None
    master_id = db.upsert_master(title, kind, "", input_path.stat().st_size,
                                 w, h, dur, fps)
    mkey = f"masters/{master_id}/{input_path.name}"
    storage.upload_file(input_path, mkey)
    db.set_master_s3_key(master_id, mkey)

    job_id = db.create_job(master_id)
    try:
        db.finish_job(job_id, "running", "uploading outputs", 50)
        for o in outputs:
            okey = f"outputs/{job_id}/{o['path'].name}"
            storage.upload_file(o["path"], okey)
            pkey = None
            if o.get("preview_path"):
                pkey = f"outputs/{job_id}/{o['preview_path'].name}"
                storage.upload_file(o["preview_path"], pkey)
            oid = db.insert_output(job_id, master_id, o["platform"], o["ratio"],
                                   o["kind"], okey, pkey, o.get("speaker_pct"))
            db.insert_validations(oid, [r for r in rows if r["output"] == o["path"].name])
        db.insert_decisions(job_id, None, log.to_list())
        db.finish_job(job_id, "done", "done", 100)
    except Exception as exc:
        db.finish_job(job_id, "error", "error", 0, error=str(exc)[:500])
        raise
    return job_id


# -------------------------------------------------------------------- main ----

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="reframe")
    ap.add_argument("input_path", type=Path)
    ap.add_argument("--title", required=True)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args(argv)

    load_config()
    spec = json.loads(SPEC_PATH.read_text())
    platforms = spec["platforms"]
    inp = args.input_path.resolve()
    if not inp.exists():
        raise SystemExit(f"input not found: {inp}")

    is_video = inp.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm")
    kind = "video" if is_video else "image"
    outdir = (args.outdir or P4_ROOT / "work" / "runs" / f"{kind}_master").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    log = DecisionLog()
    timings: dict[str, float] = {}
    print(f"[reframe] master={inp.name} kind={kind} -> {outdir}")

    if kind == "image":
        outputs = run_image_master(inp, outdir, platforms, log, timings)
    else:
        outputs = run_video_master(inp, outdir, platforms, log, timings)

    t0 = time.monotonic()
    rows = validate.validate_all(spec, [
        {"platform": o["platform"], "path": o["path"], "name": Path(o["path"]).name}
        for o in outputs
    ])
    timings["validate"] = round(time.monotonic() - t0, 2)
    timings["total"] = round(time.monotonic() - t_start, 2)
    for o in outputs:
        o["validations"] = [r for r in rows if r["output"] == Path(o["path"]).name]

    manifest = {
        "master": {"path": str(inp), "title": args.title, "kind": kind},
        "outputs": [
            {
                "platform": o["platform"], "ratio": o["ratio"], "kind": o["kind"],
                "path": str(o["path"]), "preview_path": str(o["preview_path"]),
                "size": o["size"], "speaker_pct": o.get("speaker_pct"),
                "validations": o["validations"],
            } for o in outputs
        ],
        "decisions": log.to_list(),
        "timings": timings,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))

    npass = sum(1 for r in rows if r["passed"])
    print(f"[reframe] outputs={len(outputs)} validations {npass}/{len(rows)} passed "
          f"decisions={len(log)} total={timings['total']}s")
    for o in outputs:
        ok = all(r["passed"] for r in o["validations"])
        print(f"  {o['platform']:<20} {o['ratio']:<5} {'PASS' if ok else 'FAIL'}"
              + (f" speaker_pct={o['speaker_pct']}" if o.get("speaker_pct") is not None else ""))
    if args.publish:
        job_id = publish(inp, kind, args.title, outputs, rows, log)
        print(job_id)  # last line: the job id
    return 0
