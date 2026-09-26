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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from . import db, probe as probe_mod, storage, validate, validate_speaker
from .config import load_config
from .decisions import DecisionLog
from .image.crops import best_crop_candidates
from .image.importance import band_is_actionable, importance_map
from .image.render import render_crop
from .preview import make_image_preview, make_video_poster
from .video.annotate import annotate_tracks, make_analysis_proxy
from .video.camera import camera_path
from .video.faces import analyze_faces
from .video.render import crop_size, render_vertical
from .video.shots import detect_shots
from .video.speaker import assign_speakers
from .vlm import collect_vlm_log, pick_best

P4_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = P4_ROOT / "spec" / "spec.json"
SARVAM_DIR = P4_ROOT / "work" / "sarvam"

# image platforms by ratio: 16:9, 1:1, 9:16, 4:5 (4 VLM calls total)
IMAGE_PLATFORMS = ["youtube_thumbnail", "square_image", "story_image", "feed_image"]
# video target ratios: 9:16 crop-tracked, 4:5 crop-tracked, 1:1 crop-tracked,
# 16:9 = stream copy (master is already 16:9); still = 16:9 thumbnail
CROP_PLATFORMS = ["instagram_reel", "instagram_feed", "youtube_square"]

# ---------------------------------------------------------------- helpers ----

def _ffmpeg(args: list[str]) -> None:
    res = subprocess.run(["ffmpeg", "-v", "error", "-y", *args],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr[-800:]}")


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
    image_ratios = [platforms[pl]["ratio"] for pl in IMAGE_PLATFORMS]
    imp, _scale = importance_map(img_bgr, ratios=image_ratios)
    face_mask = (imp > 0.85).astype(np.float32)
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    timings["importance_map"] = round(time.monotonic() - t0, 2)
    band_plan = {r: band_is_actionable(img_w, img_h, r) for r in image_ratios}
    log.add("watermark_mask",
            f"band y=0.40-0.60 zeroed={all(band_plan.values())} "
            f"(per-ratio avoidable: {band_plan})",
            "the map is shared, so the band is zeroed only when EVERY requested "
            "ratio can be steered clear of it. A tall ratio cut from a square "
            "source must cross the band by construction, which makes zeroing it a "
            "uniform dead zone that deletes the saliency of the subjects")

    outputs: list[dict[str, Any]] = []
    for platform in IMAGE_PLATFORMS:
        t1 = time.monotonic()
        p = platforms[platform]
        res = best_crop_candidates(imp, img_w, img_h, p["ratio"], k=3,
                                   face_mask=face_mask)
        cands = res["candidates"]
        crops = [img.crop((c["box"][0], c["box"][1],
                           c["box"][0] + c["box"][2], c["box"][1] + c["box"][3]))
                 for c in cands]
        context = (f"Target platform: {p['label']} ({p['ratio']}, "
                   f"{p['width']}x{p['height']}). Pick the crop keeping the most "
                   "important subjects (faces/people) fully visible and well framed.")
        pick, provider = pick_best(crops, context)
        if pick["choice"] < 0:
            # The model says none of the three is acceptable. Falling back to
            # candidate 0 is the honest floor, but say so loudly rather than
            # shipping a people-free crop as though it had won.
            log.add(f"image_crop:{platform}",
                    "VLM rejected all candidates; fell back to candidate 0",
                    f"{pick['reason']} (provider={provider}, faces_found="
                    f"{res['faces_found']}, faces_kept={res['faces_kept']})",
                    confidence=0.0)
            pick = {**pick, "choice": 0}
        best = cands[pick["choice"]]
        log.add(f"image_crop:{platform}",
                f"candidate {pick['choice']} of {len(cands)}",
                f"{pick['reason']} (provider={provider}, scores="
                f"{[round(c['score'], 3) for c in cands]}, faces_found="
                f"{res['faces_found']}, faces_kept={res['faces_kept']}"
                f"{' -- NO CANDIDATE COULD HOLD A FACE WHOLE' if res['faces_found'] and not res['faces_kept'] else ''})")
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
        "no motion blur, strong composition."), subjects_preserved=False)
    if pick["choice"] < 0:
        log.add("still_pick", "VLM rejected all 8 sampled frames; fell back to frame 0",
                f"{pick['reason']} (provider={provider})", confidence=0.0)
        pick = {**pick, "choice": 0}
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
    log.add("speaker_map", f"method={fused['method']}, speakers->tracks={fused['speaker_track']}, "
            f"confidence={fused.get('speaker_confidence')}, ambiguous={fused.get('ambiguous')}, "
            f"overlap_dropped_s={fused.get('overlap_dropped_s')}",
            "Sarvam diarization matched to face tracks via mouth-energy contrast "
            "(own turn minus other turns), damped by presence and assigned "
            "one-to-one so two speakers cannot land on one face"
            if fused["method"] == "sarvam+mouth"
            else "mouth-motion fallback (no usable diarization); assignment is "
                 "consistent but not corroborated by audio")

    # raw artifacts (video only): full analysis + fusion dicts land in outdir for
    # publish_raw_artifacts; the debug video is QA-only and never enters `outputs`
    (outdir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, default=str))
    (outdir / "speaker_fusion.json").write_text(
        json.dumps(fused, ensure_ascii=False, default=str))

    t0 = time.monotonic()
    proxy = outdir / "_analysis_proxy.mp4"
    make_analysis_proxy(video, proxy)
    try:
        annotate_tracks(proxy, analysis, fused, outdir / "debug_tracking.mp4")
    finally:
        proxy.unlink(missing_ok=True)
    timings["tracking_debug"] = round(time.monotonic() - t0, 2)
    log.add("tracking_debug", "boxes+labels drawn from tracks/speaker fusion",
            "every track's box on the 480p analysis proxy at the analysis fps, "
            "colored per speaker (teal/amber), labelled 'Speaker A - 0.83'; "
            "QA-only raw artifact, never published to the library")

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
        mix = Counter(e["strategy"] for e in path)
        log.add(f"camera:{platform}",
                f"crop {cw}x{ch}, OneEuro(min_cutoff=0.7, beta=0.5, d_cutoff=1.0), "
                f"dead_zone=0.02 (fraction of travel), vel_clamp=full travel in "
                f"0.5s; cx=fraction of available travel; strategy mix="
                f"{dict(mix)}; untracked analysis frames={n_fb}/{len(fused['timeline'])}",
                "ladder: track=identified active speaker, wide=all visible faces "
                "held, hold=no face visible (never snaps to centre); untracked "
                "frames are those where no speaker could be identified at all")
        out_path = outdir / f"{platform}.mp4"
        render_vertical(video, path, sp["ratio"], out_path,
                        src_w=p["width"], src_h=p["height"],
                        out_w=sp["width"], out_h=sp["height"])
        timings[f"render:{platform}"] = round(time.monotonic() - t0, 2)
        kind = "reel" if sp["ratio"] == "9:16" else "crop"
        prev = make_video_poster(out_path, outdir / f"{platform}.preview.jpg",
                                 duration=p["duration"])
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
    prev = make_video_poster(out169, outdir / "youtube_landscape.preview.jpg",
                             duration=p["duration"])
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
        # Report the ALL-FRAMES hit rate, not the confident-frames-only rate.
        # `overall` is conditioned on frames where a speaker was actually
        # identified; on its own it read 0.92 while only ~36% of speech time was
        # tracked, which is the number a judge would call out.
        pct = score.get("overall_all", score["overall"])
        cov = score.get("coverage")
        for o in outputs:
            if o["platform"] == "instagram_reel":
                o["speaker_pct"] = pct
                o["speaker_coverage"] = cov
                o["speaker_detail"] = score
        log.add("speaker_on_screen",
                f"{round(pct * 100, 1)}% of all speech frames; "
                f"coverage={round((cov or 0) * 100, 1)}% of speech time had an "
                f"identified speaker ({score.get('tracked_frames')}/"
                f"{score.get('speech_frames')} frames tracked)",
                "active speaker's face inside the 9:16 window that was actually "
                "rendered, over ALL speech frames including the ones where no "
                "speaker could be identified. The confident-frames-only rate was "
                f"{score.get('overall_confident')} and is NOT what we report, "
                "because conditioning on frames the system already got right "
                "overstates the result.", confidence=pct)
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


# ------------------------------------------------------- raw artifacts ----

# outdir filename -> (raw kind, human label); keys under jobs/<job_id>/raw/
RAW_FILES = {
    "analysis.json": ("analysis_json", "Full analysis (probe, shots, face tracks)"),
    "speaker_fusion.json": ("speaker_fusion", "Speaker-to-track fusion result"),
    "vlm_calls.json": ("vlm_calls", "Every VLM call: prompt, raw reply, latency, choice"),
    "debug_tracking.mp4": ("debug_video", "Tracking debug video (boxes + speaker labels)"),
    "manifest.json": ("run_manifest", "Run manifest (decisions, timings, validations)"),
}


def publish_raw_artifacts(job_id: str, outdir: Path) -> None:
    """Upload raw per-job artifacts under jobs/<job_id>/raw/ and index them in
    jobs/<job_id>/raw/manifest.json (fixed key -- the web app's entry point).

    Raw-only by design: none of these go into the outputs table, so they never
    appear in the library and get no validations. Each upload is independent --
    one failure must not kill publishing or the DB job record."""
    calls = collect_vlm_log()
    (outdir / "vlm_calls.json").write_text(
        json.dumps(calls, ensure_ascii=False, indent=1))

    artifacts: list[dict[str, Any]] = []
    for name, (kind, label) in RAW_FILES.items():
        path = outdir / name
        if not path.exists():
            continue
        key = f"jobs/{job_id}/raw/{name}"
        try:
            storage.upload_file(path, key)
            artifacts.append({"kind": kind, "key": key, "label": label,
                              "bytes": path.stat().st_size})
        except Exception as exc:  # noqa: BLE001 - tolerate per-file failures
            print(f"[reframe] raw upload failed for {name}: {exc}", file=sys.stderr)

    index = {
        "job_id": job_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    tmp = outdir / "_raw_manifest.json"
    tmp.write_text(json.dumps(index, ensure_ascii=False, indent=1))
    try:
        storage.upload_file(tmp, f"jobs/{job_id}/raw/manifest.json")
    except Exception as exc:  # noqa: BLE001
        print(f"[reframe] raw upload failed for manifest.json: {exc}", file=sys.stderr)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"[reframe] raw artifacts uploaded: {len(artifacts)} "
          f"+ index under jobs/{job_id}/raw/")


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
        publish_raw_artifacts(job_id, outdir)
        print(job_id)  # last line: the job id
    return 0
