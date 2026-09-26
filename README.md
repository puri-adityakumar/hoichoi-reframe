# hoichoi-reframe

**Creative Reformatting Engine** — hoichoi Hackathon'26, Problem 4.

Upload one master image or video. Get back every platform-ready version —
16:9, 1:1, 9:16, 4:5, a still pulled from the video, and a vertical reel that
**follows whoever is speaking** — with every asset validated against a
machine-readable platform spec before it enters the library, and every AI
decision logged with its reason.

**Live demo:** https://hoichoi-reframe-kctsurqvj-project-by-aditya.vercel.app

## What makes it AI-native

1. **Audio + vision fusion for "who is talking."** Sarvam batch diarization
   (Bengali/Hindi-native) gives speaker time-ranges; MediaPipe face tracking
   gives mouth-motion signals per face. Neither alone decides — the fusion
   does, with a confidence margin.
2. **A camera operator, not a crop.** The 9:16 window follows the active
   speaker with a One Euro filter, dead zone and velocity clamp, never pans
   across a shot cut, and falls back to a two-face-safe framing when unsure.
3. **A VLM critic picks the crop.** For image masters, ~200 candidate crops
   per ratio are scored by an importance map (faces + saliency, watermark
   masked out), and GLM-5.3-Flash picks the winner and explains why.
4. **A VLM picks the thumbnail.** 8 sampled frames, one pick, one reason.
5. **An honest validator.** Every output is checked against `spec/spec.json`
   (ratio, resolution, duration, fps, codec, size). Failures stay visible
   with the reason — e.g. a 190 s reel is flagged honestly against
   Instagram's 90 s limit rather than hidden.
6. **Full traceability.** Every output row carries its master id, the spec
   checks and the AI decision log (stage, choice, reason, confidence).

## Architecture

```
Browser ──presigned upload──> Neon Object Storage (S3-compatible)
   │                              │
   └─ POST /api/jobs ──────────> Neon Postgres (masters, jobs, outputs,
         │                        validations, decisions)
         └─ trigger ──> Blaxel Batch Job `reframe-worker` (CPU, 4 GB)
                          │  download master → 5 fps proxy analysis
                          │  MediaPipe faces+mouth → tracks (IoU)
                          │  Sarvam diarization → speaker fusion
                          │  One Euro camera path → single-pass ffmpeg
                          │  VLM critic / still picker (OpenRouter, GMI fallback)
                          └─ upload outputs + previews → DB rows
UI polls every 2 s; library shows previews, spec checks and decisions.
```

Stack: Next.js 15 on Vercel · Neon Postgres + Object Storage · Blaxel Batch
Jobs · Python 3.11 worker (MediaPipe, scenedetect, ffmpeg, no PyTorch) ·
Sarvam Saaras v3 · GLM-5.3-Flash via OpenRouter (GMI fallback).

## Running locally

```bash
# worker (needs ffmpeg + a P4/.env.local with the keys)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python mediapipe numpy scenedetect \
    pydantic httpx pillow boto3 "psycopg[binary]"
.venv/bin/python -m reframe <image-or-video> --title "T" --outdir work/runs/t

# web
cd web && npm install && npm run dev
```

## Repo layout

`web/` Next.js app · `worker/` pipeline + Blaxel job · `spec/spec.json`
platform rules · `HANDOVER.md` full brief · `PLAN.md` structure/phases/budgets
· `LOGS.md` plain-language decision history · `BLOCKERS.md` setup checklist.
