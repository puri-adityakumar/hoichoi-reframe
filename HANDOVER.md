# P4 Handover — Creative Reformatting Engine

Self-contained brief for building P4. Full handbook: `../HANDBOOK.md`. Decisions and research history: `LOGS.md`. Open items: `BLOCKERS.md`.

## Hackathon essentials
- 26 Sep 2026, 11:30 am → submission closes **12:30 am IST**. Solo.
- Judged on **AI-native approach** + **working end-to-end solve**.
- Submit: live demo link + public GitHub repo + explainer video under 5 minutes. Demo must stay live through judging (~2 weeks).

## The problem in plain words
Upload one master image or video. Get back every platform-specific version (vertical, square, landscape, story, thumbnail) where the **important thing is still in frame** (the speaking face, the title, the logo). Every output is checked against a platform spec file, and anything that fails is flagged. Outputs land in a library tagged by title, platform and date.

## MVP scope (from handbook)
Upload one master (image or video) → auto-generate 3 platform variants with subject-aware cropping → validate against a spec file → output library with a flag on any variant that fails.

## Required outputs (all of them)
- Ratios: **16:9, 1:1, 9:16, 4:5** from both image and video masters, with subject-aware crops.
- A **still** pulled from the video.
- A **speaker-tracked 9:16 reel**.
- Every asset validated against a machine-readable `spec.json` before it enters the library.

## Toughest test
A **two-person dialogue scene** reformatted to 9:16. The frame must follow **whoever is speaking**, not the centre, and not just the largest face.

## Auto-disqualifiers
- Centre-crop presented as smart reframing.
- No validation against platform specs.
- Outputs cannot be traced back to the master.

## Given assets (`assets/given/`, git-ignored)
- `input_image.png` (13 MB)
- `input_video.mp4` (269 MB)

## Pipeline (chosen approach)

### Image path
```
master image
  → face detect (MediaPipe) + saliency (OpenCV spectral residual) + text/logo boxes (VLM)
  → importance map
  → for each target ratio: ~200 candidate crop boxes scored by "importance kept"
  → top 3 → VLM critic (z-ai/glm-5.3-flash) picks best and explains why
  → render (sharp / Pillow), extend with blur-pad only if subject cannot fit
  → validator vs spec.json → library
```

### Video path
```
master video
  → proxy (downscale 480p, 5 fps analysis) + shot cuts (PySceneDetect)
  → faces per frame (MediaPipe FaceLandmarker) → link into tracks (IoU)
  → mouth-open signal per track (lip landmark distance over time)
  → who is speaking:
       Sarvam Batch STT with diarization (speaker time ranges)
       + match each speaker to the face whose mouth moves during their turns
       (fallback: mouth motion only, no audio model)
  → camera path per shot: centre on active speaker, One Euro smoothing,
    dead zone, velocity clamp, never pan across a cut
  → low confidence (both mouths moving, no face) → stacked 2-face layout or wide safe crop
  → one-pass FFmpeg render per ratio (crop expression from path file)
  → best-still picker (VLM) for thumbnail
  → validator vs spec.json (ratio, resolution, duration, size, codec, fps)
       + "speaker on screen %" quality score
  → library (title / platform / date / master_id / decision log)
```

### AI-native story for judges
1. Audio (diarization) + vision (mouth motion) fused to decide who speaks.
2. VLM critic chooses among crop candidates and writes the reason.
3. VLM picks the best still for thumbnails.
4. Every decision is logged and shown per output ("Speaker B at 00:12, confidence 0.83").
5. Honest validator: shows the speaker-on-screen score, does not hide failures.

## Stack
| Need | Choice |
|---|---|
| Web app | Next.js on Vercel (upload, progress, library, side-by-side preview) |
| DB | Neon Postgres (jobs, assets, variants, validation results, AI decision log). UI polls job status |
| File storage | Neon Object Storage (S3-compatible, 5 GiB per object, presigned direct uploads, multipart for big files) |
| Worker | Python 3.11 container on Blaxel (Batch Job, HTTP-triggered), 8–16 GB RAM, CPU only |
| Vision libs | mediapipe, opencv-python-headless, scenedetect, numpy |
| Video | ffmpeg / ffprobe |
| Diarization | Sarvam Saaras v3 **Batch API** (`with_diarization`, `num_speakers=2`) |
| VLM | OpenRouter `z-ai/glm-5.3-flash` (image + video input) |
| Schemas | pydantic (worker) / zod (web) |

No PyTorch. Target about 1–2 minutes per 60 s clip on CPU.

## Open-source references (methods, not dependencies)
- `artbyjazi/autoclip` (MIT): scene detect + MediaPipe + IoU tracking + mouth-aspect-ratio vs diarization + One Euro filter. Closest to our plan.
- ClipsAI `resize`: pyannote diarization + scene detect + face detection → per-segment crops.
- Light-ASD (MIT, CVPR 2023): proper active-speaker model; optional upgrade only.
- `pzanella/vertix`: lesson that a wrong confident lock is worse than a stacked fallback.
- `KazKozDev/auto-vertical-reframe`: YOLO11 + ByteTrack (needs PyTorch, skipped).
- `smartcrop.py`: image crop scoring idea.

## Risks
- Sarvam Batch diarization is async; latency unknown → measure in spike; fallback is mouth-motion only.
- Neon stays on the **free tier**: 5 GB storage and 5 GB/month egress are the hard caps. Serve small previews, downloads on demand only, delete temp objects after each job.
- Cold starts on Blaxel; keep a precomputed sample job in the library so the demo is instant.
- Given video may not contain a clean 2-person dialogue → prepare our own test clips.
