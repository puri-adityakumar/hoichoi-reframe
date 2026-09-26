# P4 Blockers Checklist

Resolve these before building. Tick when done.

## A. Keys and accounts (need from Aditya)
- [x] `SARVAM_API_KEY`: works (tested). ₹100 credits.
- [x] `OPENROUTER_API_KEY`: works; `z-ai/glm-5.3-flash` vision call correctly found the two women (~$0.0002/call). **Account shows 0 purchased credits**: add ~$5 or calls may start failing.
- [x] `GMI_API_KEY`: works; model id `zai-org/GLM-5.3-Flash`, vision call correct. Used as VLM fallback.
- [x] `BL_WORKSPACE` + `BL_API_KEY`: workspace `workspace-x` is ready, region `ap-southeast-1`, jobs API returns 200. Still to do: update CLI (0.1.105 → 0.1.117).
- [ ] Neon MCP added to Droid (`https://mcp.neon.tech/mcp`); **waiting for Aditya to authenticate via `/mcp`**. Then create project + bucket and fill:
- [ ] Neon project in **AWS Singapore** (`aws-ap-southeast-1`): `DATABASE_URL`, plus Object Storage `S3_ENDPOINT`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, bucket name (private).
- [ ] Vercel account linked (CLI already installed).
- [ ] GitHub repo name for the public submission. Local git repo created in `P4/` (no remote yet).
- [ ] (Optional) `HF_TOKEN`: only if we fall back to pyannote locally. Probably not needed.

## B. Decisions (need a yes/no)
- [x] **Where big files live → Neon** (Postgres + S3-compatible Object Storage in one project).
  - Why: 5 GiB max per object, 5 GB object storage on Free, presigned URLs for direct browser upload (skips Vercel's 4.5 MB body limit), multipart upload for >100 MB, Singapore region, upload triggers.
  - Rejected: Supabase free (50 MB per file), Appwrite free (50 MB per file; 5 GB only on Pro $25/mo).
  - Watch: Free plan egress is **5 GB/month shared** across DB + storage. One master (282 MB) plus outputs viewed by several judges can exceed it. Plan: serve small H.264 previews in the UI, full files on download only, and switch to **Launch plan** (pay-as-you-go, no minimum, 500 GB egress) before submission.
  - No built-in realtime → UI polls job status every 2 s (fine for this app).
- [x] **Outputs → everything the handbook lists**: 16:9, 1:1, 9:16, 4:5, a still pulled from the video, and a speaker-tracked vertical reel. Every asset validated against `spec.json`.

## C. Local environment
- [ ] Python 3.11 venv with `mediapipe opencv-python-headless scenedetect numpy pydantic httpx`.
- [ ] Download MediaPipe `face_landmarker.task` model file.
- [ ] Install Neon CLI (`npm i -g neonctl`) and log in.
- [x] ffmpeg/ffprobe 8.1.1, uv, node, pnpm, docker, vercel, gh present.

## D. Test material
- [x] Given assets downloaded and probed (see below).
- [ ] Cut 2–3 short two-person dialogue clips (10–30 s) for the toughest test. Source: given video (over-the-shoulder shots around 54–78 s) plus the P1/P2 sample films if needed.
- [ ] Write `spec.json` (platform rules the validator checks).

## E. Spike to prove the risky parts (after A–D)
- [ ] Sarvam Batch diarization on a 30 s clip: measure latency and accuracy.
- [ ] MediaPipe mouth signal at 5 fps on the same clip.
- [ ] Speaker ↔ face matching + One Euro camera path + FFmpeg 9:16 render.
- [ ] Decide: keep Sarvam in the loop, or use mouth motion only.

## Given assets (probed)
| File | Details |
|---|---|
| `input_image.png` | 4000×4000 PNG, 13.7 MB. Two women (older in saree, younger crouching) around a blood pool, temple interior, strong light from the right. |
| `input_video.mp4` | 1920×1080, H.264, 25 fps, AAC 48 kHz stereo, 190 s, 282 MB, ~11.9 Mbps. Many shots: courtroom corridors, crowds of 4–6 people, over-the-shoulder two-person talk, several very dark low-light scenes. |

### Things these assets tell us
- **Watermark "HOICHOI HACKATHON COPY"** is burned across the middle of both. Saliency and text detection will flag it as important. We must mask it out of the importance map so it doesn't pull the crop.
- **Dark scenes** will lower face detection confidence. Need a brightness boost on the analysis proxy only, and the stacked/wide fallback.
- **Crowd shots** have many faces. Speaker choice matters more than "largest face".
- **Over-the-shoulder shots** have one face visible and one back-of-head. Speaker may be the person facing away; fallback is to keep the visible face plus shoulder.
- **Image**: two subjects plus the blood pool (story object). For 9:16 we can't fit both people and the pool comfortably; the VLM critic should decide what tells the story.
