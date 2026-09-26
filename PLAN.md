# P4 Plan — Structure and Phases

Time budget: today, until submission closes at **00:30 IST**. Target submit **00:00** (30 min buffer).

## 1. Project structure

```
P4/                              one git repo → becomes the public submission repo
├── web/                         Next.js app on Vercel
│   ├── app/
│   │   ├── page.tsx             upload a master (image or video) + sample masters
│   │   ├── jobs/[id]/page.tsx   live progress (polls every 2 s)
│   │   ├── library/page.tsx     all outputs, filter by title / platform / date
│   │   ├── assets/[id]/page.tsx one output: preview, spec checks, AI decision log, link to master
│   │   └── api/
│   │       ├── uploads/         presigned multipart upload to Neon Object Storage
│   │       ├── jobs/            create job → trigger Blaxel job; read status
│   │       └── library/         list/filter outputs
│   └── lib/ (db.ts, s3.ts, blaxel.ts, zod schemas)
├── worker/                      Python 3.11, runs as a Blaxel Batch Job (CPU)
│   ├── reframe/
│   │   ├── main.py              entry: job_id → download master → run → upload → write DB
│   │   ├── probe.py             ffprobe/Pillow metadata
│   │   ├── image/               importance map, crop candidates, VLM critic, render
│   │   ├── video/               proxy, shots, faces+tracks, mouth signal, diarization,
│   │   │                        speaker match, camera path, FFmpeg render, still picker
│   │   ├── validate.py          checks every output against spec.json
│   │   ├── vlm.py               OpenRouter glm-5.3-flash, GMI fallback
│   │   ├── sarvam.py            Batch STT with diarization
│   │   ├── storage.py, db.py    Neon S3 + Postgres
│   │   └── decisions.py         structured AI decision log
│   ├── models/                  face_landmarker.task (downloaded, git-ignored)
│   ├── tests/                   small fixtures + golden checks
│   ├── Dockerfile, blaxel.toml, pyproject.toml
├── spec/spec.json               machine-readable platform spec (shared by web + worker)
├── db/schema.sql                tables below
├── scripts/                     cut test clips, run worker locally, seed sample jobs
└── HANDOVER.md, PLAN.md, LOGS.md, BLOCKERS.md, README.md
```

### Data model (Neon Postgres)
| Table | Holds |
|---|---|
| `masters` | uploaded file: title, kind (image/video), S3 key, metadata, uploaded_at |
| `jobs` | master_id, status, stage, progress %, error, timings |
| `outputs` | job_id, master_id, platform, ratio, kind (crop / reel / still), S3 key, preview key |
| `validations` | output_id, rule, expected, actual, pass/fail |
| `decisions` | job_id / output_id, stage, what the AI chose, why, confidence, timestamp |

Every output keeps `master_id`, so it always traces back (handbook disqualifier).

### Outputs per master
- **Image master:** 16:9, 1:1, 9:16, 4:5 subject-aware crops.
- **Video master:** 16:9, 1:1, 9:16, 4:5 subject-tracked versions; 9:16 reel follows the active speaker; one still (VLM-picked) cropped to each ratio as thumbnail/poster.
- All validated against `spec/spec.json` before entering the library. Failures stay visible with a red flag and the reason.

## 2. Phases

| # | Phase | Window (IST) | Done when |
|---|---|---|---|
| 0 | **Setup** | now → 16:30 | Neon project + private bucket + schema; Python venv with mediapipe/opencv/scenedetect; face model downloaded; `spec.json` written; 2–3 two-person test clips cut |
| 1 | **Spike: speaker tracking** | 16:30 → 17:30 | On one dialogue clip: faces + mouth signal + (Sarvam diarization) → 9:16 render that visibly follows the speaker. Decide: keep Sarvam or mouth-only |
| 2 | **Worker core (local CLI)** | 17:30 → 19:45 | `python -m reframe <file>` makes every output + validation report + decision log for both given assets |
| 3 | **Cloud worker** | 19:45 → 20:45 | Docker image deployed as Blaxel job; reads/writes Neon S3 + Postgres; triggered by HTTP |
| 4 | **Web app** | 20:45 → 22:45 | Upload (presigned multipart) → job progress → library → output detail with spec checks + decisions, deployed on Vercel |
| 5 | **Demo polish** | 22:45 → 23:30 | Given assets precomputed in library; side-by-side "centre crop vs ours"; README; verify Neon free-tier usage (storage + egress) with headroom |
| 6 | **Video + submit** | 23:30 → 00:00 | <5 min video recorded; repo made public; form submitted |

### Cut list if behind (in order)
1. Sarvam diarization → mouth-motion only.
2. VLM critic on image crops → deterministic scorer only (keep VLM for still picker).
3. Blaxel deploy → run worker on Vercel-triggered Blaxel sandbox, or precompute and show library only (last resort; must still accept uploads).
4. Library filters → plain list.

Never cut: speaker-following 9:16, spec validation, traceability to master, live demo link.

## 3. Budgets and model strategy

### Resource budgets (hard limits)
| Resource | Budget | Discipline |
|---|---|---|
| Neon | **Free tier only**: 5 GB object storage, 5 GB/month egress, 0.5 GB DB, 100 CU-hours | Small previews (about 200-500 KB) in the UI, full files as downloads only, delete temp/proxy objects after each job, compute already capped at 0.25-1 CU with auto-suspend. No paid upgrade. |
| OpenRouter | $5 | VLM critic on top-3 image crops only, VLM still picker on about 8 frames only, short JSON prompts, cap output tokens. About $0.0002 per image call, so the risk is loops, not single calls: max 1 retry per asset. |
| GMI | $5 | Fallback for the same VLM calls if OpenRouter fails or is slow. |
| Blaxel | 170 credits | One worker job definition, 8 GB (4-core) size to start, scale to 16 GB only if the spike shows we need it. Test the Docker image locally before every deploy. |
| Sarvam | about Rs 100 | Diarization spike on one 30 s clip only; if latency or cost looks bad, mouth-motion only. |

### Model roles (who thinks, who types)
- **Core model: GLM-5.3-Flash everywhere** (user switched the session model). Orchestrator and subagents all run on it; subagents inherit automatically, so no BYOK/routing setup is needed.
- **Orchestrator (this session):** plans, splits work into precise briefs, fires subagents, reviews diffs, owns architecture, demo story and all judgement calls. Token discipline: briefs are short and exact, file reading happens in subagents, the orchestrator re-reads only what it is about to edit.
  - `p4-builder` droid: implements one bounded task + self-validates.
  - `p4-checker` droid: read-only review against handbook rules.
  - built-in `explorer`: quick repo lookups so orchestrator context stays lean.
- Loop per task: orchestrator writes a precise brief, builder implements and self-validates, checker reviews, orchestrator re-runs the one key check and integrates. Independent tasks run in parallel in the background.

The droid files live in `.factory/droids/` and are committed to the repo.

## 4. Checkpoints
- **17:30**: spike works? If not, switch to mouth-only + stacked fallback immediately.
- **20:45**: worker running in cloud? If not, freeze features and fix deploy.
- **22:45**: feature freeze. Only bug fixes and demo after this.
