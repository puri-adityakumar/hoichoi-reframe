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
| 5 | **Demo polish** | 22:45 → 23:30 | Given assets precomputed in library; side-by-side "centre crop vs ours"; README; Neon on Launch plan |
| 6 | **Video + submit** | 23:30 → 00:00 | <5 min video recorded; repo made public; form submitted |

### Cut list if behind (in order)
1. Sarvam diarization → mouth-motion only.
2. VLM critic on image crops → deterministic scorer only (keep VLM for still picker).
3. Blaxel deploy → run worker on Vercel-triggered Blaxel sandbox, or precompute and show library only (last resort; must still accept uploads).
4. Library filters → plain list.

Never cut: speaker-following 9:16, spec validation, traceability to master, live demo link.

## 3. How the work gets done: orchestrator + cheap subagents\n\nCredits are limited, so the expensive model does the least possible.\n\n**Roles**\n- **Orchestrator (this session, Opus):** plans, splits work into small well-specified tasks, fires subagents, reviews every diff, runs validation, owns all architecture decisions, demo story and anything visual/risky. Never writes boilerplate itself.\n- **`p4-builder` droid (GLM-5.3-Flash / DeepSeek via BYOK):** implements one bounded task at a time, given exact files, interfaces and acceptance checks. Runs its own validation and reports back.\n- **`p4-checker` droid (same cheap model):** read-only review of each builder output against handbook rules before the orchestrator even looks.\n- **`explorer` (built-in, lightest):** quick lookups in the repo so the orchestrator's context stays lean.\n\n**Loop per task:** orchestrator writes a precise brief → `p4-builder` implements + self-validates → `p4-checker` reviews → orchestrator re-runs the key check and integrates. Two or three independent tasks can run in parallel in the background (e.g. image path while video path is built).\n\n**Setup needed from Aditya (one time, in `/settings` → Subagents):**\n1. Add GLM-5.3-Flash and DeepSeek as BYOK custom models (OpenRouter key is already in `.env.local`; Factory BYOK needs it in Settings → Models).\n2. Map complexity routing: **Light → GLM-5.3-Flash, Medium → GLM-5.3-Flash, Heavy → inherit (stays on Opus).**\n3. Subagent autonomy: **Medium** (builders need to edit files and run tests).\n\nThe droid files live in `.factory/droids/` and are committed to the repo.\n\n## 4. Checkpoints
- **17:30**: spike works? If not, switch to mouth-only + stacked fallback immediately.
- **20:45**: worker running in cloud? If not, freeze features and fix deploy.
- **22:45**: feature freeze. Only bug fixes and demo after this.
