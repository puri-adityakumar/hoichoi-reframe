# P4 Logs

Plain-language record of decisions and research, newest at the bottom.

## 1. Choosing the problem
- Read all four problems from the handbook. Ranked easiest to hardest: P3, P1, P4, P2.
- P3 had no assets and a clear web-app shape. P4 had the most visible "wow" (the frame follows the speaker) and a hard, testable bar.
- Decision: the user builds **P3** in a separate session. This workspace focuses **only on P4**.

## 2. What "toughest test" means
- A two-person conversation reframed to vertical 9:16. The frame must stay on whoever is talking.
- Centre crop fails (both people half-cut). Largest-face crop fails (locks onto the closer person even when silent).
- So we need to know **who is speaking and when**, then move the crop smoothly.

## 3. Research: how others solve it
- **autoclip** (MIT): shot cuts, MediaPipe faces, track linking, mouth movement matched with diarization, One Euro smoothing. Best template.
- **ClipsAI resize**: pyannote diarization + face detection per segment. Proves the audio + face approach.
- **Light-ASD**: accurate active-speaker model (94% mAP) but needs PyTorch. Kept as optional.
- **vertix**: when unsure, show both faces stacked instead of guessing. We adopted this.
- **auto-vertical-reframe**: YOLO + ByteTrack. Too heavy for CPU-only hosting.
- **smartcrop.py**: score many candidate crops; used for the image path.

## 4. Pipeline decisions
- Analyse a small proxy (480p, 5 fps) instead of every full-res frame. Big speed win, no quality loss for tracking.
- Detect speakers with audio (Sarvam diarization) and confirm with video (mouth motion). Either alone is weaker.
- Never pan across a shot cut; jump instead.
- Smooth the camera (One Euro filter, dead zone, max speed) so it feels like a camera operator, not jitter.
- Low confidence → stacked 2-face layout or wide safe crop. Report it honestly.
- Render each ratio in one FFmpeg pass.
- Validator checks the spec file and also reports "speaker on screen %".
- No PyTorch. Keeps the worker small and CPU-friendly.

## 5. Services checked
- **Sarvam** key works (HTTP 200; language ID returned `bn-IN`). ₹100 free credits.
  - Diarization is **only in the Batch API** (async, up to 2 h files, up to 20 speakers). Sync REST is 30 s max and has no diarization.
  - Cost: STT with diarization about ₹45/hour of audio. Credits are enough for testing.
- **OpenRouter GLM**: `z-ai/glm-5.3-flash` accepts text, image and video. $0.04/M input, $0.50/M output. Good as the VLM critic. `glm-5.3` and `glm-5.3-prime` are text-only.
- **Blaxel**: sandboxes are CPU-only; 8 GB → 4 cores, 16 GB → 6 cores. Batch Jobs can be triggered by HTTP (`POST /jobs/{name}/executions`). Fits the worker.
- **Supabase storage**: chunked (TUS) upload does **not** get around the 50 MB per-file limit on the free plan. The limit applies to the stored file. TUS also needs 6 MB chunks, not 25 MB.

## 6. Local machine
- Apple M4, 16 GB RAM, 34 GB free disk.
- Have: ffmpeg/ffprobe 8.1.1, Python 3.11, uv, node 26, pnpm, bun, Blaxel CLI 0.1.105 (update available), Vercel CLI, gh, git, docker.
- Missing: Supabase CLI.
- MediaPipe needs Python ≤3.12, so use a 3.11 venv.

## 7. Setup done
- Created `P3/` and `P4/` with `.gitignore`, `.env.example`, `.env.local` (Sarvam key only).
- Downloaded given assets into `P4/assets/given/`.
- Wrote `HANDOVER.md` for both, and `BLOCKERS.md` for P4.
- Building has **not** started.

## 8. Given assets inspected
- Image: 4000×4000 PNG, two women and a blood pool in a temple. Video: 190 s, 1080p, 25 fps, 282 MB, many shots including crowds, over-the-shoulder talk and very dark scenes.
- Both carry a burned-in "HOICHOI HACKATHON COPY" watermark across the centre. Decision: mask it out of the importance map, or it will drag every crop to the middle.
- Video is 282 MB, so the Supabase free 50 MB cap is a real blocker, not theoretical.
- Dark scenes → brighten the analysis proxy only (not the output).

## 9. Storage decision: Neon
- Compared three options for storing a 282 MB master:
  - Supabase Free: 50 MB per file. Fails.
  - Appwrite Free: 50 MB per file, 2 GB total. Fails (Pro is $25/mo for 5 GB files).
  - **Neon**: Postgres plus S3-compatible Object Storage in one project. 5 GiB per object, 5 GB storage on Free, presigned URLs, multipart upload, Singapore region. Chosen.
- Catch: Neon Free egress is 5 GB/month shared across everything. Plan: small previews in the UI and switch to the Launch plan (pay only for use) before submission.
- Neon has no realtime push, so the UI will poll job status. Acceptable.

## 10. Output scope: all of them
- Aditya chose to produce everything the handbook lists, not just 3 variants: 16:9, 1:1, 9:16, 4:5, a video still, and a speaker-tracked vertical reel, all validated against `spec.json`. Next step after blockers: a 30–60 min spike on one 2-person clip.

## 11. Keys validated
- OpenRouter: `z-ai/glm-5.3-flash` saw the image correctly (two women, centre-left and centre-right). Cost about $0.0002 per image call. Account has no purchased credits yet.
- GMI: same model as `zai-org/GLM-5.3-Flash`, also correct. Kept as fallback.
- Blaxel: `workspace-x` ready in `ap-southeast-1` (Singapore), same region we want for Neon, so worker ↔ storage traffic stays local.
- Neon MCP added to Droid; waiting for sign-in.

## 12. Plan and repo
- `P4/` is its own git repo, since it becomes the public submission. Secrets, assets and work files are git-ignored.
- Phases and timings are in `PLAN.md`. Key rule: never cut speaker-following, spec validation, traceability, or the live link.

## 13. Neon set up
- Project `p4-reframe` in AWS Singapore, Postgres 17, compute capped at 0.25–1 CU with auto-suspend (Neon balance is only $5, so everything stays lean).
- Private bucket `p4-media`; S3 credential `p4-app` with read+write scope. Endpoint is branch-scoped and needs path-style URLs.
- Database tables created: masters, jobs, outputs, validations, decisions.
- All values saved to `.env.local` (git-ignored).

## 14. Orchestrator + subagent approach
- Aditya asked for a multi-agent setup to save credits: Opus orchestrates and reviews; cheap fast models (GLM-5.3-Flash / DeepSeek via BYOK) do the implementation.
- Created two project droids in `.factory/droids/`: `p4-builder` (implements one bounded task + self-validates) and `p4-checker` (read-only review against handbook rules).
- Aditya maps Light/Medium complexity to the cheap models in Settings → Subagents; the orchestrator then delegates with `complexity: light/medium`.

## 15. Repo name
- Chosen: **hoichoi-reframe** (explicit hackathon tie-in). GitHub repo stays private/local until submission.

## 16. Real budgets, revised model plan
- Actual budgets: Neon **free tier** (no paid upgrade — the earlier Launch-plan idea is dropped), OpenRouter $5, GMI $5, Blaxel 170 credits, Sarvam about Rs 100.
- Opus credits are exhausted, so the orchestrator is now **Kimi K3** (this session), with strict token discipline: short exact briefs, subagents do the file reading and writing.
- Implementation, review and testing all run on **DeepSeek v4.1-Flash** and **GLM-5.3-Flash** (BYOK) through the `p4-builder` and `p4-checker` droids.
- PLAN.md section 3 now has the full budget table with per-service discipline rules (small previews, capped VLM retries, 8 GB Blaxel worker, local Docker test before deploy).

## 17. Model switch resolves the last setup item
- Core model switched to **GLM-5.3-Flash** for the orchestrator and all subagents (subagents inherit). The BYOK/complexity-routing setup is no longer needed.
- Verified: Vercel CLI already logged in (`wonderboyxtreme-4811`), Blaxel CLI already logged in. No account actions left from the user.
- One watch item: OpenRouter's balance endpoint reports $0 even though paid-model calls succeed. GMI serves the same model as fallback, so a sudden OpenRouter rejection cannot stop the pipeline.

## 18. Phases 0-2 done (autonomous run)
- Phase 0: venv (mediapipe 0.10.21 — 1.0.1 aborts on macOS), face model, spec.json, 3 test clips cut; clip_c picked (two dominant faces, mouth std 0.161/0.119).
- Phase 1: Sarvam batch diarization kept (30 s in ~5 s, 2 speakers, 6 turns; gotcha: Azure upload needs x-ms-blob-type header). Speaker fusion + One Euro camera path + sendcmd single-pass render: clip_c 95.4% and clip_a 95.8% speaker-on-screen (target 85%), fallbacks engaged correctly on shot/reverse-shot.
- Phase 2: `python -m reframe` end-to-end. Video: 5 outputs, 92.3% speaker-on-screen, 30/33 validations pass (3 honest fails: 190 s reel > Instagram 90 s/100 MB, 16:9 copy > 200 MB — flagged, not hidden). Image: 4 outputs, 20/20 pass. VLM used 5 calls total.
- Phase 2b (Neon publish): fixed three real bugs — `_one()` int()-casting uuid ids, `outputs_kind_check` missing 'copy' kind, DB connection dying across long S3 uploads (added rollback+reconnect+retry). Both masters published: image job 9ff0f8c1, video job e4b32ef3. ~1.6 GB of orphaned objects from failed attempts deleted (multipart objects need exact-key deletes, prefix delete was not enough). Live bucket ~1.2 GB of 5 GB.
- Verification is programmatic this session (the session model has no vision input): face-box math, validator rows, DB checks. Human stills review pending Aditya's return (work/render/stills/, work/outputs/image/candidates_sheet.jpg).
