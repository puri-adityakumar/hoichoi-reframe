# Submission video — hoichoi-reframe (Problem 4)

Handbook requirement: **under 5 minutes**, covering what you built, how it works, and how it
addresses the problem statement. Judged on **AI-native approach** and a **working end-to-end solve**.

Target cut: **4:30** (leave margin under the hard cap). Narration below is ~620 words ≈ 4:35 at a
conversational 135 wpm. Do one timed dry run before the real take.

---

## 0:00–0:25 — The problem (title card over the upload page)

**On screen:** the app's Create page, then a quick split: centre-cropped vertical vs your reel.

> Every streaming team knows this problem. You finish one master video — and then you need a
> landscape cut for YouTube, a square for feed, a vertical for Reels, and a thumbnail. Do it by
> hand, or worse, with a dumb centre crop, and you cut your speaker in half. This is
> hoichoi-reframe: one master in, every platform-ready version out — and the important thing
> always stays in frame.

## 0:25–1:00 — What it is (live app)

**On screen:** Create page. Type a title, pick the given image master, hit "Upload & run".
Show the job row appear, then cut to a **pre-run** job mid-progress showing the live stage
and progress bar (do not make judges watch a 282 MB upload).

> Here it is, live. I upload one master — title, file, upload and run. The file goes straight to
> object storage on a presigned URL, a job lands on a Blaxel batch worker, and the dashboard
> shows live stage and progress. In about a minute we get every output: sixteen-by-nine, square,
> vertical, four-by-five, a still for the thumbnail — and the star of the show: a speaker-tracked
> nine-by-sixteen reel.

## 1:00–2:15 — How it works (the AI-native core)

**On screen:** library grouped by job; open an image output; then the worker pipeline diagram
(from README) or quick B-roll of the decision log while you talk.

> Everything here is AI-native, not bolted on.
>
> For images: I build an importance map — faces, saliency, text and logos, with the watermark
> masked out so it doesn't drag the crop. Then roughly two hundred candidate crops per ratio are
> scored by how much importance they keep, and a vision-language model — GLM — acts as the
> critic: it picks the winner and writes down why.
>
> For video: I analyse a low-res proxy at five frames a second, track faces with MediaPipe, and
> read mouth motion per face. In parallel, Sarvam diarization — built for Bengali and Hindi —
> says who is speaking and when. Neither signal alone decides; the fusion does, with a confidence
> margin. The crop window then moves like a camera operator: One-Euro smoothing, a dead zone, a
> speed limit — and it never pans across a shot cut. If it isn't sure who's talking, it falls
> back to a two-face-safe framing instead of guessing. A second VLM call picks the thumbnail
> still.

## 2:15–3:15 — The toughest test: two-person dialogue

**On screen:** the speaker-tracked reel, full size. Then the asset page's "Speaker on screen" bar.

> The handbook's toughest test: a two-person dialogue, reframed to nine-by-sixteen. A centre crop
> keeps both people half-cut. A largest-face lock grabs whoever is closer — even when they're
> silent. Here, the frame follows the speaker.
>
> *(let the reel play ~15 s: A speaks, camera on A; B answers, camera moves to B)*
>
> And when I open the output, there's a number for this: speaker-on-screen — the percentage of
> the time the active speaker stayed in frame.

## 3:15–4:10 — Honest validation + traceability

**On screen:** asset detail page — spec checks table (rule / expected / actual / result), then
scroll to the AI decision log cards with stage, choice, reason, confidence. Highlight one
**failing** check if you have one (e.g. the 190 s reel vs Instagram's 90 s limit).

> Every output is validated against a machine-readable spec — ratio, resolution, fps, codec,
> duration, size — before it's allowed into the library. And failures are not hidden: this reel
> is 190 seconds against Instagram's 90-second limit, and the app says so honestly.
>
> Every output also traces back to its master, with the full AI decision log: the stage, the
> choice, the reason, and the confidence. Judges — or a content-ops team — can audit every AI
> decision the system made.

## 4:10–4:40 — Close

**On screen:** library grid of outputs, then end card with demo URL + repo URL.

> So: one master in, every platform version out — with audio-plus-vision speaker tracking, a VLM
> critic on every crop, honest spec validation, and full traceability. Built on Next.js, Neon
> Postgres, Blaxel and ffmpeg — CPU only. The live demo and the public repo are linked in the
> submission form. Thank you.

---

## Shot list / recording prep

| # | Shot | Need |
|---|---|---|
| 1 | Create page with form filled | Title + file picker visible; use a **sample master's "Run job"** instead of a live 282 MB upload |
| 2 | Job in progress | Stage + progress % visible — run the job **right before recording** so it's mid-flight |
| 3 | Library, grouped by job | Pre-run jobs so the grid is full and instant |
| 4 | Image asset page | Spec checks table + VLM critic decision card with its reason |
| 5 | Two-person dialogue reel | Cue it to a speaker change; full-size player |
| 6 | Asset page "Speaker on screen" bar | Note the real % before recording so you can say it |
| 7 | A failing validation | Pre-rerun the 190 s master so the honest-fail story is real |
| 8 | End card | Demo URL + repo URL |

**Recording tips**
- Record at 2560×1440 or 1080p+, 30/60 fps; hide bookmarks bar; close notifications.
- Zoom browser to ~110–125% so text in the spec table and decision log is readable.
- One idea per cut; silence is fine — cut it in editing.
- Dry-run once with a timer. If over 4:45, trim section 1's narration first.

## Submission checklist (handbook "How to Submit")

- [ ] **Live demo link** — Vercel URL openable in incognito; Neon + Blaxel kept alive through judging (~2 weeks); watch Neon free-tier 5 GB egress while judges click through.
- [ ] **Public GitHub repo** — `hoichoi-reframe` currently has **uncommitted changes** (`git status` shows modified web/worker files): commit, push, and flip the repo public. Confirm no secrets in history (`.env.local` is git-ignored; given assets in `assets/given/` stay git-ignored).
- [ ] **Video under 5 min** — upload as unlisted YouTube or Drive, test the link in a private window, keep the raw file as backup.
- [ ] **Submit the form** — hoichoi AI Builders Hackathon submission form (MS Forms link in the handbook) with all three links.
- [ ] Deadline: submission window closes **12:30 am IST** (1 h after the 12-hour mark).
