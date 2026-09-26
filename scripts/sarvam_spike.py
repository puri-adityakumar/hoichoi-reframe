"""Sarvam diarization spike: 30s two-person dialogue clip.

Loads P4/.env.local, extracts nothing (expects work/sarvam/clip_a_mono.wav
already cut by ffmpeg), submits exactly one batch job, polls, saves
work/sarvam/clip_a.json, prints latency + diarization summary.

Usage: python3 scripts/sarvam_spike.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

P4 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P4))

WAV = P4 / "work/sarvam/clip_a_mono.wav"
OUT = P4 / "work/sarvam/clip_a.json"


def load_env(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


from worker.reframe.sarvam import JobParameters, SarvamBatchClient, SarvamError


def main() -> int:
    from worker.reframe.sarvam import JobParameters as _JP  # noqa: F401

    env = load_env(P4 / ".env.local")
    import os

    os.environ.setdefault("SARVAM_API_KEY", env.get("SARVAM_API_KEY", ""))

    t0 = time.monotonic()
    with SarvamBatchClient(api_key=os.environ["SARVAM_API_KEY"]) as client:
        resume_id = os.environ.get("RESUME_JOB_ID")
        if resume_id:
            job_id = resume_id
            t_submit = 0.0
            t_run = 0.0
            status = {"job_state": "Completed"}
            print(f"resuming completed job: {job_id}")
        else:
            job_id = client.submit_batch(
                WAV,
                JobParameters(model="saaras:v3", mode="transcribe", with_timestamps=True,
                              with_diarization=True, num_speakers=2),
            )
            t_submit = time.monotonic() - t0
            print(f"job_id: {job_id}")
            print(f"submit latency (init+upload+start): {t_submit:.1f}s")

            t1 = time.monotonic()
            status = client.wait_until_complete(job_id, poll_interval=5.0, timeout=600.0)
            t_run = time.monotonic() - t1
            print(f"queue+run latency: {t_run:.1f}s (state={status['job_state']})")

        result = client.fetch_result(job_id)

    total_speech = sum(
        (s.end_time_seconds or 0) - (s.start_time_seconds or 0) for s in result.segments
    )
    speakers = sorted({s.speaker_id for s in result.segments})
    payload = {
        "job_id": job_id,
        "submit_latency_s": round(t_submit, 2),
        "queue_run_latency_s": round(t_run, 2),
        "num_segments": len(result.segments),
        "speakers": speakers,
        "total_speech_seconds": round(total_speech, 2),
        "segments": [s.model_dump() for s in result.segments],
        "raw": result.raw,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved: {OUT}")
    print(f"segments: {len(result.segments)}  speakers: {speakers}  speech: {total_speech:.1f}s")
    print("first segments:")
    for s in result.segments[:3]:
        print(f"  [{s.start_time_seconds:.2f}-{s.end_time_seconds:.2f}] SPEAKER_{s.speaker_id}: {s.transcript[:60]}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SarvamError as e:
        print(f"SARVAM SPIKE ERROR: {e}", file=sys.stderr)
        sys.exit(1)
