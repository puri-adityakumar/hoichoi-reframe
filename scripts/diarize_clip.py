#!/usr/bin/env python3
"""CLI: diarize one clip via Sarvam Batch (exactly ONE submission).

Usage: .venv/bin/python scripts/diarize_clip.py <wav> <out.json> [num_speakers]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

P4 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P4))

from worker.reframe import config  # noqa: E402
from worker.reframe.sarvam import (  # noqa: E402
    JobParameters,
    SarvamBatchClient,
    SarvamError,
)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    wav, out = Path(sys.argv[1]), Path(sys.argv[2])
    num_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    config.load_config()
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            with SarvamBatchClient() as client:
                job_id = client.submit_batch(
                    wav,
                    JobParameters(model="saaras:v3", mode="transcribe",
                                  with_timestamps=True, with_diarization=True,
                                  num_speakers=num_speakers),
                )
                print(f"job_id: {job_id}")
                status = client.wait_until_complete(job_id, poll_interval=5.0, timeout=600.0)
                print(f"queue+run: {time.monotonic() - t0:.1f}s (state={status['job_state']})")
                result = client.fetch_result(job_id)
            break
        except SarvamError as e:
            print(f"attempt {attempt} failed: {e}", file=sys.stderr)
            if attempt >= 2:
                print("SARVAM FAILED after 1 retry; falling back to mouth-only.", file=sys.stderr)
                out.write_text(json.dumps({"error": str(e), "segments": []}))
                return 1

    segments = [s.model_dump() for s in result.segments]
    payload = {
        "job_id": job_id,
        "latency_s": round(time.monotonic() - t0, 2),
        "num_segments": len(segments),
        "speakers": sorted({s["speaker_id"] for s in segments if s["speaker_id"] is not None}),
        "segments": segments,
    }
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"saved: {out} ({len(segments)} segments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
