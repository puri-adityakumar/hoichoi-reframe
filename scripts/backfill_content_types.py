#!/usr/bin/env python3
"""One-off: give already-uploaded objects a real Content-Type.

Every output was written by `storage.upload_file` without `ExtraArgs`, so S3
stored `binary/octet-stream`. The web serves objects through
`/api/serve`, and a `<video>`/`<img>` refuses octet-stream outright — the
browser does not sniff — so assets rendered as a black 0:00 player and
thumbnails as broken-image icons.

`upload_file` now sets ContentType from the extension, so nothing new lands
broken. This repairs the existing objects: for each key whose stored
ContentType is not already correct, re-stamp it with a server-side
`copy_object` (MetadataDirective=REPLACE). No object bytes move, and no
download/upload round trip is involved.

Neon Object Storage ignores `response-content-type` on a presigned GET, so
overriding the header at read time is not an option — the metadata itself has
to be corrected.

Usage:
  .venv/bin/python scripts/backfill_content_types.py --dry-run
  .venv/bin/python scripts/backfill_content_types.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.reframe import db, storage  # noqa: E402
from worker.reframe.config import load_config  # noqa: E402

# octet-stream is also what an object with no ContentType reads back as, so it
# is the marker for "never stamped".
UNSET = {"binary/octet-stream", "application/octet-stream", ""}


def candidate_keys(s3, bucket: str) -> list[str]:
    """Every key the web can serve: masters, outputs, previews, raw artifacts."""
    keys: set[str] = set(db.all_servable_keys())
    # Raw per-job QA artifacts (analysis JSON, VLM log, tracking debug video).
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="jobs/"):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if "/raw/" in k and not k.endswith("/"):
                keys.add(k)
    return sorted(keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_config()
    s3 = storage._s3_client()
    bucket = storage._bucket()
    keys = candidate_keys(s3, bucket)
    print(f"scanned {len(keys)} servable keys")

    bad: list[tuple[str, str, str]] = []
    for k in keys:
        try:
            head = s3.head_object(Bucket=bucket, Key=k)
        except Exception as e:  # noqa: BLE001 - a missing key is not fatal
            print(f"  skip (head failed) {k}: {type(e).__name__}")
            continue
        have = head.get("ContentType") or ""
        want = storage.content_type_for(k)
        if have in UNSET or have != want:
            bad.append((k, have, want))

    if not bad:
        print("nothing to fix: every object already carries a usable Content-Type")
        return 0

    for k, have, want in bad:
        print(f"  {have or '(none)'} -> {want}  {k}")

    if args.dry_run:
        print(f"dry run: {len(bad)} would be re-stamped")
        return 0

    fixed = 0
    for k, _have, want in bad:
        try:
            s3.copy_object(
                Bucket=bucket,
                Key=k,
                CopySource={"Bucket": bucket, "Key": k},
                MetadataDirective="REPLACE",
                ContentType=want,
            )
            fixed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {k}: {type(e).__name__}: {e}")
    print(f"re-stamped {fixed}/{len(bad)} objects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
