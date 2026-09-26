#!/usr/bin/env python3
"""One-off: replace video preview_keys with JPEG poster frames.

Early pipeline runs stored a downscaled mp4 as `outputs.preview_key`, which the
web renders in an <img> — so those thumbnails showed as broken images. The
worker now writes poster stills (worker/reframe/preview.py) and db.insert_output
rejects non-image keys, so nothing new can land. This repairs the old rows:
for every output whose preview_key is not an image, download the full output
(or the legacy preview), grab a poster frame, upload it as
`<name>.preview.jpg` and repoint preview_key.

Usage:
  .venv/bin/python scripts/backfill_preview_posters.py --dry-run
  .venv/bin/python scripts/backfill_preview_posters.py
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.reframe import db, storage  # noqa: E402
from worker.reframe.config import load_config  # noqa: E402
from worker.reframe.preview import (  # noqa: E402
    is_image_preview, make_video_poster, preview_ext,
)

SELECT_BAD = """
SELECT id::text AS id, job_id::text AS job_id, platform, s3_key, preview_key
FROM outputs
WHERE preview_key IS NOT NULL
ORDER BY created_at ASC
"""


def poster_key_for(preview_key: str) -> str:
    """outputs/<job>/<platform>.preview.mp4 -> outputs/<job>/<platform>.preview.jpg"""
    p = PurePosixPath(preview_key)
    stem = p.name
    for ext in (".preview.mp4", ".mp4"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return str(p.with_name(f"{stem}.preview.jpg"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_config()
    conn = db.connect()
    with conn.cursor() as cur:
        cur.execute(SELECT_BAD)
        rows = cur.fetchall()
    conn.commit()

    bad = [r for r in rows if r[4] and not is_image_preview(r[4])]
    print(f"{len(rows)} outputs, {len(bad)} with a non-image preview_key", flush=True)
    if not bad:
        return 0

    fixed = failed = 0
    for oid, job_id, platform, s3_key, preview_key in bad:
        new_key = poster_key_for(preview_key)
        try:
            with tempfile.TemporaryDirectory() as td:
                poster = Path(td) / PurePosixPath(new_key).name
                # The legacy preview mp4 is a downscaled copy of the same clip
                # and is ~100x smaller to fetch, so prefer it; fall back to the
                # full output if that object is gone.
                src = None
                for candidate in (preview_key, s3_key):
                    try:
                        local = Path(td) / PurePosixPath(candidate).name
                        storage.download_file(candidate, local)
                        src = local
                        break
                    except Exception:  # noqa: BLE001, S0301
                        continue
                if src is None:
                    raise RuntimeError(f"no readable source for {oid}")
                make_video_poster(src, poster)
                if args.dry_run:
                    print(f"  would write {new_key} ({poster.stat().st_size} B) "
                          f"for {platform} [{oid[:8]}]", flush=True)
                    continue
                storage.upload_file(poster, new_key)
            db.set_output_preview_key(oid, new_key)
            print(f"  {platform} [{oid[:8]}] {preview_ext(preview_key)} -> jpg  {new_key}",
                  flush=True)
            fixed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {platform} [{oid[:8]}]: {exc}", flush=True)
            failed += 1

    if args.dry_run:
        print(f"dry run: {len(bad)} would be rewritten")
    else:
        print(f"done: {fixed} fixed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
