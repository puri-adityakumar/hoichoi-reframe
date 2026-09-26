"""Neon Postgres access (psycopg 3). Schema: db/schema.sql tables
masters / jobs / outputs / validations / decisions."""
from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse, urlunparse

import psycopg

from worker.reframe.config import load_config
from worker.reframe.preview import assert_image_preview

_conn: Optional[psycopg.Connection] = None


def connect() -> psycopg.Connection:
    global _conn
    if _conn is not None and not _conn.closed:
        return _conn
    load_config()
    url = os.environ["DATABASE_URL"]
    try:
        _conn = psycopg.connect(url)
    except Exception:
        # Fallback: strip query params except sslmode (also keep channel_binding if present)
        u = urlparse(url)
        q = parse_qs(u.query)
        keep: dict[str, str] = {}
        if "sslmode" in q:
            keep["sslmode"] = q["sslmode"][0]
        if "channel_binding" in q:
            keep["channel_binding"] = q["channel_binding"][0]
        clean = urlunparse(u._replace(query="&".join(f"{k}={v}" for k, v in keep.items())))
        _conn = psycopg.connect(clean)
    return _conn


def _reset_conn() -> None:
    global _conn
    if _conn is not None:
        try:
            _conn.rollback()
        except Exception:
            pass
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None


def _run_with_retry(fn) -> Any:
    """Long S3 uploads can outlive Neon's idle connection; retry once on a fresh one."""
    last: Optional[Exception] = None
    for attempt in range(2):
        try:
            return fn()
        except psycopg.Error as exc:
            last = exc
            _reset_conn()
    raise last  # type: ignore[misc]


def _one(sql: str, params: tuple) -> Any:
    def go() -> Any:
        conn = connect()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            (val,) = cur.fetchone()
        conn.commit()
        return val
    return _run_with_retry(go)


def _exec(sql: str, params: tuple) -> None:
    def go() -> None:
        conn = connect()
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
    _run_with_retry(go)


def upsert_master(title: str, kind: str, s3_key: str, size: int, w: int, h: int,
                  dur: Optional[float], fps: Optional[float]) -> int:
    return _one(
        """
        INSERT INTO masters (title, kind, s3_key, size_bytes, width, height, duration_sec, fps)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (title, kind, s3_key, size, w, h, dur, fps),
    )


def set_master_s3_key(master_id: Any, s3_key: str) -> None:
    _exec("UPDATE masters SET s3_key = %s WHERE id = %s::uuid", (s3_key, str(master_id)))


def create_job(master_id: Any) -> Any:
    return _one(
        "INSERT INTO jobs (master_id, status, stage, progress) VALUES (%s::uuid, 'running', 'start', 0) RETURNING id",
        (str(master_id),),
    )


def finish_job(job_id: Any, status: str, stage: str, progress: int,
               error: Optional[str] = None) -> None:
    _exec(
        """UPDATE jobs SET status=%s, stage=%s, progress=%s, error=%s, finished_at=now()
           WHERE id=%s::uuid""",
        (status, stage, progress, error, str(job_id)),
    )


def set_job(job_id: Any, status: str, stage: str, progress: int,
            error: Optional[str] = None) -> None:
    _exec(
        """UPDATE jobs SET status=%s, stage=%s, progress=%s, error=%s
           WHERE id=%s::uuid""",
        (status, stage, progress, error, str(job_id)),
    )


def get_master(master_id: Any) -> dict[str, Any]:
    def go() -> dict[str, Any]:
        conn = connect()
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """SELECT id::text AS id, title, kind, s3_key, size_bytes, width,
                          height, duration_sec, fps
                   FROM masters WHERE id = %s::uuid""",
                (str(master_id),),
            )
            row = cur.fetchone()
        conn.commit()
        if row is None:
            raise RuntimeError(f"master {master_id} not found")
        return dict(row)

    return _run_with_retry(go)


def insert_output(job_id: Any, master_id: Any, platform: str, ratio: str, kind: str,
                  s3_key: str, preview_key: Optional[str],
                  speaker_pct: Optional[float]) -> Any:
    # The web renders preview_key in an <img>. A video there is a broken image,
    # so reject it here rather than letting a bad row reach the UI.
    if preview_key is not None:
        assert_image_preview(preview_key)
    return _one(
        """
        INSERT INTO outputs (job_id, master_id, platform, ratio, kind, s3_key, preview_key,
                             speaker_on_screen_pct)
        VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (str(job_id), str(master_id), platform, ratio, kind, s3_key, preview_key, speaker_pct),
    )


def set_output_preview_key(output_id: Any, preview_key: str) -> None:
    assert_image_preview(preview_key)
    _exec("UPDATE outputs SET preview_key = %s WHERE id = %s::uuid",
          (preview_key, str(output_id)))


def all_servable_keys() -> list[str]:
    """Every s3_key / preview_key the web can serve, de-duplicated.

    Used by scripts/backfill_content_types.py to repair stored Content-Type
    on objects that predate upload_file setting one.
    """
    def go() -> list[str]:
        conn = connect()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT s3_key FROM masters WHERE s3_key IS NOT NULL
                   UNION
                   SELECT s3_key FROM outputs WHERE s3_key IS NOT NULL
                   UNION
                   SELECT preview_key FROM outputs WHERE preview_key IS NOT NULL"""
            )
            rows = [r[0] for r in cur.fetchall()]
        conn.commit()
        return sorted(set(rows))

    return _run_with_retry(go)


def insert_validations(output_id: Any, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    def go() -> None:
        conn = connect()
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(
                    "INSERT INTO validations (output_id, rule, expected, actual, passed) "
                    "VALUES (%s::uuid, %s, %s, %s, %s)",
                    (str(output_id), r["rule"], r["expected"], r["actual"], bool(r["passed"])),
                )
        conn.commit()

    _run_with_retry(go)


def insert_decisions(job_id: Any, output_id: Optional[Any], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    def go() -> None:
        conn = connect()
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(
                    "INSERT INTO decisions (job_id, output_id, stage, choice, reason, confidence) "
                    "VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s)",
                    (str(job_id), str(output_id) if output_id else None,
                     r["stage"], r["choice"], r["reason"], r.get("confidence")),
                )
        conn.commit()

    _run_with_retry(go)
