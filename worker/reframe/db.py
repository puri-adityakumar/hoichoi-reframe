"""Neon Postgres access (psycopg 3). Schema: db/schema.sql tables
masters / jobs / outputs / validations / decisions."""
from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse, urlunparse

import psycopg

from worker.reframe.config import load_config

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


def _one(sql: str, params: tuple) -> int:
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        (val,) = cur.fetchone()
    conn.commit()
    return int(val)


def _exec(sql: str, params: tuple) -> None:
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()


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


def set_master_s3_key(master_id: int, s3_key: str) -> None:
    _exec("UPDATE masters SET s3_key = %s WHERE id = %s", (s3_key, master_id))


def create_job(master_id: int) -> int:
    return _one(
        "INSERT INTO jobs (master_id, status, stage, progress) VALUES (%s, 'running', 'start', 0) RETURNING id",
        (master_id,),
    )


def finish_job(job_id: int, status: str, stage: str, progress: int,
               error: Optional[str] = None) -> None:
    _exec(
        """UPDATE jobs SET status=%s, stage=%s, progress=%s, error=%s, finished_at=now()
           WHERE id=%s""",
        (status, stage, progress, error, job_id),
    )


def insert_output(job_id: int, master_id: int, platform: str, ratio: str, kind: str,
                  s3_key: str, preview_key: Optional[str],
                  speaker_pct: Optional[float]) -> int:
    return _one(
        """
        INSERT INTO outputs (job_id, master_id, platform, ratio, kind, s3_key, preview_key,
                             speaker_on_screen_pct)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (job_id, master_id, platform, ratio, kind, s3_key, preview_key, speaker_pct),
    )


def insert_validations(output_id: int, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    conn = connect()
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                "INSERT INTO validations (output_id, rule, expected, actual, passed) "
                "VALUES (%s, %s, %s, %s, %s)",
                (output_id, r["rule"], r["expected"], r["actual"], bool(r["passed"])),
            )
    conn.commit()


def insert_decisions(job_id: int, output_id: Optional[int], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    conn = connect()
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                "INSERT INTO decisions (job_id, output_id, stage, choice, reason, confidence) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (job_id, output_id, r["stage"], r["choice"], r["reason"], r.get("confidence")),
            )
    conn.commit()
