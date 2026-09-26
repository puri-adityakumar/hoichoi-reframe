"""Blaxel batch-job entrypoint for the reframe pipeline.

Runs as a Blaxel Batch Job (name: reframe-worker). Inputs {job_id, master_id}
reach the handler as task kwargs via bl_start_job.start(handler); defensive
fallbacks read env JSON / argv for local runs. Reuses the exact pipeline in
worker/reframe/main.py, publishes under the EXISTING job_id (master is already
in S3, never re-uploaded).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent  # /app in the container
for _p in (APP_ROOT, APP_ROOT.parent):  # container: /app, local: P4 root
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from worker.reframe import db, storage, validate  # noqa: E402
from worker.reframe.config import load_config  # noqa: E402
from worker.reframe.decisions import DecisionLog  # noqa: E402
from worker.reframe.main import run_image_master, run_video_master  # noqa: E402

SPEC_PATH = APP_ROOT / "spec" / "spec.json"
TMP_BASE = Path(os.environ.get("REFRAME_TMP", "/tmp/reframe"))

ENV_INPUT_VARS = (
    "BLAXEL_TASK_INPUTS", "BLAXEL_EXECUTION_INPUTS", "BL_TASK_INPUTS",
    "TASK_INPUTS", "BLAXEL_INPUTS", "JOB_INPUTS",
)


def _inputs_from_env_or_argv() -> dict[str, Any]:
    """Defensive fallback: find {job_id, master_id} in env JSON or argv."""
    for var in ENV_INPUT_VARS:
        raw = os.environ.get(var)
        if not raw:
            continue
        try:
            data = json.loads(raw)
            print(f"[job_entry] inputs read from env {var}", flush=True)
            if isinstance(data, dict):
                if isinstance(data.get("inputs"), dict):
                    data = data["inputs"]
                return data
        except Exception:
            print(f"[job_entry] env {var} set but not parseable JSON", flush=True)
    for arg in sys.argv[1:]:
        if arg.lstrip("-").startswith(("job_id", "inputs", "{")):
            try:
                data = json.loads(arg)
                if isinstance(data, dict):
                    print("[job_entry] inputs read from argv JSON", flush=True)
                    return data.get("inputs", data) if "inputs" in data else data
            except Exception:
                pass
    raise RuntimeError(
        "no inputs found (tried handler kwargs, env "
        f"{ENV_INPUT_VARS}, argv)")


def handler(*args: Any, **kwargs: Any) -> None:
    payload: dict[str, Any] = dict(kwargs)
    if not payload and args and isinstance(args[0], dict):
        payload = dict(args[0])
    if isinstance(payload.get("inputs"), dict):
        payload = dict(payload["inputs"])
    if not payload.get("job_id") or not payload.get("master_id"):
        envp = _inputs_from_env_or_argv()
        payload = {**envp, **{k: v for k, v in payload.items() if v}}
    if "job_id" in payload and "inputs" not in payload:
        print("[job_entry] inputs read from task kwargs", flush=True)
    run_job(str(payload["job_id"]), str(payload["master_id"]))


def _download(url: str, dst: Path) -> None:
    import httpx
    with httpx.stream("GET", url, timeout=httpx.Timeout(60.0, read=1800.0)) as r:
        r.raise_for_status()
        with open(dst, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")


def _ensure_models() -> None:
    """Blaxel code-upload drops binary files, so fetch the public model at start."""
    target = APP_ROOT / "worker" / "models" / "face_landmarker.task"
    if target.exists() and target.stat().st_size > 1_000_000:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    import httpx
    print(f"[job_entry] downloading face model to {target}", flush=True)
    with httpx.stream("GET", MODEL_URL, timeout=httpx.Timeout(60.0, read=300.0)) as r:
        r.raise_for_status()
        tmp = target.with_suffix(".task.tmp")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
        tmp.replace(target)


def run_job(job_id: str, master_id: str) -> None:
    load_config()
    _ensure_models()
    spec = json.loads(SPEC_PATH.read_text())
    platforms = spec["platforms"]

    m = db.get_master(master_id)
    s3_key = m["s3_key"]
    kind = m["kind"]
    print(f"[job_entry] job={job_id} master={master_id} kind={kind} "
          f"key={s3_key}", flush=True)

    workdir = TMP_BASE / job_id
    outdir = workdir / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    local = workdir / Path(s3_key).name

    try:
        db.set_job(job_id, "running", "downloading master", 10)
        t0 = time.monotonic()
        if not local.exists():
            _download(storage.presign_get(s3_key, exp=6 * 3600), local)
        print(f"[job_entry] downloaded {local.stat().st_size} bytes in "
              f"{time.monotonic() - t0:.1f}s", flush=True)

        db.set_job(job_id, "running", "analyzing", 25)
        log = DecisionLog()
        timings: dict[str, float] = {}
        if kind == "video":
            outputs = run_video_master(local, outdir, platforms, log, timings)
        else:
            outputs = run_image_master(local, outdir, platforms, log, timings)

        rows = validate.validate_all(spec, [
            {"platform": o["platform"], "path": o["path"],
             "name": Path(o["path"]).name} for o in outputs])
        npass = sum(1 for r in rows if r["passed"])
        print(f"[job_entry] outputs={len(outputs)} validations {npass}/{len(rows)}"
              f" passed", flush=True)

        db.set_job(job_id, "running", "uploading outputs", 60)
        for o in outputs:
            okey = f"outputs/{job_id}/{Path(o['path']).name}"
            storage.upload_file(o["path"], okey)
            pkey = None
            if o.get("preview_path"):
                pkey = f"outputs/{job_id}/{Path(o['preview_path']).name}"
                storage.upload_file(o["preview_path"], pkey)
            oid = db.insert_output(job_id, master_id, o["platform"], o["ratio"],
                                   o["kind"], okey, pkey, o.get("speaker_pct"))
            db.insert_validations(
                oid, [r for r in rows if r["output"] == Path(o["path"]).name])
        db.insert_decisions(job_id, None, log.to_list())
        db.set_job(job_id, "done", "done", 100)
        print(f"[job_entry] job {job_id} done", flush=True)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        try:
            db.set_job(job_id, "failed", "error", 0, error=str(exc)[:500])
        except Exception:
            traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        from blaxel.core.jobs import bl_start_job
    except ImportError:
        bl_start_job = None
        print("[job_entry] blaxel SDK not installed; direct-CLI mode", flush=True)
    if bl_start_job is not None:
        bl_start_job.start(handler)
    else:
        # local dev: python job_entry.py '{"job_id": "...", "master_id": "..."}'
        # or REFRAME_JOB_ID / REFRAME_MASTER_ID env vars
        if len(sys.argv) > 1:
            payload = json.loads(sys.argv[1])
        else:
            payload = {"job_id": os.environ["REFRAME_JOB_ID"],
                       "master_id": os.environ["REFRAME_MASTER_ID"]}
        handler(**payload)
