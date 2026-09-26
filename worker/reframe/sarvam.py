"""Sarvam Batch Speech-to-Text client with diarization (saaras:v3).

Flow: init job -> get presigned upload URL -> PUT file -> start job ->
poll status -> get presigned download URL -> parse diarized transcript.

Docs: https://docs.sarvam.ai/api-reference/speech-to-text/stt/job/initiate
Auth: `api-subscription-key` header.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field

BASE_URL = "https://api.sarvam.ai"
API_KEY_ENV = "SARVAM_API_KEY"

JobState = Literal["Accepted", "Pending", "Running", "Completed", "Failed"]


class JobParameters(BaseModel):
    model: str = "saaras:v3"
    mode: str = "transcribe"
    with_timestamps: bool = True
    with_diarization: bool = True
    num_speakers: Optional[int] = 2
    language_code: str = "unknown"


class DiarizedSegment(BaseModel):
    transcript: str = ""
    start_time_seconds: Optional[float] = None
    end_time_seconds: Optional[float] = None
    speaker_id: Optional[str] = None


class TranscriptFile(BaseModel):
    transcript: Optional[str] = None
    language_code: Optional[str] = None
    diarized_transcript: list[DiarizedSegment] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class BatchResult(BaseModel):
    segments: list[DiarizedSegment]
    raw: dict[str, Any]


class SarvamError(RuntimeError):
    pass


class SarvamBatchClient:
    def __init__(self, api_key: Optional[str] = None, timeout: float = 60.0):
        self.api_key = api_key or os.environ.get(API_KEY_ENV)
        if not self.api_key:
            raise SarvamError(f"{API_KEY_ENV} not set")
        self._client = httpx.Client(
            headers={"api-subscription-key": self.api_key}, timeout=timeout
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SarvamBatchClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _check(self, resp: httpx.Response, what: str) -> dict[str, Any]:
        if resp.status_code >= 400:
            raise SarvamError(f"{what} failed: HTTP {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    # -- step 1: init ------------------------------------------------------
    def submit_batch(self, wav_path: str | os.PathLike[str], params: JobParameters | None = None) -> str:
        """Create a batch job; upload the file; start the job. Returns job_id."""
        params = params or JobParameters()
        resp = self._client.post(
            f"{BASE_URL}/speech-to-text/job/v1",
            json={"job_parameters": params.model_dump(exclude_none=True)},
        )
        data = self._check(resp, "init job")
        job_id: str = data["job_id"]

        file_name = os.path.basename(str(wav_path))
        resp = self._client.post(
            f"{BASE_URL}/speech-to-text/job/v1/upload-files",
            json={"job_id": job_id, "files": [file_name]},
        )
        up = self._check(resp, "upload-files")
        upload_url = up["upload_urls"][file_name]["file_url"]
        with open(wav_path, "rb") as fh:
            put = httpx.put(
                upload_url,
                content=fh.read(),
                headers={"x-ms-blob-type": "BlockBlob"},
                timeout=120.0,
            )
        if put.status_code >= 400:
            raise SarvamError(f"upload PUT failed: HTTP {put.status_code}: {put.text[:300]}")

        resp = self._client.post(f"{BASE_URL}/speech-to-text/job/v1/{job_id}/start")
        self._check(resp, "start job")
        return job_id

    # -- step 2: poll ------------------------------------------------------
    def poll_batch(self, job_id: str) -> dict[str, Any]:
        """One status poll. Returns the raw status response."""
        resp = self._client.get(f"{BASE_URL}/speech-to-text/job/v1/{job_id}/status")
        return self._check(resp, "get status")

    def wait_until_complete(
        self, job_id: str, poll_interval: float = 5.0, timeout: float = 600.0
    ) -> dict[str, Any]:
        """Poll every `poll_interval` seconds until terminal state or timeout."""
        import time

        deadline = time.monotonic() + timeout
        while True:
            status = self.poll_batch(job_id)
            state: JobState = status.get("job_state")
            if state in ("Completed", "Failed"):
                return status
            if time.monotonic() > deadline:
                raise SarvamError(f"job {job_id} timed out after {timeout}s (state={state})")
            time.sleep(poll_interval)

    # -- step 3: fetch result ----------------------------------------------
    def fetch_result(self, job_id: str) -> BatchResult:
        """Download the transcript JSON(s) of a completed job."""
        status = self.poll_batch(job_id)
        if status.get("job_state") != "Completed":
            raise SarvamError(
                f"job {job_id} not Completed: {status.get('job_state')} {status.get('error_message')}"
            )
        file_names = [
            out["file_name"]
            for task in status.get("job_details") or []
            for out in task.get("outputs") or []
        ]
        if not file_names:
            raise SarvamError(f"job {job_id} has no output files: {status}")
        resp = self._client.post(
            f"{BASE_URL}/speech-to-text/job/v1/download-files",
            json={"job_id": job_id, "files": file_names},
        )
        down = self._check(resp, "download-files")

        all_segments: list[DiarizedSegment] = []
        raw: dict[str, Any] = {"files": {}}
        for name, detail in down.get("download_urls", {}).items():
            file_resp = httpx.get(detail["file_url"], timeout=60.0)
            if file_resp.status_code >= 400:
                raise SarvamError(f"download {name} failed: HTTP {file_resp.status_code}")
            payload = file_resp.json()
            raw["files"][name] = payload
            # diarized_transcript is {"entries": [...]} in the real payload
            payload = dict(payload)
            dt = payload.get("diarized_transcript")
            if isinstance(dt, dict):
                payload["diarized_transcript"] = dt.get("entries") or []
            parsed = TranscriptFile.model_validate({**payload, "raw": {}})
            all_segments.extend(parsed.diarized_transcript)
        return BatchResult(segments=all_segments, raw=raw)
