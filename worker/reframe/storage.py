"""Neon Object Storage (S3-compatible) client. Config from env; never logs secrets."""
from __future__ import annotations

import os
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

from worker.reframe.config import load_config

_client = None


def _s3_client():
    global _client
    if _client is not None:
        return _client
    load_config()
    _client = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        region_name=os.environ.get("S3_REGION", ""),
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        config=BotoConfig(s3={"addressing_style": "path"}),
    )
    return _client


def _bucket() -> str:
    load_config()
    return os.environ["S3_BUCKET"]


def upload_file(path: str | Path, key: str) -> str:
    s3 = _s3_client()
    s3.upload_file(str(path), _bucket(), key)
    return key


def presign_get(key: str, exp: int = 3600) -> str:
    s3 = _s3_client()
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": _bucket(), "Key": key}, ExpiresIn=exp
    )
