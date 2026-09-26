"""Reusable VLM client: OpenRouter first, GMI fallback. Never logs secrets."""
from __future__ import annotations

import base64
import io
import json
import re

import httpx
from PIL import Image

from worker.reframe.config import load_config

load_config()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GMI_URL = "https://api.gmi-serving.com/v1/chat/completions"
OPENROUTER_MODEL = "z-ai/glm-5.3-flash"
GMI_MODEL = "zai-org/GLM-5.3-Flash"
MAX_EDGE = 640
MAX_TOKENS = 2000  # GLM-5.3-Flash is a reasoning model; small budgets starve content after reasoning
TIMEOUT = 90


def _inline_image(img: Image.Image) -> str:
    im = img.convert("RGB")
    im.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _content(images: list[Image.Image], prompt: str) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": _inline_image(im)}} for im in images]
    return content


def _call(url: str, api_key: str, model: str, images: list[Image.Image], prompt: str) -> str:
    resp = httpx.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": _content(images, prompt)}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    content = msg.get("content")
    if not content:
        # reasoning models can put the whole answer in the reasoning field
        reasoning = msg.get("reasoning") or ""
        if reasoning:
            return reasoning
        raise RuntimeError("empty completion (content was null)")
    return content


def ask_vlm(images: list[Image.Image], prompt: str) -> tuple[str, str]:
    """Returns (text, provider). One provider-fallback retry max."""
    import os
    provider = "openrouter"
    key = os.environ.get("OPENROUTER_API_KEY", "")
    try:
        if key:
            return _call(OPENROUTER_URL, key, OPENROUTER_MODEL, images, prompt), provider
        raise RuntimeError("no OPENROUTER_API_KEY")
    except Exception as exc:  # noqa: BLE001 - fallback on any error
        print(f"vlm: {provider} failed ({type(exc).__name__}); falling back to gmi")
        gmi_key = os.environ.get("GMI_API_KEY", "")
        if not gmi_key:
            raise
        return _call(GMI_URL, gmi_key, GMI_MODEL, images, prompt), "gmi"


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in VLM reply: {text[:200]!r}")
    return json.loads(m.group(0))


def pick_best(crops: list[Image.Image], context: str) -> tuple[dict, str]:
    """Ask the VLM to pick the best crop. Returns ({choice, reason}, provider)."""
    options = "\n".join(f"{i}: candidate crop {i}" for i in range(len(crops)))
    prompt = (
        f"{context}\n\n"
        f"You are shown {len(crops)} candidate crops of the same master image:\n{options}\n\n"
        "Pick the single best crop: the one that keeps the most important subjects "
        "(faces/people) fully and well framed, feels least cropped, and works for the "
        "target aspect ratio. Reply with STRICT JSON only, no prose:\n"
        '{"choice": 0, "reason": "one short sentence"}\n'
        "where choice is the 0-based index of the best candidate."
    )
    text, provider = ask_vlm(crops, prompt)
    try:
        data = _parse_json(text)
        choice = int(data.get("choice", 0))
        if not 0 <= choice < len(crops):
            choice = 0
        return {"choice": choice, "reason": str(data.get("reason", "")).strip()}, provider
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"choice": 0, "reason": f"unparseable VLM reply: {text[:120]}"}, provider
