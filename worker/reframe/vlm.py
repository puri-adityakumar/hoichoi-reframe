"""Reusable VLM client: OpenRouter first, GMI fallback. Never logs secrets."""
from __future__ import annotations

import base64
import io
import json
import re
import time
from typing import Any

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

# raw VLM call log (prompt/reply/provider/model/latency/parsed choice), drained
# by collect_vlm_log() at publish time into jobs/<job_id>/raw/vlm_calls.json
_VLM_LOG: list[dict[str, Any]] = []


def collect_vlm_log() -> list[dict[str, Any]]:
    """Return every captured VLM call since the last drain, and clear it."""
    out = list(_VLM_LOG)
    _VLM_LOG.clear()
    return out


def _log_choice(choice: Any) -> None:
    """Attach the parsed choice to the most recent captured call."""
    if _VLM_LOG:
        _VLM_LOG[-1]["choice"] = choice


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


def _timed_call(url: str, api_key: str, model: str,
                images: list[Image.Image], prompt: str) -> str:
    """_call + capture into _VLM_LOG (successful calls only)."""
    provider = "openrouter" if url == OPENROUTER_URL else "gmi"
    t0 = time.monotonic()
    text = _call(url, api_key, model, images, prompt)
    _VLM_LOG.append({
        "prompt": prompt,
        "reply": text,
        "provider": provider,
        "model": model,
        "latency_s": round(time.monotonic() - t0, 3),
        "choice": None,
    })
    return text


def ask_vlm(images: list[Image.Image], prompt: str) -> tuple[str, str]:
    """Returns (text, provider). One provider-fallback retry max."""
    import os
    provider = "openrouter"
    key = os.environ.get("OPENROUTER_API_KEY", "")
    try:
        if key:
            return _timed_call(OPENROUTER_URL, key, OPENROUTER_MODEL, images, prompt), provider
        raise RuntimeError("no OPENROUTER_API_KEY")
    except Exception as exc:  # noqa: BLE001 - fallback on any error
        print(f"vlm: {provider} failed ({type(exc).__name__}); falling back to gmi")
        gmi_key = os.environ.get("GMI_API_KEY", "")
        if not gmi_key:
            raise
        return _timed_call(GMI_URL, gmi_key, GMI_MODEL, images, prompt), "gmi"


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in VLM reply: {text[:200]!r}")
    return json.loads(m.group(0))


def pick_best(crops: list[Image.Image], context: str,
              subjects_preserved: bool = True) -> tuple[dict, str]:
    """Ask the VLM to pick the best crop. Returns ({choice, reason}, provider).

    `choice` is the 0-based index of the chosen candidate, or -1 when the model
    rejected ALL of them, in which case the dict also carries
    "rejected": True and the reason explains why. -1 is never coerced to 0: the
    caller must check `pick["choice"] < 0` and decide what to do (a subject-free
    crop is worse than an honest failure, and on the run that shipped an empty
    pillar the model was handed three people-free options with no way to say so).

    `subjects_preserved` tells the model the candidates were already filtered to
    keep the detected subjects whole, so it judges composition and prominence
    only. Pass False for candidate sets that were NOT filtered (the video still
    pick hands over 8 raw frames).
    """
    options = "\n".join(f"{i}: candidate crop {i}" for i in range(len(crops)))
    filtered = (
        "All of these candidates were already filtered by our own detector to keep the "
        "detected subjects' faces whole inside the frame, so do not spend this answer on "
        "whether a subject is present - judge composition, prominence and placement.\n\n"
        if subjects_preserved else ""
    )
    prompt = (
        f"{context}\n\n"
        f"You are shown {len(crops)} candidate crops of the same master image:\n{options}\n\n"
        f"{filtered}"
        "Pick the single best crop by composition, in this order of weight:\n"
        "1. The subjects' FACES are the largest and most complete in the frame.\n"
        "2. No face is clipped, cut in half, or touching a crop edge.\n"
        "3. The subject sits well for the target aspect ratio (for a tall crop, keep "
        "the faces in the upper-middle rather than lost at the top or bottom).\n"
        "A crop full of empty background, bare floor or architecture scores badly here "
        "even if it is tidy and evenly exposed.\n\n"
        "Reply with STRICT JSON only, no prose:\n"
        '{"choice": 0, "reason": "one short sentence"}\n'
        "where choice is the 0-based index of the best candidate, or -1 if NONE of "
        "them is acceptable - use -1 rather than endorsing a bad option."
    )
    text, provider = ask_vlm(crops, prompt)
    try:
        data = _parse_json(text)
        choice = int(data.get("choice", 0))
        reason = str(data.get("reason", "")).strip()
        if choice == -1:
            _log_choice(-1)
            return {"choice": -1, "reason": reason, "rejected": True}, provider
        if not 0 <= choice < len(crops):
            # keep the old clamp, but make the coercion visible in the log
            _log_choice(0)
            return ({"choice": 0, "rejected": False,
                     "reason": f"{reason} (out-of-range choice, clamped to 0)".strip()},
                    provider)
        _log_choice(choice)
        return {"choice": choice, "reason": reason, "rejected": False}, provider
    except (ValueError, TypeError, json.JSONDecodeError):
        _log_choice(None)
        return ({"choice": 0, "rejected": False,
                 "reason": f"unparseable VLM reply: {text[:120]}"}, provider)
