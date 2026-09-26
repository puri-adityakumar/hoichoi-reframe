#!/usr/bin/env python3
"""Image reformatting pipeline: importance map -> candidates -> VLM pick ->
render -> validate. Exactly one VLM call per ratio."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

P4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P4_ROOT))

from worker.reframe.image.importance import importance_map  # noqa: E402
from worker.reframe.image.crops import best_crop_candidates  # noqa: E402
from worker.reframe.image.render import render_crop  # noqa: E402
from worker.reframe.vlm import pick_best  # noqa: E402

MASTER = P4_ROOT / "assets" / "given" / "input_image.png"
SPEC = P4_ROOT / "spec" / "spec.json"
OUT_DIR = P4_ROOT / "work" / "outputs" / "image"
IMAGE_RATIOS = ["youtube_thumbnail", "feed_image", "story_image", "square_image"]
ASPECT_TOL = 0.01


def main() -> None:
    t0 = time.time()
    spec = json.loads(SPEC.read_text())["platforms"]
    img_bgr = cv2.imread(str(MASTER), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"cannot read {MASTER}")
    img_h, img_w = img_bgr.shape[:2]
    print(f"master {img_w}x{img_h}")

    imp, scale = importance_map(img_bgr)
    # face mask = importance > threshold where it looks like a face peak
    face_mask = (imp > 0.85).astype(np.float32)
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    report: dict = {}
    contact: list[Image.Image] = []
    for name in IMAGE_RATIOS:
        p = spec[name]
        ratio = p["ratio"]
        out_w, out_h = p["width"], p["height"]
        cands = best_crop_candidates(imp, img_w, img_h, ratio, k=3, face_mask=face_mask)
        crops = [img.crop((c["box"][0], c["box"][1],
                           c["box"][0] + c["box"][2], c["box"][1] + c["box"][3])) for c in cands]
        context = (
            f"Target platform: {p['label']} ({ratio}, {out_w}x{out_h}). Master shows two women "
            "(older woman standing centre-left, younger woman crouching centre-right) around a "
            "blood pool in a temple, lit strongly from the right. The burned-in watermark band "
            "must not dominate; prefer crops keeping both faces fully visible."
        )
        pick, provider = pick_best(crops, context)
        best = cands[pick["choice"]]
        out_path = OUT_DIR / f"{name}.jpg"
        render_crop(img, best["box"], out_w, out_h, out_path)
        size_bytes = out_path.stat().st_size

        actual_ar = out_w / out_h
        spec_ar = out_w / out_h  # target aspect from spec width/height
        aspect_check = abs((best["box"][2] / best["box"][3]) - spec_ar) / spec_ar <= ASPECT_TOL
        ratio_ok = abs(actual_ar - spec_ar) < 1e-9
        max_size_ok = size_bytes <= p["max_size_mb"] * 1024 * 1024
        report[name] = {
            "chosen_box": best["box"],
            "candidate_scores": [round(c["score"], 4) for c in cands],
            "vlm_choice": pick["choice"],
            "vlm_reason": pick["reason"],
            "vlm_provider": provider,
            "output_size_bytes": size_bytes,
            "validation": {"ratio_ok": bool(ratio_ok and aspect_check), "max_size_ok": max_size_ok},
        }
        print(f"{name}: box={best['box']} provider={provider} reason={pick['reason']}")
        for i, c in enumerate(crops):
            thumb = c.copy()
            thumb.thumbnail((256, 256))
            contact.append(_label(thumb, f"{name} #{i}{' *' if i == pick['choice'] else ''}"))

    # contact sheet: rows of labelled candidate thumbnails
    cols = 3
    rows = (len(contact) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 266 + 10, rows * 306 + 10), "white")
    for i, th in enumerate(contact):
        sheet.paste(th, (10 + (i % cols) * 266, 10 + (i // cols) * 306))
    sheet_path = OUT_DIR / "candidates_sheet.jpg"
    sheet.save(sheet_path, "JPEG", quality=85)

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"done in {time.time() - t0:.1f}s -> {OUT_DIR}")


def _label(img: Image.Image, text: str) -> Image.Image:
    from PIL import ImageDraw
    out = Image.new("RGB", (266, 306), "white")
    img = img.copy()
    img.thumbnail((256, 276))
    out.paste(img, (5, 25))
    ImageDraw.Draw(out).text((6, 5), text, fill="black")
    return out


if __name__ == "__main__":
    main()
