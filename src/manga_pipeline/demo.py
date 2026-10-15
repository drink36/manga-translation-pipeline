"""Synthetic fixture and scripted responses; never represent model quality."""
from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from .storage import write_json

FONT = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"


def create_demo(root: Path) -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "input"
    source.mkdir(exist_ok=True)
    font_path = FONT if Path(FONT).is_file() else None
    try:
        font = ImageFont.truetype(font_path or "DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default(size=28)
    cases = [("Hello!", "你好！"), ("Wait for me!", "等等我！")]
    for number, (original, translated) in enumerate(cases, 1):
        image = Image.new("RGB", (640, 800), (225, 225, 225))
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 620, 780), outline="black", width=4)
        draw.rounded_rectangle((100, 100, 540, 270), radius=35, fill="white", outline="black", width=3)
        draw.text((145, 155), original, font=font, fill="black")
        draw.text((60, 700), "SYNTHETIC DEMO", font=font, fill="black")
        image.save(source / f"{number:03}.png")
        write_json(source / f"{number:03}.fixture.json", {"original_text": original, "translation": translated, "text_bbox": [140, 150, 510, 210], "bubble_bbox": [100, 100, 540, 270]})
    config = {
        "mode": "demo", "input": "input", "output": "output", "source_language": "en", "reading_direction": "ltr",
        "detector": {"backend": "demo"}, "ocr": {"readers": {"paddle_vl": {"backend": "demo"}, "manga_ocr": {"backend": "demo"}}, "routes": {"en": ["paddle_vl", "manga_ocr"]}},
        "audit": {"enabled": True, "tile_grid": [2, 2]}, "vlm": {"model": "SCRIPTED-DEMO-NO-MODEL"},
        "retries": {"region_retries": 1, "ocr_retries": 1, "translation_retries": 1, "layout_retries": 1},
        "context": {"pages_per_chunk": 2}, "render": {"font_path": font_path, "font_index": 3 if font_path else 0, "font_size_min": 16, "font_size_max": 32, "padding": 12, "line_spacing": 1.2, "mask_dilation": 1, "inpaint": {"backend": "plain"}, "render_uncertain": False, "sfx_policy": "preserve"},
        "qa": {"visual_enabled": True}, "export": {"cbz": True},
    }
    # A missing CJK font must not be passed off as a successful Chinese render.
    if not font_path:
        for number, (original, translated) in enumerate(cases, 1):
            write_json(source / f"{number:03}.fixture.json", {"original_text": original, "translation": f"Demo {number}", "text_bbox": [140, 150, 510, 210], "bubble_bbox": [100, 100, 540, 270]})
    path = root / "demo.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


class DemoVLM:
    def ask(self, task, payload, images=None):
        if task == "coverage":
            return {"regions": payload.get("demo_regions", []), "uncertain": False}
        if task == "reading_order":
            return {"region_ids": [r["region_id"] for r in payload["regions"]], "uncertain": False, "reason": "scripted demo"}
        if task in ("ocr", "reconcile"):
            return {"text": payload.get("demo_text", "Hello!"), "uncertain": False, "reason": "scripted demo"}
        if task == "context":
            return {"summary": "Synthetic fixture. No model was run.", "glossary": [], "characters": []}
        if task == "translate":
            return {"translations": [{"region_id": r["region_id"], "text": r.get("demo_translation", "Demo"), "uncertain": False, "reason": "scripted demo"} for r in payload["regions"]]}
        if task == "verify":
            return {"checks": [{"region_id": r["region_id"], "passed": True, "issues": []} for r in payload["regions"]]}
        if task == "visual_qa":
            return {"issues": [], "passed": True}
        raise ValueError(f"Unsupported demo task: {task}")
