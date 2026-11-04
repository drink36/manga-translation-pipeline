"""Conservative text removal and measured horizontal typesetting.

No weights are downloaded here. An unsafe or unrenderable region is left intact
and appears in the report. Every render starts from the source image.
"""

from __future__ import annotations

import math
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


FONT_SIZE_MIN = 10
FONT_SIZE_MAX = 48


class FontError(ValueError):
    """A font cannot render the requested translation."""


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", str(value)).strip(".") or "page"


def _bbox(value: Any, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("Expected bbox [left, top, right, bottom].")
    x1, y1, x2, y2 = (float(v) for v in value)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("Bounding box contains non-finite coordinates.")
    result = (max(0, math.floor(x1)), max(0, math.floor(y1)),
              min(width, math.ceil(x2)), min(height, math.ceil(y2)))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("Bounding box has no area inside the original image.")
    return result


def _rect_mask(box: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((size[1], size[0]), dtype=bool)
    mask[box[1]:box[3], box[0]:box[2]] = True
    return mask


def _text_region_mask(region: dict, box: tuple[int, int, int, int],
                      size: tuple[int, int]) -> np.ndarray:
    mask = _rect_mask(box, size)
    polygon = region.get("polygon")
    if polygon:
        if len(polygon) < 3 or any(len(point) != 2 for point in polygon):
            raise ValueError("Text polygon must contain at least three [x, y] points.")
        points = [tuple(float(value) for value in point) for point in polygon]
        if not all(math.isfinite(value) for point in points for value in point):
            raise ValueError("Text polygon contains non-finite coordinates.")
        image = Image.new("L", size)
        ImageDraw.Draw(image).polygon(points, fill=255)
        mask &= np.asarray(image) > 127
        if not mask.any():
            raise ValueError("Text polygon does not intersect the text bounding box.")
    return mask


def _read_mask(path: str, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError(f"Mask {path} must use original-page dimensions {size}.")
        return np.asarray(image.convert("L")) > 127


def _morph(mask: np.ndarray, radius: int, dilate: bool = True) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    result = np.zeros(mask.shape, dtype=bool)
    rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return result
    # Only pixels within `radius` of the mask can change, so filter just that window
    # instead of the whole page. Rank filters replicate edge pixels; one extra pixel of
    # margin keeps the window border empty, so the result equals a full-page filter.
    margin = radius + 1
    top, bottom = max(0, rows[0] - margin), min(mask.shape[0], rows[-1] + 1 + margin)
    left, right = max(0, columns[0] - margin), min(mask.shape[1], columns[-1] + 1 + margin)
    operation = ImageFilter.MaxFilter if dilate else ImageFilter.MinFilter
    window = Image.fromarray(mask[top:bottom, left:right].astype(np.uint8) * 255)
    result[top:bottom, left:right] = np.asarray(window.filter(operation(2 * radius + 1))) > 127
    return result


def _infer_white_glyph_mask(rgb: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    """A last-resort threshold mask, allowed only on a mostly white text crop."""
    darkest = np.min(rgb[region_mask], axis=1)  # Darkest channel of each pixel in the region.
    if not len(darkest) or np.mean(darkest >= 235) < 0.60:
        raise ValueError("No glyph mask and the text crop is not mostly near-white.")
    mask = np.zeros_like(region_mask)
    mask[region_mask] = darkest < 210
    if not mask.any():
        raise ValueError("No original glyphs could be isolated in the white text crop.")
    return mask


def _white_background(rgb: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    """Require a nearly uniform white neighborhood around all erased pixels."""
    ring = _morph(mask, 3) & ~mask
    samples = rgb[ring]
    if len(samples) < 4:
        raise ValueError("Insufficient background around the glyph mask.")
    if np.mean(np.min(samples, axis=1) >= 235) < 0.97:
        raise ValueError("Glyphs touch a nonuniform or nonwhite background; LaMa is required.")
    bright = samples[np.min(samples, axis=1) >= 235]
    if np.max(np.std(bright.astype(float), axis=0)) > 5:
        raise ValueError("Background varies too much for plain fill; LaMa is required.")
    return tuple(int(v) for v in np.median(bright, axis=0))


def _font(cfg: dict, text: str, size: int) -> ImageFont.ImageFont:
    path = cfg.get("font_path")
    if path:
        try:
            font = ImageFont.truetype(str(path), size, index=int(cfg.get("font_index", 0)))
        except (OSError, ValueError) as error:
            raise FontError(f"Cannot load configured font: {error}") from error
        # FreeType normally renders missing characters as the same .notdef glyph.
        missing = font.getmask(chr(0x10FFFF))
        fingerprint = (missing.size, bytes(missing))
        for char in set(text):
            if ord(char) > 127 and not char.isspace():
                glyph = font.getmask(char)
                if (glyph.size, bytes(glyph)) == fingerprint:
                    raise FontError(f"Configured font has no glyph for {char!r}.")
        return font
    if any(ord(char) > 127 for char in text):
        raise FontError("Configure font_path to a font with Traditional Chinese glyphs.")
    # Pillow's packaged scalable default is suitable for a Latin-only smoke demo.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _wrap(text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        tokens = re.findall(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*|[ \t]+|[^\s]", paragraph)
        for token in tokens:
            if font.getlength(current + token) <= width:
                current += token
                continue
            if current.strip():
                lines.append(current.rstrip())
                current = ""
            token = token.lstrip()
            for char in token:
                if current and font.getlength(current + char) > width:
                    lines.append(current)
                    current = ""
                current += char
        lines.append(current.rstrip())
    return lines


def _fit_text(text: str, size: tuple[int, int], box: tuple[int, int, int, int],
              allowed: np.ndarray, cfg: dict) -> tuple[Image.Image, dict]:
    padding = max(0, int(cfg.get("padding", 5)))
    usable_width = box[2] - box[0] - 2 * padding
    usable_height = box[3] - box[1] - 2 * padding
    if usable_width < 1 or usable_height < 1:
        raise ValueError("No text area remains after padding.")
    minimum = max(1, int(cfg.get("font_size_min", FONT_SIZE_MIN)))
    maximum = max(minimum, int(cfg.get("font_size_max", FONT_SIZE_MAX)))
    spacing = max(1.0, float(cfg.get("line_spacing", 1.15)))
    safe_area = _morph(allowed, padding, dilate=False)
    for font_size in range(maximum, minimum - 1, -1):
        font = _font(cfg, text, font_size)
        lines = _wrap(text, font, usable_width)
        dimensions = [font.getbbox(line or " ") for line in lines]
        heights = [max(1, bounds[3] - bounds[1]) for bounds in dimensions]
        line_height = max(heights + [font_size])
        gap = round(line_height * (spacing - 1))
        total_height = sum(heights) + gap * (len(lines) - 1)
        if total_height > usable_height:
            continue
        overlay = Image.new("RGBA", size)
        draw = ImageDraw.Draw(overlay)
        y = box[1] + (box[3] - box[1] - total_height) // 2
        for line, bounds, height in zip(lines, dimensions, heights):
            text_width = bounds[2] - bounds[0]
            x = box[0] + (box[2] - box[0] - text_width) // 2
            draw.text((x - bounds[0], y - bounds[1]), line, font=font, fill=(0, 0, 0, 255))
            y += height + gap
        ink = np.asarray(overlay.getchannel("A")) > 0
        if ink.any() and not np.any(ink & ~safe_area):
            return overlay, {"font_size": font_size, "lines": lines,
                             "text_bbox": list(overlay.getbbox()), "overflow_pixels": 0}
    raise ValueError("Translation does not fit the permitted text area at the minimum font size.")


@lru_cache(maxsize=1)
def _load_lama(path: str, device: str):
    import torch
    return torch.jit.load(path, map_location=device).eval()


def _lama_fill(original: Image.Image, mask: np.ndarray, options: dict) -> Image.Image:
    """Run explicitly supplied LaMa TorchScript weights on a padded local crop."""
    import torch
    import torch.nn.functional as functional

    path = options.get("torchscript_path")
    if not path or not Path(path).is_file():
        raise ValueError("LaMa requires an existing local inpaint.torchscript_path.")
    device = str(options.get("device", "cpu"))
    model = _load_lama(str(Path(path).resolve()), device)
    rows, columns = np.where(mask)
    margin = max(8, int(options.get("context_padding", 64)))
    box = (max(0, int(columns.min()) - margin), max(0, int(rows.min()) - margin),
           min(original.width, int(columns.max()) + 1 + margin),
           min(original.height, int(rows.max()) + 1 + margin))
    pixels = np.array(original.crop(box)).astype(np.float32) / 255
    local_mask = mask[box[1]:box[3], box[0]:box[2]]
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(local_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    height, width = pixels.shape[:2]
    pad = (0, (-width) % 8, 0, (-height) % 8)
    tensor = functional.pad(tensor, pad, mode="replicate")
    mask_tensor = functional.pad(mask_tensor, pad, mode="replicate")
    with torch.inference_mode():
        if options.get("signature", "image_mask") == "dict":
            result = model({"image": tensor, "mask": mask_tensor})
        else:
            result = model(tensor, mask_tensor)
    if isinstance(result, dict):
        result = result["inpainted"] if "inpainted" in result else result["image"]
    if isinstance(result, (tuple, list)):
        result = result[0]
    result = result[0, :, :height, :width].detach().float().cpu().numpy().transpose(1, 2, 0)
    if result.shape != pixels.shape or not np.isfinite(result).all():
        raise ValueError("LaMa returned an invalid image tensor.")
    output = original.copy()
    output.paste(Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8)), box[:2])
    return output


def render_page(page: dict, output_dir: Path, cfg: dict) -> dict:
    """Render approved regions, returning an explicit per-region QA report.

    ``cfg`` is the rendering section. ``inpaint`` may be a backend name or a
    mapping with ``backend`` (auto/plain/lama), local ``torchscript_path``, and
    optional ``device`` / ``signature``. Source and mask coordinates must refer
    to the original image. SFX preservation is intentional and is not a QA error.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_name(page["page_id"])
    output_path = output_dir / f"{name}.translated.png"
    source_path = Path(page["source_path"]).resolve()
    if output_path.resolve() == source_path:
        raise ValueError("Translated output would overwrite the source image; choose a separate output directory.")
    with Image.open(page["source_path"]) as image:
        original = image.convert("RGB")
    rgb = np.asarray(original)
    rendered = original.copy()
    occupied = np.zeros((original.height, original.width), dtype=bool)
    report: dict = {"page_id": page["page_id"], "issues": [], "regions": []}
    original_path = output_dir / f"{name}.original.png"
    if original_path.resolve() != source_path:
        original.save(original_path)
    report["original_path"] = str(original_path.resolve())
    inpaint = cfg.get("inpaint", {"backend": "auto"})
    options = {"backend": inpaint} if isinstance(inpaint, str) else dict(inpaint)
    backend = options.get("backend", "auto")
    mismatch = any(page.get(key, actual) != actual
                   for key, actual in (("width", original.width), ("height", original.height)))

    def fail(record: dict, code: str, message: str):
        record.update(status="needs_review", reason=message)
        report["issues"].append({"region_id": record["region_id"], "code": code, "message": message})

    for region in page.get("regions", []):
        record = {"region_id": region["region_id"], "status": "pending"}
        report["regions"].append(record)
        if region.get("kind") == "sfx" and cfg.get("sfx_policy", "preserve") == "preserve":
            record.update(status="preserved", reason="SFX intentionally preserved.")
            continue
        if mismatch:
            fail(record, "coordinate_mismatch", "Stored page dimensions differ from the source image.")
            continue
        translation = region.get("translation") or {}
        text = str(translation.get("text") or "").strip()
        if not text:
            fail(record, "missing_translation", "No translated text; original lettering retained.")
            continue
        uncertain = (region.get("uncertainty") or region.get("status") == "needs_review"
                     or translation.get("status") == "needs_review" or translation.get("issues"))
        if uncertain and not cfg.get("render_uncertain", False):
            fail(record, "uncertain_text", "Text requires review; original lettering retained.")
            continue
        if cfg.get("direction", "horizontal") != "horizontal":
            fail(record, "unsupported_direction", "This renderer supports horizontal Traditional Chinese typesetting.")
            continue
        try:
            text_box = _bbox(region["bbox"], *original.size)
            region_mask = _text_region_mask(region, text_box, original.size)
            bubble = (_read_mask(region["bubble_mask_path"], original.size)
                      if region.get("bubble_mask_path") else None)
            if region.get("bubble_bbox"):
                fit_box = _bbox(region["bubble_bbox"], *original.size)
            elif bubble is not None and bubble.any():
                ys, xs = np.where(bubble)
                fit_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            else:
                fit_box = text_box
            allowed = _rect_mask(fit_box, original.size)
            if bubble is not None:
                allowed &= bubble
            if not allowed.any():
                raise ValueError("Bubble mask does not contain a usable text area.")
        except (OSError, ValueError, KeyError, TypeError) as error:
            fail(record, "invalid_geometry", str(error))
            continue
        try:
            overlay, metrics = _fit_text(text, original.size, fit_box, allowed, cfg)
        except (OSError, ValueError) as error:
            code = "font_error" if isinstance(error, (FontError, OSError)) else "text_overflow"
            fail(record, code, str(error))
            continue
        try:
            if region.get("mask_path"):
                mask = _read_mask(region["mask_path"], original.size) & region_mask
                record["mask_source"] = "provided"
            else:
                mask = _infer_white_glyph_mask(rgb, region_mask)
                record["mask_source"] = "white_background_threshold"
            if not mask.any():
                raise ValueError("Glyph mask has no pixels inside the text region.")
            mask = _morph(mask, max(0, int(cfg.get("mask_dilation", 1))))
            if bubble is not None and np.any(mask & ~bubble):
                raise ValueError("Dilated glyph mask extends beyond the bubble boundary.")
        except (OSError, ValueError) as error:
            fail(record, "unsafe_mask", str(error))
            continue
        ink = np.asarray(overlay.getchannel("A")) > 0
        if np.any((mask | ink) & occupied):
            fail(record, "region_overlap", "Erasure or translated lettering overlaps an already rendered region.")
            continue
        try:
            if backend not in {"plain", "lama", "auto"}:
                raise ValueError(f"Unknown inpainting backend: {backend}")
            color = None
            if backend in {"plain", "auto"}:
                try:
                    color = _white_background(rgb, mask)
                except ValueError:
                    if backend == "plain" or not options.get("torchscript_path"):
                        raise
            if color is not None:
                cleaned = Image.new("RGB", original.size, color)
                record["inpaint_backend"] = "plain"
            else:
                cleaned = _lama_fill(original, mask, options)
                record["inpaint_backend"] = "lama"
        except Exception as error:
            fail(record, "inpaint_unavailable", str(error))
            continue
        rendered.paste(cleaned, (0, 0), Image.fromarray(mask.astype(np.uint8) * 255))
        rendered = Image.alpha_composite(rendered.convert("RGBA"), overlay).convert("RGB")
        occupied |= mask | ink
        record.update(metrics, status="auto_passed")
        if uncertain:
            fail(record, "uncertain_rendered", "Uncertain translation rendered by explicit render_uncertain setting.")
    temporary = output_path.with_suffix(".tmp.png")
    rendered.save(temporary)
    temporary.replace(output_path)
    report.update(output_path=str(output_path.resolve()),
                  status="needs_review" if report["issues"] else "auto_passed")
    return report


def export_cbz(pages: list[Path], output_path: Path) -> Path:
    """Export pages in supplied reading order with deterministic, safe ZIP names."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, path in enumerate(pages, 1):
            path = Path(path)
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                raise ValueError(f"Unsupported CBZ page format: {path.suffix}")
            info = zipfile.ZipInfo(f"{index:05d}{path.suffix.lower()}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, path.read_bytes())
    temporary.replace(output_path)
    return output_path
