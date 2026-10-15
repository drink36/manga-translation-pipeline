"""Page-coordinate geometry. VLM coordinates are always normalized to 0..1000."""
from __future__ import annotations

import math


def checked_box(box, width: int, height: int) -> list[int]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("A box must have four coordinates")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box):
        raise ValueError("Box coordinates must be finite numbers")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Inverted or empty box: {box}")
    out = [max(0, int(math.floor(x1))), max(0, int(math.floor(y1))), min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))]
    if out[2] <= out[0] or out[3] <= out[1]:
        raise ValueError(f"Box outside page: {box}")
    return out


def polygon(box):
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def overlap(a, b):
    return intersection(a, b) / max(1, min(area(a), area(b)))


def iou(a, b):
    common = intersection(a, b)
    return common / max(1, area(a) + area(b) - common)


def covered_fraction(target, boxes):
    """Exact rectangle-union coverage; duplicate detections cannot inflate it."""
    clipped = [[max(target[0], b[0]), max(target[1], b[1]), min(target[2], b[2]), min(target[3], b[3])] for b in boxes if intersection(target, b) > 0]
    xs = sorted({x for b in clipped for x in (b[0], b[2])})
    total = 0
    for left, right in zip(xs, xs[1:]):
        spans = sorted((b[1], b[3]) for b in clipped if b[0] < right and b[2] > left)
        length, end = 0, float("-inf")
        for start, stop in spans:
            length += max(0, stop - max(start, end))
            end = max(end, stop)
        total += (right - left) * length
    return total / max(1, area(target))


def normalize(box, width, height):
    return [round(box[0] * 1000 / width), round(box[1] * 1000 / height), round(box[2] * 1000 / width), round(box[3] * 1000 / height)]


def denormalize(box, width, height, offset=(0, 0)):
    if len(box) != 4 or any(not isinstance(x, (float, int)) or isinstance(x, bool) or not math.isfinite(x) or x < 0 or x > 1000 for x in box):
        raise ValueError("VLM boxes must be in 0..1000 image coordinates")
    local = checked_box([box[0] * width / 1000, box[1] * height / 1000, box[2] * width / 1000, box[3] * height / 1000], width, height)
    return [local[0] + offset[0], local[1] + offset[1], local[2] + offset[0], local[3] + offset[1]]


def padded(box, width, height, pixels=12):
    return checked_box([box[0] - pixels, box[1] - pixels, box[2] + pixels, box[3] + pixels], width, height)


def associate(regions: list[dict], layouts: list[dict], direction="rtl") -> list[dict]:
    """Associate by containment; keep separate source regions to avoid merging speakers."""
    for region in regions:
        for kind in ("bubble", "panel"):
            choices = [x for x in layouts if x["kind"] == kind and intersection(region["bbox"], x["bbox"]) / max(1, area(region["bbox"])) > 0.65]
            if choices:
                parent = min(choices, key=lambda x: area(x["bbox"]))
                region[f"{kind}_id"] = parent["region_id"]
                region[f"{kind}_bbox"] = parent["bbox"]
                if parent.get("mask_path"):
                    region[f"{kind}_mask_path"] = parent["mask_path"]
    # Geometric fallback only. Qwen can supply a validated permutation later.
    ordered = sorted(regions, key=lambda r: (r.get("panel_bbox", r["bbox"])[1], -r.get("panel_bbox", r["bbox"])[0] if direction == "rtl" else r.get("panel_bbox", r["bbox"])[0], r["bbox"][1], -r["bbox"][0] if direction == "rtl" else r["bbox"][0]))
    for index, region in enumerate(ordered):
        region["reading_order"] = index
    return ordered
