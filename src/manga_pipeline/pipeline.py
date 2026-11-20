"""Stage orchestration. Heavy inference occurs only inside an explicit run."""
from __future__ import annotations

import copy
import csv
import io
import json
import re
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps

from .config import STAGES, digest
from .geometry import area, associate, checked_box, covered_fraction, denormalize, intersection, normalize, overlap, padded, polygon
from .storage import Store, natural_key, read_json, write_json
from .vlm import VLMError

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PREVIOUS = {"detect": "prepare", "audit": "detect", "ocr": "audit", "reconcile": "ocr", "translate": "reconcile", "render": "translate", "qa": "render"}


def is_page_name(name: str) -> bool:
    """An image page, not a folder, hidden file or macOS metadata such as __MACOSX/._001.jpg."""
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return (bool(parts) and not name.endswith("/") and PurePosixPath(parts[-1]).suffix.lower() in IMAGE_SUFFIXES
            and not any(part.startswith(".") or part == "__MACOSX" for part in parts))


def flag(region: dict, reason: str):
    if reason not in region.setdefault("uncertainty", []):
        region["uncertainty"].append(reason)
    region["status"] = "needs_review"


# Page work in these stages is independent and mostly waits on the VLM server, so
# pages run concurrently, up to the server's request slots (server.max_num_seqs).
CONCURRENT_STAGES = ("audit", "reconcile", "translate", "qa")


def normalized_text(text: str) -> str:
    # Preserve digits, punctuation, kana, negation, and full/half-width differences.
    return re.sub(r"\s+", "", text)


def readings_agree(readings: list[str]) -> bool:
    """At least two independent, nonempty readings that match apart from whitespace."""
    return len(readings) >= 2 and all(r.strip() for r in readings) and len({normalized_text(r) for r in readings}) == 1


def _merge_context(running: dict, reply: dict) -> dict:
    """Add one chunk's reply to the chapter context; a reply only covers the pages in view."""
    def key(text) -> str:
        return " ".join(str(text).split()).casefold()

    glossary = {key(item["source"]): item for item in running["glossary"]}
    for item in reply["glossary"]:
        known = glossary.get(key(item["source"]))
        if known is None:
            glossary[key(item["source"])] = item
            continue
        # The first accepted target stays, so a name keeps one rendering for the chapter.
        known["evidence"] = list(dict.fromkeys([*known["evidence"], *item["evidence"]]))[:5]
        if item["status"] == "confirmed":
            known["status"] = "confirmed"
    characters = {key(item["name"]): item for item in running["characters"]}
    characters.update({key(item["name"]): item for item in reply["characters"]})
    return {"summary": reply["summary"], "glossary": list(glossary.values()), "characters": list(characters.values())}


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.store = Store(cfg)
        self.demo = cfg.get("mode") == "demo"
        self._vlm = None
        self._detector = None
        self._detector_lock = threading.Lock()
        self._readers = {}
        self.overrides = {}
        override_path = cfg.get("review", {}).get("overrides_path")
        if override_path and Path(override_path).is_file():
            self.overrides = read_json(Path(override_path))
            if not isinstance(self.overrides, dict):
                raise ValueError("Review overrides must map region_id to edited fields")
            for edit in self.overrides.values():
                if not isinstance(edit, dict):
                    raise ValueError("Each region override must be an object")
                for key in ("mask_path", "bubble_mask_path"):
                    if edit.get(key):
                        edit[key] = str((Path(override_path).parent / edit[key]).resolve())

    @property
    def vlm(self):
        if self._vlm is None:
            if self.demo:
                from .demo import DemoVLM
                self._vlm = DemoVLM()
            else:
                from .vlm import VLMClient
                self._vlm = VLMClient(self.cfg["vlm"], self.store.root / "requests" / "qwen")
        return self._vlm

    @property
    def detector(self):
        if self._detector is None:
            from .adapters import build_detector
            self._detector = build_detector(self.cfg["detector"])
        return self._detector

    def close(self):
        for obj in [self._detector, *self._readers.values()]:
            if obj is not None and hasattr(obj, "close"):
                obj.close()
        self._detector = None
        self._readers = {}
        rendering = sys.modules.get(f"{__package__}.rendering")
        if rendering is not None:
            rendering.release_models()

    def run(self, stage="all", force=False):
        stages = STAGES if stage == "all" else (stage,)
        if any(s not in STAGES for s in stages):
            raise ValueError(f"Unknown stage {stage}")
        self.store.forget_verified()  # Files may have changed since a previous run.
        try:
            for current in stages:
                print(f"[{current}] mode={'SCRIPTED DEMO' if self.demo else 'real'}", flush=True)
                if current in ("reconcile", "translate", "render", "qa") and self.overrides:
                    self.validate_overrides()
                if current == "prepare":
                    self.prepare(force)
                elif current == "context":
                    self.context(force)
                elif current == "export":
                    self.export()
                else:
                    self.page_stages(current, force)
                # Do not keep OCR/layout networks resident between stages.
                self.close()
        finally:
            self.close()

    def page_ids(self):
        path = self.store.root / "manifest.json"
        if not path.is_file():
            raise ValueError("Run prepare first")
        manifest = read_json(path)
        if manifest["input"] != self.cfg["input"]:
            raise ValueError("Output folder belongs to a different input; choose another output")
        return manifest["page_ids"]

    def validate_overrides(self):
        valid = set()
        for page_id in self.page_ids():
            path = self.store.path(page_id, "audit")
            if not path.is_file():
                raise ValueError("Run audit before applying review overrides")
            valid.update(r["region_id"] for r in read_json(path)["regions"])
        unknown = set(self.overrides) - valid
        if unknown:
            raise ValueError(f"Review overrides reference unknown/grouped region IDs: {sorted(unknown)}")

    def prepare(self, force=False):
        source = Path(self.cfg["input"])
        records = []
        if source.is_file() and source.suffix.lower() in (".cbz", ".zip"):
            with zipfile.ZipFile(source) as zf:
                names = sorted([n for n in zf.namelist() if is_page_name(n)], key=natural_key)
                for name in names:
                    records.append((f"{source}!{name}", zf.read(name), None))
        elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            records = [(str(source), source.read_bytes(), source.with_suffix(".fixture.json"))]
        elif source.is_dir():
            for path in sorted([p for p in source.iterdir() if p.is_file() and is_page_name(p.name)], key=natural_key):
                records.append((str(path), path.read_bytes(), path.with_suffix(".fixture.json")))
        else:
            raise ValueError(f"Input must be an image folder, image, or CBZ/ZIP: {source}")
        if not records:
            raise ValueError(f"No images found in {source}")
        page_ids = []
        for number, (name, raw, fixture_path) in enumerate(records, 1):
            import hashlib
            page_id = f"p{number:04}"
            page_ids.append(page_id)
            fixture = read_json(fixture_path) if self.demo and fixture_path and fixture_path.is_file() else None
            inputs = {"source": name, "sha256": hashlib.sha256(raw).hexdigest(), "fixture": fixture}
            fingerprint = self.store.fingerprint("prepare", inputs)
            artifact = self.store.path(page_id, "prepare")
            if self.store.cached(artifact, fingerprint, force):
                continue
            directory = artifact.parent
            directory.mkdir(parents=True, exist_ok=True)
            original_path = directory / ("original" + Path(name).suffix.lower())
            original_path.write_bytes(raw)
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            canonical = directory / "source.png"
            image.save(canonical)
            page = {"schema_version": 1, "page_id": page_id, "source_name": name, "source_sha256": inputs["sha256"], "original_path": str(original_path), "source_path": str(canonical), "width": image.width, "height": image.height, "coordinate_system": "pixels in EXIF-oriented source.png", "regions": [], "layouts": [], "issues": [], "mode": "demo" if self.demo else "real"}
            if fixture:
                page["fixture"] = fixture
            self.store.save(artifact, page, fingerprint, "prepare", inputs)
        write_json(self.store.root / "manifest.json", {"schema_version": 1, "input": self.cfg["input"], "page_ids": page_ids, "mode": "demo" if self.demo else "real"})

    def page_stages(self, stage, force=False):
        page_ids = self.page_ids()
        workers = min(len(page_ids), int(self.cfg.get("server", {}).get("max_num_seqs", 1))) if stage in CONCURRENT_STAGES else 1
        if workers <= 1:
            for page_id in page_ids:
                self.page_stage(stage, page_id, force)
            return
        self.vlm  # create the shared client before worker threads start
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(self.page_stage, stage, page_id, force) for page_id in page_ids]
            try:
                for future in futures:
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    def page_stage(self, stage, page_id, force=False):
        previous = PREVIOUS[stage]
        page = copy.deepcopy(self.store.required(page_id, previous))
        inputs = {"artifacts": {str(self.store.path(page_id, previous)): self.store.hash(self.store.path(page_id, previous))}}
        if stage == "translate":
            context_path = self.store.root / "context.json"
            if not context_path.is_file():
                raise ValueError("Run context before translate")
            self.read_context()
            inputs["artifacts"][str(context_path)] = self.store.hash(context_path)
        fingerprint = self.store.fingerprint(stage, inputs)
        path = self.store.path(page_id, stage)
        if self.store.cached(path, fingerprint, force):
            print(f"  {page_id}: cached", flush=True)
            return
        page.pop("_meta", None)
        method = getattr(self, f"stage_{stage}")
        page = method(page)
        self.store.save(path, page, fingerprint, stage, inputs)
        print(f"  {page_id}: saved ({len(page['regions'])} regions)", flush=True)

    def _region(self, detection, page, number):
        box = checked_box(detection["bbox"], page["width"], page["height"])
        kind = detection.get("kind", "text")
        if kind not in ("text", "sfx", "bubble", "panel"):
            raise ValueError(f"Unsupported detection kind: {kind}")
        return {**detection, "region_id": f"{page['page_id']}_r{number:04}", "bbox": box, "polygon": detection.get("polygon") or polygon(box), "kind": kind, "language": self.cfg.get("source_language", "ja"), "status": "auto_passed", "uncertainty": [], "ocr_candidates": {}, "original_text": "", "orientation": "unknown", "coordinate_transform": {"crop_origin": box[:2], "scale": [1, 1]}}

    def stage_detect(self, page):
        if self.demo:
            fixture = page["fixture"]
            detections = [{"bbox": fixture["text_bbox"], "kind": "text", "score": 1.0}, {"bbox": fixture["bubble_bbox"], "kind": "bubble", "score": 1.0}]
        else:
            detections = self.detector.detect(Path(page["source_path"]), self.store.path(page["page_id"], "detect").parent / "masks")
            fallback = self.cfg["detector"].get("fallback", {})
            if fallback.get("enabled") and (fallback.get("always", False) or not any(d["kind"] == "text" for d in detections)):
                from .adapters import build_detector
                other = build_detector(fallback)
                try:
                    for item in other.detect(Path(page["source_path"]), self.store.path(page["page_id"], "detect").parent / "fallback_masks"):
                        if not any(d["kind"] == item["kind"] and overlap(d["bbox"], item["bbox"]) > 0.8 for d in detections):
                            detections.append(item)
                finally:
                    other.close()
        regions = [self._region(x, page, i) for i, x in enumerate(detections, 1)]
        page["layouts"] = [x for x in regions if x["kind"] in ("bubble", "panel")]
        page["regions"] = associate([x for x in regions if x["kind"] in ("text", "sfx")], page["layouts"], self.cfg.get("reading_direction", "rtl"))
        if not page["regions"]:
            page["issues"].append("detector_found_no_text")
        return page

    def _add_proposals(self, page, proposals, size, offset=(0, 0), source="qwen"):
        added = []
        existing_numbers = [int(r["region_id"].rsplit("r", 1)[-1]) for r in page["regions"] + page["layouts"]]
        next_number = max(existing_numbers, default=0) + 1
        for proposal in proposals:
            box = denormalize(proposal["bbox"], size[0], size[1], offset)
            # Require coverage of the PROPOSED region; a tiny contained box must
            # not hide a large new text proposal. Keep one-to-many matches.
            coverage = covered_fraction(box, [region["bbox"] for region in page["regions"]])
            if coverage >= self.cfg.get("audit", {}).get("match_overlap", 0.6):
                continue
            # Optional: a loose VLM box around text a detector already outlined is that text.
            contained = self.cfg.get("audit", {}).get("contained_overlap")
            if contained and any(intersection(box, region["bbox"]) >= contained * max(1, area(region["bbox"]))
                                 for region in page["regions"] if region["kind"] == "text"):
                continue
            region = self._region({"bbox": box, "kind": proposal.get("kind", "text"), "score": None, "detected_by": source}, page, next_number)
            flag(region, "vlm_only_region_requires_confirmation")
            page["regions"].append(region)
            added.append(region)
            next_number += 1
        return added

    def stage_audit(self, page):
        cfg = self.cfg.get("audit", {})
        if not cfg.get("enabled", True):
            page["coverage"] = {"status": "not_checked"}
            page["issues"].append("coverage_not_checked")
            # Grouping is geometric; it must not depend on the VLM coverage scan.
            self._group_bubble_text(page)
            return page
        image = Image.open(page["source_path"]).convert("RGB")
        scan_payload = {"page_id": page["page_id"], "known_regions": [], "attempt": 0}
        if self.demo:
            scan_payload["demo_regions"] = [{"bbox": normalize(page["fixture"]["text_bbox"], *image.size), "kind": "text", "reason": "synthetic fixture"}]
        result = self.vlm.ask("coverage", scan_payload, [image])
        detector_regions = list(page["regions"])
        scan_boxes = [denormalize(p["bbox"], *image.size) for p in result["regions"]]
        detector_only = [r for r in detector_regions if covered_fraction(r["bbox"], scan_boxes) < cfg.get("match_overlap", 0.6)]
        for region in detector_only:
            flag(region, "detector_only_region_not_confirmed_by_scan")
        added = self._add_proposals(page, result["regions"], image.size)
        scans = [{"source": "qwen", "attempt": 0, "result": result}]
        uncertain = result["uncertain"] or bool(detector_only)
        # Tile review can discover omissions shared by initial detector and VLM.
        retries = self.cfg.get("retries", {}).get("region_retries", 1)
        for attempt in range(retries):
            if not (added or uncertain or cfg.get("always_tile", False)):
                break
            cols, rows = cfg.get("tile_grid", [2, 2])
            if not 1 <= cols <= 8 or not 1 <= rows <= 8:
                raise ValueError("audit.tile_grid values must be 1..8")
            added = []
            uncertain = False
            for row in range(rows):
                for col in range(cols):
                    box = padded([col * image.width / cols, row * image.height / rows, (col + 1) * image.width / cols, (row + 1) * image.height / rows], image.width, image.height, cfg.get("tile_overlap_pixels", 32))
                    crop = image.crop(box)
                    scan = self.vlm.ask("coverage", {"page_id": page["page_id"], "known_regions": [], "attempt": attempt + 1}, [crop])
                    scans.append({"source": "qwen", "attempt": attempt + 1, "tile_bbox": box, "result": scan})
                    added.extend(self._add_proposals(page, scan["regions"], crop.size, box[:2]))
                    tile_boxes = [denormalize(p["bbox"], *crop.size, box[:2]) for p in scan["regions"]]
                    scan_boxes.extend(tile_boxes)
                    for region in detector_only:
                        if covered_fraction(region["bbox"], scan_boxes) >= cfg.get("match_overlap", 0.6):
                            region["uncertainty"] = [u for u in region["uncertainty"] if u != "detector_only_region_not_confirmed_by_scan"]
                            region["status"] = "needs_review" if region["uncertainty"] else "auto_passed"
                    uncertain |= scan["uncertain"]
        uncertain |= any("detector_only_region_not_confirmed_by_scan" in r["uncertainty"] for r in detector_only)
        if cfg.get("mangalmm", {}).get("enabled", False) and not self.demo:
            from .vlm import VLMClient
            auditor = VLMClient(cfg["mangalmm"], self.store.root / "requests" / "mangalmm")
            scan = auditor.ask("coverage", {"page_id": page["page_id"], "known_regions": [], "attempt": 0}, [image])
            self._add_proposals(page, scan["regions"], image.size, source="mangalmm")
            uncertain |= scan["uncertain"]
            scans.append({"source": "mangalmm", "result": scan})
        page["coverage"] = {"status": "needs_review" if uncertain else "auto_passed", "scans": scans, "note": "Agreement does not establish exhaustive coverage."}
        if uncertain:
            page["issues"].append("coverage_uncertain_after_retries")
        if not self.demo and cfg.get("detector_retry", False):
            self._confirm_proposals(page, image)
        page["regions"] = associate(page["regions"], page["layouts"], self.cfg.get("reading_direction", "rtl"))
        self._group_bubble_text(page)
        return page

    def _confirm_proposals(self, page, image):
        """Retry the detector on suspicious original-resolution crops and map masks back."""
        directory = self.store.path(page["page_id"], "audit").parent / "audit_crops"
        directory.mkdir(parents=True, exist_ok=True)
        for region in page["regions"]:
            if "vlm_only_region_requires_confirmation" not in region["uncertainty"]:
                continue
            box = padded(region["bbox"], *image.size, pixels=32)
            path = directory / f"{region['region_id']}.png"
            image.crop(box).save(path)
            with self._detector_lock:  # one local detector, shared by concurrent pages
                candidates = self.detector.detect(path, directory / region["region_id"])
            global_mask = np.zeros((image.height, image.width), dtype=bool)
            found = False
            for candidate in candidates:
                if candidate["kind"] != region["kind"]:
                    continue
                b = candidate["bbox"]
                global_box = [b[0] + box[0], b[1] + box[1], b[2] + box[0], b[3] + box[1]]
                if intersection(global_box, region["bbox"]) / max(1, area(region["bbox"])) < 0.6:
                    continue
                found = True
                if candidate.get("mask_path"):
                    with Image.open(candidate["mask_path"]) as mask:
                        # Union: a later candidate must not erase an earlier candidate's glyphs.
                        global_mask[box[1]:box[3], box[0]:box[2]] |= np.asarray(mask.convert("L")) > 127
            if found:
                region["uncertainty"].remove("vlm_only_region_requires_confirmation")
                region["status"] = "needs_review" if region["uncertainty"] else "auto_passed"
                if global_mask.any():
                    mask_path = directory / f"{region['region_id']}.full_mask.png"
                    Image.fromarray(global_mask.astype("uint8") * 255).save(mask_path)
                    region["mask_path"] = str(mask_path)

    def _group_bubble_text(self, page):
        groups = {}
        for region in page["regions"]:
            if region["kind"] == "text" and region.get("bubble_id"):
                groups.setdefault(region["bubble_id"], []).append(region)
        removed = set()
        for members in groups.values():
            if len(members) < 2:
                continue
            representative = members[0]
            # Keep constituent region IDs and geometry for audits/review.
            representative["member_regions"] = copy.deepcopy(members)
            boxes = [r["bbox"] for r in members]
            representative["bbox"] = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
            representative["polygon"] = polygon(representative["bbox"])
            representative["uncertainty"] = sorted({u for r in members for u in r["uncertainty"]})
            representative["status"] = "needs_review" if representative["uncertainty"] else "auto_passed"
            if all(r.get("mask_path") for r in members):
                mask = np.zeros((page["height"], page["width"]), dtype=bool)
                for region in members:
                    mask |= np.asarray(Image.open(region["mask_path"]).convert("L")) > 127
                path = self.store.path(page["page_id"], "audit").parent / f"{representative['region_id']}.group_mask.png"
                Image.fromarray(mask.astype("uint8") * 255).save(path)
                representative["mask_path"] = str(path)
            else:
                representative["mask_path"] = None
            removed.update(r["region_id"] for r in members[1:])
        page["regions"] = [r for r in page["regions"] if r["region_id"] not in removed]

    def _reader(self, name):
        if name not in self._readers:
            from .adapters import build_reader
            self._readers[name] = build_reader(self.cfg["ocr"]["readers"][name])
        return self._readers[name]

    def stage_ocr(self, page):
        image = Image.open(page["source_path"]).convert("RGB")
        cfg = self.cfg.get("ocr", {})
        for region in page["regions"]:
            if region["kind"] == "sfx" and self.cfg.get("render", {}).get("sfx_policy", "preserve") == "preserve":
                region["status"] = "preserved"
                continue
            box = padded(region["bbox"], *image.size, pixels=cfg.get("crop_padding", 8))
            crop = image.crop(box)
            crop_path = self.store.path(page["page_id"], "ocr").parent / "crops" / f"{region['region_id']}.png"
            crop_path.parent.mkdir(exist_ok=True)
            crop.save(crop_path)
            region["crop_path"] = str(crop_path)
            region["crop_bbox"] = box
            region["coordinate_transform"] = {"crop_origin": box[:2], "scale": [1, 1]}
            language = region["language"]
            names = cfg.get("routes", {}).get(language)
            if not names:
                raise ValueError(f"No OCR route configured for language '{language}'")
            for name in names:
                model_cfg = cfg["readers"][name]
                text = page["fixture"]["original_text"] if self.demo else self._reader(name).read(crop)
                if not isinstance(text, str):
                    raise ValueError(f"Reader {name} returned non-text output")
                region["ocr_candidates"][name] = {"text": text, "model_id": model_cfg.get("model_id", name), "revision": model_cfg.get("revision"), "independent_input": True}
            if self.demo:
                region["demo_translation"] = page["fixture"]["translation"]
        return page

    def stage_reconcile(self, page):
        image = Image.open(page["source_path"]).convert("RGB")
        policy = self.cfg.get("ocr", {}).get("qwen_blind_read", "disagreements")
        for region in page["regions"]:
            if region.get("status") == "preserved":
                continue
            override = self.overrides.get(region["region_id"], {})
            if "original_text" in override:
                if not isinstance(override["original_text"], str) or not override["original_text"].strip():
                    raise ValueError("Manual original_text must be nonempty text")
                region["original_text"] = override["original_text"]
                region["status"] = "human_reviewed"
                region["resolved_uncertainty"] = region["uncertainty"]
                region["uncertainty"] = []
                continue
            candidates = {k: v["text"] for k, v in region["ocr_candidates"].items()}
            values = list(candidates.values())
            agreement = readings_agree(values)
            crop = image.crop(region["crop_bbox"])
            if policy == "always" or not agreement:
                response = self.vlm.ask("ocr", {"region_id": region["region_id"], "language": region["language"]}, [crop])
                candidates["qwen_blind"] = response["text"]
                region["ocr_candidates"]["qwen_blind"] = {"text": response["text"], "model_id": self.cfg["vlm"]["model"], "independent_input": True, "uncertain": response["uncertain"], "reason": response["reason"]}
                # The blind read is itself an independent reading. On a single-reader
                # route (en/auto/mixed) it is the second reading that can confirm the first.
                agreement = not response["uncertain"] and readings_agree([*values, response["text"]])
            if agreement:
                region["original_text"] = values[0]
                region["ocr_decision"] = "independent_transcriptions_agree"
            else:
                budget = self.cfg.get("retries", {}).get("ocr_retries", 1)
                response = None
                attempts = []
                for attempt in range(budget + 1):
                    # Re-crop from original pixels, never generative super-resolution.
                    box = padded(region["bbox"], *image.size, pixels=8 + attempt * 12)
                    retry_crop = image.crop(box)
                    surrounding = image.crop(padded(region["bbox"], *image.size, pixels=96))
                    if attempt:
                        reread = self.vlm.ask("ocr", {"region_id": region["region_id"], "language": region["language"]}, [retry_crop])
                        candidates[f"qwen_retry_{attempt}"] = reread["text"]
                    response = self.vlm.ask("reconcile", {"region_id": region["region_id"], "language": region["language"], "candidates": candidates}, [retry_crop, surrounding])
                    attempts.append(response)
                    if not response["uncertain"] and response["text"].strip():
                        break
                region["original_text"] = response["text"]
                region["ocr_review"] = attempts
                region["ocr_decision"] = "vlm_arbitrated"
                # Arbitration is not independent verification; keep configurable review gate.
                if response["uncertain"] or self.cfg.get("review", {}).get("flag_arbitrated_ocr", True):
                    flag(region, "ocr_disagreement_" + ("unresolved" if response["uncertain"] else "arbitrated"))
            if not region["original_text"].strip():
                flag(region, "empty_transcription")
        active = [r for r in page["regions"] if r.get("status") != "preserved"]
        if active:
            try:
                order = self.vlm.ask("reading_order", {"regions": [{"region_id": r["region_id"], "bbox": normalize(r["bbox"], *image.size), "original_text": r["original_text"], "bubble_id": r.get("bubble_id"), "panel_id": r.get("panel_id")} for r in active], "direction": self.cfg.get("reading_direction", "rtl")}, [image])
            except VLMError as error:
                # Ordering is advisory: keep the geometric order and send the page to review.
                order = {"region_ids": [r["region_id"] for r in active], "uncertain": True, "reason": f"Kept geometric order: {error}"}
            if sorted(order["region_ids"]) != sorted(r["region_id"] for r in active):
                raise ValueError("Reading order must contain every region ID exactly once")
            for i, region_id in enumerate(order["region_ids"]):
                region = next(r for r in active if r["region_id"] == region_id)
                region["reading_order"] = i
                if order["uncertain"]:
                    flag(region, "reading_order_uncertain")
            page["reading_order_review"] = order
        page["regions"].sort(key=lambda r: r.get("reading_order", 0))
        return page

    def context(self, force=False):
        pages = [self.store.required(p, "reconcile") for p in self.page_ids()]
        inputs = {"artifacts": {str(self.store.path(p["page_id"], "reconcile")): self.store.hash(self.store.path(p["page_id"], "reconcile")) for p in pages}}
        fingerprint = self.store.fingerprint("context", inputs)
        path = self.store.root / "context.json"
        if self.store.cached(path, fingerprint, force):
            print("  chapter: cached", flush=True)
            return
        size = self.cfg.get("context", {}).get("pages_per_chunk", 3)
        if not isinstance(size, int) or not 1 <= size <= 20:
            raise ValueError("context.pages_per_chunk must be 1..20")
        context = {"summary": "", "glossary": [], "characters": []}
        chunks = []
        for start in range(0, len(pages), size):
            chunk = [{"page_id": p["page_id"], "regions": [{"region_id": r["region_id"], "original_text": r["original_text"], "uncertainty": r["uncertainty"]} for r in p["regions"] if r.get("status") != "preserved"]} for p in pages[start:start + size]]
            reply = self.vlm.ask("context", {"pages": chunk, "previous_context": context})
            # Bound future prompts; full chunk outputs remain in the artifact.
            chunks.append(copy.deepcopy(reply))
            context = _merge_context(context, reply)
            context["summary"] = context["summary"][:self.cfg.get("context", {}).get("max_summary_chars", 4000)]
            context["glossary"] = context["glossary"][:self.cfg.get("context", {}).get("max_glossary_entries", 100)]
            context["characters"] = context["characters"][:self.cfg.get("context", {}).get("max_characters", 40)]
        self.store.save(path, {"context": context, "chunks": chunks}, fingerprint, "context", inputs)

    def read_context(self):
        # Verifies the context and, transitively, the OCR of every page it was built from.
        return self.store.required_path(self.store.root / "context.json")["context"]

    def stage_translate(self, page):
        image = Image.open(page["source_path"]).convert("RGB")
        context = self.read_context()
        active = [r for r in page["regions"] if r.get("status") != "preserved" and r["original_text"].strip()]
        pending = []
        for region in active:
            override = self.overrides.get(region["region_id"], {})
            if "translation" in override:
                if not isinstance(override["translation"], str) or not override["translation"].strip():
                    raise ValueError("Manual translation must be nonempty text")
                region["translation"] = {"text": override["translation"], "status": "human_reviewed", "issues": [], "origin": "review_overrides"}
            else:
                pending.append(region)
        history = []
        prior_issues = {}
        for attempt in range(self.cfg.get("retries", {}).get("translation_retries", 1) + 1):
            if not pending:
                break
            payload_regions = [{"region_id": r["region_id"], "original_text": r["original_text"], "uncertainty": r["uncertainty"], **({"demo_translation": r["demo_translation"]} if self.demo else {})} for r in pending]
            response = self.vlm.ask("translate", {"page_id": page["page_id"], "regions": payload_regions, "context": context, "prior_issues": prior_issues}, [image])
            translations = {r["region_id"]: r for r in response["translations"]}
            if len(translations) != len(response["translations"]) or set(translations) != {r["region_id"] for r in pending}:
                raise ValueError("Translation must contain each requested region exactly once")
            verification_regions = []
            for region in pending:
                item = translations[region["region_id"]]
                region["translation"] = {"text": item["text"], "status": "needs_review" if item["uncertain"] else "auto_passed", "issues": [item["reason"]] if item["uncertain"] else [], "attempt": attempt}
                verification_regions.append({"region_id": region["region_id"], "original_text": region["original_text"], "translation": item["text"]})
            verdict = self.vlm.ask("verify", {"page_id": page["page_id"], "regions": verification_regions, "context": context}, [image])
            checks = {r["region_id"]: r for r in verdict["checks"]}
            if len(checks) != len(verdict["checks"]) or set(checks) != {r["region_id"] for r in pending}:
                raise ValueError("Verification must check each requested region exactly once")
            retry = []
            for region in pending:
                check = checks[region["region_id"]]
                translation = region["translation"]
                if not check["passed"] or check["issues"] or not translation["text"].strip() or translation["status"] == "needs_review":
                    translation["status"] = "needs_review"
                    translation["issues"].extend(check["issues"] or ["verification_failed_or_uncertain"])
                    prior_issues[region["region_id"]] = translation["issues"]
                    retry.append(region)
                elif region["uncertainty"]:
                    translation["status"] = "needs_review"
                    translation["issues"].append("source_requires_review")
            history.append({"attempt": attempt, "translation": response, "verification": verdict})
            pending = retry
        page["translation_history"] = history
        return page

    def stage_render(self, page):
        from .rendering import render_page
        for region in page["regions"]:
            edit = self.overrides.get(region["region_id"], {})
            for key in ("mask_path", "bubble_bbox", "bubble_mask_path"):
                if key in edit:
                    region[key] = edit[key]
        directory = self.store.root / "rendered" / "initial"
        page["render"] = render_page(page, directory, self.cfg.get("render", {}))
        return page

    def stage_qa(self, page):
        from .rendering import FONT_SIZE_MAX, FONT_SIZE_MIN, render_page
        cfg = self.cfg.get("qa", {})
        page["visual_qa"] = {"status": "not_checked", "attempts": []}
        if not cfg.get("visual_enabled", True):
            page["issues"].append("visual_qa_not_checked")
            return page
        original = Image.open(page["source_path"]).convert("RGB")
        render_cfg = copy.deepcopy(self.cfg.get("render", {}))
        for attempt in range(self.cfg.get("retries", {}).get("layout_retries", 1) + 1):
            rendered = Image.open(page["render"]["output_path"]).convert("RGB")
            rendered_regions = {r["region_id"]: r for r in page["render"].get("regions", [])}
            regions = [{"region_id": r["region_id"], "bbox": normalize(r["bbox"], *original.size), "translation": r.get("translation", {}).get("text", ""), "sfx_policy": render_cfg.get("sfx_policy", "preserve"), "status": r.get("status"), "render_result": rendered_regions.get(r["region_id"], {}), "render_intent": "preserve_sfx" if r.get("status") == "preserved" else "translate_or_flag_if_skipped"} for r in page["regions"]]
            verdict = self.vlm.ask("visual_qa", {"page_id": page["page_id"], "regions": regions, "deterministic_issues": page["render"]["issues"]}, [original, rendered])
            valid_ids = {"page", *[r["region_id"] for r in page["regions"]]}
            if any(issue["region_id"] not in valid_ids for issue in verdict["issues"]):
                raise ValueError("Visual QA referenced an unknown region ID")
            page["visual_qa"]["attempts"].append(verdict)
            if verdict["passed"] and not verdict["issues"] and not page["render"]["issues"]:
                page["visual_qa"]["status"] = "auto_passed"
                break
            page["visual_qa"]["status"] = "needs_review"
            if attempt >= self.cfg.get("retries", {}).get("layout_retries", 1):
                break
            categories = {issue["category"] for issue in verdict["issues"]}
            adjusted = copy.deepcopy(render_cfg)
            if "mask" in categories:
                # Grow the erasure by one pixel up to 3, but never shrink a configured dilation.
                dilation = int(render_cfg.get("mask_dilation", 1))
                adjusted["mask_dilation"] = max(dilation, min(3, dilation + 1))
            if "layout" in categories:
                minimum = int(render_cfg.get("font_size_min", FONT_SIZE_MIN))
                maximum = int(render_cfg.get("font_size_max", FONT_SIZE_MAX))
                adjusted["font_size_max"] = min(maximum, max(minimum, int(maximum * 0.85)))
            if adjusted == render_cfg:
                break  # Nothing left to adjust; an identical re-render would prove nothing.
            render_cfg = adjusted
            # Re-run cleaning AND typesetting from original for either feedback class.
            page["render"] = render_page(page, self.store.root / "rendered" / "qa" / f"attempt-{attempt + 1:02}", render_cfg)
        return page

    def export(self):
        from .rendering import export_cbz
        pages = [self.store.required(p, "qa") for p in self.page_ids()]
        directory = self.store.root / "export"
        directory.mkdir(exist_ok=True)
        review = []
        for page in pages:
            for region in page["regions"]:
                issues = list(region.get("uncertainty", [])) + region.get("translation", {}).get("issues", [])
                issues.extend(i.get("code", i.get("message", "render_issue")) for i in page["render"]["issues"] if i.get("region_id") == region["region_id"])
                for attempt in page.get("visual_qa", {}).get("attempts", [])[-1:]:
                    issues.extend(i["reason"] for i in attempt["issues"] if i.get("region_id") == region["region_id"])
                if issues:
                    review.append({"page_id": page["page_id"], "region_id": region["region_id"], "original_text": region["original_text"], "translation": region.get("translation", {}).get("text", ""), "issues": issues})
            if page["issues"] or page.get("visual_qa", {}).get("status") != "auto_passed":
                page_reasons = [i["reason"] for a in page.get("visual_qa", {}).get("attempts", [])[-1:] for i in a["issues"] if i["region_id"] == "page"]
                review.append({"page_id": page["page_id"], "region_id": "", "original_text": "", "translation": "", "issues": page["issues"] + page_reasons + (["page_visual_qa_requires_review"] if page.get("visual_qa", {}).get("status") != "auto_passed" else [])})
        # No final document is labelled human-approved by an automated pass.
        status = "needs_review" if review else "auto_passed"
        result = {"schema_version": 1, "mode": "demo" if self.demo else "real", "status": status, "pages": pages, "context": self.read_context(), "review_items": review}
        write_json(directory / "chapter.json", result)
        write_json(directory / "review.json", review)
        with (directory / "review.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["page_id", "region_id", "original_text", "translation", "issues"])
            writer.writeheader()
            writer.writerows({**r, "issues": "; ".join(r["issues"])} for r in review)
        if self.cfg.get("export", {}).get("cbz", True):
            export_cbz([Path(p["source_path"]) for p in pages], directory / "original.cbz")
            name = "translated.demo.cbz" if self.demo else "translated.cbz"
            export_cbz([Path(p["render"]["output_path"]) for p in pages], directory / name)
        print(f"  export: {directory} ({status}, {len(review)} review items)", flush=True)
        return result
