"""Atomic artifacts, provenance fingerprints, and immutable source copies."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .config import digest


ALL_MODULES = None
_BASE = {"__init__.py", "config.py", "geometry.py", "pipeline.py", "storage.py"}
_MODEL = {"adapters.py", "demo.py", "prompts.py", "vlm.py"}
# Modules whose code can change a stage's artifact (pipeline.py holds every stage).
STAGE_MODULES = {
    "prepare": _BASE,
    "detect": _BASE | {"adapters.py"},
    "audit": _BASE | _MODEL,
    "ocr": _BASE | _MODEL,
    "reconcile": _BASE | _MODEL,
    "context": _BASE | _MODEL,
    "translate": _BASE | _MODEL,
    "render": _BASE | {"rendering.py"},
    "qa": _BASE | _MODEL | {"rendering.py"},
    "export": _BASE | {"rendering.py"},
}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def natural_key(path):
    return [(0, int(x)) if x.isdigit() else (1, x.casefold()) for x in re.split(r"(\d+)", str(path))]


class Store:
    def __init__(self, config: dict):
        self.config = config
        self.root = Path(config["output"])
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, page_id: str, stage: str) -> Path:
        return self.root / "pages" / page_id / f"{stage}.json"

    def fingerprint(self, stage: str, inputs) -> str:
        # Include the code each stage runs, so changed processing cannot reuse old
        # artifacts, while e.g. a renderer change leaves OCR and translation reusable.
        names = STAGE_MODULES.get(stage, ALL_MODULES)
        modules = sorted(p for p in Path(__file__).parent.glob("*.py") if names is ALL_MODULES or p.name in names)
        keys = {
            "prepare": [], "detect": ["detector", "source_language", "reading_direction"],
            "audit": ["audit", "detector", "vlm", "reading_direction"], "ocr": ["ocr", "render"],
            "reconcile": ["ocr", "vlm", "reading_direction"], "context": ["context", "vlm"],
            "translate": ["vlm"], "render": ["render"], "qa": ["qa", "render", "vlm"], "export": ["export"],
        }.get(stage, [])
        cfg = {k: self.config.get(k) for k in ["mode", "input", "output", *keys]}
        if stage == "ocr":
            cfg["render"] = {"sfx_policy": self.config.get("render", {}).get("sfx_policy", "preserve")}
        retry_key = {"audit": "region_retries", "reconcile": "ocr_retries", "translate": "translation_retries", "qa": "layout_retries"}.get(stage)
        if retry_key:
            cfg["retries"] = self.config.get("retries", {}).get(retry_key, 1)
        if stage == "reconcile":
            cfg["flag_arbitrated_ocr"] = self.config.get("review", {}).get("flag_arbitrated_ocr", True)
        override = self.config.get("review", {}).get("overrides_path")
        edits = read_json(Path(override)) if override and Path(override).is_file() else {}
        fields = {"reconcile": ["original_text"], "translate": ["translation"], "render": ["mask_path", "bubble_bbox", "bubble_mask_path"]}.get(stage, [])
        relevant = {rid: {k: value for k, value in edit.items() if k in fields} for rid, edit in edits.items() if any(k in edit for k in fields)}
        override_hash = digest(relevant)
        external = {}
        if stage in ("render", "qa"):
            render = self.config.get("render", {})
            paths = [render.get("font_path"), render.get("inpaint", {}).get("torchscript_path")]
            for edit in relevant.values():
                paths.extend(str((Path(override).parent / edit[k]).resolve()) for k in ("mask_path", "bubble_mask_path") if edit.get(k))
            external = {p: file_hash(Path(p)) for p in paths if p and Path(p).is_file()}
        return digest({"stage": stage, "config": cfg, "inputs": inputs, "overrides": override_hash, "external": external, "code": {p.name: file_hash(p) for p in modules}})

    def cached(self, path: Path, fingerprint: str, force=False):
        if force or not path.is_file():
            return None
        value = read_json(path)
        if value.get("_meta", {}).get("fingerprint") != fingerprint:
            return None
        # Artifact's referenced images/masks must still exist.
        refs = value.get("_meta", {}).get("files", {})
        if any(not Path(p).is_file() or file_hash(Path(p)) != h for p, h in refs.items()):
            return None
        return value

    def save(self, path: Path, value: dict, fingerprint: str, stage: str, inputs=None):
        refs = set()
        def collect(obj):
            if isinstance(obj, dict):
                for key, val in obj.items():
                    if key in ("source_path", "original_path", "mask_path", "bubble_mask_path", "output_path", "crop_path") and isinstance(val, str) and Path(val).is_file():
                        refs.add(val)
                    elif key != "_meta":
                        collect(val)
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)
        collect(value)
        value["_meta"] = {"fingerprint": fingerprint, "stage": stage, "mode": self.config.get("mode", "real"), "inputs": inputs, "files": {p: file_hash(Path(p)) for p in sorted(refs)}}
        write_json(path, value)
        return value

    def required(self, page_id: str, stage: str):
        path = self.path(page_id, stage)
        return self.required_path(path)

    def required_path(self, path: Path, visited=None):
        visited = set() if visited is None else visited
        if path in visited:
            raise ValueError(f"Cyclic artifact dependency at {path}")
        visited = {*visited, path}
        if not path.is_file():
            raise ValueError(f"Run prerequisite stage first: missing {path}")
        value = read_json(path)
        meta = value.get("_meta", {})
        stage = meta.get("stage", path.stem)
        inputs = meta.get("inputs")
        if isinstance(inputs, dict):
            for dep, expected in inputs.get("artifacts", {}).items():
                if not Path(dep).is_file() or file_hash(Path(dep)) != expected:
                    raise ValueError(f"Stale prerequisite {path}; rerun stage '{stage}'")
                self.required_path(Path(dep), visited)
        if self.cached(path, self.fingerprint(stage, inputs)) is None:
            raise ValueError(f"Config, code or source changed; rerun stage '{stage}' (or run --stage all)")
        return value
