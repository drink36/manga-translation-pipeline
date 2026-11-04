"""Atomic artifacts, provenance fingerprints, and immutable source copies."""
from __future__ import annotations

import hashlib
import json
import re
from functools import cache
from pathlib import Path

from .config import digest


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@cache
def code_hashes() -> dict[str, str]:
    """Hash the package once per process: the code that was loaded is the code that runs."""
    return {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


# VLM client settings that change how a request is delivered, not what the model answers.
TRANSPORT_KEYS = frozenset({"endpoint", "api_key_env", "timeout_seconds", "request_retries"})


def _without_transport(section: dict) -> dict:
    return {key: value for key, value in section.items() if key not in TRANSPORT_KEYS}


def _signature(path: Path) -> tuple[int, ...]:
    # Changes whenever a file is replaced or rewritten.
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


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
        # Without these memos every stage re-verifies the whole chapter for every page
        # (each page's translation depends on context.json, which depends on every page),
        # re-hashing the same images quadratically. Hashes are keyed by file identity and
        # modification time. A verified artifact is forgotten when it, or any artifact it
        # depends on, is saved, and everything is re-verified at the start of each run.
        self._hashes: dict[tuple[int, ...], str] = {}
        self._verified: dict[Path, tuple[tuple[int, ...], frozenset[Path]]] = {}

    def hash(self, path: Path) -> str:
        """SHA-256 of a file, computed once while the file is unchanged on disk."""
        key = _signature(path)
        if key not in self._hashes:
            self._hashes[key] = file_hash(Path(path))
        return self._hashes[key]

    def forget_verified(self) -> None:
        self._verified.clear()

    def _forget(self, path: Path) -> None:
        self._verified = {key: memo for key, memo in self._verified.items() if key != path and path not in memo[1]}

    def path(self, page_id: str, stage: str) -> Path:
        return self.root / "pages" / page_id / f"{stage}.json"

    def fingerprint(self, stage: str, inputs) -> str:
        # Include code in fingerprints so changed processing cannot reuse old artifacts.
        keys = {
            "prepare": [], "detect": ["detector", "source_language", "reading_direction"],
            "audit": ["audit", "detector", "vlm", "reading_direction"], "ocr": ["ocr", "render"],
            "reconcile": ["ocr", "vlm", "reading_direction"], "context": ["context", "vlm"],
            "translate": ["vlm"], "render": ["render"], "qa": ["qa", "render", "vlm"], "export": ["export"],
        }.get(stage, [])
        cfg = {k: self.config.get(k) for k in ["mode", "input", "output", *keys]}
        # Raising a timeout or moving the server must not discard hours of cached model work.
        if isinstance(cfg.get("vlm"), dict):
            cfg["vlm"] = _without_transport(cfg["vlm"])
        if isinstance(cfg.get("audit"), dict) and isinstance(cfg["audit"].get("mangalmm"), dict):
            cfg["audit"] = {**cfg["audit"], "mangalmm": _without_transport(cfg["audit"]["mangalmm"])}
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
            external = {p: self.hash(Path(p)) for p in paths if p and Path(p).is_file()}
        return digest({"stage": stage, "config": cfg, "inputs": inputs, "overrides": override_hash, "external": external, "code": code_hashes()})

    def cached(self, path: Path, fingerprint: str, force=False):
        if force or not path.is_file():
            return None
        value = read_json(path)
        if value.get("_meta", {}).get("fingerprint") != fingerprint:
            return None
        # Artifact's referenced images/masks must still exist.
        refs = value.get("_meta", {}).get("files", {})
        if any(not Path(p).is_file() or self.hash(Path(p)) != h for p, h in refs.items()):
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
        value["_meta"] = {"fingerprint": fingerprint, "stage": stage, "mode": self.config.get("mode", "real"), "inputs": inputs, "files": {p: self.hash(Path(p)) for p in sorted(refs)}}
        write_json(path, value)
        self._forget(Path(path))
        return value

    def required(self, page_id: str, stage: str):
        path = self.path(page_id, stage)
        return self.required_path(path)

    def required_path(self, path: Path, visited=None):
        path = Path(path)
        visited = set() if visited is None else visited
        if path in visited:
            raise ValueError(f"Cyclic artifact dependency at {path}")
        visited = {*visited, path}
        if not path.is_file():
            raise ValueError(f"Run prerequisite stage first: missing {path}")
        signature = _signature(path)
        value = read_json(path)
        if self._verified.get(path, (None,))[0] == signature:
            return value
        meta = value.get("_meta", {})
        stage = meta.get("stage", path.stem)
        inputs = meta.get("inputs")
        dependencies: set[Path] = set()
        if isinstance(inputs, dict):
            for dep, expected in inputs.get("artifacts", {}).items():
                dep = Path(dep)
                if not dep.is_file() or self.hash(dep) != expected:
                    raise ValueError(f"Stale prerequisite {path}; rerun stage '{stage}'")
                self.required_path(dep, visited)
                dependencies |= {dep, *self._verified[dep][1]}
        if self.cached(path, self.fingerprint(stage, inputs)) is None:
            raise ValueError(f"Config, code or source changed; rerun stage '{stage}' (or run --stage all)")
        self._verified[path] = (signature, frozenset(dependencies))
        return value
