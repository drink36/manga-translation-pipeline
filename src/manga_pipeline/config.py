"""Configuration loading, with paths resolved relative to the YAML file."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

STAGES = ("prepare", "detect", "audit", "ocr", "reconcile", "context", "translate", "render", "qa", "export")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_config(path: str | Path, _seen=None) -> dict:
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Cyclic config inheritance: {path}")
    seen = {*seen, path}
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    cfg = copy.deepcopy(cfg)
    parent = cfg.pop("extends", None)
    if parent:
        if (path.parent / parent).resolve() == path:
            raise ValueError("Config cannot extend itself")
        base = load_config(path.parent / parent, seen)
        def merge(left, right):
            result = copy.deepcopy(left)
            for key, value in right.items():
                result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
            return result
        cfg = merge(base, cfg)
    for key in ("input", "output"):
        if not cfg.get(key):
            raise ValueError(f"Missing config field: {key}")
        cfg[key] = str((path.parent / cfg[key]).resolve())
    if Path(cfg["input"]) == Path(cfg["output"]) or Path(cfg["input"]) in Path(cfg["output"]).parents:
        raise ValueError("Output must be outside the input folder")
    for section, key in (("review", "overrides_path"), ("render", "font_path")):
        value = cfg.get(section, {}).get(key)
        if value:
            cfg[section][key] = str((path.parent / value).resolve())
    inpaint = cfg.get("render", {}).get("inpaint", {})
    if isinstance(inpaint, str):
        inpaint = {"backend": inpaint}
        cfg["render"]["inpaint"] = inpaint
    if inpaint.get("torchscript_path"):
        inpaint["torchscript_path"] = str((path.parent / inpaint["torchscript_path"]).resolve())
    for key in ("region_retries", "ocr_retries", "translation_retries", "layout_retries"):
        value = cfg.get("retries", {}).get(key, 1)
        if not isinstance(value, int) or not 0 <= value <= 5:
            raise ValueError(f"retries.{key} must be an integer from 0 to 5")
    if cfg.get("source_language", "ja") not in ("ja", "en", "auto", "mixed"):
        raise ValueError("source_language must be ja, en, auto, or mixed")
    if cfg.get("mode", "real") not in ("real", "demo"):
        raise ValueError("mode must be real or demo")
    cfg["_config_path"] = str(path)
    return cfg
