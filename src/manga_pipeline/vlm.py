"""Bounded JSON-only client for local OpenAI-compatible vision servers.

No model is downloaded or loaded by this module. Qwen and an optional MangaLMM
server can use separate instances/configuration. Configuration keys: endpoint,
model, api_key_env, timeout_seconds, max_tokens, temperature, top_p, top_k, min_p,
presence_penalty, repetition_penalty, image_max_pixels,
request_retries, enable_thinking, and response_format_json (default true: each
request carries its task's JSON schema for constrained decoding).
"""

from __future__ import annotations

import base64
import copy
import io
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
import urllib.error
import urllib.request
import uuid

from .prompts import TASK_SCHEMAS, system_prompt

if TYPE_CHECKING:
    from PIL import Image


class VLMError(RuntimeError):
    """The model request did not produce a usable task result."""


class VLMResponseError(VLMError):
    """A response was truncated, empty, invalid, or violated its task schema."""


def _parse_json(content: str) -> dict[str, Any]:
    """Parse one JSON object, allowing only a surrounding markdown fence."""
    if not isinstance(content, str) or not content.strip():
        raise VLMResponseError("Model returned empty text")
    value = content.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"} or lines[-1].strip() != "```":
            raise VLMResponseError("Malformed JSON code fence")
        value = "\n".join(lines[1:-1])
    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = item
        return result

    def reject_constant(constant):
        raise ValueError(f"Nonfinite JSON constant: {constant}")

    try:
        parsed = json.loads(value, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise VLMResponseError("Model response is not one complete JSON object") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise VLMResponseError("Model response must be a nonempty JSON object")
    return parsed


def _require(value: Any, expected: type, label: str) -> None:
    # bool is an int subclass; use exact types for primitive schema fields.
    valid = type(value) is expected if expected in {str, bool, int, float} else isinstance(value, expected)
    if not valid:
        raise VLMResponseError(f"Invalid or missing {label}")


def _source_ids(payload: dict[str, Any]) -> list[str]:
    regions = payload.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError("payload.regions must be a list")
    ids = [item.get("region_id") for item in regions if isinstance(item, dict)]
    if len(ids) != len(regions) or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("Every input region needs a nonempty region_id")
    if len(set(ids)) != len(ids):
        raise ValueError("Input region IDs must be unique")
    return ids


def _check_ids(actual: list[str], expected: list[str], label: str) -> None:
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise VLMResponseError(f"{label} must contain every input region ID exactly once")


def _request_schema(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Task schema narrowed to this request's region IDs, so replies cannot drop or invent them."""
    schema = copy.deepcopy(TASK_SCHEMAS[task])
    if task not in {"reading_order", "translate", "verify", "visual_qa"}:
        return schema
    ids = _source_ids(payload)
    if not ids:
        return schema
    properties = schema["properties"]
    if task == "reading_order":
        properties["region_ids"].update({"items": {"enum": ids}, "minItems": len(ids), "maxItems": len(ids)})
    elif task == "visual_qa":
        properties["issues"]["items"]["properties"]["region_id"] = {"enum": [*ids, "page"]}
    else:
        array = properties["translations" if task == "translate" else "checks"]
        array["items"]["properties"]["region_id"] = {"enum": ids}
        array.update({"minItems": len(ids), "maxItems": len(ids)})
    return schema


def _text_result(result: dict[str, Any], label: str) -> None:
    _require(result.get("text"), str, f"{label}.text")
    _require(result.get("uncertain"), bool, f"{label}.uncertain")
    _require(result.get("reason"), str, f"{label}.reason")
    if not result["text"].strip() and not result["uncertain"]:
        raise VLMResponseError(f"{label}: empty text cannot pass as certain")
    if result["uncertain"] and not result["reason"].strip():
        raise VLMResponseError(f"{label}: uncertainty requires a reason")


def _validate_result(task: str, result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    if task == "coverage":
        _require(result.get("regions"), list, "regions")
        _require(result.get("uncertain"), bool, "uncertain")
        for region in result["regions"]:
            _require(region, dict, "region")
            bbox = region.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise VLMResponseError("Coverage bbox needs four normalized coordinates")
            if any(type(n) not in {int, float} or not math.isfinite(n) or not 0 <= n <= 1000 for n in bbox):
                raise VLMResponseError("Coverage bbox coordinates must be finite and within 0..1000")
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                raise VLMResponseError("Coverage bbox has no positive area")
            if region.get("kind") not in {"text", "sfx"}:
                raise VLMResponseError("Coverage region kind must be text or sfx")
            _require(region.get("reason"), str, "region.reason")
    elif task in {"ocr", "reconcile"}:
        _text_result(result, task)
    elif task == "reading_order":
        _require(result.get("region_ids"), list, "region_ids")
        for region_id in result["region_ids"]:
            _require(region_id, str, "region_id")
        _check_ids(result["region_ids"], _source_ids(payload), "Reading order")
        _require(result.get("uncertain"), bool, "uncertain")
        _require(result.get("reason"), str, "reason")
    elif task == "context":
        _require(result.get("summary"), str, "summary")
        _require(result.get("glossary"), list, "glossary")
        _require(result.get("characters"), list, "characters")
        evidence_ids = set()

        def collect_ids(value):
            if isinstance(value, dict):
                if isinstance(value.get("region_id"), str):
                    evidence_ids.add(value["region_id"])
                for child in value.values():
                    collect_ids(child)
            elif isinstance(value, list):
                for child in value:
                    collect_ids(child)

        collect_ids(payload.get("pages", []))
        previous = payload.get("previous_context", {})
        if isinstance(previous, dict):
            for item in previous.get("glossary", []):
                if isinstance(item, dict):
                    evidence_ids.update(i for i in item.get("evidence", []) if isinstance(i, str))
        for item in result["glossary"]:
            _require(item, dict, "glossary entry")
            for key in ("source", "target"):
                _require(item.get(key), str, f"glossary.{key}")
                if not item[key].strip():
                    raise VLMResponseError(f"Glossary {key} cannot be empty")
            if item.get("status") not in {"tentative", "confirmed"}:
                raise VLMResponseError("Glossary status must be tentative or confirmed")
            _require(item.get("evidence"), list, "glossary.evidence")
            for region_id in item["evidence"]:
                _require(region_id, str, "glossary evidence region_id")
                if region_id not in evidence_ids:
                    raise VLMResponseError("Glossary cites evidence absent from the supplied context")
            if item["status"] == "confirmed" and not item["evidence"]:
                raise VLMResponseError("Confirmed glossary entries need source evidence")
    elif task in {"translate", "verify"}:
        key = "translations" if task == "translate" else "checks"
        _require(result.get(key), list, key)
        actual = []
        for item in result[key]:
            _require(item, dict, key + " item")
            _require(item.get("region_id"), str, key + ".region_id")
            actual.append(item["region_id"])
            if task == "translate":
                _text_result(item, "translation")
            else:
                _require(item.get("passed"), bool, "check.passed")
                _require(item.get("issues"), list, "check.issues")
                for issue in item["issues"]:
                    _require(issue, str, "check issue")
                    if not issue.strip():
                        raise VLMResponseError("Verification issues cannot be empty strings")
                if item["passed"] != (len(item["issues"]) == 0):
                    raise VLMResponseError("Verification passed must agree with issues")
        _check_ids(actual, _source_ids(payload), key)
    elif task == "visual_qa":
        _require(result.get("issues"), list, "issues")
        _require(result.get("passed"), bool, "passed")
        expected = set(_source_ids(payload)) | {"page"}
        for issue in result["issues"]:
            _require(issue, dict, "visual QA issue")
            if issue.get("region_id") not in expected:
                raise VLMResponseError("Visual QA issue references an unknown region")
            if issue.get("category") not in {"mask", "layout", "translation", "other"}:
                raise VLMResponseError("Unknown visual QA category")
            _require(issue.get("reason"), str, "visual QA reason")
            if not issue["reason"].strip():
                raise VLMResponseError("Visual QA issues need a reason")
        if result["passed"] != (len(result["issues"]) == 0):
            raise VLMResponseError("Visual QA passed must agree with issues")
    return result


class VLMClient:
    def __init__(self, cfg: dict[str, Any], log_dir: Path | None = None):
        self.cfg = dict(cfg)
        self.endpoint = str(cfg.get("endpoint", "http://127.0.0.1:8000/v1")).rstrip("/")
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("VLM endpoint must use http or https")
        self.url = self.endpoint if self.endpoint.endswith("/chat/completions") else self.endpoint + "/chat/completions"
        self.model = cfg.get("model")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("VLM config needs a model served by the endpoint")
        self.timeout = float(cfg.get("timeout_seconds", 180))
        self.retries = int(cfg.get("request_retries", 1))
        self.max_tokens = int(cfg.get("max_tokens", 4096))
        self.max_pixels = int(cfg.get("image_max_pixels", 1600000))
        if not 0 <= self.retries <= 4:
            raise ValueError("request_retries must be between 0 and 4")
        if not math.isfinite(self.timeout) or self.timeout <= 0 or self.max_tokens <= 0 or self.max_pixels <= 0:
            raise ValueError("Timeout, max_tokens, and image_max_pixels must be positive")
        self.log_dir = Path(log_dir) if log_dir is not None else None

    def _image_url(self, original: Image.Image) -> str:
        # Pillow is optional until a request actually contains images.
        from PIL import Image, ImageOps

        prepared = ImageOps.exif_transpose(original)
        if prepared.width * prepared.height > self.max_pixels:
            scale = math.sqrt(self.max_pixels / (prepared.width * prepared.height))
            size = (max(1, int(prepared.width * scale)), max(1, int(prepared.height * scale)))
            prepared = prepared.resize(size, Image.Resampling.LANCZOS)
        if prepared.mode in {"RGBA", "LA"} or "transparency" in prepared.info:
            rgba = prepared.convert("RGBA")
            canvas = Image.new("RGB", rgba.size, "white")
            canvas.paste(rgba, mask=rgba.getchannel("A"))
            prepared = canvas
        else:
            prepared = prepared.convert("RGB")
        buffer = io.BytesIO()
        prepared.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def _save_response(self, task: str, request_id: str, attempt: int, raw: bytes) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Only response bytes are retained: never config, authorization, or requests.
        path = self.log_dir / f"{task}-{request_id}-attempt-{attempt + 1}.response.txt"
        path.write_bytes(raw)

    def ask(self, task: str, payload: dict[str, Any], images: list[Image.Image] | None = None) -> dict[str, Any]:
        prompt = system_prompt(task)
        if not isinstance(payload, dict):
            raise ValueError("VLM payload must be a dictionary")
        if task in {"translate", "verify", "reading_order", "visual_qa"}:
            _source_ids(payload)
        if task == "ocr" and any(key in payload for key in ("candidates", "original_text", "text", "transcription")):
            raise ValueError("Independent OCR payload must not contain candidate transcriptions")
        content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
        for image in images or []:
            content.append({"type": "image_url", "image_url": {"url": self._image_url(image)}})
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            "temperature": float(self.cfg.get("temperature", 0)),
            "max_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": bool(self.cfg.get("enable_thinking", False))},
        }
        for key in ("top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
            if key in self.cfg:
                body[key] = self.cfg[key]
        if "repetition_penalty" in self.cfg:
            # llama.cpp's name for it; unset, llama-server applies its own 1.1 default.
            body["repeat_penalty"] = self.cfg["repetition_penalty"]
        if self.cfg.get("response_format_json", True):
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": task, "schema": _request_schema(task, payload), "strict": True}}
        headers = {"Content-Type": "application/json"}
        key_env = self.cfg.get("api_key_env")
        if key_env:
            key = os.environ.get(str(key_env))
            if not key:
                raise VLMError(f"Configured API key environment variable {key_env} is unset")
            headers["Authorization"] = "Bearer " + key
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request_id = uuid.uuid4().hex
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                self._save_response(task, request_id, attempt, raw)
                try:
                    envelope = json.loads(raw)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise VLMResponseError("Endpoint returned an invalid JSON response envelope") from exc
                if not isinstance(envelope, dict):
                    raise VLMResponseError("Endpoint response envelope must be an object")
                choices = envelope.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise VLMResponseError("Endpoint returned no completion choices")
                choice = choices[0]
                finish_reason = choice.get("finish_reason")
                if finish_reason == "length":
                    raise VLMResponseError("Model output hit max_tokens; increase the budget or reduce the request")
                if finish_reason not in {None, "stop"}:
                    raise VLMResponseError(f"Unsupported completion finish reason: {finish_reason}")
                message = choice.get("message")
                if not isinstance(message, dict):
                    raise VLMResponseError("Endpoint returned no assistant message")
                parsed = _parse_json(message.get("content"))
                return _validate_result(task, parsed, payload)
            except urllib.error.HTTPError as exc:
                self._save_response(task, request_id, attempt, exc.read())
                last_error = VLMError(f"VLM endpoint returned HTTP {exc.code}")
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = VLMError(f"VLM request failed ({type(exc).__name__})")
            except VLMResponseError as exc:
                last_error = exc
        raise VLMError(f"{task} failed after {self.retries + 1} attempt(s): {last_error}") from last_error
