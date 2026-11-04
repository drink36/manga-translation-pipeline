"""Optional local model adapters; importing this module never loads model weights.

Coordinates and masks returned by detectors always refer to the input image.
Factories instantiate models only when their stage is explicitly run.
"""
from __future__ import annotations

import gc
import json
import math
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

KOHARU_MODEL = "mayocream/koharu-layout-rfdetr-seg-2xl-1152"
KOHARU_REVISION = "aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2"
PADDLE_MODEL = "jzhang533/PaddleOCR-VL-For-Manga"
PADDLE_REVISION = "1e8aa5f1dd90cc86fe9137c9c0b26ebde613cfe8"
MANGA_MODEL = "kha-white/manga-ocr-base"
MANGA_REVISION = "aa6573bd10b0d446cbf622e29c3e084914df9741"


class AdapterError(RuntimeError):
    """An optional dependency, model, or external tool could not be used."""


def _reference(cfg: dict, default_id: str, default_revision: str) -> tuple[str, str | None]:
    model_id = str(cfg.get("model_id", default_id))
    if Path(model_id).expanduser().is_dir():
        return str(Path(model_id).expanduser().resolve()), None
    revision = cfg.get("revision", default_revision if model_id == default_id else None)
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise AdapterError(f"Set a full 40-character revision for remote model {model_id!r}.")
    return model_id, revision


def _hub_options(cfg: dict, revision: str | None) -> dict:
    options: dict[str, Any] = {"revision": revision, "local_files_only": bool(cfg.get("local_files_only", False))}
    if cfg.get("cache_dir"):
        options["cache_dir"] = str(Path(cfg["cache_dir"]).expanduser())
    return options


def _release(owner: Any) -> None:
    for name in ("model", "processor"):
        if hasattr(owner, name):
            setattr(owner, name, None)
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _rectangle(box: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class KoharuDetector:
    def __init__(self, cfg: dict):
        try:
            from huggingface_hub import hf_hub_download
            from rfdetr import RFDETRSeg2XLarge
            from rfdetr.config import PretrainWeightsCompatibilityWarning
            from safetensors.torch import load_file
        except ImportError as exc:
            raise AdapterError("Koharu needs rfdetr==1.7.0, safetensors and huggingface-hub; see docs/models.md.") from exc
        self.model_id, self.revision = _reference(cfg, KOHARU_MODEL, KOHARU_REVISION)
        self.thresholds = {"text": 0.25, "sfx": 0.20, "bubble": 0.50, "panel": 0.50}
        self.thresholds.update(cfg.get("thresholds", {}))
        if any(not 0 <= float(v) <= 1 for v in self.thresholds.values()):
            raise AdapterError("Detector thresholds must be between 0 and 1.")
        weights = Path(self.model_id) / "model.safetensors" if self.revision is None else Path(
            hf_hub_download(self.model_id, "model.safetensors", **_hub_options(cfg, self.revision))
        )
        # Reproduce the published strict loader locally. Do not fetch/execute Hub Python.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PretrainWeightsCompatibilityWarning)
            self.model = RFDETRSeg2XLarge(
                pretrain_weights=None, resolution=1152, num_select=160,
                num_classes=4, device=str(cfg.get("device", "cpu")),
            )
        self.model.model.model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        self.model.model.class_names = ["text", "onomatopoeia", "bubble", "panel"]

    @property
    def metadata(self) -> dict:
        return {"backend": "koharu", "model_id": self.model_id, "revision": self.revision}

    def detect(self, image_path: Path, mask_dir: Path) -> list[dict]:
        import numpy as np
        from PIL import Image
        mask_dir = Path(mask_dir).resolve()
        mask_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        detections = self.model.predict(
            image, threshold=min(map(float, self.thresholds.values())),
            shape=(1152, 1152), include_source_image=False,
        )
        kinds = ["text", "sfx", "bubble", "panel"]
        result = []
        for i, (box, label, confidence) in enumerate(zip(
            detections.xyxy, detections.class_id, detections.confidence,
        )):
            if int(label) not in range(4):
                raise AdapterError(f"Unexpected Koharu class id {label}")
            kind = kinds[int(label)]
            if float(confidence) < float(self.thresholds[kind]):
                continue
            box = [max(0.0, min(float(v), float(width if j % 2 == 0 else height))) for j, v in enumerate(box)]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            mask_path = None
            if detections.mask is not None:
                mask = Image.fromarray((np.asarray(detections.mask[i]) > 0).astype("uint8") * 255)
                if mask.size != image.size:
                    mask = mask.resize(image.size, Image.Resampling.NEAREST)
                path = mask_dir / f"{i:04d}_{kind}.png"
                mask.save(path)
                mask_path = str(path)
            result.append({"kind": kind, "bbox": box, "polygon": _rectangle(box),
                           "score": float(confidence), "mask_path": mask_path})
        return result

    def close(self) -> None:
        _release(self)


class MangaOCRReader:
    def __init__(self, cfg: dict):
        try:
            from manga_ocr import MangaOcr
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise AdapterError("Install manga-ocr==0.1.16 and torch>=2.6; see docs/models.md.") from exc
        self.model_id, self.revision = _reference(cfg, MANGA_MODEL, MANGA_REVISION)
        path = self.model_id if self.revision is None else snapshot_download(
            self.model_id, allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"],
            **_hub_options(cfg, self.revision),
        )
        # Load on CPU first, then honor the explicit device (MangaOcr otherwise auto-selects a GPU).
        self.model = MangaOcr(pretrained_model_name_or_path=path, force_cpu=True)
        self.model.model.to(str(cfg.get("device", "cpu"))).eval()

    @property
    def metadata(self) -> dict:
        return {"backend": "manga_ocr", "model_id": self.model_id, "revision": self.revision}

    def read(self, image: Any) -> str:
        return str(self.model(image.convert("RGB"))).strip()

    def close(self) -> None:
        _release(self)


def _paddle_processor(model_id: str, hub_options: dict):
    """Convert legacy crop-model processor metadata to the native Transformers API."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    from transformers import AutoProcessor, AutoTokenizer
    from transformers.models.paddleocr_vl.image_processing_paddleocr_vl import PaddleOCRVLImageProcessor
    from transformers.models.paddleocr_vl.processing_paddleocr_vl import PaddleOCRVLProcessor

    def metadata_path(filename: str) -> Path:
        if Path(model_id).is_dir():
            return Path(model_id) / filename
        return Path(hf_hub_download(model_id, filename, **hub_options))

    common = {**hub_options, "trust_remote_code": False}
    try:
        metadata = json.loads(metadata_path("preprocessor_config.json").read_text(encoding="utf-8"))
    except (EntryNotFoundError, FileNotFoundError):
        return AutoProcessor.from_pretrained(model_id, **common)
    size = metadata.get("size", {})
    if not isinstance(size, dict) or not {"min_pixels", "max_pixels"}.intersection(size):
        return AutoProcessor.from_pretrained(model_id, **common)
    limits = {key: metadata.get(key, size.get(key)) for key in ("min_pixels", "max_pixels")}
    if any(value is None for value in limits.values()):
        raise AdapterError("Legacy Paddle processor is missing min_pixels/max_pixels.")
    image_processor = PaddleOCRVLImageProcessor.from_pretrained(
        model_id, size=None, **{key: int(value) for key, value in limits.items()}, **common,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
    tokenizer.image_token = getattr(tokenizer, "image_token", None) or "<|IMAGE_PLACEHOLDER|>"
    token_id = tokenizer.convert_tokens_to_ids(tokenizer.image_token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise AdapterError("Paddle tokenizer lacks its required image placeholder token.")
    tokenizer.image_token_id = token_id
    return PaddleOCRVLProcessor(
        image_processor=image_processor, tokenizer=tokenizer,
        chat_template=metadata_path("chat_template.jinja").read_text(encoding="utf-8"),
    )


class PaddleVLReader:
    def __init__(self, cfg: dict):
        try:
            import torch
            from transformers import AutoConfig, AutoModelForImageTextToText
        except ImportError as exc:
            raise AdapterError("Paddle VL needs torch>=2.6 and transformers==5.16.0; see docs/models.md.") from exc
        self.model_id, self.revision = _reference(cfg, PADDLE_MODEL, PADDLE_REVISION)
        hub_options = _hub_options(cfg, self.revision)
        common = {**hub_options, "trust_remote_code": False}
        try:
            self.processor = _paddle_processor(self.model_id, hub_options)
        except ImportError as exc:
            raise AdapterError("Native PaddleOCR-VL processor unavailable; install transformers==5.16.0.") from exc
        config = AutoConfig.from_pretrained(self.model_id, **common)
        if getattr(config, "text_config", None) is None:
            raise AdapterError("Paddle config was not converted by native Transformers; install transformers==5.16.0.")
        # These published checkpoints have distinct input/output embeddings. Native flat-config
        # conversion can restore the wrong default unless both nested/top-level flags are set.
        config.tie_word_embeddings = False
        config.text_config.tie_word_embeddings = False
        device = str(cfg.get("device", "cpu"))
        dtype_name = str(cfg.get("dtype", "float32" if device == "cpu" else "bfloat16"))
        if dtype_name not in {"float32", "float16", "bfloat16"}:
            raise AdapterError("Paddle dtype must be float32, float16 or bfloat16.")
        self.model, load_info = AutoModelForImageTextToText.from_pretrained(
            self.model_id, config=config, dtype=getattr(torch, dtype_name),
            attn_implementation="sdpa", use_safetensors=True, output_loading_info=True, **common,
        )
        problems = {k: v for k, v in load_info.items() if k in {
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
        } and v}
        if problems:
            _release(self)
            raise AdapterError(f"Paddle checkpoint did not load exactly: {problems}")
        self.model.to(device).eval()
        self.max_new_tokens = int(cfg.get("max_new_tokens", 512))
        if self.max_new_tokens < 1:
            raise AdapterError("max_new_tokens must be positive.")

    @property
    def metadata(self) -> dict:
        return {"backend": "paddle_vl", "model_id": self.model_id, "revision": self.revision}

    def read(self, image: Any) -> str:
        import torch
        image = image.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": "OCR:"},
        ]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            # The published manga checkpoints ship use_cache: false (a training setting).
            # Without the KV cache every new token re-processes the whole image prompt.
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True)
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        if generated.shape[1] >= self.max_new_tokens and int(generated[0, -1]) not in eos:
            raise AdapterError("Paddle OCR reached max_new_tokens; retry a smaller crop or increase the limit.")
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def close(self) -> None:
        _release(self)


class PaddleOCRReader:
    def __init__(self, cfg: dict):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise AdapterError("Install paddleocr>=3.4,<4 and a compatible PaddlePaddle runtime; see docs/models.md.") from exc
        self.cfg = dict(cfg)
        device = str(cfg.get("device", "cpu"))
        if device.startswith("cuda"):
            device = "gpu" + device[4:]
        self.model = PaddleOCR(
            lang=str(cfg.get("language", "en")), device=device,
            use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
            **cfg.get("options", {}),
        )

    @property
    def metadata(self) -> dict:
        return {"backend": "paddle_ocr", "language": self.cfg.get("language", "en")}

    def read(self, image: Any) -> str:
        import numpy as np
        # PaddleOCR ndarray inputs use OpenCV BGR convention.
        results = self.model.predict(np.asarray(image.convert("RGB"))[:, :, ::-1].copy())
        texts = []
        for result in results:
            if "rec_texts" not in result:
                raise AdapterError("Unexpected PaddleOCR output: missing rec_texts. Use PaddleOCR 3.x.")
            texts.extend(str(text) for text in result["rec_texts"])
        return "\n".join(texts).strip()

    def close(self) -> None:
        _release(self)
        paddle = sys.modules.get("paddle")
        if paddle is not None and paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()


def _run_command(cfg: dict, replacements: dict[str, str]) -> str:
    command = cfg.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise AdapterError("command must be a non-empty YAML list of argument strings, not a shell command.")
    argv = list(command)
    for placeholder, value in replacements.items():
        argv = [arg.replace("{" + placeholder + "}", value) for arg in argv]
    try:
        completed = subprocess.run(argv, check=True, capture_output=True, text=True,
                                   timeout=float(cfg.get("timeout_seconds", 300)), shell=False)
    except subprocess.TimeoutExpired as exc:
        raise AdapterError("External model command timed out.") from exc
    except FileNotFoundError as exc:
        raise AdapterError(f"External executable not found: {argv[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise AdapterError(f"External model command exited {exc.returncode}: {exc.stderr[-2000:]}") from exc
    return completed.stdout


class CommandReader:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)

    def read(self, image: Any) -> str:
        with tempfile.TemporaryDirectory(prefix="manga-ocr-") as directory:
            path = Path(directory) / "crop.png"
            output = Path(directory) / "text.json"
            image.convert("RGB").save(path)
            stdout = _run_command(self.cfg, {"image": str(path), "output": str(output)})
            raw = output.read_text(encoding="utf-8") if output.exists() else stdout
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AdapterError('OCR command must produce JSON {"text": "..."} or a JSON string.') from exc
            text = value.get("text") if isinstance(value, dict) else value
            if not isinstance(text, str):
                raise AdapterError('OCR command output must contain a string "text".')
            return text.strip()

    def close(self) -> None:
        pass


class CommandDetector:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)

    def detect(self, image_path: Path, mask_dir: Path) -> list[dict]:
        from PIL import Image
        mask_dir = Path(mask_dir).resolve()
        mask_dir.mkdir(parents=True, exist_ok=True)
        image_path = Path(image_path).resolve()
        with Image.open(image_path) as image:
            width, height = image.size
        with tempfile.TemporaryDirectory(prefix="manga-detector-") as directory:
            output = Path(directory) / "regions.json"
            stdout = _run_command(self.cfg, {"image": str(image_path), "mask_dir": str(mask_dir), "output": str(output)})
            raw = output.read_text(encoding="utf-8") if output.exists() else stdout
            try:
                regions = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AdapterError("Detector command must produce a JSON list; write logging to stderr.") from exc
        if not isinstance(regions, list):
            raise AdapterError("Detector command must return a JSON list of regions.")
        for region in regions:
            if not isinstance(region, dict) or region.get("kind") not in {"text", "sfx", "bubble", "panel"}:
                raise AdapterError("Detector region kind must be text, sfx, bubble or panel.")
            box = region.get("bbox")
            if not isinstance(box, list) or len(box) != 4 or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in box):
                raise AdapterError("Detector bbox must be four finite pixel coordinates.")
            if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise AdapterError("Detector bbox lies outside the original image or has no area.")
            polygon = region.setdefault("polygon", _rectangle(box))
            if not isinstance(polygon, list) or len(polygon) < 3 or not all(
                isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)
                and 0 <= p[0] <= width and 0 <= p[1] <= height for p in polygon
            ):
                raise AdapterError("Detector polygon must contain at least three finite in-image points.")
            score = region.setdefault("score", 1.0)
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise AdapterError("Detector score must be a probability between 0 and 1.")
            if region.get("mask_path"):
                path = Path(region["mask_path"])
                path = path if path.is_absolute() else mask_dir / path
                with Image.open(path) as mask:
                    if mask.size != (width, height):
                        raise AdapterError("Detector masks must match the original image dimensions.")
                    if any(mask.convert("L").histogram()[1:255]):
                        raise AdapterError("Detector masks must be binary (pixel values 0 or 255).")
                region["mask_path"] = str(path.resolve())
            else:
                region["mask_path"] = None
        return regions

    def close(self) -> None:
        pass


def build_detector(cfg: dict):
    backend = cfg.get("backend", "koharu")
    if backend == "koharu":
        return KoharuDetector(cfg)
    if backend == "command":
        return CommandDetector(cfg)
    raise AdapterError(f"Unknown detector backend {backend!r}; expected koharu or command.")


def build_reader(cfg: dict):
    readers = {"manga_ocr": MangaOCRReader, "paddle_vl": PaddleVLReader,
               "paddle_ocr": PaddleOCRReader, "command": CommandReader}
    backend = cfg.get("backend")
    if backend not in readers:
        raise AdapterError(f"Unknown reader backend {backend!r}; expected one of {list(readers)}.")
    return readers[backend](cfg)
