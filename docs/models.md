# Model adapters and provenance

The adapters are optional, lazy imports. Importing the package, planning a run,
and running the synthetic demo do not download or instantiate model weights.
Creating a real model adapter when running its stage can download its pinned
checkpoint. No real model inference or GPU memory benchmark was performed while
creating this folder.

## Checkpoint choices

| Role | Model ID | Pinned revision |
| --- | --- | --- |
| Layout | `mayocream/koharu-layout-rfdetr-seg-2xl-1152` | `aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2` |
| Independent Japanese OCR | `kha-white/manga-ocr-base` | `aa6573bd10b0d446cbf622e29c3e084914df9741` |
| Public manga Paddle VL reader | `jzhang533/PaddleOCR-VL-For-Manga` | `1e8aa5f1dd90cc86fe9137c9c0b26ebde613cfe8` |
| Requested manga 1.6 candidate | `AlphaBeta07/PaddleOCR-VL-1.6-For-Manga` | `103c97c277d688b31b8adb1bb2228380b77a640b` |
| Public general-purpose 1.6 alternative | `PaddlePaddle/PaddleOCR-VL-1.6` | `c5630abae1d940eafe0697512a0325494b02ab42` |

The runnable public manga default is **the older PaddleOCR-VL manga fine-tune,
not version 1.6**. The AlphaBeta07 model is the promoted 1.6 revision documented
by its training project, but direct public metadata/config access returned HTTP
401 during setup. `gigass03/PaddleOCR-VL-1.6-Manga-Finetuned` also returned HTTP
401. Access to either checkpoint was not established. If you have access, select
the desired `model_id` and full `revision` explicitly. The official public 1.6
alternative is a general-purpose model, not a manga fine-tune. A configured
checkpoint failing to load is an error; the adapter never silently substitutes
another model.

All remote PyTorch model references require a full commit revision. A local
model directory is also supported. Add `local_files_only: true` to require an
already cached checkpoint, and optionally set `cache_dir`.

Sources: [Koharu model card](https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152),
[Koharu strict loader](https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152/blob/aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2/load_model.py),
[MangaOCR](https://github.com/kha-white/manga-ocr),
[public older manga checkpoint](https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga),
[1.6 training project](https://github.com/boomerkuigi/paddleocr-vl-1.6-manga-sft),
[official 1.6 checkpoint](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6).

## Dependencies and devices

Use Python 3.11 or 3.12 and a PyTorch/torchvision pair compatible with the node's
CUDA environment. Keep Qwen serving in its own environment. These are API
targets derived from upstream sources, not a claim that the full GPU dependency
stack has been tested here.

On Windows, `configs/windows.yaml` sets `server.backend: llama.cpp`. `serve` then
starts `server.executable` (llama.cpp's `llama-server`) with the GGUF weights in
`server.model_path` and the vision projector in `server.mmproj_path` instead of
vLLM. The pipeline sends the same OpenAI-compatible requests to either server.

| Adapter | Additional dependencies |
| --- | --- |
| `koharu` | `rfdetr==1.7.0`, `safetensors>=0.8`, `huggingface-hub>=1.5,<2`, PyTorch/torchvision |
| `manga_ocr` | `manga-ocr==0.1.16`, `torch>=2.6`, `transformers==5.16.0`, `huggingface-hub>=1.5,<2` |
| `paddle_vl` | `torch>=2.6`, compatible `torchvision`, `transformers==5.16.0`, `safetensors>=0.8`, `huggingface-hub>=1.5,<2` |
| `paddle_ocr` | `paddleocr>=3.4,<4` and a compatible `paddlepaddle>=3,<4` CPU runtime or Paddle GPU build |
| `command` | The external tool's own environment; it can be separate from this package |

RF-DETR 1.7.0 requires Transformers `>=5.1,<6`. The manga training project's
[dependency constraints](https://github.com/boomerkuigi/paddleocr-vl-1.6-manga-sft/blob/main/constraints.txt)
pin Transformers 5.16.0 and MangaOCR 0.1.16. Transformers 5.16.0 itself requires
the newer Hub and SafeTensors bounds listed above, and its native Paddle image
processor uses torchvision. Paddle VL uses native Transformers,
`trust_remote_code=False`, SafeTensors weights, and `OCR:` crop prompts. It
repairs legacy processor metadata and preserves untied token embeddings as
described in the project's
[reference inference implementation](https://github.com/boomerkuigi/paddleocr-vl-1.6-manga-sft/blob/main/manga_sft/inference.py).
It rejects missing/unexpected checkpoint keys and truncated generation.
MangaOCR's published revision has a PyTorch `.bin` checkpoint; use modern
PyTorch's restricted loading path, and do not bypass Transformers' version checks.

Local adapters default to `device: cpu`. This permits Qwen to occupy the single
12 GB RTX 5070 (formerly a 40 GB A100) while local adapters run on the CPU, at a potentially substantial
speed cost. In GPU stage scheduling, explicitly select `device: cuda` only after
releasing the Qwen server's GPU memory. Calling `close()` releases the local
adapter, but does not stop an independently running Qwen process. For Paddle VL,
the default dtype is `float32` on CPU and `bfloat16` on CUDA. Regular PaddleOCR
maps `cuda:0` to its own `gpu:0` device notation; its GPU runtime needs a separate
installation matching the CUDA version. See the
[Paddle installation guide](https://www.paddlepaddle.org.cn/install/quick).

## Geometry contract

`build_detector(config).detect(image_path, mask_dir)` returns a list containing
text, SFX, bubble and panel detections. Coordinates are in original input pixels.
Koharu's `onomatopoeia` class becomes `sfx`. Masks are original-resolution PNGs
with values 0 and 255, saved under the absolute `mask_dir`.

Koharu's `polygon` field is a rectangle derived from its bounding box. Its actual
segmentation is preserved separately in `mask_path`; do not treat that rectangle
as a glyph erasure mask. Text and SFX masks represent lettering; bubble and panel
masks are geometry for grouping/layout. The model does not supply OCR, reading
order, or text-to-bubble relationships. Default thresholds are text 0.25, SFX
0.20, bubble 0.50, and panel 0.50, configurable under `thresholds` by those names.

The adapter reconstructs the published RF-DETR Seg 2XL configuration locally,
then loads SafeTensors with `strict=True`. It never imports downloaded
`load_model.py` or enables arbitrary Hub model code.

## CTD or external model bridge

CTD is an optional external detector. Its native outputs need a small wrapper
that implements this JSON contract; installing CTD alone does not automatically
provide the wrapper. You can use a different virtual environment for it:

```yaml
backend: command
command:
  - /absolute/path/to/ctd-venv/bin/python
  - /absolute/path/to/your_ctd_bridge.py
  - --image
  - "{image}"
  - --mask-dir
  - "{mask_dir}"
  - --output
  - "{output}"
timeout_seconds: 300
```

Arguments are passed directly to `subprocess.run(..., shell=False)`. Placeholders
are literal replacements. Shell syntax, environment expansion, pipes, and shell
quoting are not evaluated. Paths containing spaces remain single arguments.
The command must write UTF-8 JSON to `{output}`, or return UTF-8 JSON alone on stdout;
send log messages to stderr. For a detector, return:

```json
[
  {
    "kind": "text",
    "bbox": [20, 30, 80, 160],
    "polygon": [[20, 30], [80, 30], [80, 160], [20, 160]],
    "score": 0.95,
    "mask_path": "0000_text.png"
  }
]
```

Allowed kinds are `text`, `sfx`, `bubble`, `panel`. Bboxes and polygons must lie
inside the original image; mask dimensions must match it. Relative mask paths
are resolved against `{mask_dir}` and must persist after the command exits.
`polygon` may be omitted to use the bbox rectangle; `mask_path` may be null.
Use binary PNGs for masks. Masks must describe the correct class, not a generic
filled text bbox.

For an OCR reader, `backend: command` accepts `{image}` (temporary crop PNG) and
`{output}` (temporary JSON output). Return `{"text": "日本語"}` or a JSON string.
`build_reader(config).read(PIL_image)` always returns a string. Reader instances
and detector instances expose `close()`.

Source: [Comic Text Detector](https://github.com/dmMaze/comic-text-detector).

## Optional MangaLMM

MangaLMM remains an optional page auditor, not a required local adapter here.
Its page-level grounding needs to satisfy the same region-coordinate contract
as the Qwen coverage stage before it can replace that stage. Do not feed an OCR
candidate into an allegedly independent first visual reading. See the
[official MangaLMM repository](https://github.com/manga109/MangaLMM).
