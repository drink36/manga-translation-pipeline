# Manga translation pipeline

Japanese/English manga → Traditional Chinese, with independent OCR readings,
coverage checks, chapter context, bounded verification loops, and editable JSON.
Designed around one **NVIDIA A100 40 GB on OSC Ascend**.

This folder contains executable orchestration and model adapters, not downloaded
weights. The CPU demo and automated tests run without model inference. Actual OCR,
translation quality, GPU compatibility, memory peaks, and speed have not been
validated on your manga or on an A100.

## What is included

| Role | Default |
|---|---|
| Layout | KoharuLayout: text/SFX/bubbles/panels and masks |
| Coverage | Independent Qwen page scan, optional overlapping tiles and detector retries |
| Japanese OCR | PaddleOCR-VL manga checkpoint and MangaOCR, both blind |
| English/mixed OCR | Regular English PaddleOCR / multilingual PaddleOCR-VL |
| OCR arbitration | Qwen blind reread, then reconciliation with alternatives |
| Context/translation | Chunked chapter context, page-level zh-TW translation |
| Verification | Strict region-ID checks, Qwen review, bounded targeted retries |
| Rendering | Glyph masks, conservative plain fill or local LaMa, horizontal CJK fitting |
| QA/export | Deterministic rendering checks plus VLM inspection; PNG, CBZ, JSON, review CSV |

**Checkpoint distinction:** public access to the previously discussed
`AlphaBeta07/PaddleOCR-VL-1.6-For-Manga` returned HTTP 401 during setup. The default
uses the accessible **older** `jzhang533/PaddleOCR-VL-For-Manga`, with a pinned
revision. It is not presented as 1.6. Alternatives:

- `configs/paddle-vl16-candidate.yaml`: the requested manga-specific 1.6 candidate;
  you must verify access first. Loading errors never silently select another model.
- `configs/paddle-vl16-base.yaml`: official public 1.6 base, with independent
  MangaOCR still enabled; this base is not manga-fine-tuned.
- `configs/quality.yaml`: always perform Qwen blind crop OCR and overlapping
  coverage tiles, retaining the public default manga checkpoint.

MangaLMM is an optional second HTTP endpoint. CTD is an optional external detector
bridge. Neither is required or automatically downloaded. See [models](docs/models.md).

## Try the CPU demo now

The `.venv` already created in this folder contains only the lightweight runtime
and test dependencies. From this folder:

```bash
.venv/bin/python -m manga_pipeline demo --output output/demo
.venv/bin/python -m pytest -q
```

The demo generates two synthetic pages and uses scripted OCR/translation replies.
It exercises real stage storage, grouping, reconciliation plumbing, CJK rendering,
QA plumbing and export. It is **not evidence of model accuracy**.

Open `output/demo/output/rendered/initial/p0001.translated.png` and inspect
`output/demo/output/export/chapter.json`. The CBZ is explicitly named
`translated.demo.cbz`. If the configured Noto CJK font is unavailable, the demo
uses clearly labelled Latin placeholder translations instead.

For a fresh lightweight environment (environments are managed with
[uv](https://docs.astral.sh/uv/), which also fetches the Python version):

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv -e '.[test]'
```

## Configure a real chapter

1. Put naturally numbered PNG/JPEG/WebP pages in `input/`, or change `input` to an
   image/CBZ/ZIP path. A folder is read at its top level, in natural filename order.
   Hidden files and macOS metadata (`__MACOSX/`, `._*`) are skipped.
2. Edit `configs/pipeline.yaml`: input/output, language, reading direction, model
   choices, font, and retry budgets. Paths are relative to the YAML file.
3. Use `source_language: ja` or `en` when known. `auto` and `mixed` route all crops
   through the configured multilingual reader; there is no automatic per-region
   language classifier in this version.
4. Keep `render_uncertain: false` to preserve original pixels for unresolved text.

```bash
.venv/bin/python -m manga_pipeline plan --config configs/pipeline.yaml
.venv/bin/python -m manga_pipeline doctor --config configs/pipeline.yaml
.venv/bin/python -m manga_pipeline serve --config configs/pipeline.yaml --print-command
```

These three inspection commands do not load/download models or contact a server.

## Install model environments when ready

Use Python 3.11 or 3.12 for the model environments. Keep the serving engine
separate from the local OCR dependencies. The existing CPU test `.venv` does not
contain the model packages. These commands install packages, not an evaluated
CUDA stack; use the CUDA/PyTorch build appropriate for your allocation.

```bash
uv venv --python 3.12 .venv-models
uv pip install -p .venv-models -e '.[layout,manga,vl,english]'

uv venv --python 3.12 .venv-vllm
uv pip install -p .venv-vllm vllm -e .
```

The adapters target `rfdetr==1.7.0`, `transformers==5.16.0`, and
`manga-ocr==0.1.16`. Installing/running these optional packages was not part of
the CPU checks. Record resolved versions with `uv pip freeze -p <env>` for a real run.

## Run on Ascend

The default model configuration uses CPU for layout/OCR and one GPU for Qwen.
This allows the Qwen server to remain loaded across audit, reconciliation,
translation, and QA. **40 GB is device memory; `--mem=64G` is host RAM.**

The launcher requests Ascend's `nextgen` partition, one GPU, and starts/stops its
own Qwen server. It uses your script-first sbatch convention and environment stage
selection. Replace `YOUR_PROJECT` with your Slurm allocation:

```bash
cd /users/PAS2119/drink36/manga-translation-pipeline
MANGA_STAGE=all sbatch scripts/ascend.sh --account=YOUR_PROJECT
```

For a different configuration:

```bash
MANGA_STAGE=all MANGA_CONFIG=configs/quality.yaml \
  sbatch scripts/ascend.sh --account=YOUR_PROJECT
```

**No job has been submitted.** The script never submits additional jobs. Account,
environment installations, model availability, and server startup must be ready
before a real launch. The default 16K context, one request at a time, two images per
request, and 0.85 GPU memory fraction are starting settings, not measured optima.
Reduce image pixels/context if needed after measuring. Local readers stay loaded
within the OCR stage; if you move them to CUDA, budget their combined peak memory
alongside Qwen or run the relevant stages with the server stopped.

To use an existing local endpoint, set `MANGA_START_SERVER=0`. You may override
`MANGA_PYTHON` and `MANGA_SERVER_PYTHON` with absolute interpreter paths.

An explicit run inside an allocated compute node can also use two terminals:

```bash
# Terminal 1: starts model serving and may download uncached weights.
.venv-vllm/bin/python -m manga_pipeline serve --config configs/pipeline.yaml

# Terminal 2: runs the chapter; may download missing detector/OCR weights.
.venv-models/bin/python -m manga_pipeline run --config configs/pipeline.yaml --stage all
```

## Stages and resume

Execution order is `prepare → detect → audit → ocr → reconcile → context →
translate → render → qa → export`. Use `--stage STAGE` to run one stage after its
prerequisites. `--stage all` reuses matching artifacts; `--force` recomputes the
requested stages. Layout/OCR models are released between stages.

Each artifact records configuration, source/code hashes, input dependencies, and
referenced file hashes. Stale prerequisites fail with a rerun instruction instead
of silently mixing outputs. VLM connection settings (`endpoint`, `api_key_env`,
`timeout_seconds`, `request_retries`) are excluded, so raising a timeout keeps
cached model results. Outputs are scoped by chapter/page and are never
written over input pages. Do not run two writers against the same output folder.

```text
output/chapter/
  manifest.json
  pages/p0001/
    original.jpg          # original source bytes (extension follows input)
    source.png            # EXIF-oriented, full-resolution canonical page
    prepare.json ... qa.json
    masks/  crops/  audit_crops/
  context.json
  requests/qwen/          # raw model replies, including malformed ones
  rendered/initial/      # initial PNGs
  rendered/qa/attempt-01/ # separate retries; initial artifact is not overwritten
  export/
    chapter.json
    review.json
    review.csv
    original.cbz         # canonical page PNGs; raw originals remain under pages/
    translated.cbz
```

## Review and correct

Inspect `review.csv` and the corresponding crops/images. Create
`review/overrides.json` (outside the output folder by default):

```json
{
  "p0001_r0001": {
    "original_text": "本当の原文",
    "translation": "人工校對後的繁體中文"
  }
}
```

These explicit edits receive `human_reviewed` provenance. Automated passes use
`auto_passed` or `needs_review`; they never claim human approval. Overrides can
also supply `mask_path`, `bubble_bbox`, and `bubble_mask_path` for rendering.
Mask paths resolve relative to the overrides file; masks must match the canonical
page's pixel dimensions. Unknown region IDs are errors.

Run `--stage all` again. Translation-only edits reuse detection and OCR; rendering
settings/mask edits reuse translation. Source transcription edits rerun chapter
context and translation because their meaning may change. See
[rendering](docs/rendering.md) for exact rules and limitations.

## What still needs real validation

- Model loading and all CUDA/vLLM kernels on the actual A100 environment.
- Missed-text recall, OCR character errors, translation accuracy, and pages/minute.
- MangaLMM's extra benefit. Its endpoint must emit the documented JSON contracts;
  the published model was not tested with this prompting layer.
- LaMa weights and restoration quality. Without local compatible weights, artwork
  that cannot be safely plain-filled is retained and flagged.
- Horizontal typesetting is implemented. Vertical Chinese, curved SFX lettering,
  font-style matching, and a graphical editor are not implemented.
- Automatic panel/bubble association and reading order can remain ambiguous.
  Agreement between recognizers is evidence, not ground truth.

See [pipeline design](docs/pipeline.md), [model provenance](docs/models.md), and
[rendering behavior](docs/rendering.md). The [validation record](docs/validation.md)
distinguishes the completed CPU checks from the remaining real-model checks.
