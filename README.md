# Manga translation pipeline

Japanese/English manga → Traditional Chinese, with independent OCR readings,
coverage checks, chapter context, bounded verification loops, and editable JSON.
The same code runs on two machines. Each has its own config file, and every command
takes it with `--config`:

| Machine | Config | Qwen server | How to run |
|---|---|---|---|
| OSC Ascend, A100 40 GB (Linux) | `configs/ascend.yaml` (Japanese), `configs/ascend-en.yaml` (English) | vLLM, `Qwen3.8-27B` (4-bit AWQ) | Slurm or two terminals: [Run on Ascend](#run-on-ascend) |
| RTX 5070 12 GB (Windows) | `configs/windows.yaml` (Japanese), `configs/windows-en.yaml` (English) | llama.cpp, `Qwen3.5-9B` (4-bit GGUF) | Two terminals: [Run on Windows](#run-on-windows-rtx-5070) |

`configs/pipeline.yaml` is the shared base that both extend. It holds everything that
is the same on both machines (stages, OCR readers, sampling, retries, rendering) and
is not run on its own. Put a setting in the base when both machines should get it,
and in a machine config when only that machine should.

This folder contains executable orchestration and model adapters, not downloaded
weights. The CPU demo and automated tests run without model inference. One English
chapter has been run end to end on the RTX 5070. On the A100, a 98-page English
chapter has run through `reconcile` with `configs/ascend-en.yaml`; `context` and the
later stages have not completed there yet. OCR and translation quality and memory
peaks have not been measured on your manga.

## How the pipeline works

A chapter goes through ten stages, in this order. Each stage saves its result per page
under `output/`, so a run that stops can be started again with the same command and
continues from the pages already done ([Stages and resume](#stages-and-resume)).

| # | Stage | What it does | Model |
|---|---|---|---|
| 1 | `prepare` | Reads the pages and saves each as a full-size, correctly oriented PNG | none |
| 2 | `detect` | Finds text, sound effects, bubbles and panels, with masks | KoharuLayout; the English configs take text from PaddleOCR text lines |
| 3 | `audit` | Scans each page independently for text the detector missed, with one retry on overlapping tiles | Qwen |
| 4 | `ocr` | Reads every text crop | Japanese: PaddleOCR-VL manga and MangaOCR. English: PaddleOCR. Mixed or unknown: PaddleOCR-VL 1.6 |
| 5 | `reconcile` | Where the readers disagree, or there is only one, Qwen reads the crop without seeing their text and then settles the reading. It also sets the reading order | Qwen |
| 6 | `context` | Builds a chapter summary, glossary and character list, a few pages at a time | Qwen |
| 7 | `translate` | Translates each page to Traditional Chinese, checks every region, and retries the regions that fail | Qwen |
| 8 | `render` | Erases the original lettering (white fill, or LaMa over artwork) and fits the Chinese text | LaMa, optional |
| 9 | `qa` | Checks the layout by rule, then has Qwen compare the original and the rendered page | Qwen |
| 10 | `export` | Writes PNG, CBZ, the full JSON and a review CSV | none |

Whatever the models cannot settle is flagged for review and left untouched on the page,
never guessed ([Review and correct](#review-and-correct)).

Qwen always runs on the GPU, behind a local server that stays loaded for the whole run.
The other models run where the config puts them:

| Config | Qwen | Layout (Koharu) | OCR | LaMa |
|---|---|---|---|---|
| `ascend.yaml` | `Qwen3.8-27B`, 4-bit, vLLM | CPU | CPU | not set up |
| `ascend-en.yaml` | same | GPU | GPU | GPU |
| `windows.yaml` | `Qwen3.5-9B`, 4-bit, llama.cpp | CPU | CPU | not set up |
| `windows-en.yaml` | same | CPU | GPU | CPU |

- `audit`, `reconcile`, `translate` and `qa` work on `server.max_num_seqs` pages at the
  same time. `context` cannot: each chunk builds on the one before it.
- LaMa needs its weights file at the path named in the config. Without it, lettering
  that sits on artwork keeps its original text.
- Design details and the saved-file format are in [docs/pipeline.md](docs/pipeline.md),
  the model adapters in [docs/models.md](docs/models.md), and rendering in
  [docs/rendering.md](docs/rendering.md).

### Model choices

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
and test dependencies. From this folder (on Windows, use `.venv\Scripts\python`):

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
2. Edit your machine's config (`configs/ascend.yaml` or `configs/windows.yaml`) for
   the Qwen model, server and font. Edit `configs/pipeline.yaml` for settings both
   machines share: input/output, language, reading direction, OCR readers, and retry
   budgets. A machine config can override any of them. Paths are relative to the
   YAML file.
3. Use `source_language: ja` or `en` when known. `auto` and `mixed` route all crops
   through the configured multilingual reader; there is no automatic per-region
   language classifier in this version.
4. Keep `render_uncertain: false` to preserve original pixels for unresolved text.

```bash
.venv/bin/python -m manga_pipeline plan --config configs/ascend.yaml
.venv/bin/python -m manga_pipeline doctor --config configs/ascend.yaml
.venv/bin/python -m manga_pipeline serve --config configs/ascend.yaml --print-command
```

These three inspection commands do not load/download models or contact a server.
`--config` is required; on Windows use `.venv\Scripts\python` and
`configs/windows.yaml`.

## Run on Windows (RTX 5070)

vLLM is Linux-only, and its 9.1 GB checkpoint does not fit beside a Windows desktop
on a 12 GB card. `configs/windows.yaml` therefore serves Qwen3.5-9B as a 4-bit GGUF
(~6.6 GB with its vision projector) through llama.cpp's OpenAI-compatible server,
keeps layout/OCR on the CPU, and renders with Microsoft JhengHei. Run everything
from this folder in PowerShell.

One-time setup, about 8 GB of downloads:

```powershell
# Layout/OCR packages (CPU PyTorch) and the `hf` downloader
uv venv --python 3.12 .venv     # skip if .venv already exists
uv pip install -p .venv -e ".[layout,manga,vl]"

# llama.cpp v0.6.0 (build b11429), CUDA 13 build; needs a driver reporting CUDA 13.x
$base = "https://github.com/ggml-org/llama.cpp/releases/download/b11429"
foreach ($zip in "llama-b11429-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip") {
    curl.exe -fL -o "$env:TEMP\$zip" "$base/$zip"
    Expand-Archive "$env:TEMP\$zip" bin\llama.cpp -Force
}

# Qwen3.5-9B Q4_K_M and its vision projector, at the revision pinned in configs/windows.yaml
.venv\Scripts\hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf mmproj-F16.gguf `
    --revision 3885219b6810b007914f3a7950a8d1b469d598a5 --local-dir models\qwen3.5-9b

.venv\Scripts\python -m manga_pipeline doctor --config configs/windows.yaml
```

`doctor` should report `font_exists`, `input_exists`, and all three
`llama_cpp_files_present` entries as `true`. Put the chapter's pages in `input\`,
then use two terminals:

```powershell
# Terminal 1: model server. Keep it open; Ctrl+C stops it.
.venv\Scripts\python -m manga_pipeline serve --config configs/windows.yaml

# Terminal 2: start once `curl.exe http://127.0.0.1:8000/health` prints {"status":"ok"}
.venv\Scripts\python -m manga_pipeline run --config configs/windows.yaml --stage all
```

Results go to `output\chapter\` (layout below); `--stage` and `--force` work as
described in [Stages and resume](#stages-and-resume). The first run also downloads
the layout and OCR checkpoints (about 2.5 GB). They run on the CPU, so expect
minutes per page.

llama.cpp keeps 1 GB of GPU memory free and places layers that do not fit on the
CPU, which is slower. Closing GPU-heavy apps (browsers, Discord, Steam) keeps the
whole model on the GPU. If the server runs out of memory on large pages, lower
`vlm.image_max_pixels`.

### English pages

Install the English reader once, then use `configs/windows-en.yaml` in both
terminals (results go to `output\chapter-en\`):

```powershell
uv pip install -p .venv -e ".[english]"
.venv\Scripts\python -m manga_pipeline serve --config configs/windows-en.yaml
.venv\Scripts\python -m manga_pipeline run --config configs/windows-en.yaml --stage all
```

PaddleOCR's dependencies move numpy from 2.5 to 2.3 and PyYAML to 6.0.2, and add
`opencv-contrib-python` beside `opencv-python`. The tests, layout detection and
English OCR were checked with that combination. `windows.yaml` turns off
PaddleOCR's oneDNN CPU acceleration, which crashes in PaddlePaddle 3.3.
`reading_direction` stays `rtl` for translated manga; set `ltr` for Western comics.

## Install model environments when ready

For the A100 (Linux) with vLLM; on Windows, follow the previous section instead.
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

`configs/ascend.yaml` serves `Qwen3.8-27B` (4-bit AWQ) through vLLM on one A100 and
uses the CPU for layout/OCR. This allows the Qwen server to remain loaded across
audit, reconciliation, translation, and QA. **40 GB is device memory; `--mem=64G`
is host RAM.**

The launcher uses `configs/ascend.yaml` unless `MANGA_CONFIG` is set. It requests
Ascend's `nextgen` partition, one GPU, and starts/stops its own Qwen server. It uses
your script-first sbatch convention and environment stage selection. Replace
`YOUR_PROJECT` with your Slurm allocation:

```bash
cd /users/PAS2119/drink36/manga-translation-pipeline
MANGA_STAGE=all sbatch scripts/ascend.sh --account=YOUR_PROJECT
```

For a different configuration:

```bash
MANGA_STAGE=all MANGA_CONFIG=configs/quality.yaml \
  sbatch scripts/ascend.sh --account=YOUR_PROJECT
```

`quality.yaml` and the two `paddle-vl16-*.yaml` variants extend `ascend.yaml`. To use
one on Windows, change its `extends:` line to `windows.yaml`.

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
.venv-vllm/bin/python -m manga_pipeline serve --config configs/ascend.yaml

# Terminal 2: runs the chapter; may download missing detector/OCR weights.
.venv-models/bin/python -m manga_pipeline run --config configs/ascend.yaml --stage all
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

- The A100 setup after the merge with the Windows changes: vLLM loading
  `Qwen3.8-27B`, and vLLM accepting the fixed-order (`prefixItems`) JSON schema used
  for translate and verify replies. That schema has only been checked on llama.cpp.
  The shared sampling settings (temperature 0.7) are also new for the A100.
- Peak GPU memory on either machine. If a server runs out of memory, lower
  `server.max_model_len` or `vlm.image_max_pixels`.
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
