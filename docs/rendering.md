# Rendering and review

The renderer supports horizontal Traditional Chinese lettering, conservative
white-background erasure, and an optional local LaMa TorchScript model. It does
not download an inpainting model. The CPU synthetic tests cover preservation,
font fitting, masks, overlap handling, rerendering and CBZ output. Real manga
quality and the LaMa model path have not been validated in this workspace.

## Configuration

Use a mapping for `render.inpaint` in the pipeline YAML:

```yaml
render:
  font_path: /usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc
  font_index: 3
  font_size_min: 10
  font_size_max: 48
  padding: 5
  line_spacing: 1.15
  mask_dilation: 1
  direction: horizontal
  sfx_policy: preserve
  render_uncertain: false
  inpaint:
    backend: auto
    device: cpu
    torchscript_path: null
    signature: image_mask
    context_padding: 64
```

Paths in the YAML are resolved relative to that YAML file. The Noto TTC path
above exists on the inspected system; face index 3 was exercised with
Traditional Chinese in the CPU test suite. Use the matching path and face for a
different font collection. Without an explicit font, only the Latin-only demo
uses Pillow's packaged font. Missing Traditional Chinese glyphs are reported
instead of rendering replacement boxes.

Font sizes, padding, mask dilation and context padding use original-image
pixels. `line_spacing` is a multiplier of at least 1.0. The renderer wraps
characters and Latin words, preserves explicit newlines, and tries integer
font sizes from maximum down to minimum. It centers each line and the text
block. It does not implement vertical lettering, ruby/furigana, translated
lettering that follows a curved baseline, decorative SFX fonts, or professional
CJK line-breaking rules. Translations are not automatically shortened to fit.

## Geometry and cleaning

Every bbox is `[left, top, right, bottom]` in the canonical original-image
coordinate system. The bbox and optional polygon restrict the source glyph
mask; they are not themselves erasure masks. The renderer expects binary,
full-page PNG masks, with white pixels denoting the selected area:

| Field | Meaning |
| --- | --- |
| `bbox`, `polygon` | Text region in original pixels |
| `mask_path` | Original lettering to erase |
| `bubble_bbox` | Rectangle available for translated text |
| `bubble_mask_path` | Bubble interior available for translated text |

Glyph masks are dilated by `mask_dilation`. If the dilation crosses an available
bubble mask, the region is left intact and flagged. Rasterized translated
lettering must fit inside the padded bubble mask. Without a bubble mask, the
renderer can check only its placement rectangle; it cannot establish that text
fits a curved bubble. Without bubble geometry, it uses the text bbox, which
can be much too small for a Chinese translation.

The `plain` backend fills only masked pixels with an estimated white
background. It requires at least 97% of the three-pixel surrounding ring to
have every RGB channel at least 235, with low variation among those bright
pixels. It will not fill a colored or textured region white.

If no glyph mask is supplied, a threshold fallback is allowed only when at
least 60% of pixels in the text region are near-white. Pixels with a channel
below 210 become the candidate lettering mask, followed by the same
neighborhood check. This is a conservative image heuristic, not learned glyph
segmentation: an inaccurate detection box can still include a border or small
piece of artwork. Inspect threshold-derived masks on real pages and prefer
accurate glyph segmentation where available.

`auto` tries plain fill first, then LaMa only when a local TorchScript path is
configured. `lama` explicitly requests LaMa. Neither setting provides a way to
infer a safe glyph mask on a textured crop: supply a correct mask first. Failed
mask checks, missing weights, model errors, font errors, overlap and overflow
leave that region's original lettering unchanged.

The renderer fits translated text before erasing anything. It also rejects a
region if its erasure or new lettering overlaps an already rendered region.
Multiple text detections sharing the same bubble may therefore need grouping
or a manually assigned layout; this renderer does not independently partition
a bubble between dialogue blocks.

## Optional local LaMa model

For a separate CPU-only rendering environment, from the project folder:

```bash
uv venv --python 3.12 .venv-render
uv pip install -p .venv-render -e .
uv pip install -p .venv-render --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6'
```

When using an existing CUDA environment, use its compatible PyTorch installation
and set `device: cuda` only when memory is available. The renderer retains one
loaded TorchScript model in-process. It does not stop an independently running
Qwen server or manage that server's GPU allocation.

Set `torchscript_path` to an existing local file loadable by
`torch.jit.load`. A training checkpoint or an arbitrary file with a `.pt` suffix
is not sufficient. The original [LaMa repository](https://github.com/advimman/lama)
provides its own model and inference workflow; this package does not convert
that workflow's checkpoints to TorchScript or install its historical training
environment. Obtain or export a compatible TorchScript module separately and
record its provenance.

The adapter's exact contract is:

| Item | Required form |
| --- | --- |
| Image input | Float32 RGB tensor, `[1, 3, H, W]`, values in `[0, 1]` |
| Mask input | Float32 tensor, `[1, 1, H, W]`, 1 means erase |
| Default call | `model(image, mask)` with `signature: image_mask` |
| Alternative call | `model({"image": image, "mask": mask})` with `signature: dict` |
| Output | RGB tensor `[1, 3, H, W]` in `[0, 1]`; alternatively a dict with `inpainted` or `image`, or a tuple/list whose first item is that tensor |

The adapter extracts a local crop around the glyph mask, adds
`context_padding`, and pads height and width to multiples of eight. It crops
the result back and copies generated pixels only inside the glyph mask. Each
region uses the original source image as its inpainting input, so previous
inpainting results are never fed into subsequent inpainting calls.

A local model compatibility check, after setting the path, is:

```python
from pathlib import Path
import torch

weights = Path("/absolute/path/to/lama-torchscript.pt")
model = torch.jit.load(str(weights), map_location="cpu").eval()
image = torch.ones(1, 3, 64, 64)
mask = torch.zeros(1, 1, 64, 64)
mask[:, :, 24:40, 24:40] = 1
with torch.inference_mode():
    output = model(image, mask)  # Adapt this call for signature: dict.
print(type(output))
```

This checks that the configured call executes, not restoration quality.

## Outputs and statuses

`render_page(page, output_dir, cfg)` writes `<page_id>.original.png` and
`<page_id>.translated.png`. Its original PNG is a canonical RGB copy, not a
byte-identical copy of a source JPEG, TIFF or PNG. The preparation stage
separately retains original input bytes. Output names are derived from stable
page IDs; calling the function again in the same directory replaces those PNGs
using the original source image. It rejects an output path that would overwrite
that source.

The report contains `output_path`, `original_path`, `page_id`, `status`,
`issues`, and per-region results. Successfully typeset regions also include
the font size, wrapped lines, text bbox, zero measured overflow pixels, mask
source, and selected inpainting backend. Each issue has `region_id`, `code`
and `message`.

| Status | Meaning |
| --- | --- |
| `auto_passed` | The implemented checks passed; this is not human approval |
| `needs_review` | A region was skipped or an unresolved issue remains |
| `preserved` | The region is SFX intentionally kept under `sfx_policy: preserve` |

Intentional SFX preservation is not an error. An unresolved ordinary dialogue
region is a review item even though its original lettering is safely retained.
With `render_uncertain: true`, a region may be drawn despite unresolved text
uncertainty, but its result remains `needs_review` with `uncertain_rendered`.
This setting cannot turn an uncertain translation into an approved one.

The CBZ helper writes pages in the supplied reading order with safe sequential
names and fixed ZIP timestamps. The pipeline exports original and translated
archives plus chapter JSON and review JSON/CSV. Review output must be checked
before treating an automatically produced chapter as finished.

## Corrections and visual QA

Use `review.overrides_path` for reviewed text corrections. Editing a review CSV
alone does not apply a correction. An overrides file maps stable region IDs to
fields, for example:

```json
{
  "p0001_r0001": {
    "original_text": "確認した日本語",
    "translation": "人工確認的繁體中文譯文",
    "mask_path": "/absolute/path/to/reviewed-glyph-mask.png",
    "bubble_bbox": [20, 30, 200, 240],
    "bubble_mask_path": "/absolute/path/to/reviewed-bubble-mask.png"
  }
}
```

A source-text override acknowledges that source transcription and clears its
existing text uncertainty. A translation override alone does not resolve
uncertainty in the source. Reading-order checks can still raise a separate
review requirement. Set `review.overrides_path` in `configs/pipeline.yaml`,
then rerun from the project directory with your machine config:

```bash
manga-pipeline run --config configs/ascend.yaml --stage all
```

Changing the overrides changes affected artifact provenance. A full resumable
stage run also refreshes dependent context, translations and exports as needed.

Text overrides do not themselves correct geometry. Optional `mask_path`,
`bubble_bbox` and `bubble_mask_path` overrides supply reviewed erasure and layout
geometry during rendering; omit any field that does not need correction.
Use absolute mask paths, original-page coordinates and full-page mask
dimensions. Other detector geometry changes require corrected upstream
detector output. Do not edit cached stage JSON as if it were an overrides file;
cached artifacts have dependency hashes.

Final visual QA compares the original and rendered page and can request bounded
mask or layout retries. Those retries regenerate cleaning and typesetting from
the original. A larger dilation can help residual edge pixels; a smaller font
maximum can help visually oversized text. The latter cannot fix deterministic
overflow when the renderer has already tried the configured minimum size.
Such a region needs a better text area, an explicitly lower minimum, or a
reviewed shorter translation.

The retry adjustments currently apply to the page's render configuration, so a
retry can also change the font or dilation of otherwise successful regions.
These are bounded heuristic repairs, not automatic redrawing of arbitrary masks
or complete typography optimization. Deterministic checks do not independently
read the final pixels to prove every source character was erased; final visual
inspection remains necessary for inaccurate segmentation and artistic pages.
