# Validation record

Checked on 2026-10-14 using the lightweight project `.venv`.

```text
python -m pytest -q
58 passed, 16 subtests passed in 17.43s
```

Also completed:

- `python -m compileall -q src`
- `bash -n scripts/ascend.sh`
- Configuration planning and local dependency inspection.
- Two-page synthetic end-to-end run, followed by a fully cached rerun.
- Visual inspection of the rendered Traditional Chinese demo page.

The tests cover strict model response/region-ID validation, malformed and
truncated responses, blind OCR input isolation, bounded retries, unresolved
source propagation, detector/coverage disagreement, coordinate mapping, command
bridges, mocked Koharu output handling, immutable render retries, transitive
artifact invalidation, manual edits, CJK wrapping/fitting, masks, source
preservation, and PNG/CBZ/JSON exports.

End-to-end tests verify that a cached rerun makes zero scripted model calls;
translation-only edits reuse OCR/reconciliation/context; rendering-only changes
reuse translation. These checks use scripted fixtures, not real OCR or translation.

Public model revision metadata and optional-package dependency metadata were
inspected. No model weights, actual model forward passes, vLLM server, LaMa
weights, GPU benchmarks, or Slurm jobs were used. Hardware compatibility and
translation quality remain to be measured on representative manga.
