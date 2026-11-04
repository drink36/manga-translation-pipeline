"""Offline orchestration regression tests using synthetic pages and model mocks."""

import copy
import io
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from manga_pipeline.config import load_config
from manga_pipeline.demo import DemoVLM, create_demo
from manga_pipeline.geometry import associate, normalize
from manga_pipeline.pipeline import Pipeline
from manga_pipeline.storage import file_hash, write_json


class ScriptedVLM:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def ask(self, task, payload, images=None):
        self.calls.append((task, copy.deepcopy(payload), len(images or [])))
        return self.handler(task, payload)


@pytest.fixture
def subject(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 120), "white").save(source)
    cfg = {
        "mode": "real", "input": str(source), "output": str(tmp_path / "output"),
        "source_language": "ja", "reading_direction": "rtl",
        "vlm": {"model": "mock-vision"},
        "audit": {"enabled": True, "match_overlap": 0.6},
        "ocr": {"qwen_blind_read": "always"},
        "review": {"flag_arbitrated_ocr": True},
        "retries": {"region_retries": 0, "ocr_retries": 1,
                    "translation_retries": 1, "layout_retries": 1},
        "render": {"font_size_min": 10, "font_size_max": 24, "sfx_policy": "preserve"},
        "qa": {"visual_enabled": True}, "export": {"cbz": False},
    }
    pipeline = Pipeline(cfg)
    page = {
        "page_id": "p0001", "width": 160, "height": 120, "source_path": str(source),
        "regions": [], "layouts": [], "issues": [],
    }
    region = pipeline._region({"bbox": [20, 20, 70, 70], "kind": "text"}, page, 1)
    region.update(crop_bbox=[20, 20, 70, 70], original_text="猫", ocr_candidates={
        "reader_a": {"text": "猫"}, "reader_b": {"text": "猫"},
    })
    page["regions"] = [region]
    return pipeline, page


def test_prepare_skips_macos_metadata_and_hidden_files(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(buffer, "PNG")
    page = buffer.getvalue()
    archive = tmp_path / "chapter.cbz"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("ch1/002.png", page)
        zf.writestr("ch1/001.png", page)
        zf.writestr("__MACOSX/ch1/._001.png", b"AppleDouble metadata, not an image")
        zf.writestr("ch1/.hidden.png", b"not an image")
    folder = tmp_path / "pages"
    folder.mkdir()
    (folder / "001.png").write_bytes(page)
    (folder / "._001.png").write_bytes(b"AppleDouble metadata, not an image")
    expected = {archive: [f"{archive}!ch1/001.png", f"{archive}!ch1/002.png"], folder: [str(folder / "001.png")]}
    for source, names in expected.items():
        pipeline = Pipeline({"mode": "real", "input": str(source), "output": str(tmp_path / f"out-{source.stem}")})
        pipeline.prepare()
        assert [pipeline.store.required(p, "prepare")["source_name"] for p in pipeline.page_ids()] == names


def test_blind_ocr_receives_no_candidate_transcriptions(subject):
    pipeline, page = subject
    page["regions"][0]["ocr_candidates"]["reader_b"]["text"] = "犬"

    def respond(task, payload):
        if task in {"ocr", "reconcile"}:
            return {"text": "猫", "uncertain": False, "reason": "visible characters"}
        if task == "reading_order":
            return {"region_ids": ["p0001_r0001"], "uncertain": False, "reason": "single region"}
        raise AssertionError(task)

    pipeline._vlm = ScriptedVLM(respond)
    result = pipeline.stage_reconcile(page)
    first_ocr = next(payload for task, payload, _ in pipeline.vlm.calls if task == "ocr")
    assert set(first_ocr) == {"region_id", "language"}
    reconciliation = next(payload for task, payload, _ in pipeline.vlm.calls if task == "reconcile")
    assert reconciliation["candidates"] == {"reader_a": "猫", "reader_b": "犬", "qwen_blind": "猫"}
    assert result["regions"][0]["status"] == "needs_review"
    assert "ocr_disagreement_arbitrated" in result["regions"][0]["uncertainty"]


def test_ocr_retries_are_bounded_and_unresolved_text_stays_flagged(subject):
    pipeline, page = subject
    page["regions"][0]["ocr_candidates"]["reader_b"]["text"] = "犬"

    def respond(task, payload):
        if task in {"ocr", "reconcile"}:
            return {"text": "猫?", "uncertain": True, "reason": "small ambiguous character"}
        return {"region_ids": ["p0001_r0001"], "uncertain": False, "reason": "single region"}

    pipeline._vlm = ScriptedVLM(respond)
    result = pipeline.stage_reconcile(page)
    assert sum(task == "reconcile" for task, *_ in pipeline.vlm.calls) == 2
    assert sum(task == "ocr" for task, *_ in pipeline.vlm.calls) == 2
    assert "ocr_disagreement_unresolved" in result["regions"][0]["uncertainty"]


@pytest.mark.parametrize("policy", ["disagreements", "always"])
def test_single_reader_route_is_confirmed_by_matching_blind_read(subject, policy):
    pipeline, page = subject
    pipeline.cfg["ocr"]["qwen_blind_read"] = policy
    page["regions"][0]["ocr_candidates"] = {"paddle_english": {"text": "WAIT!"}}

    def respond(task, payload):
        if task == "ocr":
            return {"text": "WAIT!", "uncertain": False, "reason": "visible characters"}
        if task == "reading_order":
            return {"region_ids": ["p0001_r0001"], "uncertain": False, "reason": "single region"}
        raise AssertionError(f"Two matching independent readings need no {task}")

    pipeline._vlm = ScriptedVLM(respond)
    region = pipeline.stage_reconcile(page)["regions"][0]
    assert region["ocr_decision"] == "independent_transcriptions_agree"
    assert region["original_text"] == "WAIT!"
    assert region["uncertainty"] == [] and region["status"] == "auto_passed"


def test_single_reader_route_still_arbitrates_a_mismatching_blind_read(subject):
    pipeline, page = subject
    page["regions"][0]["ocr_candidates"] = {"paddle_english": {"text": "WAIT!"}}

    def respond(task, payload):
        if task == "ocr":
            return {"text": "WAIT?", "uncertain": False, "reason": "visible characters"}
        if task == "reconcile":
            return {"text": "WAIT!", "uncertain": False, "reason": "exclamation mark is visible"}
        return {"region_ids": ["p0001_r0001"], "uncertain": False, "reason": "single region"}

    pipeline._vlm = ScriptedVLM(respond)
    region = pipeline.stage_reconcile(page)["regions"][0]
    assert region["ocr_decision"] == "vlm_arbitrated"
    assert "ocr_disagreement_arbitrated" in region["uncertainty"]


def test_source_uncertainty_survives_passing_translation_verifier(subject):
    pipeline, page = subject
    page["regions"][0]["uncertainty"] = ["ocr_disagreement_arbitrated"]

    def respond(task, payload):
        if task == "translate":
            return {"translations": [{"region_id": "p0001_r0001", "text": "貓", "uncertain": False, "reason": ""}]}
        return {"checks": [{"region_id": "p0001_r0001", "passed": True, "issues": []}]}

    pipeline._vlm = ScriptedVLM(respond)
    with patch.object(pipeline, "read_context", return_value={}):
        result = pipeline.stage_translate(page)
    translation = result["regions"][0]["translation"]
    assert translation["status"] == "needs_review"
    assert "source_requires_review" in translation["issues"]


def test_translation_uncertainty_is_not_cleared_by_verifier_and_retry_is_bounded(subject):
    pipeline, page = subject

    def respond(task, payload):
        if task == "translate":
            return {"translations": [{"region_id": "p0001_r0001", "text": "貓", "uncertain": True, "reason": "uncertain name"}]}
        return {"checks": [{"region_id": "p0001_r0001", "passed": True, "issues": []}]}

    pipeline._vlm = ScriptedVLM(respond)
    with patch.object(pipeline, "read_context", return_value={}):
        result = pipeline.stage_translate(page)
    assert sum(task == "translate" for task, *_ in pipeline.vlm.calls) == 2
    assert result["regions"][0]["translation"]["status"] == "needs_review"
    second = [payload for task, payload, _ in pipeline.vlm.calls if task == "translate"][1]
    assert "uncertain name" in second["prior_issues"]["p0001_r0001"]


def test_audit_flags_detector_only_regions_instead_of_silent_agreement(subject):
    pipeline, page = subject
    pipeline._vlm = ScriptedVLM(lambda *_: {"regions": [], "uncertain": False})
    result = pipeline.stage_audit(page)
    assert len(result["regions"]) == 1  # Preserve the detector's evidence.
    assert (result["coverage"]["status"] == "needs_review"
            or result["regions"][0]["status"] == "needs_review")


def test_audit_does_not_discard_material_extension_of_detected_text(subject):
    pipeline, page = subject
    page["regions"][0]["bbox"] = [20, 20, 40, 40]
    proposal = {"bbox": normalize([20, 20, 120, 40], 160, 120), "kind": "text", "reason": "entire line"}
    pipeline._vlm = ScriptedVLM(lambda *_: {"regions": [proposal], "uncertain": False})
    result = pipeline.stage_audit(page)
    assert (any(r["bbox"][2] >= 120 for r in result["regions"])
            or result["coverage"]["status"] == "needs_review"
            or any(r["status"] == "needs_review" for r in result["regions"]))


def test_detector_confirmation_merges_every_matching_mask(subject):
    pipeline, page = subject
    region = page["regions"][0]
    region["uncertainty"] = ["vlm_only_region_requires_confirmation"]

    class TwoMaskDetector:
        """Confirms the crop twice, with the glyphs split between two masks."""

        def detect(self, image_path, mask_dir):
            mask_dir = Path(mask_dir)
            mask_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(image_path) as crop:
                size = crop.size
            results = []
            for index, (left, right) in enumerate([(10, 30), (30, 50)]):
                mask = Image.new("L", size)
                ImageDraw.Draw(mask).rectangle((left, 10, right - 1, 40), fill=255)
                mask.save(mask_dir / f"{index}.png")
                results.append({"kind": "text", "bbox": [0, 0, *size], "mask_path": str(mask_dir / f"{index}.png")})
            return results

    pipeline._detector = TwoMaskDetector()
    pipeline.store.path("p0001", "audit").parent.mkdir(parents=True)
    pipeline._confirm_proposals(page, Image.open(page["source_path"]).convert("RGB"))
    assert region["status"] == "auto_passed"
    with Image.open(region["mask_path"]) as merged:
        columns = np.flatnonzero((np.asarray(merged) > 127).any(axis=0))
    # The padded crop starts at the page origin, so crop and page columns coincide.
    assert (columns.min(), columns.max()) == (10, 49)


def test_bubble_text_is_grouped_even_when_coverage_audit_is_disabled(subject):
    pipeline, page = subject
    pipeline.cfg["audit"]["enabled"] = False
    bubble = pipeline._region({"bbox": [10, 10, 150, 110], "kind": "bubble"}, page, 2)
    second = pipeline._region({"bbox": [80, 20, 130, 70], "kind": "text"}, page, 3)
    page["layouts"] = [bubble]
    page["regions"] = associate([page["regions"][0], second], [bubble], "rtl")

    def respond(task, payload):
        raise AssertionError("A disabled audit must not call the VLM")

    pipeline._vlm = ScriptedVLM(respond)
    result = pipeline.stage_audit(page)
    assert len(result["regions"]) == 1
    members = {member["region_id"] for member in result["regions"][0]["member_regions"]}
    assert members == {"p0001_r0001", "p0001_r0003"}
    assert "coverage_not_checked" in result["issues"]


def test_qa_rerender_does_not_mutate_render_stage_artifact(subject):
    pipeline, page = subject
    initial = pipeline.store.root / "rendered" / "p0001.png"
    initial.parent.mkdir(parents=True)
    Image.new("RGB", (160, 120), "white").save(initial)
    page["render"] = {"output_path": str(initial), "issues": []}
    inputs = {"artifacts": {}}
    path = pipeline.store.path("p0001", "render")
    pipeline.store.save(path, page, pipeline.store.fingerprint("render", inputs), "render", inputs)
    before = file_hash(initial)
    results = iter([
        {"passed": False, "issues": [{"region_id": "p0001_r0001", "category": "mask", "reason": "residual text"}]},
        {"passed": True, "issues": []},
    ])
    pipeline._vlm = ScriptedVLM(lambda *_: next(results))

    def rerender(page, directory, cfg):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "p0001.png"
        Image.new("RGB", (160, 120), "black").save(output)
        return {"output_path": str(output), "issues": []}

    with patch("manga_pipeline.rendering.render_page", side_effect=rerender):
        result = pipeline.stage_qa(copy.deepcopy(page))
    assert result["visual_qa"]["status"] == "auto_passed"
    assert file_hash(initial) == before
    assert pipeline.store.required("p0001", "render")


@pytest.mark.parametrize("category,setting,configured,expected", [
    ("mask", "mask_dilation", 1, [2]),
    ("mask", "mask_dilation", 3, []),
    ("mask", "mask_dilation", 5, []),  # Must not be "adjusted" down to 3.
    ("layout", "font_size_max", 20, [17]),
    ("layout", "font_size_max", 10, []),  # Already at font_size_min.
])
def test_visual_retry_rerenders_only_when_a_setting_can_still_improve(subject, category, setting, configured, expected):
    pipeline, page = subject
    pipeline.cfg["render"][setting] = configured
    page["render"] = {"output_path": page["source_path"], "issues": []}
    issue = {"region_id": "p0001_r0001", "category": category, "reason": "visible defect"}
    pipeline._vlm = ScriptedVLM(lambda *_: {"passed": False, "issues": [issue]})

    def rerender(page, directory, cfg):
        return {"output_path": page["source_path"], "issues": []}

    with patch("manga_pipeline.rendering.render_page", side_effect=rerender) as mocked:
        result = pipeline.stage_qa(page)
    assert [call.args[2][setting] for call in mocked.call_args_list] == expected
    assert result["visual_qa"]["status"] == "needs_review"


def test_page_scoped_visual_issue_is_accepted_and_exported(subject):
    pipeline, page = subject
    page["render"] = {"output_path": page["source_path"], "issues": []}
    pipeline._vlm = ScriptedVLM(lambda *_: {
        "passed": False,
        "issues": [{"region_id": "page", "category": "other", "reason": "image contrast insufficient to inspect"}],
    })
    result = pipeline.stage_qa(page)
    assert result["visual_qa"]["status"] == "needs_review"
    with patch.object(pipeline, "page_ids", return_value=["p0001"]), \
         patch.object(pipeline.store, "required", return_value=result), \
         patch.object(pipeline, "read_context", return_value={}):
        exported = pipeline.export()
    assert exported["status"] == "needs_review"
    assert any("image contrast insufficient to inspect" in item["issues"]
               for item in exported["review_items"])


def test_required_rejects_transitively_stale_artifact(subject):
    pipeline, page = subject
    previous = None
    for stage in ("prepare", "detect", "audit"):
        inputs = {"artifacts": {str(previous): file_hash(previous)}} if previous else {"artifacts": {}}
        path = pipeline.store.path("p0001", stage)
        pipeline.store.save(path, copy.deepcopy(page), pipeline.store.fingerprint(stage, inputs), stage, inputs)
        previous = path
    assert pipeline.store.required("p0001", "audit")
    changed = copy.deepcopy(page)
    changed["new_source_annotation"] = "updated"
    inputs = {"artifacts": {}}
    pipeline.store.save(pipeline.store.path("p0001", "prepare"), changed,
                        pipeline.store.fingerprint("prepare", inputs), "prepare", inputs)
    with pytest.raises(ValueError, match="[Ss]tale|changed|rerun"):
        pipeline.store.required("p0001", "audit")


def test_export_keeps_source_review_flag_even_when_visual_qa_passes(subject):
    pipeline, page = subject
    page["regions"][0]["uncertainty"] = ["reading_order_uncertain"]
    page["regions"][0]["translation"] = {"text": "貓", "status": "needs_review", "issues": ["source_requires_review"]}
    page["render"] = {"output_path": page["source_path"], "issues": []}
    page["visual_qa"] = {"status": "auto_passed", "attempts": [{"passed": True, "issues": []}]}
    with patch.object(pipeline, "page_ids", return_value=["p0001"]), \
         patch.object(pipeline.store, "required", return_value=page), \
         patch.object(pipeline, "read_context", return_value={}):
        exported = pipeline.export()
    assert exported["status"] == "needs_review"
    assert "reading_order_uncertain" in exported["review_items"][0]["issues"]


def _artifact_snapshot(output, stages):
    paths = [path for path in (output / "pages").glob("*/*.json") if path.stem in stages]
    if "context" in stages:
        paths.append(output / "context.json")
    return {str(path.relative_to(output)): (file_hash(path), path.stat().st_mtime_ns) for path in paths}


def _edit_demo_config(path, update):
    value = yaml.safe_load(path.read_text())
    update(value)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False))
    return load_config(path)


def test_demo_full_export_and_cached_rerun_need_no_model_requests(tmp_path):
    config_path = create_demo(tmp_path / "demo")
    cfg = load_config(config_path)
    with patch("urllib.request.urlopen", side_effect=AssertionError("Demo must not access a server")):
        Pipeline(cfg).run()
    output = Path(cfg["output"])
    chapter = json.loads((output / "export" / "chapter.json").read_text())
    assert chapter["mode"] == "demo"
    assert chapter["status"] == "auto_passed"
    assert chapter["review_items"] == []
    assert [page["page_id"] for page in chapter["pages"]] == ["p0001", "p0002"]
    for page in chapter["pages"]:
        assert Path(page["render"]["output_path"]).is_file()
        assert page["regions"][0]["translation"]["text"]
        assert page["visual_qa"]["status"] == "auto_passed"
    for name in ("original.cbz", "translated.demo.cbz"):
        with zipfile.ZipFile(output / "export" / name) as archive:
            assert len(archive.namelist()) == 2
            assert archive.testzip() is None
    stages = {"prepare", "detect", "audit", "ocr", "reconcile", "context", "translate", "render", "qa"}
    before = _artifact_snapshot(output, stages)
    with patch.object(DemoVLM, "ask", side_effect=AssertionError("Cached rerun must not request inference")):
        Pipeline(load_config(config_path)).run()
    assert _artifact_snapshot(output, stages) == before


def test_cached_rerun_hashes_each_file_at_most_once(tmp_path):
    # Verification used to re-walk the whole chapter for every page, re-hashing images quadratically.
    config_path = create_demo(tmp_path / "demo")
    Pipeline(load_config(config_path)).run()
    hashed = []

    def recording_hash(path):
        hashed.append(str(path))
        return file_hash(path)

    with patch("manga_pipeline.storage.file_hash", side_effect=recording_hash):
        Pipeline(load_config(config_path)).run()
    assert hashed and len(hashed) == len(set(hashed))


def test_new_run_rechecks_prerequisites_edited_since_the_previous_run(tmp_path):
    pipeline = Pipeline(load_config(create_demo(tmp_path / "demo")))
    pipeline.run()
    ocr = pipeline.store.path("p0001", "ocr")
    edited = json.loads(ocr.read_text(encoding="utf-8"))
    edited["regions"][0]["original_text"] = "edited outside the pipeline"
    write_json(ocr, edited)
    with pytest.raises(ValueError, match="Stale prerequisite"):
        pipeline.run("translate")


def test_demo_translation_override_reuses_ocr_reconcile_and_context(tmp_path):
    config_path = create_demo(tmp_path / "demo")
    cfg = load_config(config_path)
    Pipeline(cfg).run()
    output = Path(cfg["output"])
    upstream = {"prepare", "detect", "audit", "ocr", "reconcile", "context"}
    before = _artifact_snapshot(output, upstream)
    overrides = config_path.parent / "review.json"
    overrides.write_text(json.dumps({"p0001_r0001": {"translation": "Reviewed text"}}))
    cfg = _edit_demo_config(config_path, lambda value: value.update(review={"overrides_path": "review.json"}))
    calls = []
    original_ask = DemoVLM.ask

    def recording_ask(self, task, payload, images=None):
        calls.append(task)
        return original_ask(self, task, payload, images)

    with patch.object(DemoVLM, "ask", recording_ask):
        Pipeline(cfg).run()
    assert _artifact_snapshot(output, upstream) == before
    assert not set(calls).intersection({"coverage", "ocr", "reconcile", "reading_order", "context"})
    assert "visual_qa" in calls
    chapter = json.loads((output / "export" / "chapter.json").read_text())
    translation = chapter["pages"][0]["regions"][0]["translation"]
    assert translation["text"] == "Reviewed text"
    assert translation["status"] == "human_reviewed"
    assert translation["origin"] == "review_overrides"
    assert chapter["status"] == "auto_passed"  # Automation never labels the whole chapter human-reviewed.


def test_demo_render_config_change_reuses_all_model_work_except_visual_qa(tmp_path):
    config_path = create_demo(tmp_path / "demo")
    cfg = load_config(config_path)
    Pipeline(cfg).run()
    output = Path(cfg["output"])
    upstream = {"prepare", "detect", "audit", "ocr", "reconcile", "context", "translate"}
    before = _artifact_snapshot(output, upstream)
    render_before = _artifact_snapshot(output, {"render", "qa"})

    def change_font(value):
        value["render"]["font_size_max"] = value["render"]["font_size_max"] - 2

    cfg = _edit_demo_config(config_path, change_font)
    calls = []
    original_ask = DemoVLM.ask

    def recording_ask(self, task, payload, images=None):
        calls.append(task)
        return original_ask(self, task, payload, images)

    with patch.object(DemoVLM, "ask", recording_ask):
        Pipeline(cfg).run()
    assert _artifact_snapshot(output, upstream) == before
    assert _artifact_snapshot(output, {"render", "qa"}) != render_before
    assert calls == ["visual_qa", "visual_qa"]
    chapter = json.loads((output / "export" / "chapter.json").read_text())
    assert chapter["status"] == "auto_passed"
