from pathlib import Path
import copy
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from manga_pipeline.rendering import export_cbz, render_page


def make_page(tmp_path, background="white"):
    image = Image.new("RGB", (240, 140), background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 50, 65, 65), fill="black")
    draw.rectangle((75, 50, 90, 65), fill="black")
    source = tmp_path / "source.png"
    image.save(source)
    return {"page_id": "page001", "source_path": str(source), "width": 240,
            "height": 140, "regions": [{"region_id": "r1", "bbox": [45, 45, 96, 71],
             "bubble_bbox": [15, 15, 225, 125], "kind": "text", "original_text": "AB",
             "uncertainty": [], "status": "auto_passed",
             "translation": {"text": "Hello", "status": "auto_passed", "issues": []}}]}


def config(**overrides):
    return {"font_size_min": 10, "font_size_max": 20, "padding": 5,
            "mask_dilation": 1, "inpaint": {"backend": "auto"}, **overrides}


def image_array(path):
    with Image.open(path) as image:
        return np.array(image)


def test_plain_fill_preserves_source_and_only_changes_erasure_or_type_area(tmp_path):
    page = make_page(tmp_path)
    before = Path(page["source_path"]).read_bytes()
    report = render_page(page, tmp_path / "out", config())
    assert report["status"] == "auto_passed", report
    assert report["regions"][0]["mask_source"] == "white_background_threshold"
    assert report["regions"][0]["overflow_pixels"] == 0
    assert Path(page["source_path"]).read_bytes() == before
    rendered = image_array(report["output_path"])
    original = image_array(page["source_path"])
    assert np.all(rendered[50:60, 50:60] == 255)
    assert np.array_equal(rendered[:10], original[:10])
    assert np.array_equal(image_array(report["original_path"]), original)


def test_uncertain_and_missing_translation_retain_original(tmp_path):
    page = make_page(tmp_path)
    page["regions"][0]["uncertainty"] = ["OCR disagreement"]
    report = render_page(page, tmp_path / "uncertain", config())
    assert report["issues"][0]["code"] == "uncertain_text"
    assert np.array_equal(image_array(report["output_path"]), image_array(page["source_path"]))
    page["regions"][0]["translation"]["text"] = ""
    report = render_page(page, tmp_path / "missing", config())
    assert report["issues"][0]["code"] == "missing_translation"


def test_overflow_does_not_erase_original(tmp_path):
    page = make_page(tmp_path)
    page["regions"][0]["bubble_bbox"] = [45, 45, 60, 60]
    page["regions"][0]["translation"]["text"] = "A very long translation that cannot fit"
    report = render_page(page, tmp_path / "out", config(font_size_min=20))
    assert report["issues"][0]["code"] == "text_overflow"
    assert np.array_equal(image_array(report["output_path"]), image_array(page["source_path"]))


def test_artwork_never_receives_plain_fill_without_white_background(tmp_path):
    page = make_page(tmp_path, background=(150, 130, 100))
    mask = Image.new("L", (240, 140))
    draw = ImageDraw.Draw(mask)
    draw.rectangle((50, 50, 65, 65), fill=255)
    draw.rectangle((75, 50, 90, 65), fill=255)
    mask_path = tmp_path / "glyph.png"
    mask.save(mask_path)
    page["regions"][0]["mask_path"] = str(mask_path)
    report = render_page(page, tmp_path / "out", config())
    assert report["issues"][0]["code"] == "inpaint_unavailable"
    assert np.array_equal(image_array(report["output_path"]), image_array(page["source_path"]))


def test_sfx_preservation_is_intentional_not_failure(tmp_path):
    page = make_page(tmp_path)
    page["regions"][0]["kind"] = "sfx"
    report = render_page(page, tmp_path / "out", config())
    assert report["status"] == "auto_passed"
    assert report["regions"][0]["status"] == "preserved"
    assert np.array_equal(image_array(report["output_path"]), image_array(page["source_path"]))


def test_overlap_is_reported_and_first_render_preserved(tmp_path):
    page = make_page(tmp_path)
    first = render_page(page, tmp_path / "first", config())
    duplicate = copy.deepcopy(page["regions"][0])
    duplicate["region_id"] = "r2"
    page["regions"].append(duplicate)
    report = render_page(page, tmp_path / "two", config())
    assert report["regions"][1]["status"] == "duplicate" and not report["issues"]
    assert np.array_equal(image_array(first["output_path"]), image_array(report["output_path"]))
    page["regions"][1].update(region_id="r3", bbox=[80, 45, 140, 71], original_text="Something else")
    report = render_page(page, tmp_path / "three", config())
    assert report["issues"][0]["code"] == "region_overlap"
    assert np.array_equal(image_array(first["output_path"]), image_array(report["output_path"]))


def test_preferred_source_wins_overlap(tmp_path):
    page = make_page(tmp_path)
    twin = copy.deepcopy(page["regions"][0])
    twin.update(region_id="r2", detected_by="qwen", translation={"text": "Hi", "status": "auto_passed", "issues": []})
    page["regions"].append(twin)
    statuses = lambda report: {r["region_id"]: r["status"] for r in report["regions"]}
    assert statuses(render_page(page, tmp_path / "order", config())) == {"r1": "auto_passed", "r2": "duplicate"}
    assert statuses(render_page(page, tmp_path / "qwen", config(prefer_on_overlap="qwen"))) == {"r1": "duplicate", "r2": "auto_passed"}


def test_dilation_past_bubble_edge_is_clipped_not_rejected(tmp_path):
    page = make_page(tmp_path)
    bubble = Image.new("L", (240, 140))
    ImageDraw.Draw(bubble).rectangle((15, 15, 225, 125), fill=255)
    ImageDraw.Draw(bubble).line((49, 15, 49, 125), fill=0)  # bubble edge touching the glyphs
    bubble.save(tmp_path / "bubble.png")
    page["regions"][0]["bubble_mask_path"] = str(tmp_path / "bubble.png")
    report = render_page(page, tmp_path / "out", config())
    assert report["status"] == "auto_passed", report
    changed = np.any(image_array(report["output_path"]) != image_array(page["source_path"]), axis=2)
    assert changed[50:66, 50:66].any() and not changed[:, 49].any()


def test_bubble_mask_constrains_text_not_only_bounding_box(tmp_path):
    page = make_page(tmp_path)
    bubble = Image.new("L", (240, 140))
    ImageDraw.Draw(bubble).ellipse((15, 15, 225, 125), fill=255)
    bubble_path = tmp_path / "bubble.png"
    bubble.save(bubble_path)
    page["regions"][0]["bubble_mask_path"] = str(bubble_path)
    page["regions"][0]["translation"]["text"] = "A short line inside a curved bubble"
    report = render_page(page, tmp_path / "out", config())
    assert report["status"] == "auto_passed", report
    changed = np.any(image_array(report["output_path"]) != image_array(page["source_path"]), axis=2)
    assert not np.any(changed & (np.array(bubble) == 0))


def test_missing_cjk_font_keeps_original(tmp_path):
    page = make_page(tmp_path)
    page["regions"][0]["translation"]["text"] = "你好"
    report = render_page(page, tmp_path / "out", config())
    assert report["issues"][0]["code"] == "font_error"
    assert np.array_equal(image_array(report["output_path"]), image_array(page["source_path"]))


def test_rerender_starts_from_original_and_cbz_is_deterministic(tmp_path):
    page = make_page(tmp_path)
    report = render_page(page, tmp_path / "out", config())
    first = Path(report["output_path"]).read_bytes()
    report = render_page(page, tmp_path / "out", config())
    assert Path(report["output_path"]).read_bytes() == first
    path = Path(report["output_path"])
    one = export_cbz([path, path], tmp_path / "one.cbz")
    two = export_cbz([path, path], tmp_path / "two.cbz")
    assert one.read_bytes() == two.read_bytes()
    with zipfile.ZipFile(one) as archive:
        assert archive.namelist() == ["00001.png", "00002.png"]


def test_bad_mask_dimensions_and_page_dimensions_are_rejected(tmp_path):
    page = make_page(tmp_path)
    mask_path = tmp_path / "small_mask.png"
    Image.new("L", (100, 100), 255).save(mask_path)
    page["regions"][0]["mask_path"] = str(mask_path)
    report = render_page(page, tmp_path / "badmask", config())
    assert report["issues"][0]["code"] == "unsafe_mask"
    page["width"] = 200
    report = render_page(page, tmp_path / "baddims", config())
    assert report["issues"][0]["code"] == "coordinate_mismatch"


def test_translation_cannot_overwrite_source(tmp_path):
    page = make_page(tmp_path)
    source = tmp_path / "page001.translated.png"
    Path(page["source_path"]).rename(source)
    page["source_path"] = str(source)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="overwrite the source"):
        render_page(page, tmp_path, config())
    assert source.read_bytes() == before


def test_traditional_chinese_font_can_render_when_available(tmp_path):
    font = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
    if not font.exists():
        pytest.skip("Optional system Noto CJK font is unavailable.")
    page = make_page(tmp_path)
    page["regions"][0]["translation"]["text"] = "臺灣繁體中文測試"
    report = render_page(page, tmp_path / "out", config(font_path=str(font), font_index=3))
    assert report["status"] == "auto_passed", report
    assert report["regions"][0]["overflow_pixels"] == 0
