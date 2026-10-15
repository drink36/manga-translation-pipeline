"""Offline adapter contract tests: synthetic images and fake model modules only."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from manga_pipeline import adapters


def test_import_does_not_import_heavy_model_libraries():
    source = Path(adapters.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(source)}
    code = (
        "import sys; import manga_pipeline.adapters; "
        "assert not {'torch','transformers','paddleocr','rfdetr','manga_ocr'}.intersection(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=10)


def test_remote_references_are_pinned_and_local_directories_work(tmp_path):
    assert adapters._reference({}, adapters.PADDLE_MODEL, adapters.PADDLE_REVISION) == (
        adapters.PADDLE_MODEL, adapters.PADDLE_REVISION,
    )
    assert adapters._reference({"model_id": str(tmp_path)}, "unused", "unused") == (str(tmp_path), None)
    for cfg in ({"revision": "main"}, {"model_id": "someone/other-model"}):
        with pytest.raises(adapters.AdapterError, match="40-character revision"):
            adapters._reference(cfg, adapters.PADDLE_MODEL, adapters.PADDLE_REVISION)


def test_koharu_strict_loader_geometry_and_class_thresholds(tmp_path, monkeypatch):
    captured = {}

    class FakeInner:
        def load_state_dict(self, state, strict):
            captured["strict"] = strict

    class FakeRF:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = SimpleNamespace(model=FakeInner())

        def predict(self, image, **kwargs):
            captured["predict"] = kwargs
            return SimpleNamespace(
                xyxy=np.array([[1, 2, 30, 40], [0, 0, 50, 90], [40, 50, 60, 70]]),
                class_id=np.array([0, 2, 1]), confidence=np.array([0.9, 0.1, 0.3]),
                mask=np.ones((3, 10, 10), dtype=bool),
            )

    def download(model_id, filename, **kwargs):
        captured["download"] = (model_id, filename, kwargs)
        return str(tmp_path / "fake.safetensors")

    for name, module in {
        "rfdetr": SimpleNamespace(RFDETRSeg2XLarge=FakeRF),
        "rfdetr.config": SimpleNamespace(PretrainWeightsCompatibilityWarning=UserWarning),
        "huggingface_hub": SimpleNamespace(hf_hub_download=download),
        "safetensors.torch": SimpleNamespace(load_file=lambda *a, **kw: {}),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 200)).save(image_path)
    detector = adapters.build_detector({})
    result = detector.detect(image_path, tmp_path / "masks")
    assert captured["strict"] is True
    assert captured["device"] == "cpu"
    assert captured["pretrain_weights"] is None
    assert captured["num_classes"] == 4
    assert captured["download"][:2] == (adapters.KOHARU_MODEL, "model.safetensors")
    assert captured["download"][2]["revision"] == adapters.KOHARU_REVISION
    assert captured["predict"]["threshold"] == 0.2
    assert [region["kind"] for region in result] == ["text", "sfx"]
    assert result[0]["polygon"] == [[1.0, 2.0], [30.0, 2.0], [30.0, 40.0], [1.0, 40.0]]
    for region in result:
        assert Path(region["mask_path"]).is_absolute()
        with Image.open(region["mask_path"]) as mask:
            assert mask.size == (100, 200)
            assert mask.getextrema() == (255, 255)


def test_detector_command_receives_literal_paths_and_persistent_masks(tmp_path):
    directory = tmp_path / "folder with spaces;$(literal)"
    directory.mkdir()
    image_path = directory / "page with spaces.png"
    Image.new("RGB", (100, 200)).save(image_path)
    script = directory / "detector.py"
    script.write_text(
        "import json,sys\nfrom pathlib import Path\nfrom PIL import Image\n"
        "assert Path(sys.argv[1]).name == 'page with spaces.png'\n"
        "Image.new('L',(100,200),255).save(Path(sys.argv[2])/'mask.png')\n"
        "Path(sys.argv[3]).write_text(json.dumps([{'kind':'text','bbox':[2,3,40,50],"
        "'mask_path':'mask.png'}]))\n",
        encoding="utf-8",
    )
    detector = adapters.build_detector({"backend": "command", "command": [
        sys.executable, str(script), "{image}", "{mask_dir}", "{output}",
    ]})
    regions = detector.detect(image_path, directory / "masks")
    assert regions[0]["polygon"] == [[2, 3], [40, 3], [40, 50], [2, 50]]
    assert regions[0]["mask_path"] == str(directory / "masks" / "mask.png")
    assert Path(regions[0]["mask_path"]).is_file()
    detector.close()


@pytest.mark.parametrize("invalid", [
    [{"kind": "text", "bbox": [-1, 0, 20, 40]}],
    [{"kind": "text", "bbox": [1, 1, 101, 40]}],
    [{"kind": "text", "bbox": [1, 1, 20, 40], "score": float("nan")}],
    [{"kind": "text", "bbox": [1, 1, 20, 40], "polygon": [[0, 0], [1, 1]]}],
    [{"kind": "unknown", "bbox": [1, 1, 20, 40]}],
    {"regions": []},
])
def test_command_detector_rejects_invalid_geometry_and_schema(tmp_path, invalid):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 200)).save(image_path)
    command = [sys.executable, "-c", "import sys; print(sys.argv[1])", json.dumps(invalid)]
    detector = adapters.build_detector({"backend": "command", "command": command})
    with pytest.raises(adapters.AdapterError):
        detector.detect(image_path, tmp_path / "masks")


@pytest.mark.parametrize("mask_size,value", [((10, 10), 255), ((100, 200), 127)])
def test_command_detector_rejects_wrong_size_and_nonbinary_masks(tmp_path, mask_size, value):
    image_path, mask_path = tmp_path / "page.png", tmp_path / "mask.png"
    Image.new("RGB", (100, 200)).save(image_path)
    Image.new("L", mask_size, value).save(mask_path)
    records = [{"kind": "text", "bbox": [1, 1, 20, 40], "mask_path": str(mask_path)}]
    detector = adapters.build_detector({"backend": "command", "command": [
        sys.executable, "-c", "import sys; print(sys.argv[1])", json.dumps(records),
    ]})
    with pytest.raises(adapters.AdapterError, match="masks must"):
        detector.detect(image_path, tmp_path / "masks")


@pytest.mark.parametrize("write_file", [False, True])
def test_ocr_command_json_and_temporary_image_lifecycle(write_file):
    code = (
        "import json,sys; from pathlib import Path; "
        "p=Path(sys.argv[1]); assert p.is_file(); "
        "value=json.dumps({'text':'日本語'},ensure_ascii=False); "
        + ("Path(sys.argv[2]).write_text(value,encoding='utf-8')" if write_file else "print(value)")
    )
    reader = adapters.build_reader({"backend": "command", "command": [
        sys.executable, "-c", code, "{image}", "{output}",
    ]})
    assert reader.read(Image.new("RGB", (20, 20))) == "日本語"
    reader.close()


def test_ocr_command_rejects_shell_strings_and_bad_json():
    reader = adapters.build_reader({"backend": "command", "command": "echo unsafe"})
    with pytest.raises(adapters.AdapterError, match="argument strings"):
        reader.read(Image.new("RGB", (10, 10)))
    reader = adapters.build_reader({"backend": "command", "command": [
        sys.executable, "-c", "print('not JSON')",
    ]})
    with pytest.raises(adapters.AdapterError, match="must produce JSON"):
        reader.read(Image.new("RGB", (10, 10)))


def test_external_failure_reports_exit_and_stderr():
    reader = adapters.build_reader({"backend": "command", "command": [
        sys.executable, "-c", "import sys; print('model unavailable',file=sys.stderr); sys.exit(7)",
    ]})
    with pytest.raises(adapters.AdapterError, match="exited 7: model unavailable"):
        reader.read(Image.new("RGB", (10, 10)))
