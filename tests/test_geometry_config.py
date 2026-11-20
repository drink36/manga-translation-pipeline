from pathlib import Path

import pytest

from manga_pipeline import storage
from manga_pipeline.cli import server_command
from manga_pipeline.config import load_config
from manga_pipeline.geometry import covered_fraction, denormalize, normalize


def test_duplicate_boxes_do_not_overstate_coverage():
    assert covered_fraction([0, 0, 100, 100], [[0, 0, 20, 100]] * 5) == 0.2
    assert covered_fraction([0, 0, 100, 100], [[0, 0, 50, 100], [50, 0, 100, 100]]) == 1.0


def test_crop_coordinates_map_to_original():
    assert denormalize([100, 200, 500, 800], 200, 100, (300, 400)) == [320, 420, 400, 480]
    with pytest.raises(ValueError):
        denormalize([0, 0, 1200, 100], 200, 100)


def test_extends_paths_and_cycle(tmp_path):
    (tmp_path / "base.yaml").write_text("input: input\noutput: out\nrender:\n  inpaint: plain\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\noutput: other\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["output"] == str(tmp_path / "other")
    assert cfg["render"]["inpaint"] == {"backend": "plain"}
    (tmp_path / "base.yaml").write_text("extends: child.yaml\ninput: input\noutput: out\n")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(tmp_path / "child.yaml")


def test_server_backends(tmp_path):
    (tmp_path / "base.yaml").write_text("input: input\noutput: out\nvlm: {endpoint: 'http://127.0.0.1:8000/v1', model: m}\n")
    assert server_command(load_config(tmp_path / "base.yaml"))[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    (tmp_path / "llama.yaml").write_text("extends: base.yaml\nserver:\n  backend: llama.cpp\n  executable: bin/llama-server.exe\n  model_path: models/m.gguf\n  mmproj_path: models/mmproj.gguf\n")
    command = server_command(load_config(tmp_path / "llama.yaml"))
    assert command[0] == str(tmp_path / "bin" / "llama-server.exe")
    assert command[command.index("--model") + 1] == str(tmp_path / "models" / "m.gguf")
    assert command[command.index("--alias") + 1] == "m"
    assert command[command.index("--port") + 1] == "8000"
    assert command[command.index("--ctx-size") + 1] == "16384"
    (tmp_path / "slots.yaml").write_text("extends: llama.yaml\nserver: {max_num_seqs: 4, max_model_len: 8192}\n")
    command = server_command(load_config(tmp_path / "slots.yaml"))
    assert (command[command.index("--ctx-size") + 1], command[command.index("--parallel") + 1]) == ("32768", "4")
    (tmp_path / "path.yaml").write_text("extends: base.yaml\nserver: {backend: llama.cpp, model_path: m.gguf, mmproj_path: p.gguf}\n")
    assert server_command(load_config(tmp_path / "path.yaml"))[0] == "llama-server"
    for server in ("{backend: llama.cpp, model_path: m.gguf}", "{backend: ollama}"):
        (tmp_path / "bad.yaml").write_text(f"extends: base.yaml\nserver: {server}\n")
        with pytest.raises(ValueError, match="server.backend"):
            load_config(tmp_path / "bad.yaml")


def test_renderer_code_change_keeps_translation_fingerprint(tmp_path, monkeypatch):
    store = storage.Store({"input": str(tmp_path / "in"), "output": str(tmp_path / "out"), "mode": "real"})
    before = {stage: store.fingerprint(stage, {}) for stage in ("translate", "render", "qa")}
    real = storage.file_hash
    monkeypatch.setattr(storage, "file_hash", lambda path: "changed" if Path(path).name == "rendering.py" else real(path))
    # Code is hashed once per process, so a change is only seen by a new process.
    storage.code_hashes.cache_clear()
    try:
        assert store.fingerprint("translate", {}) == before["translate"]
        assert store.fingerprint("render", {}) != before["render"] and store.fingerprint("qa", {}) != before["qa"]
    finally:
        storage.code_hashes.cache_clear()  # later tests must not see the fake hash
