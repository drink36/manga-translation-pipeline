from pathlib import Path

import pytest

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
