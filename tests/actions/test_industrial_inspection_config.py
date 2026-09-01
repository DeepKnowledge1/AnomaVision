from pathlib import Path

import yaml


def test_industrial_inspection_config_is_valid():
    path = Path(__file__).parents[2] / "examples/actions/industrial_inspection.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["source"]["type"] == "video"
    assert config["model"]["type"] == "padim"
    assert [action["type"] for action in config["actions"]] == ["mqtt", "evidence"]
