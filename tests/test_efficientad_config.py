from pathlib import Path

from anomavision.config import load_config


def test_model_name_selects_algorithm(tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("model:\n  name: efficientad\n")
    config = load_config(str(config_path))
    assert config["algorithm"] == "efficientad"
    assert config["model"] is None


def test_legacy_algorithm_config_is_unchanged(tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("algorithm: padim\nmodel: model.pt\n")
    config = load_config(str(config_path))
    assert config["algorithm"] == "padim"
    assert config["model"] == "model.pt"
