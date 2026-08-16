from pathlib import Path

import yaml


REQUIRED_EXPORT_KEYS = {
    "format",
    "opset",
    "precision",
    "tensorrt_precision",
    "dynamic_batch",
    "static_batch",
    "min_batch",
    "opt_batch",
    "max_batch",
    "workspace_gb",
    "calib_dir",
    "calib_samples",
    "quantize_dynamic",
    "quantize_static",
    "optimize",
    "coreset_method",
    "coreset_seed",
}


def test_config_template_contains_all_export_defaults():
    config_path = Path(__file__).parents[1] / "config.yml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert REQUIRED_EXPORT_KEYS.issubset(config)
    assert config["calib_dir"] is None
    assert config["min_batch"] <= config["opt_batch"] <= config["max_batch"]
    assert config["coreset_method"] == "kcenter"
