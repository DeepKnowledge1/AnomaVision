from easydict import EasyDict

from anomavision.export import _apply_export_defaults


def test_export_defaults_fill_optional_tensorrt_fields():
    config = _apply_export_defaults(EasyDict())
    assert config.calib_dir is None
    assert config.calib_samples == 100
    assert config.workspace_gb == 2.0
    assert config.min_batch == 1
    assert config.opt_batch == 1
    assert config.max_batch == 4
    assert config.tensorrt_precision == "fp16"
    assert config.static_batch is False
