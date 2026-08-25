from unittest.mock import MagicMock

import torch

from anomavision.export import ModelExporter, create_parser
from scripts.convert_to_tensorrt import build_parser


def test_export_parser_accepts_tensorrt_options():
    args = create_parser().parse_args(
        [
            "--model_data_path",
            "./distributions",
            "--algorithm",
            "padim",
            "--model",
            "model.pt",
            "--format",
            "tensorrt",
            "--device",
            "cpu",
            "--tensorrt-precision",
            "int8",
            "--calib-dir",
            "./calibration",
            "--calib-samples",
            "25",
            "--workspace-gb",
            "1.5",
            "--min-batch",
            "1",
            "--opt-batch",
            "2",
            "--max-batch",
            "4",
        ]
    )

    assert args.format == "tensorrt"
    assert args.device == "cpu"
    assert args.tensorrt_precision == "int8"
    assert args.calib_dir == "./calibration"
    assert args.calib_samples == 25
    assert args.workspace_gb == 1.5
    assert (args.min_batch, args.opt_batch, args.max_batch) == (1, 2, 4)


def test_standalone_tensorrt_parser_accepts_cpu_validation_arguments():
    args = build_parser().parse_args(
        [
            "--model",
            "model.pth",
            "--output-dir",
            "engines",
            "--precision",
            "fp16",
            "--device",
            "cpu",
        ]
    )

    assert args.precision == "fp16"
    assert args.device == "cpu"
    assert args.input_shape == (1, 3, 224, 224)


def test_tensorrt_export_stops_cleanly_on_cpu_without_importing_tensorrt(tmp_path):
    logger = MagicMock()
    exporter = ModelExporter(
        tmp_path / "missing.pt",
        tmp_path,
        logger,
        device="cpu",
    )

    result = exporter.export_tensorrt(
        input_shape=(1, 3, 16, 16),
        output_name="model_fp16.engine",
        precision="fp16",
    )

    assert result is None
    logger.exception.assert_called_once()
    message = logger.exception.call_args.args[0]
    assert message == "tensorrt: failed after %.2fs"
    assert not (tmp_path / "model_fp16_fp32.onnx").exists()


def test_cpu_tensor_device_is_used_by_exporter(tmp_path):
    logger = MagicMock()
    exporter = ModelExporter(
        tmp_path / "missing.pt",
        tmp_path,
        logger,
        device="cpu",
    )

    assert exporter.device == torch.device("cpu")
