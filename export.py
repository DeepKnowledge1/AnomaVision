"""
Model Export Utility
Export PyTorch models to ONNX, TorchScript, and OpenVINO with dynamic batch support.
Logging is concise: key config, start/end, timings, sizes, and clear errors.
Now extended with ONNX quantization (dynamic + static INT8).
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import onnx
import torch
from easydict import EasyDict as edict

# ONNX Runtime quantization
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from PIL import Image

from anomavision.config import load_config
from anomavision.padim_lite import (  # stats-only .pth → runtime module
    build_padim_from_stats,
)
from anomavision.utils import (
    create_image_transform,
    get_logger,
    merge_config,
    setup_logging,
)

# Suppress "To copy construct from a tensor..." warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

# Suppress ONNX shape inference + constant folding warnings
warnings.filterwarnings(
    "ignore", message="The shape inference of prim::Constant type is missing"
)
warnings.filterwarnings("ignore", message="Constant folding")


from pathlib import Path

from PIL import Image

from anomavision.utils import create_image_transform


def load_calibration_images(
    img_dir: str,
    input_shape: Tuple[int, int, int, int],
    max_samples: int = 100,
    normalize: bool = True,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    """
    Load calibration images using the same preprocessing as AnodetDataset.
    Returns a list of np.ndarrays shaped (1,C,H,W).
    """
    import os
    from glob import glob

    import numpy as np

    h, w = input_shape[2], input_shape[3]
    transform = create_image_transform(
        resize=(h, w),
        crop_size=(h, w),
        normalize=normalize,
        mean=mean,
        std=std,
    )

    paths = glob(os.path.join(img_dir, "*.png"))[:max_samples]
    samples = []

    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensor = transform(img)  # torch tensor [C,H,W]
            samples.append(tensor.unsqueeze(0).numpy())  # (1,C,H,W)
        except Exception as e:
            print(f"Skipping {p}, error {e}")

    if not samples:
        # Fallback to random data if no images found
        samples = [np.random.rand(*input_shape).astype("float32") for _ in range(32)]

    return samples


# ---------------------------
# Calibration Data Reader for static quantization
# ---------------------------
class DummyDataReader(CalibrationDataReader):
    """Replace with a real data reader if you have calibration images."""

    def __init__(self, input_name: str, samples: list):
        self.input_name = input_name
        self.samples = iter(samples)

    def get_next(self):
        try:
            return {self.input_name: next(self.samples)}
        except StopIteration:
            return None


# during export
class _ExportWrapper(torch.nn.Module):
    """Call model.predict(x) for consistent outputs during export."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        # delegate device to the wrapped model
        if hasattr(m, "device"):
            self.device = m.device
        else:
            # fall back to parameter device or CPU
            try:
                self.device = next(m.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    def forward(self, x):
        scores, maps = self.m.predict(x, export=True)
        return scores, maps


class ModelExporter:
    """Professional model exporter with clean interface."""

    def __init__(self, model_path: Path, output_dir: Path, logger):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def _load_model(self) -> torch.nn.Module:
        """Load and prepare model for export (supports .pt and stats-only .pth)."""
        self.logger.info("load: %s", self.model_path)

        obj = torch.load(self.model_path, map_location="cpu", weights_only=False)

        # Debug: Print what we actually loaded
        self.logger.info(f"Loaded object type: {type(obj)}")

        if isinstance(obj, dict):
            self.logger.info(f"Dictionary keys: {list(obj.keys())}")
            required_keys = {
                "mean",
                "cov_inv",
                "channel_indices",
                "layer_indices",
                "backbone",
            }
            self.logger.info(f"Required keys: {required_keys}")
            self.logger.info(f"Has required keys: {required_keys.issubset(obj.keys())}")

        # If it's a stats-only dict, build a runtime module that exposes .predict(...)
        if isinstance(obj, dict) and {
            "mean",
            "cov_inv",
            "channel_indices",
            "layer_indices",
            "backbone",
        }.issubset(obj.keys()):
            self.logger.info("GOING INTO STATS PATH - building PadimLite")
            base = build_padim_from_stats(obj, device="cpu")
            self.logger.info("Export: built PadimLite from statistics (.pth).")
        else:
            self.logger.info("GOING INTO FULL MODEL PATH")
            base = obj
        return _ExportWrapper(base)

    # # Original: assume full model in .pt
    # def _load_model(self) -> torch.nn.Module:
    #     """Load and prepare model for export."""
    #     self.logger.info("load: %s", self.model_path)
    #     model = _ExportWrapper(torch.load(self.model_path, map_location="cpu"))

    #     # Handle DataParallel wrapper
    #     if hasattr(model, "module"):
    #         model = model.module

    #     model.eval()
    #     return model

    def _get_output_names(
        self, model: torch.nn.Module, dummy_input: torch.Tensor
    ) -> List[str]:
        """Automatically determine output names from model."""
        with torch.no_grad():
            outputs = model(dummy_input)

        if isinstance(outputs, dict):
            return list(outputs.keys())
        elif isinstance(outputs, (list, tuple)):
            return [f"output_{i}" for i in range(len(outputs))]
        else:
            return ["output"]

    def export_onnx(
        self,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        output_name: str = "model.onnx",
        opset_version: int = 17,
        dynamic_batch: bool = True,
        quantize_dynamic_flag: bool = False,  # NEW
        quantize_static_flag: bool = False,  # NEW
        calib_samples: int = 0,  # NEW
    ) -> Optional[Path]:
        """
        Export model to ONNX format (with optional quantization).
        Args:
            input_shape: Model input shape (batch, channels, height, width)
            output_name: Output filename
            opset_version: ONNX opset version
            dynamic_batch: Enable dynamic batch size
            quantize_dynamic_flag: Export dynamic INT8 quantized model
            quantize_static_flag: Export static INT8 quantized model
            calib_samples: Number of calibration samples for static quantization
        Returns:
            Path to exported file or None if failed
        """
        t0 = time.perf_counter()
        try:
            model = self._load_model()
            dummy_input = torch.randn(*input_shape)

            # Warm up model
            with torch.no_grad():
                for _ in range(2):
                    _ = model(dummy_input)

            # Determine outputs
            output_names = self._get_output_names(model, dummy_input)

            # Setup dynamic axes if requested
            dynamic_axes = None
            if dynamic_batch:
                dynamic_axes = {"input": {0: "batch_size"}}
                for name in output_names:
                    dynamic_axes[name] = {0: "batch_size"}

            output_path = self.output_dir / output_name

            # Suppress noisy tracer warnings
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

            # Export
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False,
            )

            elapsed = time.perf_counter() - t0
            size_mb = output_path.stat().st_size / (1024 * 1024)

            self.logger.info(
                "onnx: ok (%.2fs) file=%s size=%.1fMB dynamic_batch=%s opset=%d",
                elapsed,
                output_path,
                size_mb,
                dynamic_batch,
                opset_version,
            )

            # ---------------------------
            # Quantization Step
            # ---------------------------

            # ---------------------------
            # Quantization Step
            # ---------------------------
            if quantize_dynamic_flag:
                dyn_path = output_path.with_name(
                    output_path.stem + "_int8_dynamic.onnx"
                )
                self.logger.info("quant: dynamic INT8 -> %s", dyn_path)
                quantize_dynamic(
                    output_path,
                    dyn_path,
                    weight_type=QuantType.QInt8,
                )
                onnx.checker.check_model(onnx.load(dyn_path))

            # if quantize_static_flag:
            #     if calib_samples <= 0:
            #         raise ValueError("Static quantization requires --calib-samples > 0")
            #     static_path = output_path.with_name(output_path.stem + "_int8_static.onnx")
            #     self.logger.info("quant: static INT8 -> %s", static_path)
            #     dummy_data = [
            #         np.random.rand(*input_shape).astype("float32")
            #         for _ in range(calib_samples)
            #     ]
            #     dr = DummyDataReader("input", dummy_data)
            #     quantize_static(
            #         output_path,
            #         static_path,
            #         dr,
            #         quant_format=QuantFormat.QOperator,
            #         activation_type=QuantType.QInt8,
            #         weight_type=QuantType.QInt8,
            #     )
            #     onnx.checker.check_model(onnx.load(static_path))

            if quantize_static_flag:
                if calib_samples <= 0:
                    raise ValueError("Static quantization requires --calib-samples > 0")

                static_path = output_path.with_name(
                    output_path.stem + "_int8_static.onnx"
                )
                self.logger.info("quant: static INT8 -> %s", static_path)

                # Use real training images for calibration
                calib_dir = r"D:\01-DATA\bottle\train\good"
                calib_data = load_calibration_images(
                    calib_dir,
                    input_shape,
                    max_samples=calib_samples,
                    normalize=True,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )

                if not calib_data:
                    self.logger.warning(
                        "No calibration images found, falling back to random data"
                    )
                    calib_data = [
                        np.random.rand(*input_shape).astype("float32")
                        for _ in range(calib_samples)
                    ]

                dr = DummyDataReader("input", calib_data)

                quantize_static(
                    output_path,
                    static_path,
                    dr,
                    quant_format=QuantFormat.QDQ,  # QDQ is usually safer for accuracy
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                )
                onnx.checker.check_model(onnx.load(static_path))

            # if quantize_static_flag:
            #     if calib_samples <= 0:
            #         raise ValueError("Static quantization requires --calib-samples > 0")
            #     static_path = output_path.with_name(output_path.stem + "_int8_static.onnx")
            #     self.logger.info("quant: static INT8 -> %s", static_path)
            #     dummy_data = [np.random.rand(*input_shape).astype("float32") for _ in range(calib_samples)]
            #     dr = DummyDataReader("input", dummy_data)
            #     quantize_static(
            #         output_path,
            #         static_path,
            #         dr,
            #         quant_format=QuantFormat.QDQ,
            #         activation_type=QuantType.QInt8,
            #         weight_type=QuantType.QInt8,
            #     )
            #     onnx.checker.check_model(onnx.load(static_path))

            return output_path

        except Exception:
            self.logger.exception("onnx: failed after %.2fs", time.perf_counter() - t0)
            return None

    def export_torchscript(
        self,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        output_name: str = "model.torchscript",
        optimize: bool = False,
    ) -> Optional[Path]:
        """
        Export model to TorchScript format.

        Args:
            input_shape: Model input shape (batch, channels, height, width)
            output_name: Output filename
            optimize: Enable mobile optimization

        Returns:
            Path to exported file or None if failed
        """
        t0 = time.perf_counter()
        try:
            model = self._load_model()
            dummy_input = torch.randn(*input_shape)

            with torch.no_grad():
                for _ in range(2):
                    _ = model(dummy_input)

            output_path = self.output_dir / output_name
            if not output_path.suffix:
                output_path = output_path.with_suffix(".torchscript")

            self.logger.info("ts: tracing optimize=%s", optimize)
            traced_model = torch.jit.trace(model, dummy_input, strict=False)

            config_data = {"shape": list(dummy_input.shape)}
            extra_files = {"config.txt": json.dumps(config_data)}

            if optimize:
                try:
                    from torch.utils.mobile_optimizer import optimize_for_mobile

                    optimized_model = optimize_for_mobile(traced_model)
                    optimized_model._save_for_lite_interpreter(
                        str(output_path), _extra_files=extra_files
                    )
                    self.logger.info("ts: mobile optimization applied")
                except ImportError:
                    self.logger.warning(
                        "ts: mobile optimization unavailable; saving standard"
                    )
                    traced_model.save(str(output_path), _extra_files=extra_files)
            else:
                traced_model.save(str(output_path), _extra_files=extra_files)

            elapsed = time.perf_counter() - t0
            size_mb = output_path.stat().st_size / (1024 * 1024)

            self.logger.info(
                "ts: ok (%.2fs) file=%s size=%.1fMB optimized=%s",
                elapsed,
                output_path,
                size_mb,
                optimize,
            )
            return output_path

        except Exception:
            self.logger.exception("ts: failed after %.2fs", time.perf_counter() - t0)
            return None

    def export_openvino(
        self,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        output_name: str = "model_openvino",
        fp16: bool = True,
        dynamic_batch: bool = False,
    ) -> Optional[Path]:
        """
        Export model to OpenVINO format via ONNX.
        """
        t0 = time.perf_counter()
        temp_onnx = self.output_dir / "temp_model.onnx"
        try:
            onnx_path = self.export_onnx(
                input_shape=input_shape,
                output_name=temp_onnx.name,
                dynamic_batch=dynamic_batch,
            )
            if onnx_path is None:
                raise RuntimeError("ONNX export failed")

            try:
                import openvino.runtime as ov
                from openvino.tools import mo
            except ImportError as e:
                raise ImportError("OpenVINO not installed. pip install openvino") from e

            output_dir = self.output_dir / output_name
            output_dir.mkdir(exist_ok=True)

            self.logger.info(
                "ov: convert fp16=%s dynamic_batch=%s", fp16, dynamic_batch
            )
            ov_model = mo.convert_model(
                onnx_path, framework="onnx", compress_to_fp16=fp16
            )

            xml_path = output_dir / f"{output_name}.xml"
            ov.serialize(ov_model, str(xml_path))

            # Clean temp
            if temp_onnx.exists():
                temp_onnx.unlink()

            elapsed = time.perf_counter() - t0
            size_mb = xml_path.stat().st_size / (1024 * 1024)

            self.logger.info(
                "ov: ok (%.2fs) dir=%s xml=%s size=%.1fMB precision=%s dynamic_batch=%s",
                elapsed,
                output_dir,
                xml_path.name,
                size_mb,
                "FP16" if fp16 else "FP32",
                dynamic_batch,
            )
            return output_dir

        except Exception:
            self.logger.exception("ov: failed after %.2fs", time.perf_counter() - t0)
            # Clean temp on failure
            try:
                if temp_onnx.exists():
                    temp_onnx.unlink()
            except Exception:
                self.logger.warning("ov: temp cleanup failed for %s", temp_onnx)
            return None


def parse_args():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Export PaDiM models to ONNX, TorchScript, and OpenVINO (with optional quantization)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.yml/.json"
    )

    # Model & data paths
    parser.add_argument(
        "--model_data_path",
        type=str,
        default="./distributions/anomav_exp",
        help="Directory containing model and output location",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model file (.pt for PyTorch)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output filename (if None, uses model name)",
    )

    parser.add_argument(
        "--format",
        choices=["onnx", "openvino", "torchscript", "all"],
        help="Export format",
    )

    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--static-batch", action="store_true", help="Disable dynamic batch size"
    )

    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 precision for OpenVINO (default: FP16)",
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable mobile optimization for TorchScript",
    )

    # NEW: Quantization flags
    parser.add_argument(
        "--quantize-dynamic",
        action="store_false",
        help="Also export dynamic INT8 quantized ONNX model",
    )
    parser.add_argument(
        "--quantize-static",
        action="store_false",
        help="Also export static INT8 quantized ONNX model (requires --calib-samples)",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=100,
        help="Number of dummy calibration samples for static quantization",
    )

    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config is not None:
        cfg = load_config(str(args.config))
    else:
        cfg = load_config(str(Path(args.model_data_path) / "config.yml"))

    config = edict(merge_config(args, cfg))

    # Setup logging & logger
    setup_logging(enabled=True, log_level=config.log_level, log_to_file=True)
    logger = get_logger("anomavision.export")  # Force it into anomavision hierarchy

    model_path = Path(config.model_data_path) / config.model

    output_dir = Path(config.model_data_path)
    model_stem = Path(config.model).stem

    onnx_name = f"{model_stem}.onnx"
    openvino_name = f"{model_stem}_openvino"
    torchscript_name = f"{model_stem}.torchscript"

    h, w = config.crop_size if config.crop_size is not None else config.resize
    input_shape = [1, 3, h, w]

    # Run header (compact)
    logger.info(
        "=== export start === model=%s fmt=%s in_shape=%s dyn_batch=%s opset=%d fp16=%s opt=%s",
        model_path,
        config.format,
        tuple(input_shape),
        config.dynamic_batch,
        config.opset,
        (not config.fp32),
        config.optimize,
    )

    exporter = ModelExporter(model_path, output_dir, logger)
    started = time.perf_counter()
    success = True

    try:
        if config.format in ["onnx", "all"]:
            success &= (
                exporter.export_onnx(
                    input_shape=tuple(input_shape),
                    output_name=onnx_name,
                    opset_version=config.opset,
                    dynamic_batch=config.dynamic_batch,
                    quantize_dynamic_flag=config.quantize_dynamic,
                    quantize_static_flag=config.quantize_static,
                    calib_samples=config.calib_samples,
                )
                is not None
            )

        if config.format in ["openvino", "all"]:
            success &= (
                exporter.export_openvino(
                    input_shape=tuple(input_shape),
                    output_name=openvino_name,
                    fp16=not config.fp32,
                    dynamic_batch=config.dynamic_batch,
                )
                is not None
            )

        if config.format in ["torchscript", "all"]:
            success &= (
                exporter.export_torchscript(
                    input_shape=tuple(input_shape),
                    output_name=torchscript_name,
                    optimize=config.optimize,
                )
                is not None
            )

        elapsed = time.perf_counter() - started
        if success:
            logger.info("=== export done in %.2fs ===", elapsed)
        else:
            logger.error("=== export completed with failures (%.2fs) ===", elapsed)
            sys.exit(1)

    except Exception:
        logger.exception("fatal: unhandled error")
        sys.exit(1)


if __name__ == "__main__":
    main()
