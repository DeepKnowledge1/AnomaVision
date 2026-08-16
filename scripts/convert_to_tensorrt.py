"""Convert AnomaVision PaDiM or ultra-light PatchCore artifacts to TensorRT.

This command detects the compact statistics format written by PaDiM/PatchCore,
loads it through the shared :class:`ModelExporter`, and builds a native TensorRT
engine using the repository's current FP16/INT8 pipeline.

Examples:
    python scripts/convert_to_tensorrt.py \
        --model model_data/patchcore/bottle/run/model.pth \
        --output-dir engines/bottle \
        --calib-dir dataset/bottle/train/good \
        --precision int8 --device cuda

    python scripts/convert_to_tensorrt.py \
        --model model_data/padim/bottle/run/model.pth \
        --output-dir engines/bottle \
        --precision fp16 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Tuple

import torch

# Allow `python scripts/convert_to_tensorrt.py` from a source checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anomavision.export import ModelExporter


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PADIM_KEYS = {"mean", "cov_inv", "channel_indices", "layer_indices", "backbone"}
PATCHCORE_KEYS = {"memory_bank", "layer_indices", "backbone"}


def _logger() -> logging.Logger:
    """Create a concise standalone logger for conversion output."""
    logger = logging.getLogger("anomavision.convert_tensorrt")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _image_files(directory: Path) -> Iterable[Path]:
    """Yield supported calibration images in deterministic order."""
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def detect_artifact(model_path: Path) -> Tuple[str, object]:
    """Load an artifact and identify it as PaDiM, PatchCore, or a full model.

    Args:
        model_path: Path to a compact ``.pth``/``.pt`` artifact.

    Returns:
        A pair containing the detected model name and the loaded object.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        ValueError: If the artifact is not a supported AnomaVision model.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {model_path}")
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(artifact, dict) and PADIM_KEYS.issubset(artifact):
        return "padim", artifact
    if isinstance(artifact, dict) and PATCHCORE_KEYS.issubset(artifact):
        return "patchcore", artifact
    if isinstance(artifact, torch.nn.Module):
        name = artifact.__class__.__name__.lower()
        if "padim" in name:
            return "padim", artifact
        if "patchcore" in name:
            return "patchcore", artifact
    raise ValueError(
        "Unsupported artifact. Expected PaDiM statistics, PatchCore memory-bank "
        "statistics, or a serialized PaDiM/PatchCore module."
    )


def validate_engine(engine_path: Path, input_name: str = "input") -> None:
    """Deserialize a TensorRT engine and verify that it has an input tensor."""
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT is required for engine validation; the engine was still built."
        ) from exc
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    with engine_path.open("rb") as handle:
        engine = runtime.deserialize_cuda_engine(handle.read())
    if engine is None:
        raise RuntimeError(f"TensorRT could not deserialize {engine_path}")
    if hasattr(engine, "num_io_tensors"):
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    else:
        names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
    if input_name not in names:
        raise RuntimeError(f"Engine inputs/outputs {names} do not include '{input_name}'")


def build_parser() -> argparse.ArgumentParser:
    """Build the conversion CLI parser."""
    parser = argparse.ArgumentParser(
        description="Convert an AnomaVision PaDiM or PatchCore artifact to TensorRT."
    )
    parser.add_argument("--model", type=Path, required=True, help="Input .pth/.pt model artifact")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the engine and calibration cache")
    parser.add_argument("--output-name", default=None, help="Engine filename; defaults to <model>_<precision>.engine")
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="int8")
    parser.add_argument("--device", default="cuda", help="CUDA device used to build the engine")
    parser.add_argument("--input-shape", nargs=4, type=int, default=(1, 3, 224, 224), metavar=("N", "C", "H", "W"))
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--calib-dir", type=Path, default=None, help="Normal calibration images; required for INT8")
    parser.add_argument("--calib-samples", type=int, default=100)
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    parser.add_argument("--static-batch", action="store_true", help="Build a fixed-batch engine")
    parser.add_argument("--skip-validation", action="store_true")
    return parser


def main(argv=None) -> int:
    """Convert one artifact and return a process exit code."""
    args = build_parser().parse_args(argv)
    logger = _logger()
    model_type, artifact = detect_artifact(args.model)
    logger.info("detected %s artifact: %s", model_type, args.model)

    shape = tuple(args.input_shape)
    if args.precision == "int8":
        if args.calib_dir is None or not args.calib_dir.is_dir():
            raise SystemExit("INT8 conversion requires an existing --calib-dir.")
        calibration_images = list(_image_files(args.calib_dir))
        if not calibration_images:
            raise SystemExit(f"No supported calibration images found in {args.calib_dir}.")
        logger.info("using %d calibration images (limit=%d)", len(calibration_images), args.calib_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{args.model.stem}_{args.precision}.engine"
    exporter = ModelExporter(args.model, args.output_dir, logger, device=args.device)
    result = exporter.export_tensorrt(
        input_shape=shape,
        output_name=output_name,
        dynamic_batch=not args.static_batch,
        precision=args.precision,
        calib_dir=str(args.calib_dir) if args.calib_dir else None,
        calib_samples=args.calib_samples,
        workspace_gb=args.workspace_gb,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
    )
    if result is None:
        return 1
    if not args.skip_validation:
        validate_engine(result)
    logger.info("conversion complete: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
