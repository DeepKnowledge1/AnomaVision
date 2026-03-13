#!/usr/bin/env python
"""
AnomaVision - Unified Command-Line Interface
A single entry point for all anomaly detection operations.

Usage:
    anomavision train [args...]      # Train a new model
    anomavision export [args...]     # Export model to different formats
    anomavision detect [args...]     # Run inference on images
    anomavision eval [args...]       # Evaluate model performance

Examples:
    anomavision train --config config.yml
    anomavision export --model padim_model.pt --format onnx
    anomavision detect --model padim_model.onnx --img_path ./test_images
    anomavision eval --model padim_model.pt --class_name bottle
"""

import argparse
import sys

# Lazy imports — each submodule is only imported when its command is invoked.
# This keeps CLI startup fast and avoids loading torch/cv2 just to print --help.


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="anomavision",
        description="AnomaVision: Professional anomaly detection toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --config config.yml --dataset_path /data --class_name bottle
  %(prog)s export --model padim_model.pt --format onnx --quantize-dynamic
  %(prog)s detect --model padim_model.onnx --img_path ./test --enable_visualization
  %(prog)s eval --model padim_model.pt --class_name bottle --dataset_path /data

For detailed help on each command:
  %(prog)s train --help
  %(prog)s export --help
  %(prog)s detect --help
  %(prog)s eval --help
        """,
    )

    try:
        from anomavision import __version__
        version_str = f"AnomaVision {__version__}"
    except ImportError:
        version_str = "AnomaVision"

    parser.add_argument("--version", action="version", version=version_str)

    subparsers = parser.add_subparsers(
        title="commands",
        description="Available AnomaVision operations",
        dest="command",
        help="Operation to perform",
        required=True,
    )

    _add_train_parser(subparsers)
    _add_export_parser(subparsers)
    _add_detect_parser(subparsers)
    _add_eval_parser(subparsers)

    return parser


# ============================================================
# Subparser definitions
# ============================================================

def _add_train_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "train",
        help="Train a new anomaly detection model",
        description="Train a PaDiM model on normal training images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default="config.yml", help="Path to config.yml/.json")
    p.add_argument("--dataset_path", type=str, help='Path to dataset folder containing "train/good" images')
    p.add_argument("--resize", type=int, nargs="*", metavar=("W", "H"), help="Resize images (e.g., 256 or 256 192)")
    p.add_argument("--crop_size", type=int, nargs="*", metavar=("W", "H"), help="Center crop size (e.g., 224 or 224 224)")
    p.add_argument("--normalize", action="store_true", help="Enable input normalization")
    p.add_argument("--backbone", type=str, choices=["resnet18", "wide_resnet50"], default="resnet18", help="Backbone network for feature extraction")
    p.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    p.add_argument("--feat_dim", type=int, default=50, help="Number of random feature dimensions")
    p.add_argument("--layer_indices", type=int, nargs="+", default=[0], help="Layer indices for feature extraction")
    p.add_argument("--output_model", type=str, default="padim_model.pt", help="Output model filename")
    p.add_argument("--run_name", type=str, default="train_exp", help="Experiment name")
    p.add_argument("--model_data_path", type=str, default="./distributions", help="Directory to save model outputs")
    p.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    p.set_defaults(func=_dispatch_train)


def _add_export_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "export",
        help="Export trained model to different formats",
        description="Export PaDiM models to ONNX, TorchScript, or OpenVINO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, help="Path to config.yml/.json")
    p.add_argument("--model_data_path", type=str, default="./distributions/train_exp", help="Directory containing model files")
    p.add_argument("--model", type=str, required=True, help="Model file to export (.pt)")
    p.add_argument("--format", type=str, choices=["onnx", "openvino", "torchscript", "all"], required=True, help="Export format")
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device for export")
    p.add_argument("--precision", type=str, choices=["fp16", "fp32", "auto"], default="auto", help="Export precision (auto: FP16 for GPU, FP32 for CPU)")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--static-batch", action="store_true", help="Disable dynamic batch size")
    p.add_argument("--optimize", action="store_true", help="Enable mobile optimization (TorchScript)")
    p.add_argument("--quantize-dynamic", action="store_true", help="Export dynamic INT8 quantized ONNX")
    p.add_argument("--quantize-static", action="store_true", help="Export static INT8 quantized ONNX")
    p.add_argument("--calib-samples", type=int, default=100, help="Calibration samples for static quantization")
    p.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    p.set_defaults(func=_dispatch_export)


def _add_detect_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "detect",
        help="Run inference on images",
        description="Run anomaly detection inference using trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, help="Path to config.yml/.json")
    p.add_argument("--img_path", type=str, help="Path to test images folder")
    p.add_argument("--model_data_path", type=str, default="./distributions/train_exp", help="Directory containing model files")
    p.add_argument("--model", type=str, default="padim_model.pt", help="Model file (.pt, .onnx, .engine, .torchscript)")
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device to run inference")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    p.add_argument("--thresh", type=float, help="Threshold for anomaly classification")
    p.add_argument("--num_workers", type=int, default=1, help="Number of data loading workers")
    p.add_argument("--pin_memory", action="store_true", help="Use pinned memory for GPU transfers")
    p.add_argument("--enable_visualization", action="store_false", help="Enable result visualization")
    p.add_argument("--save_visualizations", action="store_false", help="Save visualization images")
    p.add_argument("--viz_output_dir", type=str, default="./visualizations", help="Visualization output directory")
    p.add_argument("--run_name", type=str, default="detect_exp", help="Experiment name")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing run directory")
    p.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    p.add_argument("--detailed_timing", action="store_true", help="Enable detailed timing measurements")
    p.set_defaults(func=_dispatch_detect)


def _add_eval_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "eval",
        help="Evaluate model performance",
        description="Evaluate anomaly detection model on test dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, help="Path to config.yml/.json")
    p.add_argument("--dataset_path", type=str, default=r"D:/01-DATA", help="Path to dataset folder")
    p.add_argument("--class_name", type=str, default="bottle", help="Class name for evaluation (MVTec)")
    p.add_argument("--model_data_path", type=str, default="./distributions/train_exp", help="Directory containing model files")
    p.add_argument("--model", type=str, default="padim_model.onnx", help="Model file to evaluate")
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device to run evaluation")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    p.add_argument("--num_workers", type=int, default=1, help="Number of data loading workers")
    p.add_argument("--pin_memory", action="store_true", help="Use pinned memory for GPU transfers")
    # FIX: was `action="store_true", default=True` which made the flag a no-op.
    # Use store_true with no default (False) — pass the flag explicitly to enable.
    p.add_argument("--enable_visualization", action="store_true", help="Enable result visualization")
    p.add_argument("--save_visualizations", action="store_true", help="Save visualization images")
    p.add_argument("--viz_output_dir", type=str, default="./eval_visualizations", help="Visualization output directory")
    p.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    p.add_argument("--detailed_timing", action="store_true", help="Enable detailed timing measurements")
    p.set_defaults(func=_dispatch_eval)


# ============================================================
# Dispatch functions — one line each, Namespace passed directly.
# No sys.argv manipulation. No double-parsing.
# Each submodule's main() must accept an args parameter.
# ============================================================

def _dispatch_train(args: argparse.Namespace) -> None:
    from anomavision import train

    train.main(args)


def _dispatch_export(args: argparse.Namespace) -> None:
    from anomavision import export
    export.main(args)


def _dispatch_detect(args: argparse.Namespace) -> None:
    from anomavision import detect
    sys.exit(detect.main(args) or 0)


def _dispatch_eval(args: argparse.Namespace) -> None:
    from anomavision import eval as eval_module  # 'eval' shadows the Python builtin — alias it
    eval_module.main(args)


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
