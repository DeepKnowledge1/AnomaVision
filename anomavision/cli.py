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
from pathlib import Path

# Add project root to Python path to import train.py, export.py, etc.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now we can import the scripts from project root
import train
import export
import detect
import eval as eval_module  # 'eval' is a Python builtin, so alias it


def create_parser():
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

    parser.add_argument(
        "--version",
        action="version",
        version="AnomaVision 1.0.0",
    )

    # Create subparsers for each operation
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available AnomaVision operations",
        dest="command",
        help="Operation to perform",
        required=True,
    )

    # ========================================
    # TRAIN subcommand
    # ========================================
    train_parser = subparsers.add_parser(
        "train",
        help="Train a new anomaly detection model",
        description="Train a PaDiM model on normal training images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    train_parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to config.yml/.json",
    )
    train_parser.add_argument(
        "--dataset_path",
        type=str,
        help='Path to dataset folder containing "train/good" images',
    )
    train_parser.add_argument(
        "--resize",
        type=int,
        nargs="*",
        metavar=("W", "H"),
        help="Resize images (e.g., 256 or 256 192)",
    )
    train_parser.add_argument(
        "--crop_size",
        type=int,
        nargs="*",
        metavar=("W", "H"),
        help="Center crop size (e.g., 224 or 224 224)",
    )
    train_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable input normalization",
    )
    train_parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet18", "wide_resnet50"],
        default="wide_resnet50",
        help="Backbone network for feature extraction",
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training",
    )
    train_parser.add_argument(
        "--feat_dim",
        type=int,
        default=50,
        help="Number of random feature dimensions",
    )
    train_parser.add_argument(
        "--layer_indices",
        type=int,
        nargs="+",
        default=[0],
        help="Layer indices for feature extraction",
    )
    train_parser.add_argument(
        "--output_model",
        type=str,
        default="padim_model.pt",
        help="Output model filename",
    )
    train_parser.add_argument(
        "--run_name",
        type=str,
        default="train_exp",
        help="Experiment name",
    )
    train_parser.add_argument(
        "--model_data_path",
        type=str,
        default="./distributions",
        help="Directory to save model outputs",
    )
    train_parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    # ========================================
    # EXPORT subcommand
    # ========================================
    export_parser = subparsers.add_parser(
        "export",
        help="Export trained model to different formats",
        description="Export PaDiM models to ONNX, TorchScript, or OpenVINO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    export_parser.add_argument(
        "--config",
        type=str,
        help="Path to config.yml/.json",
    )
    export_parser.add_argument(
        "--model_data_path",
        type=str,
        default="./distributions/anomav_exp",
        help="Directory containing model files",
    )
    export_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model file to export (.pt)",
    )
    export_parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "openvino", "torchscript", "all"],
        required=True,
        help="Export format",
    )
    export_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for export",
    )
    export_parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "fp32", "auto"],
        default="auto",
        help="Export precision (auto: FP16 for GPU, FP32 for CPU)",
    )
    export_parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    export_parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Disable dynamic batch size",
    )
    export_parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable mobile optimization (TorchScript)",
    )
    export_parser.add_argument(
        "--quantize-dynamic",
        action="store_true",
        help="Export dynamic INT8 quantized ONNX",
    )
    export_parser.add_argument(
        "--quantize-static",
        action="store_true",
        help="Export static INT8 quantized ONNX",
    )
    export_parser.add_argument(
        "--calib-samples",
        type=int,
        default=100,
        help="Calibration samples for static quantization",
    )
    export_parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    # ========================================
    # DETECT subcommand
    # ========================================
    detect_parser = subparsers.add_parser(
        "detect",
        help="Run inference on images",
        description="Run anomaly detection inference using trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    detect_parser.add_argument(
        "--config",
        type=str,
        help="Path to config.yml/.json",
    )
    detect_parser.add_argument(
        "--img_path",
        type=str,
        help="Path to test images folder",
    )
    detect_parser.add_argument(
        "--model_data_path",
        type=str,
        default="./distributions/anomav_exp",
        help="Directory containing model files",
    )
    detect_parser.add_argument(
        "--model",
        type=str,
        default="padim_model.pt",
        help="Model file (.pt, .onnx, .engine, .torchscript)",
    )
    detect_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run inference",
    )
    detect_parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    detect_parser.add_argument(
        "--thresh",
        type=float,
        help="Threshold for anomaly classification",
    )
    detect_parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of data loading workers",
    )
    detect_parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Use pinned memory for GPU transfers",
    )
    detect_parser.add_argument(
        "--enable_visualization",
        action="store_true",
        help="Enable result visualization",
    )
    detect_parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save visualization images",
    )
    detect_parser.add_argument(
        "--viz_output_dir",
        type=str,
        default="./visualizations",
        help="Visualization output directory",
    )
    detect_parser.add_argument(
        "--run_name",
        type=str,
        default="detect_exp",
        help="Experiment name",
    )
    detect_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing run directory",
    )
    detect_parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    detect_parser.add_argument(
        "--detailed_timing",
        action="store_true",
        help="Enable detailed timing measurements",
    )

    # ========================================
    # EVAL subcommand
    # ========================================
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate model performance",
        description="Evaluate anomaly detection model on test dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    eval_parser.add_argument(
        "--config",
        type=str,
        help="Path to config.yml/.json",
    )
    eval_parser.add_argument(
        "--dataset_path",
        type=str,
        default=r"D:/01-DATA",
        help="Path to dataset folder",
    )
    eval_parser.add_argument(
        "--class_name",
        type=str,
        default="bottle",
        help="Class name for evaluation (MVTec)",
    )
    eval_parser.add_argument(
        "--model_data_path",
        type=str,
        default="./distributions/anomav_exp",
        help="Directory containing model files",
    )
    eval_parser.add_argument(
        "--model",
        type=str,
        default="padim_model.onnx",
        help="Model file to evaluate",
    )
    eval_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run evaluation",
    )
    eval_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    eval_parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of data loading workers",
    )
    eval_parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Use pinned memory for GPU transfers",
    )
    eval_parser.add_argument(
        "--enable_visualization",
        action="store_true",
        default=True,
        help="Enable result visualization",
    )
    eval_parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save visualization images",
    )
    eval_parser.add_argument(
        "--viz_output_dir",
        type=str,
        default="./eval_visualizations",
        help="Visualization output directory",
    )
    eval_parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    eval_parser.add_argument(
        "--detailed_timing",
        action="store_true",
        help="Enable detailed timing measurements",
    )

    return parser


def dispatch_train(args):
    """Dispatch to train.py with parsed arguments."""
    # train is already imported at the top

    # Convert argparse.Namespace back to sys.argv format for the submodule
    sys.argv = ["train.py"]

    if args.config:
        sys.argv.extend(["--config", args.config])
    if hasattr(args, 'dataset_path') and args.dataset_path:
        sys.argv.extend(["--dataset_path", args.dataset_path])
    if hasattr(args, 'resize') and args.resize:
        sys.argv.extend(["--resize"] + [str(x) for x in args.resize])
    if hasattr(args, 'crop_size') and args.crop_size:
        sys.argv.extend(["--crop_size"] + [str(x) for x in args.crop_size])
    if hasattr(args, 'normalize') and args.normalize:
        sys.argv.append("--normalize")
    if hasattr(args, 'backbone') and args.backbone:
        sys.argv.extend(["--backbone", args.backbone])
    if hasattr(args, 'batch_size') and args.batch_size:
        sys.argv.extend(["--batch_size", str(args.batch_size)])
    if hasattr(args, 'feat_dim') and args.feat_dim:
        sys.argv.extend(["--feat_dim", str(args.feat_dim)])
    if hasattr(args, 'layer_indices') and args.layer_indices:
        sys.argv.extend(["--layer_indices"] + [str(x) for x in args.layer_indices])
    if hasattr(args, 'output_model') and args.output_model:
        sys.argv.extend(["--output_model", args.output_model])
    if hasattr(args, 'run_name') and args.run_name:
        sys.argv.extend(["--run_name", args.run_name])
    if hasattr(args, 'model_data_path') and args.model_data_path:
        sys.argv.extend(["--model_data_path", args.model_data_path])
    if hasattr(args, 'log_level') and args.log_level:
        sys.argv.extend(["--log_level", args.log_level])

    train.main()


def dispatch_export(args):
    """Dispatch to export.py with parsed arguments."""
    # export is already imported at the top

    sys.argv = ["export.py"]
    if args.config:
        sys.argv.extend(["--config", args.config])
    sys.argv.extend(["--model", args.model])
    sys.argv.extend(["--format", args.format])
    sys.argv.extend(["--model_data_path", args.model_data_path])
    if args.device:
        sys.argv.extend(["--device", args.device])
    if args.precision:
        sys.argv.extend(["--precision", args.precision])
    if args.opset:
        sys.argv.extend(["--opset", str(args.opset)])
    if args.static_batch:
        sys.argv.append("--static-batch")
    if args.optimize:
        sys.argv.append("--optimize")
    if args.quantize_dynamic:
        sys.argv.append("--quantize-dynamic")
    if args.quantize_static:
        sys.argv.append("--quantize-static")
    if args.calib_samples:
        sys.argv.extend(["--calib-samples", str(args.calib_samples)])
    if args.log_level:
        sys.argv.extend(["--log-level", args.log_level])

    export.main()


def dispatch_detect(args):
    """Dispatch to detect.py with parsed arguments."""
    # detect is already imported at the top

    sys.argv = ["detect.py"]
    if args.config:
        sys.argv.extend(["--config", args.config])
    if args.img_path:
        sys.argv.extend(["--img_path", args.img_path])
    sys.argv.extend(["--model_data_path", args.model_data_path])
    sys.argv.extend(["--model", args.model])
    if args.device:
        sys.argv.extend(["--device", args.device])
    if args.batch_size:
        sys.argv.extend(["--batch_size", str(args.batch_size)])
    if args.thresh:
        sys.argv.extend(["--thresh", str(args.thresh)])
    if args.num_workers:
        sys.argv.extend(["--num_workers", str(args.num_workers)])
    if args.pin_memory:
        sys.argv.append("--pin_memory")
    if args.enable_visualization:
        sys.argv.append("--enable_visualization")
    if args.save_visualizations:
        sys.argv.append("--save_visualizations")
    if args.viz_output_dir:
        sys.argv.extend(["--viz_output_dir", args.viz_output_dir])
    sys.argv.extend(["--run_name", args.run_name])
    if args.overwrite:
        sys.argv.append("--overwrite")
    sys.argv.extend(["--log_level", args.log_level])
    if args.detailed_timing:
        sys.argv.append("--detailed_timing")

    exit_code = detect.main()
    sys.exit(exit_code)


def dispatch_eval(args):
    """Dispatch to eval.py with parsed arguments."""
    # eval_module is already imported at the top (aliased because 'eval' is a builtin)

    sys.argv = ["eval.py"]
    if args.config:
        sys.argv.extend(["--config", args.config])
    sys.argv.extend(["--dataset_path", args.dataset_path])
    sys.argv.extend(["--class_name", args.class_name])
    sys.argv.extend(["--model_data_path", args.model_data_path])
    sys.argv.extend(["--model", args.model])
    if args.device:
        sys.argv.extend(["--device", args.device])
    if args.batch_size:
        sys.argv.extend(["--batch_size", str(args.batch_size)])
    if args.num_workers:
        sys.argv.extend(["--num_workers", str(args.num_workers)])
    if args.pin_memory:
        sys.argv.append("--pin_memory")
    if args.enable_visualization:
        sys.argv.append("--enable_visualization")
    if args.save_visualizations:
        sys.argv.append("--save_visualizations")
    if args.viz_output_dir:
        sys.argv.extend(["--viz_output_dir", args.viz_output_dir])
    sys.argv.extend(["--log_level", args.log_level])
    if args.detailed_timing:
        sys.argv.append("--detailed_timing")

    eval_module.main(args)


def main():
    """Main entry point for the unified CLI."""
    parser = create_parser()

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Dispatch to appropriate submodule
    try:
        if args.command == "train":
            dispatch_train(args)
        elif args.command == "export":
            dispatch_export(args)
        elif args.command == "detect":
            dispatch_detect(args)
        elif args.command == "eval":
            dispatch_eval(args)
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
