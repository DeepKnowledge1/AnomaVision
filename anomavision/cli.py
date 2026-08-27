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
    anomavision export --config config.yml --model model.pt --format onnx
    anomavision detect --config config.yml --model model.onnx --img_path ./test_images
    anomavision eval --config config.yml --model model.pt --class_name bottle
"""

import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="anomavision",
        description="AnomaVision: Professional anomaly detection toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --config config.yml --dataset_path /data --class_name bottle
  %(prog)s export --model model.pt --format onnx --quantize-dynamic
  %(prog)s detect --model model.onnx --img_path ./test --enable_visualization
  %(prog)s eval --model model.pt --class_name bottle --dataset_path /data

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
    _add_autopilot_parser(subparsers)
    return parser


def _add_train_parser(subparsers) -> None:
    from anomavision.train import create_parser as _cp
    subparsers.add_parser(
        "train", help="Train a new anomaly detection model",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_train)


def _add_export_parser(subparsers) -> None:
    from anomavision.export import create_parser as _cp
    subparsers.add_parser(
        "export", help="Export trained model to different formats",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_export)


def _add_detect_parser(subparsers) -> None:
    from anomavision.detect import create_parser as _cp
    subparsers.add_parser(
        "detect", help="Run inference on images",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_detect)


def _add_eval_parser(subparsers) -> None:
    from anomavision.eval import create_parser as _cp
    subparsers.add_parser(
        "eval", help="Evaluate model performance",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_eval)


def _add_autopilot_parser(subparsers) -> None:
    from anomavision.autopilot import create_parser as _cp
    subparsers.add_parser(
        "autopilot", help="Calibrate, profile, and package a production model",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_autopilot)


def _dispatch_train(args: argparse.Namespace) -> None:
    from anomavision import train
    train.main(args)


def _dispatch_export(args: argparse.Namespace) -> None:
    from anomavision import export
    export.main(args)


def _dispatch_detect(args: argparse.Namespace) -> None:
    from pathlib import Path

    from anomavision.config import load_config
    from anomavision.efficientad_threshold import load_calibrated_threshold

    cfg = load_config(args.config) if getattr(args, "config", None) else {}
    algorithm = str(getattr(args, "algorithm", None) or cfg.get("algorithm", "")).lower()

    if algorithm == "efficientad" and getattr(args, "thresh", None) is None:
        model_data_path = getattr(args, "model_data_path", None) or cfg.get("model_data_path", "./distributions")
        class_name = getattr(args, "class_name", None) or cfg.get("class_name")
        run_name = getattr(args, "run_name", None) or cfg.get("run_name")
        model_name = getattr(args, "model", None) or cfg.get("model")

        if class_name and run_name and model_name:
            model_path = Path(model_data_path) / algorithm / class_name / run_name / model_name
            args.thresh = load_calibrated_threshold(model_path)

    from anomavision import detect
    detect.main(args)


def _dispatch_eval(args: argparse.Namespace) -> None:
    from anomavision import eval as eval_module
    eval_module.main(args)


def _dispatch_autopilot(args: argparse.Namespace) -> None:
    from anomavision import autopilot
    autopilot.main(args)


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
