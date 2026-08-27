#!/usr/bin/env python
"""
AnomaVision - Unified Command-Line Interface
A single entry point for all anomaly detection operations.

Usage:
    anomavision train [args...]      # Train a new model
    anomavision export [args...]     # Export model to different formats
    anomavision detect [args...]     # Run inference on images
    anomavision validate [args...]   # Compare inference backends
    anomavision eval [args...]       # Evaluate model performance
"""

import argparse


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
  %(prog)s validate --config config.yml --models model.pt model.onnx --img_path ./test_images
  %(prog)s eval --model model.pt --class_name bottle --dataset_path /data

For detailed help on each command:
  %(prog)s train --help
  %(prog)s export --help
  %(prog)s detect --help
  %(prog)s validate --help
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
    _add_validate_parser(subparsers)
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


def _add_validate_parser(subparsers) -> None:
    from anomavision.validate import create_parser as _cp
    subparsers.add_parser(
        "validate", help="Compare anomaly scores and heatmaps across backends",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_validate)


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
    from anomavision import detect
    detect.main(args)


def _dispatch_validate(args: argparse.Namespace) -> None:
    from anomavision import validate
    validate.main(args)


def _dispatch_eval(args: argparse.Namespace) -> None:
    from anomavision import eval as eval_module
    eval_module.main(args)


def _dispatch_autopilot(args: argparse.Namespace) -> None:
    from anomavision import autopilot
    autopilot.main(args)


def main() -> None:
    """Parse command-line arguments and dispatch the selected operation."""
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
