#!/usr/bin/env python
"""
AnomaVision - Unified Command-Line Interface
A single entry point for all anomaly detection operations.

Usage:
    anomavision train [args...]           # Train a new model
    anomavision export [args...]          # Export model to different formats
    anomavision detect [args...]          # Run inference on images
    anomavision eval [args...]            # Evaluate model performance
    anomavision drift [args...]           # Compare embedding distributions
    anomavision drift-reference [args...] # Generate trusted reference embeddings
"""

import argparse
import os
import subprocess
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
  %(prog)s drift --reference reference.npy --current production.npy --output drift.json
  %(prog)s drift-reference --config config.yml --img_path ./train/good --output ./drift/reference_embeddings.npy

For detailed help on each command:
  %(prog)s train --help
  %(prog)s export --help
  %(prog)s detect --help
  %(prog)s eval --help
  %(prog)s drift --help
  %(prog)s drift-reference --help
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
    _add_drift_parser(subparsers)
    _add_drift_reference_parser(subparsers)
    return parser


def _add_train_parser(subparsers) -> None:
    from anomavision.train import create_parser as _cp
    subparsers.add_parser(
        "train", help="Train a new anomaly detection model", parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_train)


def _add_export_parser(subparsers) -> None:
    from anomavision.export import create_parser as _cp
    subparsers.add_parser(
        "export", help="Export trained model to different formats", parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_export)


def _add_detect_parser(subparsers) -> None:
    from anomavision.detect import create_parser as _cp
    subparsers.add_parser(
        "detect", help="Run inference on images", parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_detect)


def _add_eval_parser(subparsers) -> None:
    from anomavision.eval import create_parser as _cp
    subparsers.add_parser(
        "eval", help="Evaluate model performance", parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_eval)


def _add_autopilot_parser(subparsers) -> None:
    from anomavision.autopilot import create_parser as _cp
    subparsers.add_parser(
        "autopilot", help="Calibrate, profile, and package a production model", parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_autopilot)


def _add_drift_parser(subparsers) -> None:
    from anomavision.drift_cli import create_parser as _cp
    subparsers.add_parser(
        "drift", help="Compare embedding distributions for drift", parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_drift)


def _add_drift_reference_parser(subparsers) -> None:
    from anomavision.drift_reference import create_parser as _cp
    subparsers.add_parser(
        "drift-reference",
        help="Generate trusted reference embeddings for production drift monitoring",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_drift_reference)


def _dispatch_train(args: argparse.Namespace) -> None:
    from anomavision import train
    train.main(args)


def _dispatch_export(args: argparse.Namespace) -> None:
    from anomavision import export
    export.main(args)


def _dispatch_detect(args: argparse.Namespace) -> None:
    from anomavision import detect

    if getattr(args, "enable_drift_monitoring", False):
        try:
            status_file = getattr(args, "drift_output", None) or "./drift/drift_status.json"
            env = os.environ.copy()
            env["ANOMAVISION_DRIFT_STATUS_FILE"] = str(status_file)
            if getattr(args, "config", None):
                env["ANOMAVISION_DRIFT_CONFIG"] = str(args.config)

            subprocess.Popen(
                [sys.executable, "-m", "anomavision.drift_dashboard"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            print("[AnomaVision] Live drift dashboard: http://127.0.0.1:7860")
        except Exception as exc:
            # Monitoring UI must never stop anomaly inference.
            print(f"[AnomaVision] Live drift dashboard could not start: {exc}")

    detect.main(args)


def _dispatch_eval(args: argparse.Namespace) -> None:
    from anomavision import eval as eval_module
    eval_module.main(args)


def _dispatch_autopilot(args: argparse.Namespace) -> None:
    from anomavision import autopilot
    autopilot.main(args)


def _dispatch_drift(args: argparse.Namespace) -> None:
    from anomavision import drift_cli
    drift_cli.main(args)


def _dispatch_drift_reference(args: argparse.Namespace) -> None:
    from anomavision import drift_reference
    drift_reference.main(args)


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
