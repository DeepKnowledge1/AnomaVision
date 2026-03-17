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

# Submodules are imported lazily inside _add_*_parser() and _dispatch_*().
# CLI startup (including --help on the top-level parser) never touches torch/cv2.
# Note: --help on a subcommand (e.g. `anomavision train --help`) WILL import
# the submodule to build the parser — that is intentional and unavoidable if we
# want the submodule to own its argument definitions.


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

    return parser


# ============================================================
# Subparser registration
#
# Each submodule owns its argument definitions in create_parser().
# cli.py uses `parents=` to inherit all args — zero duplication.
#
# The key: call create_parser(add_help=False) so argparse doesn't
# register -h on the parent. The child subparser adds its own -h
# automatically. Setting add_help at *construction time* is the
# only reliable way — mutating .add_help after construction does
# not remove the already-registered -h action.
#
# Result: add --new-flag to detect.py and `anomavision detect --new-flag`
# works immediately with no changes needed here.
# ============================================================

def _add_train_parser(subparsers) -> None:
    from anomavision.train import create_parser as _cp
    subparsers.add_parser(
        "train",
        help="Train a new anomaly detection model",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_train)


def _add_export_parser(subparsers) -> None:
    from anomavision.export import create_parser as _cp
    subparsers.add_parser(
        "export",
        help="Export trained model to different formats",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_export)


def _add_detect_parser(subparsers) -> None:
    from anomavision.detect import create_parser as _cp
    subparsers.add_parser(
        "detect",
        help="Run inference on images",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_detect)


def _add_eval_parser(subparsers) -> None:
    from anomavision.eval import create_parser as _cp
    subparsers.add_parser(
        "eval",
        help="Evaluate model performance",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_eval)


# ============================================================
# Dispatch functions — one line each, Namespace passed directly.
# No sys.argv manipulation. No double-parsing.
# ============================================================

def _dispatch_train(args: argparse.Namespace) -> None:
    from anomavision import train
    train.main(args)


def _dispatch_export(args: argparse.Namespace) -> None:
    from anomavision import export
    export.main(args)


def _dispatch_detect(args: argparse.Namespace) -> None:
    from anomavision import detect
    detect.main(args)


def _dispatch_eval(args: argparse.Namespace) -> None:
    from anomavision import eval as eval_module  # 'eval' shadows the Python builtin
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
