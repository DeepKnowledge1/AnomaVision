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
import os
import subprocess
import sys
import urllib.error
import urllib.request


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anomavision",
        description="AnomaVision: Professional anomaly detection toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    try:
        from anomavision import __version__

        version_str = f"AnomaVision {__version__}"
    except ImportError:
        version_str = "AnomaVision"
    parser.add_argument("--version", action="version", version=version_str)
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
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
        "train",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_train)


def _add_export_parser(subparsers) -> None:
    from anomavision.export import create_parser as _cp

    subparsers.add_parser(
        "export",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_export)


def _add_detect_parser(subparsers) -> None:
    from anomavision.detect import create_parser as _cp

    subparsers.add_parser(
        "detect",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_detect)


def _add_eval_parser(subparsers) -> None:
    from anomavision.eval import create_parser as _cp

    subparsers.add_parser(
        "eval",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_eval)


def _add_autopilot_parser(subparsers) -> None:
    from anomavision.autopilot import create_parser as _cp

    subparsers.add_parser(
        "autopilot",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_autopilot)


def _add_drift_parser(subparsers) -> None:
    from anomavision.drift_cli import create_parser as _cp

    subparsers.add_parser(
        "drift",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_drift)


def _add_drift_reference_parser(subparsers) -> None:
    from anomavision.drift_reference import create_parser as _cp

    subparsers.add_parser(
        "drift-reference",
        parents=[_cp(add_help=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ).set_defaults(func=_dispatch_drift_reference)


def _dashboard_is_running() -> bool:
    """Return True when the local production dashboard already owns port 7860."""
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:7860/health", timeout=0.4
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


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
            status_file = os.path.abspath(
                getattr(args, "drift_output", None) or "./drift/drift_status.json"
            )
            env = os.environ.copy()
            env["ANOMAVISION_DRIFT_STATUS_FILE"] = status_file
            if getattr(args, "config", None):
                env["ANOMAVISION_DRIFT_CONFIG"] = os.path.abspath(args.config)
            env["ANOMAVISION_PROJECT_ROOT"] = os.getcwd()

            if _dashboard_is_running():
                # Reuse the existing dashboard. Starting a second process on
                # port 7860 previously failed silently and left the UI attached
                # to the previous run.
                print("[AnomaVision] Reusing live drift dashboard: http://127.0.0.1:7860/anomavision_dashboard.html")
            else:
                dashboard_script = os.path.join("apps", "ui", "drift_dashboard.py")
                subprocess.Popen(
                    [sys.executable, dashboard_script],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                print("[AnomaVision] Live drift dashboard: http://127.0.0.1:7860/anomavision_dashboard.html")
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
    args = create_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
