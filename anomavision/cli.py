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


def _dashboard_state():
    """Return metadata for the local dashboard, or None when it is unavailable."""
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:7860/health", timeout=0.5
        ) as response:
            if response.status != 200:
                return None
            import json

            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, ValueError):
        return None


def _dashboard_is_running() -> bool:
    """Return True when the local production dashboard is healthy."""
    return _dashboard_state() is not None


def _stop_dashboard() -> None:
    """Stop the current local dashboard before starting a fresh run."""
    try:
        urllib.request.urlopen(
            "http://127.0.0.1:7860/shutdown", timeout=0.8
        ).read()
    except (OSError, urllib.error.URLError):
        pass


def _wait_for_dashboard_stopped(timeout: float = 3.0) -> bool:
    """Wait until port 7860 no longer serves the dashboard."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _dashboard_state() is None:
            return True
        time.sleep(0.1)
    return _dashboard_state() is None


def _wait_for_dashboard(expected_root: str, expected_status: str, timeout: float = 5.0):
    """Wait for a newly started dashboard to expose the expected run metadata."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _dashboard_state()
        if state is not None:
            running_root = os.path.abspath(state.get("project_root", ""))
            running_status = os.path.abspath(state.get("status_file", ""))
            if (
                running_root.lower() == expected_root.lower()
                and running_status.lower() == expected_status.lower()
            ):
                return state
        time.sleep(0.1)
    return _dashboard_state()


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

            dashboard_state = _dashboard_state()
            expected_root = os.path.abspath(os.getcwd())
            expected_status = os.path.abspath(status_file)

            # Always restart the local dashboard for a new detect run. Reusing
            # a process from a previous run can leave a browser/server process
            # alive with stale environment state even when its project/status
            # paths happen to match the current command.
            if dashboard_state is not None:
                print("[AnomaVision] Restarting drift dashboard for current run...")
                _stop_dashboard()
                if not _wait_for_dashboard_stopped():
                    raise RuntimeError(
                        "Existing drift dashboard is still running on port 7860."
                    )

            dashboard_script = os.path.join("apps", "ui", "drift_dashboard.py")
            subprocess.Popen(
                [sys.executable, dashboard_script],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            dashboard_state = _wait_for_dashboard(expected_root, expected_status)
            if dashboard_state is None:
                raise RuntimeError(
                    "Drift dashboard did not become healthy on port 7860."
                )

            print(
                "[AnomaVision] Live drift dashboard: "
                "http://127.0.0.1:7860/anomavision_dashboard.html"
            )

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
