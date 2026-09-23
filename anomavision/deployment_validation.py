"""Non-invasive deployment validation for exported AnomaVision models.

This module observes exported artifacts only. It does not modify model files,
preprocessing, anomaly scoring, localization, or algorithm implementations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an exported model for deployment without changing inference logic.",
        add_help=add_help,
    )
    parser.add_argument("--model", required=True, help="Path to an ONNX model.")
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of inference runs for the latency check."
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=2, help="Number of warm-up runs excluded from latency."
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Print the report as JSON."
    )
    return parser


def _shape(value: Any) -> list[Any]:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(dim.dim_value)
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def _backend_status() -> dict[str, str]:
    return {
        "onnxruntime": "available",
        "openvino": "available" if importlib.util.find_spec("openvino") else "not installed",
        "tensorrt": "available" if importlib.util.find_spec("tensorrt") else "not installed",
        "hailo": (
            "available"
            if importlib.util.find_spec("hailo_platform")
            else "not installed"
        ),
    }


def validate_model(model_path: str | Path, runs: int = 10, warmup_runs: int = 2) -> dict[str, Any]:
    """Validate an ONNX artifact without changing its behavior."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    if runs < 1 or warmup_runs < 0:
        raise ValueError("runs must be >= 1 and warmup_runs must be >= 0")

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    inputs = []
    for value in model.graph.input:
        inputs.append({"name": value.name, "shape": _shape(value), "type": value.type.tensor_type.elem_type})

    outputs = []
    for value in model.graph.output:
        outputs.append({"name": value.name, "shape": _shape(value), "type": value.type.tensor_type.elem_type})

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    session_inputs = session.get_inputs()
    feed: dict[str, np.ndarray] = {}
    for item in session_inputs:
        shape = []
        for dim in item.shape:
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            else:
                shape.append(1)
        if item.type != "tensor(float)":
            raise ValueError(
                f"Unsupported validation input type for {item.name}: {item.type}. "
                "The validator currently supports float ONNX inputs."
            )
        feed[item.name] = np.zeros(shape, dtype=np.float32)

    for _ in range(warmup_runs):
        session.run(None, feed)

    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, feed)
    elapsed = time.perf_counter() - start
    latency_ms = elapsed / runs * 1000.0

    backends = _backend_status()
    checks = {
        "file_exists": True,
        "onnx_valid": True,
        "onnxruntime_inference": True,
        "static_input_shape": all(all(dim != "?" for dim in item["shape"]) for item in inputs),
    }

    return {
        "model": str(path),
        "format": "onnx",
        "inputs": inputs,
        "outputs": outputs,
        "performance": {
            "runs": runs,
            "warmup_runs": warmup_runs,
            "latency_ms": round(latency_ms, 3),
            "fps": round(1000.0 / latency_ms, 2) if latency_ms else None,
        },
        "backends": backends,
        "checks": checks,
        "ready_for_deployment": all(checks.values()),
        "note": "Validation is observational; no model or anomaly-detection logic is modified.",
    }


def _print_report(report: dict[str, Any]) -> None:
    print("AnomaVision Deployment Validation")
    print("─" * 36)
    print(f"Model: {report['model']}")
    for name, value in report["checks"].items():
        print(f"{'✓' if value else '✗'} {name.replace('_', ' ').title()}")
    perf = report["performance"]
    print(f"Latency: {perf['latency_ms']:.3f} ms")
    print(f"FPS:     {perf['fps']:.2f}")
    print()
    print("Backend compatibility")
    for name, status in report["backends"].items():
        print(f"  {name}: {status}")
    print()
    print(
        "RESULT: "
        + ("READY FOR DEPLOYMENT" if report["ready_for_deployment"] else "NOT READY")
    )


def main(args: argparse.Namespace) -> None:
    report = validate_model(args.model, args.runs, args.warmup_runs)
    if args.json_output:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
