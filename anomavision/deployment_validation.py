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
    parser.add_argument("--model", required=True, help="Path to a trained/exported model.")
    parser.add_argument("--config", default=None, help="Existing AnomaVision config used for input shape.")
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


def validate_model(model_path: str | Path, runs: int = 10, warmup_runs: int = 2, config_path: str | Path | None = None) -> dict[str, Any]:
    """Validate an exported model without changing its behavior."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    if runs < 1 or warmup_runs < 0:
        raise ValueError("runs must be >= 1 and warmup_runs must be >= 0")

    suffix = path.suffix.lower()

    if suffix == ".onnx":
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        inputs = [{"name": v.name, "shape": _shape(v), "type": v.type.tensor_type.elem_type} for v in model.graph.input]
        outputs = [{"name": v.name, "shape": _shape(v), "type": v.type.tensor_type.elem_type} for v in model.graph.output]
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feed = {}
        for item in session.get_inputs():
            shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in item.shape]
            if item.type != "tensor(float)":
                raise ValueError(f"Unsupported validation input type for {item.name}: {item.type}")
            feed[item.name] = np.zeros(shape, dtype=np.float32)
        for _ in range(warmup_runs):
            session.run(None, feed)
        start_time = time.perf_counter()
        for _ in range(runs):
            session.run(None, feed)
        latency_ms = (time.perf_counter() - start_time) / runs * 1000.0
        checks = {
            "file_exists": True,
            "onnx_valid": True,
            "onnxruntime_inference": True,
            "static_input_shape": all(
                all(dim != "?" for dim in item["shape"]) for item in inputs
            ),
        }
        model_format = "onnx"
    elif suffix in {".pt", ".pth", ".torchscript", ".engine", ".hef", ".xmodel", ".xml"}:
        from anomavision.inference.model.wrapper import ModelWrapper
        from anomavision.config import _shape as config_shape, load_config
        import torch

        wrapper = ModelWrapper(str(path), "cpu")
        try:
            inputs, outputs, latency_ms = [], [], None
            if config_path:
                cfg = load_config(str(config_path))
                size = config_shape(cfg["resize"])
                crop = cfg.get("crop_size")
                if crop:
                    size = config_shape(crop)
                batch = torch.zeros((1, 3, size[1], size[0]), dtype=torch.float32)
                wrapper.warmup(batch=batch, runs=warmup_runs)
                start_time = time.perf_counter()
                for _ in range(runs):
                    wrapper.predict(batch)
                latency_ms = (time.perf_counter() - start_time) / runs * 1000.0
                inputs = [{"name": "input", "shape": list(batch.shape), "type": "float32"}]
            checks = {
                "file_exists": True,
                "model_load": True,
                "inference": latency_ms is not None,
            }
        finally:
            wrapper.close()
        model_format = suffix.lstrip(".")
    else:
        raise ValueError(f"Unsupported model format '{suffix}'.")

    backends = _backend_status()
    return {
        "model": str(path),
        "format": model_format,
        "inputs": inputs,
        "outputs": outputs,
        "performance": {
            "runs": runs,
            "warmup_runs": warmup_runs,
            "latency_ms": round(latency_ms, 3) if latency_ms is not None else None,
            "fps": round(1000.0 / latency_ms, 2) if latency_ms else None,
        },
        "backends": backends,
        "checks": checks,
        "ready_for_deployment": all(checks.values()),
        "note": "Validation is observational and reuses AnomaVision's existing inference backends.",
    }


def _print_report(report: dict[str, Any]) -> None:
    print("AnomaVision Deployment Validation")
    print("─" * 36)
    print(f"Model: {report['model']}")
    for name, value in report["checks"].items():
        print(f"{'✓' if value else '✗'} {name.replace('_', ' ').title()}")
    perf = report["performance"]
    if perf["latency_ms"] is not None:
        print(f"Latency: {perf['latency_ms']:.3f} ms")
        print(f"FPS:     {perf['fps']:.2f}")
    else:
        print("Latency: not measured (provide --config for runtime validation)")
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
    report = validate_model(args.model, args.runs, args.warmup_runs, args.config)
    if args.json_output:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
