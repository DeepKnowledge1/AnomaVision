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
        "--reference-model",
        default=None,
        help="Existing reference model used for output-consistency validation.",
    )
    parser.add_argument(
        "--consistency-tolerance",
        type=float,
        default=1e-4,
        help="Maximum allowed absolute output difference for consistency validation.",
    )
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


def _compare_outputs(
    reference: tuple[np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray],
    tolerance: float,
) -> dict[str, Any]:
    reference_scores, reference_maps = np.asarray(reference[0]), np.asarray(reference[1])
    candidate_scores, candidate_maps = np.asarray(candidate[0]), np.asarray(candidate[1])

    if reference_scores.shape != candidate_scores.shape:
        raise ValueError(
            f"Score output shape mismatch: reference={reference_scores.shape}, "
            f"candidate={candidate_scores.shape}"
        )
    if reference_maps.shape != candidate_maps.shape:
        raise ValueError(
            f"Map output shape mismatch: reference={reference_maps.shape}, "
            f"candidate={candidate_maps.shape}"
        )

    score_diff = np.abs(reference_scores.astype(np.float64) - candidate_scores.astype(np.float64))
    map_diff = np.abs(reference_maps.astype(np.float64) - candidate_maps.astype(np.float64))

    max_score_diff = float(np.max(score_diff))
    mean_score_diff = float(np.mean(score_diff))
    max_map_diff = float(np.max(map_diff))
    mean_map_diff = float(np.mean(map_diff))

    return {
        "tolerance": tolerance,
        "score": {
            "max_abs_diff": max_score_diff,
            "mean_abs_diff": mean_score_diff,
            "within_tolerance": max_score_diff <= tolerance,
        },
        "map": {
            "max_abs_diff": max_map_diff,
            "mean_abs_diff": mean_map_diff,
            "within_tolerance": max_map_diff <= tolerance,
        },
        "within_tolerance": (
            max_score_diff <= tolerance and max_map_diff <= tolerance
        ),
    }


def _build_validation_batch(config_path: str | Path):
    from anomavision.config import _shape as config_shape, load_config
    import torch

    cfg = load_config(str(config_path))
    size = config_shape(cfg["resize"])
    crop = cfg.get("crop_size")
    if crop:
        size = config_shape(crop)
    return torch.zeros((1, 3, size[1], size[0]), dtype=torch.float32)


def _run_model(model_path: str | Path, batch):
    from anomavision.inference.model.wrapper import ModelWrapper

    wrapper = ModelWrapper(str(model_path), "cpu")
    try:
        return wrapper.predict(batch)
    finally:
        wrapper.close()


def validate_model(
    model_path: str | Path,
    runs: int = 10,
    warmup_runs: int = 2,
    config_path: str | Path | None = None,
    reference_model: str | Path | None = None,
    consistency_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Validate an exported model without changing its behavior."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    if runs < 1 or warmup_runs < 0:
        raise ValueError("runs must be >= 1 and warmup_runs must be >= 0")
    if consistency_tolerance < 0:
        raise ValueError("consistency_tolerance must be >= 0")
    if reference_model and not Path(reference_model).is_file():
        raise FileNotFoundError(f"Reference model not found: {reference_model}")
    if reference_model and not config_path:
        raise ValueError("--config is required for output-consistency validation")

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

        wrapper = ModelWrapper(str(path), "cpu")
        try:
            inputs, outputs, latency_ms = [], [], None
            if config_path:
                batch = _build_validation_batch(config_path)
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

    consistency = None
    if reference_model:
        batch = _build_validation_batch(config_path)
        reference_output = _run_model(reference_model, batch)
        candidate_output = _run_model(path, batch)
        consistency = _compare_outputs(
            reference_output, candidate_output, consistency_tolerance
        )
        checks["output_consistency"] = consistency["within_tolerance"]

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
        "consistency": consistency,
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

    consistency = report.get("consistency")
    if consistency is not None:
        print()
        print("Output consistency")
        print(f"  Score max abs diff: {consistency['score']['max_abs_diff']:.6g}")
        print(f"  Score mean abs diff: {consistency['score']['mean_abs_diff']:.6g}")
        print(f"  Map max abs diff:   {consistency['map']['max_abs_diff']:.6g}")
        print(f"  Map mean abs diff:  {consistency['map']['mean_abs_diff']:.6g}")
        print(f"  Tolerance:          {consistency['tolerance']:.6g}")

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
    report = validate_model(
        args.model,
        args.runs,
        args.warmup_runs,
        args.config,
        args.reference_model,
        args.consistency_tolerance,
    )
    if args.json_output:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
