"""Deployment adapters for AnomaVision Studio.

Studio orchestrates the existing exporter and deployment validator. It does not
implement model conversion or inference logic itself.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from anomavision.config import load_config
from anomavision.deployment_validation import validate_model
from anomavision.export import ModelExporter
from anomavision.utils import get_logger, setup_logging

TARGET_FORMATS = {"onnx": "onnx", "openvino": "openvino", "tensorrt": "tensorrt", "torchscript": "torchscript"}
TARGET_DESCRIPTIONS = {
    "cpu": "Validate the existing PyTorch model on CPU",
    "onnx": "Portable ONNX runtime artifact",
    "openvino": "OpenVINO IR for CPU/Intel deployment",
    "tensorrt": "TensorRT engine for NVIDIA deployment",
    "torchscript": "TorchScript artifact for PyTorch runtimes",
    "hailo": "Hailo deployment target",
    "kv260": "AMD/Xilinx KV260 deployment target",
}

def _config(model: dict[str, Any]) -> Any:
    path = Path(model["config"])
    if not path.is_file():
        raise FileNotFoundError(f"Model config not found: {path}")
    return load_config(str(path))

def _shape(cfg: Any) -> tuple[int, int, int, int]:
    size = cfg.get("crop_size") or cfg["resize"]
    return (1, 3, int(size[0]), int(size[1]))

def deploy_model(project_dir: Path | str, model: dict[str, Any], target: str, runs: int = 5, warmup_runs: int = 1) -> dict[str, Any]:
    project_dir = Path(project_dir)
    target = target.lower()
    if target not in {"cpu", *TARGET_FORMATS}:
        raise ValueError(f"Target '{target}' is not yet supported for automatic export")
    started = time.perf_counter()
    deployment_dir = project_dir / "deployments" / model["run_name"] / target
    deployment_dir.mkdir(parents=True, exist_ok=True)
    cfg = _config(model)
    output: Path = Path(model["path"])

    if target != "cpu":
        logger = get_logger("anomavision.studio.deployment")
        setup_logging(enabled=True, log_level="INFO", log_to_file=True)
        exporter = ModelExporter(output, deployment_dir, logger, device="cpu")
        shape = _shape(cfg)
        if target == "onnx":
            result = exporter.export_onnx(input_shape=shape, output_name="model.onnx", dynamic_batch=False, force_precision="fp32")
        elif target == "openvino":
            result = exporter.export_openvino(input_shape=shape, output_name="model_openvino", fp16=False, dynamic_batch=False)
        elif target == "tensorrt":
            result = exporter.export_tensorrt(input_shape=shape, output_name="model.engine", dynamic_batch=False, precision="fp32")
        else:
            result = exporter.export_torchscript(input_shape=shape, output_name="model.torchscript", force_precision="fp32")
        if result is None:
            raise RuntimeError(f"{target} export failed")
        output = Path(result)

    validation = validate_model(output, runs=runs, warmup_runs=warmup_runs, config_path=model["config"])
    payload = {
        "model": model, "target": target, "artifact": str(output),
        "validation": validation,
        "ready_for_deployment": validation["ready_for_deployment"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "status": "ready" if validation["ready_for_deployment"] else "failed",
    }
    (deployment_dir / "deployment.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return payload
