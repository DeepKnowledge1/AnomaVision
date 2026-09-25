"""
inference_engine.py
-------------------
Single source of truth for model loading and inference.

Loaded ONCE at process startup and imported by both:
  - api.py      (FastAPI routes)
  - ui.py       (Gradio frontend)

Neither layer touches ONNX Runtime directly — they call the
functions here and receive plain numpy arrays back.
"""

import multiprocessing
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import onnxruntime as ort
from onnxruntime import GraphOptimizationLevel, SessionOptions
from PIL import Image

from anomavision.static.AnomaVision import classification, to_batch, visualization

# -----------------------------------------------------------------------------
# Config — all overridable via environment variables
# -----------------------------------------------------------------------------
ANOMALY_THRESHOLD = float(os.getenv("ANOMAVISION_THRESHOLD", "13.0"))
MODEL_DATA_PATH = os.getenv("ANOMAVISION_MODEL_DATA_PATH", "")
MODEL_FILE = os.getenv("ANOMAVISION_MODEL_FILE", "model.onnx")
STUDIO_ROOT = os.path.expanduser(
    os.getenv("ANOMAVISION_STUDIO_ROOT", "~/.anomavision/projects")
)
VIZ_PADDING = int(os.getenv("ANOMAVISION_VIZ_PADDING", "40"))
VIZ_ALPHA = float(os.getenv("ANOMAVISION_VIZ_ALPHA", "0.5"))
VIZ_COLOR = tuple(map(int, os.getenv("ANOMAVISION_VIZ_COLOR", "128,0,128").split(",")))

# -----------------------------------------------------------------------------
# Module-level session — shared across all callers in the same process
# -----------------------------------------------------------------------------
_sess: Optional[ort.InferenceSession] = None
_input_name: Optional[str] = None


@dataclass
class InferenceResult:
    """Everything a caller needs. All arrays are uint8 RGB numpy arrays."""

    anomaly_score: float
    is_anomaly: bool
    image_np: np.ndarray  # original RGB, (H, W, 3) uint8
    heatmap_np: np.ndarray  # heatmap overlay, same shape
    boundary_np: np.ndarray  # framed boundary, same shape
    latency_ms: float


# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------
def _resolve_model_path(project_id: Optional[str] = None) -> Optional[str]:
    """Resolve the Studio project's latest exported ONNX model."""
    if MODEL_DATA_PATH:
        explicit = os.path.realpath(os.path.join(MODEL_DATA_PATH, MODEL_FILE))
        if os.path.isfile(explicit):
            return explicit
        if os.path.isfile(os.path.realpath(MODEL_DATA_PATH)) and MODEL_DATA_PATH.lower().endswith(".onnx"):
            return os.path.realpath(MODEL_DATA_PATH)

    root = os.path.realpath(STUDIO_ROOT)
    projects = []
    if project_id:
        projects = [os.path.join(root, project_id)]
    elif os.path.isdir(root):
        projects = [
            os.path.join(root, name)
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        ]

    candidates = []
    for project_dir in projects:
        metadata_path = os.path.join(project_dir, "models", "latest_training.json")
        if not os.path.isfile(metadata_path):
            continue
        try:
            import json
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            model_path = os.path.realpath(metadata.get("model", ""))
            if not os.path.isfile(model_path):
                continue
            run_name = os.path.splitext(os.path.basename(model_path))[0]
            deployment = os.path.join(
                project_dir, "deployments", os.path.basename(os.path.dirname(model_path)),
                "onnx", "model.onnx"
            )
            if os.path.isfile(deployment):
                candidates.append((os.path.getmtime(deployment), deployment))
        except (OSError, ValueError, TypeError):
            continue

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _export_latest_model(project_id: Optional[str] = None) -> Optional[str]:
    """Export the project's trained PyTorch artifact to the Studio ONNX location."""
    root = os.path.realpath(STUDIO_ROOT)
    if project_id:
        project_dirs = [os.path.join(root, project_id)]
    else:
        project_dirs = [
            os.path.join(root, name)
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        ] if os.path.isdir(root) else []

    for project_dir in project_dirs:
        metadata_path = os.path.join(project_dir, "models", "latest_training.json")
        if not os.path.isfile(metadata_path):
            continue
        try:
            import json
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            model_path = os.path.realpath(metadata.get("model", ""))
            config_path = os.path.realpath(metadata.get("config", ""))
            if not os.path.isfile(model_path) or not os.path.isfile(config_path):
                continue
            run_name = os.path.basename(os.path.dirname(model_path))
            output_dir = os.path.join(project_dir, "deployments", run_name, "onnx")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "model.onnx")
            if os.path.isfile(output_path) and os.path.getmtime(output_path) >= os.path.getmtime(model_path):
                return output_path

            from anomavision.config import load_config
            from anomavision.export import ModelExporter
            from anomavision.utils import get_logger, setup_logging
            cfg = load_config(config_path)
            size = cfg.get("crop_size") or cfg["resize"]
            logger = get_logger("anomavision.studio.inference")
            setup_logging(enabled=True, log_level="INFO", log_to_file=False)
            exporter = ModelExporter(model_path, output_dir, logger, device="cpu")
            result = exporter.export_onnx(
                input_shape=(1, 3, int(size[0]), int(size[1])),
                output_name="model.onnx",
                dynamic_batch=False,
                force_precision="fp32",
            )
            return str(result) if result else None
        except Exception as exc:
            print(f"[inference] Could not export trained model: {exc}")
    return None


def load_model(project_id: Optional[str] = None) -> str:
    """
    Load the selected Studio project's ONNX model and run two warmup passes.
    If no trained model exists yet, keep the API alive and report that state.
    """
    global _sess, _input_name

    model_path = _resolve_model_path(project_id) or _export_latest_model(project_id)
    if not model_path:
        _sess = None
        _input_name = None
        return "No trained Studio model is available yet."

    available = ort.get_available_providers()
    use_gpu = "CUDAExecutionProvider" in available
    providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

    opts = SessionOptions()
    opts.enable_mem_pattern = True
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if not use_gpu:
        opts.enable_cpu_mem_arena = True
        opts.intra_op_num_threads = multiprocessing.cpu_count()

    _sess = ort.InferenceSession(model_path, providers=providers, sess_options=opts)
    _input_name = _sess.get_inputs()[0].name

    # Warmup — run twice so JIT compile happens now, not on the first real request
    dummy_shape = tuple(
        d if isinstance(d, int) and d > 0 else 1 for d in _sess.get_inputs()[0].shape
    )
    dummy = np.zeros(dummy_shape, dtype=np.float32)
    _sess.run(None, {_input_name: dummy})  # triggers JIT compile
    t0 = time.perf_counter()
    _sess.run(None, {_input_name: dummy})  # steady-state measurement
    warmup_ms = (time.perf_counter() - t0) * 1000

    device = "GPU" if use_gpu else "CPU"
    return (
        f"Model loaded: {os.path.basename(model_path)} | "
        f"Device: {device} | "
        f"Warmup latency: {warmup_ms:.1f} ms"
    )


def is_loaded() -> bool:
    return _sess is not None


def session_info() -> dict:
    if _sess is None:
        return {"status": "not loaded"}
    return {
        "status": "loaded",
        "inputs": [(i.name, i.shape, i.type) for i in _sess.get_inputs()],
        "outputs": [(o.name, o.shape, o.type) for o in _sess.get_outputs()],
        "providers": _sess.get_providers(),
        "threshold": ANOMALY_THRESHOLD,
    }


# -----------------------------------------------------------------------------
# Core inference — called by both FastAPI and Gradio
# -----------------------------------------------------------------------------
def run(image_np: np.ndarray, threshold: float = ANOMALY_THRESHOLD) -> InferenceResult:
    """
    Run anomaly detection on a single RGB numpy array (H, W, 3) uint8.

    Accepts the raw image array — no PIL, no file I/O here.
    Returns an InferenceResult with all visualizations as numpy arrays.

    The caller (api.py or ui.py) decides how to present them:
      - api.py   → encodes to base64 PNG for JSON transport
      - ui.py    → passes numpy arrays directly to gr.Image (zero serialization)
    """
    if _sess is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")

    t0 = time.perf_counter()

    # to_batch applies standard_image_transform internally
    batch = to_batch([image_np])
    outputs = _sess.run(None, {_input_name: batch})

    # outputs[0]: image-level score  — shape (1,) or scalar
    # outputs[1]: pixel-level map    — shape (1, H, W) or (H, W)
    image_score = float(np.squeeze(outputs[0]))
    score_maps = outputs[1]

    # Visualizations
    score_map_cls = classification(score_maps, threshold)
    image_cls = classification(np.array([image_score]), threshold)
    test_images = np.array([image_np])

    boundary_np = visualization.framed_boundary_images(
        test_images, score_map_cls, image_cls, padding=VIZ_PADDING
    )[0]
    heatmap_np = visualization.heatmap_images(test_images, score_maps, alpha=VIZ_ALPHA)[
        0
    ]

    latency_ms = (time.perf_counter() - t0) * 1000

    return InferenceResult(
        anomaly_score=image_score,
        is_anomaly=image_score >= threshold,
        image_np=image_np,
        heatmap_np=heatmap_np,
        boundary_np=boundary_np,
        latency_ms=latency_ms,
    )
