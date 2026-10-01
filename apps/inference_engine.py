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

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime import GraphOptimizationLevel, SessionOptions
from PIL import Image

from anomavision.static.AnomaVision import classification, to_batch, visualization

# -----------------------------------------------------------------------------
# Config — all overridable via environment variables
# -----------------------------------------------------------------------------
# Keep an explicit environment override authoritative. Otherwise the
# threshold follows the algorithm encoded by the active model path.
_THRESHOLD_OVERRIDE = os.getenv("ANOMAVISION_THRESHOLD")
ANOMALY_THRESHOLD = float(_THRESHOLD_OVERRIDE) if _THRESHOLD_OVERRIDE else 13.0


def _threshold_for_model(model_path: str) -> float:
    """Return the algorithm-appropriate pixel threshold for a model artifact."""
    if _THRESHOLD_OVERRIDE:
        return float(_THRESHOLD_OVERRIDE)
    normalized = os.path.normpath(model_path).lower()
    if "patchcore" in normalized:
        return float(os.getenv("ANOMAVISION_PATCHCORE_THRESHOLD", "0.25"))
    if "efficientad" in normalized:
        return float(os.getenv("ANOMAVISION_EFFICIENTAD_THRESHOLD", "1.0"))
    return float(os.getenv("ANOMAVISION_PADIM_THRESHOLD", "13.0"))


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


def _is_valid_onnx(path: str) -> bool:
    """Return True only when ONNX Runtime can construct a session for the artifact."""
    try:
        ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return True
    except Exception as exc:
        print(f"[inference] Invalid ONNX artifact {path}: {exc}")
        return False


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
        if os.path.isfile(
            os.path.realpath(MODEL_DATA_PATH)
        ) and MODEL_DATA_PATH.lower().endswith(".onnx"):
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
                project_dir,
                "deployments",
                os.path.basename(os.path.dirname(model_path)),
                "onnx",
                "model.onnx",
            )
            if os.path.isfile(deployment) and _is_valid_onnx(deployment):
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
        project_dirs = (
            [
                os.path.join(root, name)
                for name in os.listdir(root)
                if os.path.isdir(os.path.join(root, name))
            ]
            if os.path.isdir(root)
            else []
        )

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
            if (
                os.path.isfile(output_path)
                and os.path.getmtime(output_path) >= os.path.getmtime(model_path)
                and _is_valid_onnx(output_path)
            ):
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
                include_embeddings=False,
            )
            if result and _is_valid_onnx(str(result)):
                return str(result)
            print(f"[inference] Exported ONNX artifact failed validation: {result}")
            return None
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

    # The previous API used a fixed PaDiM threshold (13.0) for every model.
    # PatchCore maps use a much smaller score scale, so that made the heatmap
    # show the defect while the binary localization mask was completely empty.
    global ANOMALY_THRESHOLD
    ANOMALY_THRESHOLD = _threshold_for_model(model_path)
    print(f"[inference] Localization threshold: {ANOMALY_THRESHOLD}")

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
def run(
    image_np: np.ndarray,
    threshold: float = ANOMALY_THRESHOLD,
    include_visualizations: bool = True,
) -> InferenceResult:
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

    # Visualizations are optional. Live/camera inference does not need them;
    # skipping this CPU-heavy path keeps latency close to the raw ONNX runtime.
    if include_visualizations:
        # The image score decides whether the frame is anomalous.
        # Localization is derived independently from the spatial PatchCore map.
        #
        # Do not use a fixed normalized threshold such as 0.60 here: for a
        # localized defect that can select a large portion of the object/image.
        # Instead, keep only the strongest spatial response (top 5% of pixels)
        # and then retain the connected high-score regions. This makes the
        # contour follow the defect rather than the whole image.
        raw_map = np.asarray(score_maps, dtype=np.float32)
        flat_map = raw_map.reshape(raw_map.shape[0], -1)
        map_min = flat_map.min(axis=1)[:, None, None]
        map_max = flat_map.max(axis=1)[:, None, None]
        normalized_maps = (raw_map - map_min) / np.maximum(map_max - map_min, 1e-8)

        score_map_cls = np.zeros_like(normalized_maps, dtype=np.uint8)
        for i, normalized_map in enumerate(normalized_maps):
            localization_threshold = float(np.quantile(normalized_map, 0.95))
            mask = (normalized_map >= localization_threshold).astype(np.uint8)

            # Close small gaps without expanding the defect excessively.
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Keep only connected regions containing one of the strongest
            # PatchCore responses. This removes scattered background pixels.
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            if num_labels > 1:
                peaks = []
                for label in range(1, num_labels):
                    component = normalized_map[labels == label]
                    peaks.append((float(component.max()), label))
                keep = {label for _, label in sorted(peaks, reverse=True)[:3]}
                mask = np.isin(labels, list(keep)).astype(np.uint8)

            score_map_cls[i] = mask

        # The image-level score is the anomaly decision. The score map is
        # localization data and must not determine the outer anomaly frame.
        image_cls = np.array(
            [int(float(image_score) >= float(threshold))], dtype=np.int64
        )
        score_map_cls[image_cls == 0] = 0
        test_images = np.array([image_np])
        boundary_np = visualization.framed_boundary_images(
            test_images, score_map_cls, image_cls, padding=VIZ_PADDING
        )[0]
        heatmap_np = visualization.heatmap_images(
            test_images, score_maps, alpha=VIZ_ALPHA
        )[0]
    else:
        boundary_np = np.empty((0, 0, 3), dtype=np.uint8)
        heatmap_np = np.empty((0, 0, 3), dtype=np.uint8)

    latency_ms = (time.perf_counter() - t0) * 1000

    is_anomaly = bool(float(image_score) >= float(threshold))

    return InferenceResult(
        anomaly_score=image_score,
        is_anomaly=is_anomaly,
        image_np=image_np,
        heatmap_np=heatmap_np,
        boundary_np=boundary_np,
        latency_ms=latency_ms,
    )
