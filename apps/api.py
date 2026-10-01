"""
api.py
------
FastAPI application that:

  1. Loads the model once via inference_engine.load_model()
  2. Mounts the Gradio UI at /ui  (zero-serialization path — numpy arrays in-process)
  3. Exposes a REST API at /predict for external consumers (curl, other services, batch)

Run:
    python api.py
    # or
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /              — index
    GET  /health        — liveness check
    GET  /model-info    — session metadata
    POST /config        — update threshold / resize at runtime
    POST /predict       — REST inference (base64 PNG visualizations in JSON)
    ANY  /ui            — Gradio frontend (mounted sub-app)
"""

import asyncio
import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import inference_engine as engine
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from anomavision.drift_runtime import input_drift_features
from anomavision.production_monitor import ProductionDriftMonitor

# import ui  # Gradio blocks defined in ui.py


# -----------------------------------------------------------------------------
# Lifespan — model loads once, shared by FastAPI routes AND Gradio callbacks
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start even when no Studio project/model is selected yet.
    try:
        status = engine.load_model()
        print(f"[startup] {status}")
    except Exception as exc:
        print(f"[startup] No model loaded yet: {exc}")
    yield
    print("[shutdown] cleaning up")


app = FastAPI(
    title="AnomaVision API",
    version="1.0.0",
    description="Visual anomaly detection — REST API + Gradio UI",
    lifespan=lifespan,
)

try:
    import gradio as gr
    import ui

    app = gr.mount_gradio_app(app, ui.demo, path="/ui")
except Exception as e:
    print("[warning] Gradio UI disabled:", e)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Gradio as a sub-application at /ui
# The Gradio blocks object is defined in ui.py and imported here.
# Because ui.py imports inference_engine, and inference_engine._sess is set
# during lifespan above, Gradio callbacks share the same loaded session.
# app = gr.mount_gradio_app(app, ui.demo, path="/ui")


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class PredictionResult(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    latency_ms: float
    heatmap_image_base64: Optional[str] = ""
    boundary_image_base64: Optional[str] = ""
    highlighted_image_base64: Optional[str] = ""
    drift_report: Optional[dict] = None


class ConfigModel(BaseModel):
    threshold: float = engine.ANOMALY_THRESHOLD
    resize_width: int = 224
    resize_height: int = 224


# Runtime-mutable config (REST clients can adjust without restart)
_resize_size: tuple = (224, 224)

_drift_monitor: Optional[ProductionDriftMonitor] = None
_drift_project_id: Optional[str] = None
_drift_config: dict = {}


def _project_root(project_id: str) -> Path:
    return Path(os.path.expanduser(
        os.getenv("ANOMAVISION_STUDIO_ROOT", "~/.anomavision/projects")
    )) / project_id


def _resolve_drift_path(project_id: str, config: dict, key: str, default: Path) -> Path:
    value = config.get(key)
    if not value:
        return default
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else _project_root(project_id) / path


def _build_input_drift_reference(project_id: str, config: dict) -> Path:
    import numpy as np
    output = _resolve_drift_path(project_id, config, "drift_reference",
        _project_root(project_id) / "drift" / "reference_embeddings.npy")
    dataset_path = Path(str(config.get("dataset_path", ""))).expanduser()
    class_name = str(config.get("class_name", "") or "")
    healthy = dataset_path / class_name / "train" / "good"
    if not healthy.is_dir():
        healthy = dataset_path / "train" / "good"
    if not healthy.is_dir():
        raise FileNotFoundError(f"Healthy training images not found: {healthy}")
    image_files = [p for p in sorted(healthy.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".webp"}][:500]
    if len(image_files) < 2:
        raise ValueError("At least 2 healthy training images are required for drift monitoring.")
    from anomavision.static.AnomaVision import to_batch
    chunks = []
    for image_path in image_files:
        try:
            image = np.array(Image.open(image_path).convert("RGB"))
            chunks.append(input_drift_features(to_batch([image])))
        except Exception as exc:
            print(f"[monitoring] Skipping reference image {image_path}: {exc}")
    if len(chunks) < 2:
        raise ValueError("Could not generate enough valid drift reference images.")
    reference = np.concatenate(chunks, axis=0).astype(np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, reference)
    output.with_suffix(output.suffix + ".json").write_text(
        __import__("json").dumps({
            "samples": int(reference.shape[0]),
            "feature_dimensions": int(reference.shape[1]),
            "representation": "input_statistics",
            "source": str(healthy),
            "project_id": project_id,
        }, indent=2) + "\n", encoding="utf-8")
    return output


def _load_project_drift_monitor(project_id: str, config: dict) -> None:
    global _drift_monitor, _drift_project_id, _drift_config
    import numpy as np
    reference_path = _resolve_drift_path(project_id, config, "drift_reference",
        _project_root(project_id) / "drift" / "reference_embeddings.npy")
    if not reference_path.is_file():
        reference_path = _build_input_drift_reference(project_id, config)
    reference = np.load(reference_path)
    if reference.ndim != 2 or reference.shape[0] < 2 or reference.shape[1] != 15:
        # The Studio runtime intentionally uses the backend-independent
        # input-statistics representation (3 channels × 5 statistics).
        reference_path.unlink(missing_ok=True)
        reference_path = _build_input_drift_reference(project_id, config)
        reference = np.load(reference_path)
    if reference.ndim != 2 or reference.shape[0] < 2 or reference.shape[1] != 15:
        raise ValueError(f"Invalid drift reference representation: {reference_path}")
    window = int(config.get("drift_window", 50) or 50)
    min_samples = max(2, min(int(config.get("drift_min_samples", 5) or 5), window))
    interval = max(1, int(config.get("drift_evaluation_interval", 25) or 25))
    _drift_monitor = ProductionDriftMonitor(reference, window_size=window,
        min_samples=min_samples, threshold=float(config.get("drift_threshold", 0.20) or 0.20),
        evaluation_interval=interval)
    _drift_project_id = project_id
    _drift_config = dict(config)
    status_path = _resolve_drift_path(project_id, config, "drift_output",
        _project_root(project_id) / "monitoring" / "drift_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    _drift_monitor.save_status(status_path)


def _update_project_drift(project_id: str, image_np) -> Optional[dict]:
    if not project_id:
        return None
    if _drift_monitor is None or _drift_project_id != project_id:
        try:
            _apply_project_config(project_id)
        except Exception as exc:
            # Monitoring must never prevent inference when its reference/config is unavailable.
            print(f"[monitoring] Could not initialize project monitoring: {exc}")
            return None
    if _drift_monitor is None or _drift_project_id != project_id:
        return None
    from anomavision.static.AnomaVision import to_batch
    try:
        _drift_monitor.update(input_drift_features(to_batch([image_np])))
        status_path = _resolve_drift_path(project_id, _drift_config, "drift_output",
            _project_root(project_id) / "monitoring" / "drift_status.json")
        _drift_monitor.save_status(status_path)
        return _drift_monitor.status().to_dict()
    except Exception as exc:
        # Monitoring is an observer and must never break anomaly inference.
        print(f"[monitoring] Drift update skipped: {exc}")
        return None



# -----------------------------------------------------------------------------
# Helpers — only used by the REST path, not by Gradio
# -----------------------------------------------------------------------------
def _numpy_to_base64(arr, resize_to: tuple) -> str:
    """Encode a visualization numpy array as a PNG data URL payload."""
    import numpy as np

    if arr is None:
        return ""

    value = np.asarray(arr)
    if value.size == 0:
        return ""

    # Visualization helpers may return (1,H,W,3), (H,W,3), or a 2-D map.
    if value.ndim == 4:
        value = value[0]
    if value.ndim == 2:
        value = np.stack([value, value, value], axis=-1)
    if value.ndim != 3 or value.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported visualization shape: {value.shape}")

    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=-1)
    elif value.shape[-1] == 4:
        value = value[:, :, :3]

    if value.dtype != np.uint8:
        value = np.nan_to_num(value, nan=0.0, posinf=255.0, neginf=0.0)
        if float(value.max()) <= 1.0:
            value = value * 255.0
        value = np.clip(value, 0, 255).astype(np.uint8)

    img = Image.fromarray(value, mode="RGB")
    if resize_to:
        img = img.resize((int(resize_to[0]), int(resize_to[1])), Image.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def _encode_async(arr, resize_to: tuple) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _numpy_to_base64, arr, resize_to)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "AnomaVision API",
        "ui": "/ui",
        "docs": "/docs",
        "endpoints": ["/health", "/model-info", "/reload-model", "/config", "/predict", "/disconnect"],
    }


_DEBUG_ENDPOINTS_ENABLED = os.getenv("ANOMAVISION_DEBUG_ENDPOINTS", "1") == "1"

if _DEBUG_ENDPOINTS_ENABLED:

    @app.get("/disconnect")
    async def disconnect():
        """
        Debug-only: terminates the process to test liveness/restart behavior.

        Responds 200 immediately, then exits ~0.5s later via os._exit(0) —
        a clean exit code, no traceback, no exception. Kubernetes' default
        restartPolicy (Always) restarts the container in the SAME pod, so
        `kubectl get pods` shows RESTARTS += 1 a few seconds later.

        Disable in real deployments by setting ANOMAVISION_DEBUG_ENDPOINTS=0.
        """

        async def _exit_soon():
            await asyncio.sleep(0.5)  # let the HTTP response flush first
            os._exit(0)

        asyncio.create_task(_exit_soon())
        return {"status": "disconnecting", "exit_in_seconds": 0.5}


_FORCE_UNHEALTHY_FILE = "/tmp/force_unhealthy"


@app.get("/health")
async def health():
    # Debug-only kill switch for testing liveness/readiness probe behavior.
    # Trigger:  kubectl exec <pod> -c api -- touch /tmp/force_unhealthy
    # Reset:    kubectl exec <pod> -c api -- rm /tmp/force_unhealthy
    if os.path.exists(_FORCE_UNHEALTHY_FILE):
        raise HTTPException(status_code=503, detail="forced unhealthy (debug)")

    return {
        "status": "healthy" if engine.is_loaded() else "unhealthy",
        "model_loaded": engine.is_loaded(),
        "threshold": engine.ANOMALY_THRESHOLD,
    }


def _apply_project_config(project_id: Optional[str]) -> None:
    """Apply runtime inference settings from the selected Studio training config."""
    if not project_id:
        return

    root = Path(
        os.path.expanduser(
            os.getenv("ANOMAVISION_STUDIO_ROOT", "~/.anomavision/projects")
        )
    )
    metadata_path = root / project_id / "models" / "latest_training.json"
    if not metadata_path.is_file():
        return

    import json
    from anomavision.config import load_config

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_value = str(metadata.get("config", "")).strip()
    config_path = Path(config_value).expanduser()
    if not config_path.is_file() and config_value:
        config_path = root / project_id / config_value
    if not config_path.is_file():
        return

    cfg = load_config(str(config_path)) or {}
    _load_project_drift_monitor(project_id, cfg)
    algorithm = str(cfg.get("algorithm", "padim")).lower()
    threshold_keys = {
        "padim": "thresh_padim",
        "patchcore": "thresh_patchcore",
        "efficientad": "thresh_efficientad",
    }
    key = threshold_keys.get(algorithm)
    if key and cfg.get(key) is not None:
        engine.ANOMALY_THRESHOLD = float(cfg[key])


@app.post("/reload-model")
async def reload_model(project_id: Optional[str] = None):
    """Load the selected Studio project's latest trained model."""
    try:
        status = engine.load_model(project_id=project_id)
        if not engine.is_loaded():
            raise HTTPException(status_code=404, detail=status)
        _apply_project_config(project_id)
        return {
            "status": "loaded",
            "message": status,
            "project_id": project_id,
            "model": engine.session_info(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load model: {exc}") from exc


@app.get("/monitoring/status")
async def monitoring_status(project_id: Optional[str] = None):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required")
    if _drift_project_id != project_id or _drift_monitor is None:
        raise HTTPException(status_code=404, detail="Monitoring is not initialized for this project")
    return _drift_monitor.status().to_dict()


@app.get("/model-info")
async def model_info():
    if not engine.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return engine.session_info()


@app.post("/config")
async def update_config(config: ConfigModel):
    global _resize_size
    engine.ANOMALY_THRESHOLD = config.threshold
    _resize_size = (config.resize_width, config.resize_height)
    return {"threshold": engine.ANOMALY_THRESHOLD, "resize_size": _resize_size}


@app.post("/predict", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(...),
    include_visualizations: bool = True,
):
    """
    REST endpoint for external consumers.

    Accepts a multipart image upload, returns JSON with anomaly score
    and optional base64-encoded PNG visualizations.

    For the Gradio UI use /ui — it bypasses this entire serialization path.
    """
    if not engine.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image_np = _load_image_np(contents)
        result = engine.run(image_np, threshold=engine.ANOMALY_THRESHOLD, include_visualizations=include_visualizations)
        drift_report = _update_project_drift(_drift_project_id or "", image_np)

        heatmap_b64 = ""
        boundary_b64 = ""
        highlighted_b64 = ""

        if include_visualizations:
            heatmap_b64, boundary_b64, highlighted_b64 = await asyncio.gather(
                _encode_async(result.heatmap_np, _resize_size),
                _encode_async(result.boundary_np, _resize_size),
                _encode_async(result.highlighted_np, _resize_size),
            )

        return PredictionResult(
            anomaly_score=result.anomaly_score,
            is_anomaly=result.is_anomaly,
            latency_ms=result.latency_ms,
            heatmap_image_base64=heatmap_b64,
            boundary_image_base64=boundary_b64,
            highlighted_image_base64=highlighted_b64,
            drift_report=drift_report,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _load_image_np(contents: bytes):
    import numpy as np

    return np.array(Image.open(io.BytesIO(contents)).convert("RGB"))


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
