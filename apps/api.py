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
from typing import Optional

import inference_engine as engine
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# import ui  # Gradio blocks defined in ui.py


# -----------------------------------------------------------------------------
# Lifespan — model loads once, shared by FastAPI routes AND Gradio callbacks
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    status = engine.load_model()
    print(f"[startup] {status}")
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


class ConfigModel(BaseModel):
    threshold: float = engine.ANOMALY_THRESHOLD
    resize_width: int = 224
    resize_height: int = 224


# Runtime-mutable config (REST clients can adjust without restart)
_resize_size: tuple = (224, 224)


# -----------------------------------------------------------------------------
# Helpers — only used by the REST path, not by Gradio
# -----------------------------------------------------------------------------
def _numpy_to_base64(arr, resize_to: tuple) -> str:
    """numpy uint8 array → resized PNG → base64 string."""
    if arr is None:
        return ""
    try:
        import numpy as np

        if arr.dtype != np.uint8:
            arr = (
                (arr * 255).clip(0, 255).astype(np.uint8)
                if arr.max() <= 1.0
                else arr.clip(0, 255).astype(np.uint8)
            )
        img = Image.fromarray(arr)
        if resize_to:
            img = img.resize(resize_to, Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=1)  # fast encode
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


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
        "endpoints": ["/health", "/model-info", "/config", "/predict", "/disconnect"],
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
        result = engine.run(image_np, threshold=engine.ANOMALY_THRESHOLD)

        heatmap_b64 = ""
        boundary_b64 = ""

        if include_visualizations:
            heatmap_b64, boundary_b64 = await asyncio.gather(
                _encode_async(result.heatmap_np, _resize_size),
                _encode_async(result.boundary_np, _resize_size),
            )

        return PredictionResult(
            anomaly_score=result.anomaly_score,
            is_anomaly=result.is_anomaly,
            latency_ms=result.latency_ms,
            heatmap_image_base64=heatmap_b64,
            boundary_image_base64=boundary_b64,
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
