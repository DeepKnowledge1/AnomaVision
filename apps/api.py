"""
api.py
FastAPI application that:
Loads the model once via inference_engine.load_model()
Mounts the Gradio UI at /ui  (zero-serialization path — numpy arrays in-process)
Exposes a REST API at /predict for external consumers (curl, other services, batch)
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
GET  /uploads/      — Browse saved images in browser
GET  /metrics       — Prometheus metrics
"""
import asyncio
import base64
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import db
from db import DB_ENABLED
import inference_engine as engine
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# Optional Prometheus support
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_ENABLED = True
except ImportError:
    PROMETHEUS_ENABLED = False

# -----------------------------------------------------------------------------
# Lifespan — model loads once, shared by FastAPI routes AND Gradio callbacks
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    status = engine.load_model()
    print(f"[startup] {status}")

    if DB_ENABLED:
        await db.init_pool()

    yield
    print("[shutdown] cleaning up")

app = FastAPI(
    title="AnomaVision API",
    version="1.0.0",
    description="Visual anomaly detection — REST API + Gradio UI",
    lifespan=lifespan,
)

# Add Prometheus metrics if the library is installed
if PROMETHEUS_ENABLED:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

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

# -----------------------------------------------------------------------------
# Image Storage Setup (Solves "I can't see the images")
# -----------------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount the uploads directory so you can view images in your browser at http://localhost:8000/uploads/
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR, html=True), name="uploads")

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class PredictionResult(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    latency_ms: float
    heatmap_image_base64: Optional[str] = ""
    boundary_image_base64: Optional[str] = ""

    # Added URLs to easily access saved images in the browser
    original_image_url: Optional[str] = ""
    heatmap_image_url: Optional[str] = ""
    boundary_image_url: Optional[str] = ""

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

def _save_numpy_to_disk(arr, path: Path):
    """Saves a numpy array to disk as a PNG file safely."""
    if arr is None:
        return
    try:
        import numpy as np
        if arr.dtype != np.uint8:
            arr = (
                (arr * 255).clip(0, 255).astype(np.uint8)
                if arr.max() <= 1.0
                else arr.clip(0, 255).astype(np.uint8)
            )
        Image.fromarray(arr).save(path)
    except Exception as e:
        print(f"[warning] Failed to save image to {path}: {e}")

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
        "metrics": "/metrics" if PROMETHEUS_ENABLED else "disabled",
        "images": "/uploads/",
        "endpoints": ["/health", "/model-info", "/config", "/predict", "/disconnect"],
    }

_DEBUG_ENDPOINTS_ENABLED = os.getenv("ANOMAVISION_DEBUG_ENDPOINTS", "1") == "1"

if _DEBUG_ENDPOINTS_ENABLED:
    @app.get("/disconnect")
    async def disconnect():
        """
        Debug-only: terminates the process to test liveness/restart behavior.
        """
        async def _exit_soon():
            await asyncio.sleep(0.5)  # let the HTTP response flush first
            os._exit(0)

        asyncio.create_task(_exit_soon())
        return {"status": "disconnecting", "exit_in_seconds": 0.5}

_FORCE_UNHEALTHY_FILE = "/tmp/force_unhealthy"

@app.get("/health")
async def health():
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
    """
    if not engine.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image_np = _load_image_np(contents)
        result = engine.run(image_np, threshold=engine.ANOMALY_THRESHOLD)

        # --- Save images to disk so you can view them in the browser ---
        timestamp = int(time.time() * 1000)
        original_path = UPLOAD_DIR / f"{timestamp}_original.png"
        heatmap_path = UPLOAD_DIR / f"{timestamp}_heatmap.png"
        boundary_path = UPLOAD_DIR / f"{timestamp}_boundary.png"

        # Save original image
        Image.fromarray(image_np).save(original_path)

        # Save heatmap and boundary (helper handles numpy type conversion)
        _save_numpy_to_disk(result.heatmap_np, heatmap_path)
        _save_numpy_to_disk(result.boundary_np, boundary_path)

        # Save the paths to PostgreSQL (FIXED: passes all 3 paths)
        if DB_ENABLED:
            await db.insert_prediction(
                image_filename=file.filename,
                original_path=str(original_path),
                heatmap_path=str(heatmap_path),
                boundary_path=str(boundary_path),
                prediction="Defective" if result.is_anomaly else "Normal",
                anomaly_score=result.anomaly_score,
                model_version="padim-v1.0.2",
                processing_ms=round(result.latency_ms),
            )

        heatmap_b64 = ""
        boundary_b64 = ""

        if include_visualizations:
            heatmap_b64, boundary_b64 = await asyncio.gather(
                _encode_async(result.heatmap_np, _resize_size),
                _encode_async(result.boundary_np, _resize_size), # FIXED TYPO
            )

        return PredictionResult(
            anomaly_score=result.anomaly_score,
            is_anomaly=result.is_anomaly,
            latency_ms=result.latency_ms,
            heatmap_image_base64=heatmap_b64,
            boundary_image_base64=boundary_b64,
            # Return URLs so the client can easily open them
            original_image_url=f"/uploads/{original_path.name}",
            heatmap_image_url=f"/uploads/{heatmap_path.name}",
            boundary_image_url=f"/uploads/{boundary_path.name}",
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


### To run it with Postgresql, you can use the following command:
# docker run -d --name anomavision-postgres \
#   -e POSTGRES_USER=anomavision \
#   -e POSTGRES_PASSWORD=anomavision \
#   -e POSTGRES_DB=anomavision \
#   -p 5432:5432 \
#   postgres:16-alpine


# Then

# $env:DATABASE_URL="postgresql://anomavision:anomavision@127.0.0.1:5432/anomavision"
# python apps\api.py


# To see the results run :
# docker exec -it anomavision-postgres psql -U anomavision -d anomavision -c "SELECT * FROM predictions;"


##### Important

# To view and manage your PostgreSQL database directly in your web browser, the easiest way is to spin up a lightweight web-based database manager using Docker.
# Run the following command

# docker run -it --rm --name adminer --link anomavision-postgres:db -p 8080:8080 adminer


# open http://localhost:8080
# Fill in the login form with these exact details:
# System: PostgreSQL
# Server: db (This is the alias we created with the --link flag)
# Username: anomavision
# Password: anomavision
# Database: anomavision



#  docker exec -it anomavision-postgres psql -U anomavision -d anomavision
# this command will give you access to the PostgreSQL command
# line interface where you can run SQL queries and manage
# your database.
