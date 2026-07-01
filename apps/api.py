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
GET  /browse        — Browse predictions with navigation
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
from fastapi.responses import HTMLResponse
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
# Browse Interface HTML
# -----------------------------------------------------------------------------

def _generate_browse_html(prediction: dict, next_id: Optional[int], prev_id: Optional[int],
                          has_next: bool, has_prev: bool,
                          current_page: int, total_pages: int, total_count: int) -> str:
    """Generate HTML for browsing predictions."""

    pred_id = prediction["id"]

    # Determine status color
    status_color = "#ef4444" if prediction["prediction"] == "Defective" else "#22c55e"
    status_text = "DEFECTIVE" if prediction["prediction"] == "Defective" else "NORMAL"

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AnomaVision - Browse Results #{pred_id}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: #f5f6fa;
            color: #1e1b4b;
            padding: 2rem;
        }}

        .header {{
            background: linear-gradient(135deg, #ffffff 0%, #eef0ff 50%, #f5f6ff 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            border: 1px solid #e2e4f0;
        }}

        .header h1 {{
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.5rem;
        }}

        .nav-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            background: white;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}

        .nav-btn {{
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.2s;
            font-size: 0.9rem;
            display: inline-block;
        }}

        .nav-btn.primary {{
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
        }}

        .nav-btn.primary:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        }}

        .nav-btn.secondary {{
            background: #f1f5f9;
            color: #475569;
        }}

        .nav-btn.secondary:hover {{
            background: #e2e8f0;
        }}

        .nav-btn:disabled, .nav-btn.disabled {{
            opacity: 0.4;
            cursor: not-allowed;
            transform: none !important;
            pointer-events: none;
        }}

        .page-info {{
            font-weight: 600;
            color: #64748b;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .info-card {{
            background: white;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}

        .info-card label {{
            display: block;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #7c82a8;
            margin-bottom: 0.5rem;
        }}

        .info-card value {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #1e1b4b;
        }}

        .status-badge {{
            display: inline-block;
            padding: 0.4rem 1rem;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.9rem;
            background: {status_color}20;
            color: {status_color};
        }}

        .images-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .image-card {{
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}

        .image-card header {{
            padding: 1rem 1.5rem;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            font-weight: 700;
            font-size: 0.95rem;
        }}

        .image-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}

        .image-card footer {{
            padding: 0.75rem 1.5rem;
            background: #f8fafc;
            font-size: 0.85rem;
            color: #64748b;
        }}

        .actions {{
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-top: 2rem;
        }}

        @media (max-width: 768px) {{
            body {{
                padding: 1rem;
            }}

            .images-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 AnomaVision Results</h1>
        <div style="color: #7c82a8;">Browse prediction history</div>
    </div>

    <div class="nav-bar">
        {"<a href='/browse/" + str(prev_id) + "' class='nav-btn secondary'>" if has_prev else "<span class='nav-btn secondary disabled'"}>
            ← Newer
        {"</a>" if has_prev else "</span>"}

        <div class="page-info">
            Result #{pred_id} | Page {current_page} of {total_pages} | Total: {total_count}
        </div>

        {"<a href='/browse/" + str(next_id) + "' class='nav-btn primary'>" if has_next else "<span class='nav-btn primary disabled'"}>
            Older →
        {"</a>" if has_next else "</span>"}
    </div>

    <div class="info-grid">
        <div class="info-card">
            <label>Status</label>
            <value><span class="status-badge">{status_text}</span></value>
        </div>
        <div class="info-card">
            <label>Anomaly Score</label>
            <value>{prediction["anomaly_score"]:.4f}</value>
        </div>
        <div class="info-card">
            <label>Filename</label>
            <value>{prediction["image_filename"]}</value>
        </div>
        <div class="info-card">
            <label>Processing Time</label>
            <value>{prediction["processing_ms"]} ms</value>
        </div>
        <div class="info-card">
            <label>Date</label>
            <value>{prediction["created_at"].strftime("%Y-%m-%d %H:%M:%S")}</value>
        </div>
        <div class="info-card">
            <label>Model Version</label>
            <value>{prediction["model_version"]}</value>
        </div>
    </div>

    <div class="images-grid">
        <div class="image-card">
            <header>📷 Original Image</header>
            <img src="/{prediction["original_path"]}" alt="Original">
            <footer>Uploaded image</footer>
        </div>

        <div class="image-card">
            <header>🔥 Anomaly Heatmap</header>
            <img src="/{prediction["heatmap_path"]}" alt="Heatmap">
            <footer>Heat overlay visualization</footer>
        </div>

        <div class="image-card">
            <header>📐 Boundary Detection</header>
            <img src="/{prediction["boundary_path"]}" alt="Boundary">
            <footer>Defect boundary framing</footer>
        </div>
    </div>

    <div class="actions">
        <a href="/browse" class="nav-btn secondary">📋 All Results</a>
        <a href="/ui" class="nav-btn primary">🎨 New Analysis</a>
    </div>
</body>
</html>
    """
    return html

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "AnomaVision API",
        "ui": "/ui",
        "docs": "/docs",
        "browse": "/browse",
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
    threshold: float = engine.ANOMALY_THRESHOLD,
    resize_width: int = 224,
    resize_height: int = 224,
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

        # Resize image if requested by UI
        import numpy as np
        if resize_width and resize_height and (image_np.shape[1] != resize_width or image_np.shape[0] != resize_height):
            image_np = np.array(Image.fromarray(image_np).resize((resize_width, resize_height), Image.BILINEAR))

        # Use the threshold passed from the UI instead of the global default
        result = engine.run(image_np, threshold=threshold)

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
                _encode_async(result.boundary_np, _resize_size),
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

@app.get("/browse", response_class=HTMLResponse)
async def browse_predictions(id: Optional[int] = None):
    """
    Browse predictions with next/previous navigation.
    If no ID is provided, shows the most recent prediction.
    """
    if not DB_ENABLED:
        return HTMLResponse(
            content="<h1>Database is disabled. Enable with DB_ENABLED=1</h1>",
            status_code=503
        )

    total_count = await db.get_predictions_count()

    if total_count == 0:
        return HTMLResponse(
            content="""
            <div style="font-family: sans-serif; padding: 2rem; text-align: center;">
                <h1>No predictions found</h1>
                <p>Upload an image at <a href="/ui">/ui</a> to create predictions.</p>
            </div>
            """,
            status_code=404
        )

    # If no ID provided, get the latest (highest ID)
    if id is None:
        predictions = await db.get_predictions_paginated(limit=1, offset=0)
        if not predictions:
            return HTMLResponse(content="<h1>No predictions found</h1>", status_code=404)
        id = predictions[0]["id"]

    # Get the specific prediction
    prediction = await db.get_prediction_by_id(id)
    if not prediction:
        return HTMLResponse(
            content=f"<h1>Prediction #{id} not found</h1><p><a href='/browse'>Back to browse</a></p>",
            status_code=404
        )

    # Get all IDs ordered by created_at DESC (newest first)
    all_predictions = await db.get_predictions_paginated(limit=total_count, offset=0)
    all_ids = [p["id"] for p in all_predictions]

    # Find current position
    current_index = all_ids.index(id) if id in all_ids else 0
    current_page = current_index + 1
    total_pages = total_count

    # Navigation logic:
    # - "Next" goes to OLDER results (higher index in the list)
    # - "Previous" goes to NEWER results (lower index in the list)
    has_next = current_index < total_count - 1  # Can go to older
    has_prev = current_index > 0  # Can go to newer

    # Get next/previous IDs
    next_id = all_ids[current_index + 1] if has_next else None
    prev_id = all_ids[current_index - 1] if has_prev else None

    html = _generate_browse_html(
        prediction=prediction,
        next_id=next_id,
        prev_id=prev_id,
        has_next=has_next,
        has_prev=has_prev,
        current_page=current_page,
        total_pages=total_pages,
        total_count=total_count
    )

    return HTMLResponse(content=html)

@app.get("/browse/list")
async def browse_list(limit: int = 20, offset: int = 0):
    """
    API endpoint to get predictions list (for building custom UIs).
    """
    if not DB_ENABLED:
        return {"error": "Database disabled"}

    predictions = await db.get_predictions_paginated(limit=limit, offset=offset)
    total = await db.get_predictions_count()

    # Convert paths to URLs
    for pred in predictions:
        if pred.get("original_path"):
            pred["original_url"] = f"/{pred['original_path']}"
        if pred.get("heatmap_path"):
            pred["heatmap_url"] = f"/{pred['heatmap_path']}"
        if pred.get("boundary_path"):
            pred["boundary_url"] = f"/{pred['boundary_path']}"

    return {
        "predictions": predictions,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total
    }

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
