import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Optional

import matplotlib
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

import anomavision
from anomavision.general import determine_device
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.inference.modelType import ModelType

matplotlib.use("Agg")  # non-interactive backend

# -----------------------------
# Globals / Config
# -----------------------------
model: Optional[ModelWrapper] = None
model_type: Optional[ModelType] = None

ANOMALY_THRESHOLD = 13.0
RESIZE_SIZE = (224, 224)

# You can override these via environment variables
MODEL_DATA_PATH = os.getenv("ANOMAVISION_MODEL_DATA_PATH", "distributions/anomav_exp")
MODEL_FILE = os.getenv("ANOMAVISION_MODEL_FILE", "padim_model.onnx")
DEVICE = os.getenv("ANOMAVISION_DEVICE", "auto")  # "auto"|"cpu"|"cuda"

# Visualization parameters (match detect.py defaults)
VIZ_PADDING = int(os.getenv("ANOMAVISION_VIZ_PADDING", "40"))
VIZ_ALPHA = float(os.getenv("ANOMAVISION_VIZ_ALPHA", "0.5"))
VIZ_COLOR = tuple(map(int, os.getenv("ANOMAVISION_VIZ_COLOR", "128,0,128").split(",")))


async def load_model():
    """
    Load model exactly like detect.py:
      model = ModelWrapper(model_path, device_str)
      model_type = ModelType.from_extension(model_path)
    """
    global model, model_type

    device_str = determine_device(DEVICE)  # "cpu" or "cuda"
    model_path = os.path.realpath(os.path.join(MODEL_DATA_PATH, MODEL_FILE))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # ModelType is inferred from extension (.pt/.onnx/.engine/...)
    model_type = ModelType.from_extension(model_path)
    model = ModelWrapper(model_path, device_str)

    # Optional warmup (keeps it lightweight; ModelWrapper may implement warmup)
    try:
        # Create a dummy batch similar to detect.py input shape.
        # We reuse anomavision.to_batch in predict() for actual inputs.
        dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device_str)
        model.warmup(batch=dummy, runs=1)
    except Exception:
        # Warmup isn't critical; ignore if not supported by backend
        pass

    print(f"Model loaded: {model_path} ({model_type.value}) on {device_str}")


async def cleanup():
    global model, model_type
    if model is not None:
        try:
            model.close()
        except Exception:
            pass
    model = None
    model_type = None
    print("Model cleanup completed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_model()
    yield
    await cleanup()


app = FastAPI(title="Anomaly Detection API", version="1.0.0", lifespan=lifespan)


class PredictionResult(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    anomaly_map_base64: Optional[str] = ""
    boundary_image_base64: Optional[str] = ""
    heatmap_image_base64: Optional[str] = ""
    highlighted_image_base64: Optional[str] = ""


class ConfigModel(BaseModel):
    threshold: float = ANOMALY_THRESHOLD
    resize_width: int = 224
    resize_height: int = 224


@app.get("/")
async def root():
    return {
        "message": "Anomaly Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict-batch",
            "model_info": "/model-info",
            "config": "/config",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


@app.get("/health")
async def health_check():
    loaded = model is not None
    return {
        "status": "healthy" if loaded else "unhealthy",
        "model_type": model_type.value if model_type else "none",
        "threshold": ANOMALY_THRESHOLD,
        "resize_size": RESIZE_SIZE,
    }


@app.post("/config")
async def update_config(config: ConfigModel):
    global ANOMALY_THRESHOLD, RESIZE_SIZE
    ANOMALY_THRESHOLD = config.threshold
    RESIZE_SIZE = (config.resize_width, config.resize_height)
    return {"message": f"Threshold updated to {ANOMALY_THRESHOLD}, Resize size set to {RESIZE_SIZE}"}


def preprocess_image_from_upload(file_contents: bytes) -> np.ndarray:
    # detect.py ultimately works with RGB images; keep same behavior
    image_pil = Image.open(io.BytesIO(file_contents)).convert("RGB")
    return np.array(image_pil)


def numpy_to_base64(image_array: np.ndarray, resize_to: tuple[int, int] = None) -> str:
    if image_array is None:
        return ""
    try:
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        img = Image.fromarray(image_array)
        if resize_to is not None:
            img = img.resize(resize_to, Image.BILINEAR)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


def create_visualizations(image_np: np.ndarray, score_maps: torch.Tensor, image_scores: torch.Tensor):
    """
    Mirror detect.py's visualization path.
    """
    score_map_classifications = anomavision.classification(score_maps, ANOMALY_THRESHOLD)
    image_classifications = anomavision.classification(image_scores, ANOMALY_THRESHOLD)

    test_images = np.array([image_np])

    boundary_images = anomavision.visualization.framed_boundary_images(
        test_images,
        score_map_classifications,
        image_classifications,
        padding=VIZ_PADDING,
    )

    heatmap_images = anomavision.visualization.heatmap_images(
        test_images,
        score_maps,
        alpha=VIZ_ALPHA,
    )

    highlighted_images = anomavision.visualization.highlighted_images(
        [image_np],
        score_map_classifications,
        color=VIZ_COLOR,
    )

    return boundary_images[0], heatmap_images[0], highlighted_images[0]


@app.post("/predict", response_model=PredictionResult)
async def predict_anomaly(file: UploadFile = File(...), include_visualizations: bool = True):
    if model is None:
        raise HTTPException(status_code=500, detail="No model loaded.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image_np = preprocess_image_from_upload(contents)

        # --- Like detect.py: create batch using anomavision transforms
        device_str = model.device  # ModelWrapper stores device string in detect.py usage
        batch = anomavision.to_batch([image_np], anomavision.standard_image_transform, torch.device(device_str))

        # detect.py uses half precision on cuda; do same
        if device_str == "cuda":
            batch = batch.half()

        # --- Run prediction via ModelWrapper (works across PT/ONNX/OpenVINO/TensorRT per your wrapper)
        with torch.no_grad():
            image_scores, score_maps = model.predict(batch)

        score_map_classifications = anomavision.classification(
            score_maps, ANOMALY_THRESHOLD
        )
        image_classifications = anomavision.classification(
            image_scores, ANOMALY_THRESHOLD
        )

        anomaly_score = float(image_scores[0])
        is_anomaly = anomaly_score >= ANOMALY_THRESHOLD

        # # Normalized anomaly map
        # score_map_np = score_maps[0].numpy()
        # if score_map_np.max() - score_map_np.min() > 0:
        #     score_map_normalized = (score_map_np - score_map_np.min()) / (score_map_np.max() - score_map_np.min())
        # else:
        #     score_map_normalized = np.zeros_like(score_map_np)

        # anomaly_map_base64 = numpy_to_base64(score_map_normalized, RESIZE_SIZE)

        boundary_image_base64 = ""
        heatmap_image_base64 = ""
        highlighted_image_base64 = ""

        if include_visualizations:
            boundary_image, heatmap_image, highlighted_image = create_visualizations(image_np, score_maps, image_scores)
            boundary_image_base64 = numpy_to_base64(boundary_image, RESIZE_SIZE)
            heatmap_image_base64 = numpy_to_base64(heatmap_image, RESIZE_SIZE)
            highlighted_image_base64 = numpy_to_base64(highlighted_image, RESIZE_SIZE)

        return PredictionResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            # If you want to return it in the UI, uncomment:
            # anomaly_map_base64=anomaly_map_base64,
            boundary_image_base64=boundary_image_base64,
            heatmap_image_base64=heatmap_image_base64,
            # highlighted_image_base64=highlighted_image_base64,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")

    results = []
    for i, f in enumerate(files):
        try:
            r = await predict_anomaly(f, include_visualizations=False)
            results.append({"file_index": i, "filename": f.filename, "result": r})
        except Exception as e:
            results.append({"file_index": i, "filename": f.filename, "error": str(e)})

    return {"batch_results": results}


@app.get("/model-info")
async def get_model_info():
    if model is None or model_type is None:
        raise HTTPException(status_code=500, detail="No model loaded")

    return {
        "model_type": model_type.value,
        "device": getattr(model, "device", "unknown"),
        "model_path": os.path.realpath(os.path.join(MODEL_DATA_PATH, MODEL_FILE)),
        "threshold": ANOMALY_THRESHOLD,
    }


if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=False)

    # To run from command line:

    # uvicorn apps.api.fastapi_app:app --host 0.0.0.0 --port 8000
    # or
    # python apps/api/fastapi_app.py --host 0.0.0.0 --port 8000
