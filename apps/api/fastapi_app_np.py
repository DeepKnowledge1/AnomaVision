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
import onnxruntime as ort
from onnxruntime import SessionOptions, GraphOptimizationLevel
import multiprocessing
from anomavision.static.AnomaVision import classification, to_batch, standard_image_transform, visualization
from anomavision.static.AnomaVision.utils import *

# import anomavision
# from anomavision.general import determine_device
# from anomavision.inference.model.wrapper import ModelWrapper
# from anomavision.inference.modelType import ModelType

matplotlib.use("Agg")  # non-interactive backend

# -----------------------------
# Globals / Config
# -----------------------------
# model: Optional[ModelWrapper] = None
# model_type: Optional[ModelType] = None

ANOMALY_THRESHOLD = 13.0
RESIZE_SIZE = (224, 224)

# You can override these via environment variables
MODEL_DATA_PATH = os.getenv(
    "ANOMAVISION_MODEL_DATA_PATH", "distributions/padim/bottle/anomav_exp"
)
MODEL_FILE = os.getenv("ANOMAVISION_MODEL_FILE", "model.onnx")
DEVICE = os.getenv("ANOMAVISION_DEVICE", "auto")  # "auto"|"cpu"|"cuda"

# Visualization parameters (match detect.py defaults)
VIZ_PADDING = int(os.getenv("ANOMAVISION_VIZ_PADDING", "40"))
VIZ_ALPHA = float(os.getenv("ANOMAVISION_VIZ_ALPHA", "0.5"))
VIZ_COLOR = tuple(map(int, os.getenv("ANOMAVISION_VIZ_COLOR", "128,0,128").split(",")))



async def load_model():
    global sess, padim_model
    try:
        # Get the list of available execution providers (e.g., CPUExecutionProvider, CUDAExecutionProvider)
        available_providers = ort.get_available_providers()
        # Check if CUDA (GPU) is available by looking for the CUDAExecutionProvider
        use_gpu = "CUDAExecutionProvider" in available_providers

        # Create session options for ONNX Runtime
        sess_options = SessionOptions()


        # Enable memory pattern optimization to improve performance on repeated inference calls
        sess_options.enable_mem_pattern = True

        # Enable graph optimization to apply advanced model graph transformations
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # Warmup the ONNX session to allocate memory and compile execution graphs
        output_names = [output.name for output in sess.get_outputs()]
        dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
        input_name = sess.get_inputs()[0].name
        _ = sess.run(output_names, {input_name: dummy_input})
        print("ONNX model warmed up successfully.")
        # These two options are useful only when using CPU
        if not use_gpu:
            # Enable memory arena for CPU allocator to optimize memory usage
            sess_options.enable_cpu_mem_arena = True

            # Set the number of threads ONNX Runtime can use for CPU operations
            sess_options.intra_op_num_threads = multiprocessing.cpu_count()

        # Set the preferred execution provider based on hardware availability
        providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

        model_path = os.path.realpath(os.path.join(MODEL_DATA_PATH, MODEL_FILE))

        # Try to load ONNX model first
        if os.path.exists(model_path):
            sess = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
            print("ONNX model loaded successfully.")
        else:
            raise FileNotFoundError("No model found. Please ensure 'padim_model.onnx' ")

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

async def cleanup():
    global sess
    sess = None
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
    model_loaded = sess is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_type": "onnx" if sess else "pytorch" if padim_model else "none",
        "threshold": ANOMALY_THRESHOLD,
        "resize_size": RESIZE_SIZE
    }


@app.post("/config")
async def update_config(config: ConfigModel):
    global ANOMALY_THRESHOLD, RESIZE_SIZE
    ANOMALY_THRESHOLD = config.threshold
    RESIZE_SIZE = (config.resize_width, config.resize_height)
    return {
        "message": f"Threshold updated to {ANOMALY_THRESHOLD}, Resize size set to {RESIZE_SIZE}"
    }


def preprocess_image_from_upload(file_contents: bytes) -> np.ndarray:
    # detect.py ultimately works with RGB images; keep same behavior
    image_pil = Image.open(io.BytesIO(file_contents)).convert("RGB")
    return np.array(image_pil)


def numpy_to_base64(image_array: np.ndarray, resize_to: tuple = None) -> str:
    if image_array is None:
        return ""
    try:
        img = image_array.astype(np.uint8)
        if resize_to is not None:
            # cv2.resize is significantly faster than PIL resize
            img = cv2.resize(img, resize_to, interpolation=cv2.INTER_LINEAR)

        # Use WebP or JPEG. WebP is ~3-5x faster to encode than PNG in OpenCV
        # quality=80 is a good tradeoff between speed and visual fidelity
        _, buffer = cv2.imencode('.webp', img, [cv2.IMWRITE_WEBP_QUALITY80])
        return base64.b64encode(buffer).decode("utf-8")
    except Exception:
        return ""

def numpy_to_base64_________(image_array: np.ndarray, resize_to: tuple[int, int] = None) -> str:
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


def create_visualizations(
    image_np: np.ndarray, score_maps: torch.Tensor, image_scores: torch.Tensor
):
    """
    Mirror detect.py's visualization path.
    """
    score_map_classifications = classification(
        score_maps, ANOMALY_THRESHOLD
    )
    image_classifications = classification(image_scores, ANOMALY_THRESHOLD)

    test_images = np.array([image_np])

    boundary_images = visualization.framed_boundary_images(
        test_images,
        score_map_classifications,
        image_classifications,
        padding=VIZ_PADDING,
    )

    heatmap_images = visualization.heatmap_images(
        test_images,
        score_maps,
        alpha=VIZ_ALPHA,
    )

    highlighted_images = visualization.highlighted_images(
        [image_np],
        score_map_classifications,
        color=VIZ_COLOR,
    )

    return boundary_images[0], heatmap_images[0], highlighted_images[0]


@app.post("/predict", response_model=PredictionResult)
async def predict_anomaly(
    file: UploadFile = File(...), include_visualizations: bool = True
):
    if sess is None:
        raise HTTPException(status_code=500, detail="No model loaded.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image_np = preprocess_image_from_upload(contents)

        if sess is not None:
            # ONNX inference - convert single image to batch
            batch = to_batch([image_np])
            input_numpy = batch

            input_name = sess.get_inputs()[0].name
            output_names = [output.name for output in sess.get_outputs()]

            if len(output_names) < 2:
                raise HTTPException(status_code=500, detail="Model must have at least 2 outputs")

            outputs = sess.run(output_names, {input_name: input_numpy})

            image_scores = np.array([outputs[0]])
            score_maps = np.array(outputs[1])





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
            boundary_image, heatmap_image, highlighted_image = create_visualizations(
                image_np, score_maps, image_scores
            )
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


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if sess is not None:
        inputs = [(inp.name, inp.shape, inp.type) for inp in sess.get_inputs()]
        outputs = [(out.name, out.shape, out.type) for out in sess.get_outputs()]
        return {
            "model_type": "onnx",
            "inputs": inputs,
            "outputs": outputs,
            "threshold": ANOMALY_THRESHOLD
        }

    else:
        raise HTTPException(status_code=500, detail="No model loaded")


# @app.get("/model-info")
# async def get_model_info():
#     if model is None or model_type is None:
#         raise HTTPException(status_code=500, detail="No model loaded")

#     return {
#         "model_type": model_type.value,
#         "device": getattr(model, "device", "unknown"),
#         "model_path": os.path.realpath(os.path.join(MODEL_DATA_PATH, MODEL_FILE)),
#         "threshold": ANOMALY_THRESHOLD,
#     }


if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=False)

    # To run from command line:

    # uvicorn apps.api.fastapi_app:app --host 0.0.0.0 --port 8000
    # or
    # python apps/api/fastapi_app.py --host 0.0.0.0 --port 8000
