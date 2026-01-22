import gradio as gr
import requests
import base64
import io
import json
import time
from PIL import Image
from typing import Optional, Tuple

# Configuration
FASTAPI_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{FASTAPI_URL}/predict"
HEALTH_ENDPOINT = f"{FASTAPI_URL}/health"
CONFIG_ENDPOINT = f"{FASTAPI_URL}/config"

def safe_get(url: str, timeout: int = 5):
    try:
        return requests.get(url, timeout=timeout)
    except:
        return None

def safe_post(url: str, *, json_payload=None, files=None, params=None, timeout: int = 30):
    try:
        return requests.post(url, json=json_payload, files=files, params=params, timeout=timeout)
    except:
        return None

def decode_base64_image(base64_str: str) -> Optional[Image.Image]:
    if not base64_str:
        return None
    try:
        img_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_bytes))
    except:
        return None

def check_backend():
    resp = safe_get(HEALTH_ENDPOINT)
    if resp and resp.status_code == 200:
        data = resp.json()
        return data.get("status") == "healthy"
    return False

def run_inference(image, threshold, resize_w, resize_h, viz):
    if image is None:
        return "Please upload an image", None, None

    # Update config
    safe_post(CONFIG_ENDPOINT, json_payload={
        "threshold": float(threshold),
        "resize_width": int(resize_w),
        "resize_height": int(resize_h)
    })

    # Prepare image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Predict
    files = {"file": ("image.png", img_byte_arr, "image/png")}
    params = {"include_visualizations": viz}

    start = time.time()
    resp = safe_post(PREDICT_ENDPOINT, files=files, params=params, timeout=60)
    elapsed = time.time() - start

    if resp and resp.status_code == 200:
        result = resp.json()
        score = float(result.get("anomaly_score", 0.0))
        is_anom = bool(result.get("is_anomaly", False))

        status = f"{'🚨 ANOMALY' if is_anom else '✅ NORMAL'}\nScore: {score:.4f} | Threshold: {threshold:.4f} | Time: {elapsed:.2f}s"

        heatmap = decode_base64_image(result.get("heatmap_image_base64", ""))
        boundary = decode_base64_image(result.get("boundary_image_base64", ""))

        return status, heatmap, boundary

    return "Error: Backend offline", None, None

# Create interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:

    gr.Markdown("# 🔍 MVTec Anomaly Detection Demo")
    gr.Markdown("Upload an image to detect anomalies using state-of-the-art deep learning models.")

    with gr.Tabs() as tabs:

        # TAB 1: Upload Image
        with gr.Tab("📤 Upload Image", id=0):
            gr.Markdown("## Upload and Analyze Images")

            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(type="pil", label="Input Image")

                    with gr.Row():
                        threshold = gr.Slider(0.1, 50.0, 13.0, step=0.1, label="Threshold")

                    with gr.Row():
                        resize_w = gr.Number(value=224, label="Width", minimum=32, maximum=2048)
                        resize_h = gr.Number(value=224, label="Height", minimum=32, maximum=2048)

                    viz_check = gr.Checkbox(value=True, label="Generate Visualizations")
                    analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")

                with gr.Column(scale=1):
                    result_text = gr.Textbox(label="Results", lines=4)

                    with gr.Row():
                        heatmap_out = gr.Image(label="Heatmap", type="pil")
                        boundary_out = gr.Image(label="Boundary", type="pil")

            analyze_btn.click(
                fn=run_inference,
                inputs=[input_img, threshold, resize_w, resize_h, viz_check],
                outputs=[result_text, heatmap_out, boundary_out]
            )

        # TAB 2: Draw Defects (placeholder)
        with gr.Tab("🎨 Draw Defects", id=1):
            gr.Markdown("## Draw Artificial Defects")
            gr.Markdown("""
            1. Upload a GOOD image (normal, without defects)
            2. Use the brush to draw artificial defects (scratches, stains, cracks, etc.)
            3. Click Analyze to see if the heatmap detects your drawn defects
            """)

            # Image editor would go here (requires canvas/drawing functionality)
            gr.Markdown("*Drawing editor coming soon - use Upload Image tab for now*")

        # TAB 3: Compare Models
        with gr.Tab("⚖️ Compare Models", id=2):
            gr.Markdown("## Compare Different Models")
            gr.Markdown("*Model comparison feature coming soon*")

        # TAB 4: Learn About Models
        with gr.Tab("📚 Learn About Models", id=3):
            gr.Markdown("## Understanding Anomaly Detection Models")

            gr.Markdown("""
            ### 🧩 PatchCore

            **Approach:** Memory Bank + K-Nearest Neighbors

            PatchCore is a memory-based anomaly detection method that works by:

            1. **Feature Extraction**: Uses a pre-trained CNN (e.g., WideResNet-50) to extract patch-level features from normal training images
            2. **Memory Bank**: Stores a representative subset of normal patch features using coreset subsampling (greedy selection to maximize coverage)
            3. **Anomaly Scoring**: For a test image, computes the distance of each patch to its nearest neighbor in the memory bank
            4. **Localization**: High distances indicate anomalous regions

            **Strengths:**
            - Very high accuracy on texture anomalies
            - No training required (only feature extraction)
            - Works well with limited normal samples

            **Weaknesses:**
            - Memory consumption grows with dataset size
            - Inference speed depends on memory bank size

            ---

            ### 📊 PaDiM (Patch Distribution Modeling)

            **Approach:** Multivariate Gaussian Distribution per Patch

            PaDiM models the distribution of normal features at each spatial location:

            1. **Feature Extraction**: Extracts features from multiple CNN layers (multi-scale approach)
            2. **Distribution Modeling**: For each patch position, fits a multivariate Gaussian distribution (mean and covariance) using normal training samples
            3. **Anomaly Scoring**: Uses Mahalanobis distance to measure how far a test patch deviates from its expected distribution
            4. **Dimensionality Reduction**: Applies random feature selection to reduce computation

            **Strengths:**
            - Memory-efficient (stores only statistics, not samples)
            - Good generalization across different defect types

            **Weaknesses:**
            - Requires calculation of covariance matrices
            - Performance depends on quality of feature extraction
            """)

        # TAB 5: Model Metrics
        with gr.Tab("📈 Model Metrics", id=4):
            gr.Markdown("## Training Metrics & Performance")
            gr.Markdown("This section shows the performance metrics of all trained models across different MVTec categories.")

            gr.Markdown("""
            ### 🏆 Best Performers

            | Metric | Best Model | Category | Value |
            |--------|------------|----------|-------|
            | Image AUROC | Fastflow | bottle | 1.0000 |
            | Pixel AUROC | Patchcore | carpet | 0.9907 |
            | F1 Score | Fastflow | bottle | 0.9920 |

            ### 📊 All Metrics

            *Performance metrics table would be dynamically loaded from backend*
            """)

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860, share=False)
