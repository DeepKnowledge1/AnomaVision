from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import anodet
from anodet import classification, visualization, to_batch

THRESH = 13
MODEL_PATH = "./distributions/padim_model.pt"

app = FastAPI()

# Allow CORS (optional for local frontend/backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
model = torch.load(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head><title>Anomaly Detection</title></head>
        <body>
            <h2>Upload image for anomaly detection</h2>
            <form id="upload-form">
                <input type="file" id="file-input" name="file" accept="image/*" required>
                <input type="submit" value="Submit">
            </form>
            <hr>
            <div id="result-container">
                <h3>Result will appear below:</h3>
                <img id="result-image" style="max-width: 90%; display: none;">
            </div>

            <script>
                const form = document.getElementById('upload-form');
                const resultImage = document.getElementById('result-image');
                const resultContainer = document.getElementById('result-container');

                form.addEventListener('submit', async (event) => {
                    event.preventDefault();
                    const fileInput = document.getElementById('file-input');
                    const formData = new FormData();
                    formData.append("file", fileInput.files[0]);

                    const response = await fetch('/process', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        const blob = await response.blob();
                        const imageUrl = URL.createObjectURL(blob);
                        resultImage.src = imageUrl;
                        resultImage.style.display = 'block';
                    } else {
                        resultContainer.innerHTML = "<p style='color:red;'>Error processing image.</p>";
                    }
                });
            </script>
        </body>
    </html>
    """

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert to RGB numpy array
    image = Image.open(BytesIO(contents)).convert("RGB")
    np_image = np.array(image)

    # Preprocess and run inference
    batch = to_batch([np_image], anodet.standard_image_transform, torch.device("cpu"))
    image_scores, score_maps = model.predict(batch)

    # Postprocess
    score_map_classifications = classification(score_maps, THRESH)
    image_classifications = classification(image_scores, THRESH)

    # Visualize results
    boundary_images = visualization.framed_boundary_images(
        [np_image], score_map_classifications, image_classifications, padding=40
    )
    heatmap_images = visualization.heatmap_images([np_image], score_maps, alpha=0.5)
    highlighted_images = visualization.highlighted_images(
        [np_image], score_map_classifications, color=(128, 0, 128)
    )

    # Compose result figure
    fig, axs = plt.subplots(1, 4, figsize=(12, 6))
    fig.suptitle("Anomaly Detection Result", y=0.75, fontsize=14)
    axs[0].imshow(np_image)
    axs[0].set_title("Original")
    axs[1].imshow(boundary_images[0])
    axs[1].set_title("Boundary")
    axs[2].imshow(heatmap_images[0])
    axs[2].set_title("Heatmap")
    axs[3].imshow(highlighted_images[0])
    axs[3].set_title("Highlighted")

    for ax in axs:
        ax.axis('off')

    # Convert figure to PNG
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
