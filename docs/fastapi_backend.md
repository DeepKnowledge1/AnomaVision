# FastAPI Backend Setup Guide

## Overview

The AnomaVision FastAPI backend provides a production-ready REST API for visual anomaly detection. It enables seamless integration with any client application through standard HTTP endpoints.

## Features

- 🚀 **Production Ready** — Built with FastAPI for high performance and reliability
- 📊 **Automatic Documentation** — Interactive Swagger UI and ReDoc interfaces
- 🎯 **Model Management** — Load and manage trained anomaly detection models
- 🖼️ **Image Processing** — Handle various image formats with automatic preprocessing
- 📈 **Batch Processing** — Process multiple images efficiently
- 🔒 **CORS Support** — Configure cross-origin requests for frontend integration
- ⚡ **Async Operations** — Non-blocking I/O for optimal performance

---

## Quick Start

### Installation

Ensure you have AnomaVision installed:

```bash
pip install AnomaVision
```

Or clone from source:

```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision
poetry install
```

### Launch the Server

Start the FastAPI backend server:

```bash
uvicorn apps.api.fastapi_app:app --host 0.0.0.0 --port 8000
```

For development with auto-reload:

```bash
uvicorn apps.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

---

## API Documentation

Once the server is running, access the interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These interfaces allow you to explore and test all endpoints directly from your browser.

---

## API Endpoints

### Health Check

**GET** `/health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Predict Anomaly

**POST** `/predict`

Detect anomalies in an uploaded image.

**Parameters:**
- `file` (required): Image file (JPEG, PNG)
- `include_visualizations` (optional, default: false): Include heatmap and boundary visualizations
- `threshold` (optional): Custom anomaly threshold

**Request Example (cURL):**
```bash
curl -X POST "http://localhost:8000/predict?include_visualizations=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

**Response:**
```json
{
  "anomaly_score": 0.85,
  "is_anomaly": true,
  "threshold": 0.5,
  "predictions": {
    "max_score": 0.85,
    "mean_score": 0.42,
    "min_score": 0.01
  },
  "visualizations": {
    "heatmap": "base64_encoded_image",
    "boundary": "base64_encoded_image"
  }
}
```

### Batch Predict

**POST** `/predict/batch`

Process multiple images in a single request.

**Parameters:**
- `files` (required): Multiple image files
- `include_visualizations` (optional, default: false)

**Response:**
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "anomaly_score": 0.85,
      "is_anomaly": true
    },
    {
      "filename": "image2.jpg",
      "anomaly_score": 0.23,
      "is_anomaly": false
    }
  ],
  "total_processed": 2
}
```

### Model Information

**GET** `/model/info`

Get information about the loaded model.

**Response:**
```json
{
  "model_type": "PaDiM",
  "backbone": "resnet18",
  "input_size": [224, 224],
  "threshold": 0.5,
  "loaded_at": "2025-01-04T10:30:00Z"
}
```

---

## Python Client Examples

### Basic Prediction

```python
import requests

# Single image prediction
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
        params={"include_visualizations": False}
    )

result = response.json()
print(f"Anomaly Score: {result['anomaly_score']}")
print(f"Is Anomaly: {result['is_anomaly']}")
```

### Prediction with Visualizations

```python
import requests
import base64
from PIL import Image
import io

with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
        params={"include_visualizations": True}
    )

result = response.json()

# Decode and display heatmap
heatmap_data = base64.b64decode(result['visualizations']['heatmap'])
heatmap_image = Image.open(io.BytesIO(heatmap_data))
heatmap_image.show()
```

### Batch Processing

```python
import requests

files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb")),
    ("files", open("image3.jpg", "rb"))
]

response = requests.post(
    "http://localhost:8000/predict/batch",
    files=files
)

results = response.json()
for result in results['results']:
    print(f"{result['filename']}: {result['anomaly_score']}")
```

---

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Model Configuration
MODEL_PATH=./models/padim_model.pth
CONFIG_PATH=./config.yml

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8501"]
CORS_ALLOW_CREDENTIALS=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/api.log
```

### Custom Configuration File

You can specify a custom configuration file when starting the server:

```bash
export CONFIG_PATH=/path/to/your/config.yml
uvicorn apps.api.fastapi_app:app --host 0.0.0.0 --port 8000
```

---

## Integration with Streamlit

To connect the Streamlit frontend with the FastAPI backend:

1. Start the FastAPI server:
```bash
uvicorn apps.api.fastapi_app:app --host 0.0.0.0 --port 8000
```

2. In a new terminal, launch Streamlit:
```bash
streamlit run apps/ui/streamlit_app.py -- --port 8000
```

3. Access Streamlit at `http://localhost:8501`

---

## Production Deployment

### Using Gunicorn

For production environments, use Gunicorn with Uvicorn workers:

```bash
gunicorn apps.api.fastapi_app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

### Systemd Service

Create a systemd service file for automatic startup:

```bash
# /etc/systemd/system/anomavision-api.service
[Unit]
Description=AnomaVision FastAPI Service
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/path/to/AnomaVision
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn apps.api.fastapi_app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable anomavision-api
sudo systemctl start anomavision-api
```

---

## Performance Optimization

### Async Processing

The API uses async operations for optimal performance:

```python
# Example: Async batch processing
@app.post("/predict/batch")
async def batch_predict(files: List[UploadFile]):
    tasks = [process_image(file) for file in files]
    results = await asyncio.gather(*tasks)
    return {"results": results}
```

### Caching

Enable response caching for frequently accessed predictions:

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="anomavision-cache")
```

### Load Balancing

Deploy multiple instances behind a load balancer for high availability:

```nginx
upstream anomavision {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
    server localhost:8004;
}

server {
    listen 80;
    location / {
        proxy_pass http://anomavision;
    }
}
```

---

## Error Handling

The API returns standard HTTP status codes:

- **200 OK**: Successful request
- **400 Bad Request**: Invalid input or parameters
- **404 Not Found**: Resource not found
- **422 Unprocessable Entity**: Validation error
- **500 Internal Server Error**: Server error

Example error response:

```json
{
  "detail": "Invalid image format. Supported formats: JPEG, PNG",
  "status_code": 400
}
```

---

## Security

### API Key Authentication

Add API key authentication:

```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/predict")
async def predict(api_key: str = Depends(api_key_header)):
    if api_key != VALID_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    # Process request
```

### Rate Limiting

Implement rate limiting to prevent abuse:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request):
    # Process request
```

---

## Monitoring

### Health Checks

Implement comprehensive health checks:

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "uptime": get_uptime(),
        "memory_usage": get_memory_usage()
    }
```

### Metrics

Expose Prometheus metrics:

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Access metrics at `http://localhost:8000/metrics`

---

## Troubleshooting

### Common Issues

**Issue: Model not loading**
```
Solution: Verify MODEL_PATH points to a valid .pth file
Check: ls -lh $MODEL_PATH
```

**Issue: CORS errors**
```
Solution: Add your frontend URL to CORS_ORIGINS
Update: CORS_ORIGINS=["http://your-frontend-url.com"]
```

**Issue: Slow inference**
```
Solution: Use ONNX or TensorRT models for faster inference
Export: python export.py --format onnx
```

**Issue: Memory errors**
```
Solution: Reduce batch size or image resolution
Configure: MAX_IMAGE_SIZE=1024 in .env
```

---

## Support

- 📖 [API Documentation](http://localhost:8000/docs)
- 💬 [GitHub Discussions](https://github.com/DeepKnowledge1/AnomaVision/discussions)
- 🐛 [Issue Tracker](https://github.com/DeepKnowledge1/AnomaVision/issues)
- 📧 [Email Support](mailto:deepp.knowledge@gmail.com)

---

## Next Steps

- ✅ [Setup Monitoring & Logging](../monitoring.md)
- ✅ [Performance Tuning](../optimization.md)
- ✅ [Security Best Practices](../security.md)

---

**Ready to integrate?** Check out our [Quick Start Guide](quickstart.md) or explore the [Streamlit Demo](streamlit_demo.md) for a visual interface!
