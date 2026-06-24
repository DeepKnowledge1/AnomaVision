


FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY anomavision/ ./anomavision/
COPY apps/ui/gradio_app.py ./apps/ui/gradio_app.py

RUN touch README.md && \
    uv pip install --system --no-cache \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system --no-cache \
        "numpy==1.26.4" "scikit-learn>=1.2,<1.4" "scikit-image==0.22.0" \
        "easydict==1.13" "tqdm>=4.67.1" "opencv-python-headless>=4.6.0" \
        "onnx>=1.15.0" "onnxruntime==1.24.2" "gradio>=6.3.0" \
        "matplotlib==3.8.3" "pydantic>=2.12.5" "scipy" && \
    uv pip install --system --no-cache --no-deps -e .

ENV ORT_LOGGING_LEVEL="3"

EXPOSE 7860
CMD ["python", "apps/ui/gradio_app.py"]


# Docker build command:

#docker build -t gradio-anomavision .


#### How to run the Docker container:

# docker run -p 7860:7860 `
#     -v "${PWD}/distributions:/app/distributions" `
#   anomavision-gradio



#### How to prevent creating leftover containers in the future — add --rm:


# docker run --rm -p 7860:7860 -v "${PWD}/distributions:/app/distributions" anomavision-gradio

# Create a Docker image with a specific tag:
# docker build -t deepknowledge/gradio-anomavision:3.4.0 .


# or

# docker run -it --rm `
#   -v "$PWD\distributions:/app/distributions" `
#   --entrypoint bash `
#   deepknowledge/gradio-anomavision:3.4.0


# create a docker tag for the image:
# docker tag gradio-anomavision:latest deepknowledge/gradio-anomavision:3.4.0


# How to push the Docker image to Docker Hub:
    # docker push deepknowledge/gradio-anomavision:3.4.0
