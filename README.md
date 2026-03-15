<div align="center">
<img src="docs/images/banner.png" width="100%" alt="AnomaVision banner"/>

# AnomaVision 🔍

**High-performance visual anomaly detection. Fast, lightweight, production-ready.**

AnomaVision detects defects without ever seeing defective examples during training.
<br>

[![PyPI](https://img.shields.io/pypi/v/anomavision?label=PyPI&color=blue)](https://pypi.org/project/anomavision/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/anomavision?color=blue)](https://pypi.org/project/anomavision/)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ONNX](https://img.shields.io/badge/ONNX-Export%20Ready-orange)](https://onnx.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-Supported-76b900)](https://developer.nvidia.com/tensorrt)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-Supported-0071C5)](https://docs.openvino.ai/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Demo-Live-yellow)](https://huggingface.co/spaces/DeepKnowledge1/mvtec-anomaly-detection)

<br>

[**Live Demo**](#-live-demo) · [**Docs**](docs/quickstart.md) · [**Quickstart**](#-quickstart) · [**Models**](#-models--performance) · [**Tasks**](#-tasks--modes) · [**Integrations**](#-integrations) · [**Issues**](https://github.com/DeepKnowledge1/AnomaVision/issues) · [**Discussions**](https://github.com/DeepKnowledge1/AnomaVision/discussions)

</div>

---

## 🤗 Live Demo

> **Try AnomaVision instantly — no installation required.**

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-xl-dark.svg)](https://huggingface.co/spaces/DeepKnowledge1/mvtec-anomaly-detection)

The live demo runs a **PaDiM model trained on MVTec bottle images** and shows:

- 🌡️ **Anomaly Heatmap** — spatial score map highlighting defect regions
- 🖼️ **Overlay** — original image with anomaly contours drawn
- 🎭 **Predicted Mask** — binary segmentation of detected defects
- ⚡ **Real-time inference** — results in milliseconds on CPU

Upload your own bottle image or pick from the provided samples to see anomaly detection in action.

---

## What is AnomaVision?

AnomaVision delivers **visual anomaly detection** optimized for production deployment. Based on PaDiM, it learns the distribution of normal images in a **single forward pass** — no labels, no segmentation masks, no lengthy training loops.

The result: a 15 MB model that runs at **43 FPS on CPU** and **547 FPS on GPU**, with higher AUROC than the existing best-in-class baseline.

---

## 🚀 Quickstart

### Install

> ⚠️ **`torch` is hardware-specific.** A plain `pip install anomavision` skips PyTorch entirely. Always install with an `[extra]` to get the right binaries for your hardware.

**Don't have `uv`?** Install it first — it's faster than pip and handles PyTorch's hardware routing correctly:

```bash
pip install uv
```

---

#### Option A — From Source (development)

```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision

# Create and activate a virtual environment
uv venv --python 3.11 .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

# Install with your hardware extra
uv sync --extra cpu              # CPU
uv sync --extra cu121            # CUDA 12.1
```

Or install from `requirements.txt` directly:

```bash
uv pip install -r requirements.txt
```

---

#### Option B — From PyPI (production / quick start)

```bash
# CPU  ·  Mac, CI runners, edge devices
uv pip install "anomavision[cpu]"

# NVIDIA GPU  ·  pick your CUDA version
uv pip install "anomavision[cu118]"   # CUDA 11.8
uv pip install "anomavision[cu121]"   # CUDA 12.1
uv pip install "anomavision[cu124]"   # CUDA 12.4
```

---

#### Option C — Already installed without extras?

If you're seeing `ModuleNotFoundError: No module named 'torch'`, add PyTorch into your current environment:

```bash
# CPU
uv pip install torch torchvision torchaudio

# GPU (CUDA 12.1)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

#### Verify

```bash
python -c "import anomavision, torch; print('✅ Ready —', torch.__version__)"
```

---

### CLI

AnomaVision ships a unified `anomavision` command — no need to run individual scripts directly.

```bash
# Train
anomavision train --config config.yml

# Detect (images or folder)
anomavision detect --config config.yml --img_path ./test_images --thresh 13.0

# Evaluate on MVTec
anomavision eval --config config.yml --enable_visualization

# Export to ONNX / TorchScript / OpenVINO / all
anomavision export --model padim_model.pt --format all --precision fp16
```

Every command has full `--help`:

```bash
anomavision --help
anomavision train --help
anomavision export --help
```

---

<details>
<summary><strong>🐍 Python API</strong></summary>
<br>

Use the Python API when you want to embed AnomaVision into a larger pipeline,
run it inside a notebook, or integrate it with your own data loading logic.

```python
import torch
import anomavision
from torch.utils.data import DataLoader

# --- 1. Dataset ---
# AnodetDataset reads normal images from a folder and applies preprocessing.
# You only need train/good/ — no labels, no anomalous images required.
dataset = anomavision.AnodetDataset(
    image_directory_path="./dataset/bottle/train/good",
    resize=(224, 224),     # resize before crop
    crop_size=(224, 224),  # center crop to this size
    normalize=True,        # ImageNet mean/std normalization
)
loader = DataLoader(dataset, batch_size=16)

# --- 2. Train ---
# fit() does a single forward pass — no gradient updates, no epochs.
# Fits a multivariate Gaussian at each spatial position of the feature map.
# Typical training time: under 10 seconds on CPU for ~200 images.
model = anomavision.Padim(
    backbone="resnet18",  # or "wide_resnet50" for higher accuracy
    device="cpu",         # or "cuda"
    feat_dim=100,         # number of random feature dimensions to keep
)
model.fit(loader)

# --- 3. Save ---
torch.save(model, "padim_model.pt")                  # full model (for export or further use)
model.save_statistics("padim_model.pth", half=True)  # stats-only (smaller, faster to load)

# --- 4. Infer ---
# scores: (batch_size,)      — scalar anomaly score per image. Higher = more anomalous.
# maps:   (batch_size, H, W) — spatial heatmap showing *where* the anomaly is.
#                              Pass directly to matplotlib.imshow() to visualize.
batch, *_ = next(iter(loader))
scores, maps = model.predict(batch)
```

See the [FAQ](#-faq) for how to pick a classification threshold from `scores`.

</details>


<details>
<summary><strong>🌐 REST API</strong></summary>
<br>

Use the REST API when you want to integrate AnomaVision into an existing service,
call it from any language, or expose it on a network without installing Python on the client.

First, start the FastAPI server (keep this terminal open):
```bash
uvicorn apps.api.fastapi_app:app --host 0.0.0.0 --port 8000
```

Then send images from any client:
```python
import requests

with open("image.jpg", "rb") as f:
    r = requests.post("http://localhost:8000/predict", files={"file": f})

print(r.json()["anomaly_score"])   # e.g. 14.3
print(r.json()["is_anomaly"])      # True / False
```

Full docs at **http://localhost:8000/docs** once the server is running.

</details>

---

<details>
<summary><strong>📊 Models & Performance</strong></summary>
<br>

### MVTec AD — Average over 15 Classes

| Model | Image AUROC ↑ | Pixel AUROC ↑ | CPU FPS ↑ | GPU FPS ↑ | Size ↓ |
|---|---|---|---|---|---|
| **AnomaVision** (resnet18) | **0.850** | **0.956** | **43.4** | **547** | **15 MB** |
| Anomalib PaDiM (baseline) | 0.810 | 0.935 | 13.0 | 356 | 40 MB |
| Δ | **+4.9%** | **+2.2%** | **+233%** | **+54%** | **−25%** |

> CPU: Intel Core i9 (single process). GPU: NVIDIA A100. Batch size 1.
> Reproduce: `anomavision eval --config config.yml`

### VisA — Average over 12 Classes

| Model | Image AUROC ↑ | Pixel AUROC ↑ | CPU FPS ↑ |
|---|---|---|---|
| **AnomaVision** | **0.812** | **0.962** | **44.8** |
| Anomalib PaDiM | 0.783 | 0.954 | 13.5 |

<details>
<summary>📋 Per-class MVTec breakdown</summary>

| Class | AV Image AUROC | AL Image AUROC | AV Pixel AUROC | AL Pixel AUROC | AV FPS |
|---|---|---|---|---|---|
| bottle | 0.997 | 0.996 | 0.984 | 0.987 | 42.2 |
| cable | 0.772 | 0.742 | 0.936 | 0.935 | 36.1 |
| capsule | 0.839 | 0.846 | 0.929 | 0.977 | 40.2 |
| carpet | 0.908 | 0.594 | 0.971 | 0.987 | 44.0 |
| grid | 0.881 | 0.832 | 0.964 | 0.965 | 41.3 |
| hazelnut | 0.984 | 0.949 | 0.978 | 0.974 | 29.0 |
| leather | 0.985 | 0.879 | 0.985 | 0.982 | 48.7 |
| metal_nut | 0.940 | 0.878 | 0.963 | 0.963 | 41.4 |
| pill | 0.793 | 0.773 | 0.957 | 0.964 | 45.4 |
| screw | 0.941 | 0.787 | 0.970 | 0.982 | 42.4 |
| tile | 0.851 | 0.876 | 0.969 | 0.971 | 46.0 |
| toothbrush | 0.978 | 0.883 | 0.993 | 0.989 | 44.8 |
| transistor | 0.800 | 0.853 | 0.968 | 0.962 | 42.2 |
| wood | 0.986 | 0.915 | 0.973 | 0.975 | 45.3 |
| zipper | 0.914 | 0.979 | 0.972 | 0.971 | 41.0 |

</details>
</details>

---

## 🎯 Tasks & Modes

| Task | Train | Detect | Eval | Export | Stream | REST |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Anomaly Detection (image score) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Anomaly Localization (pixel map) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Normal / Anomalous Classification | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Export Formats

| Format | Flag | CPU | GPU | Edge | Quantization |
|---|---|:---:|:---:|:---:|:---:|
| PyTorch `.pt` | `pt` | ✅ | ✅ | — | — |
| ONNX `.onnx` | `onnx` | ✅ | ✅ | ✅ | INT8 dynamic / static |
| TorchScript `.torchscript` | `torchscript` | ✅ | ✅ | ✅ | — |
| OpenVINO (dir) | `openvino` | ✅ | — | ✅ | FP16 |
| TensorRT `.engine` | `engine` | — | ✅ | — | FP16 |
| C++ ONNX Runtime | — | ✅ | ✅ | ✅ | — |

```bash
anomavision export \
  --model_data_path ./distributions/anomav_exp \
  --model padim_model.pt \
  --format onnx \
  --precision fp16 \
  --quantize-dynamic
```

---

<details>
<summary><strong>📺 Streaming Sources</strong></summary>
<br>

Run inference on **live sources** without changing your model or code:

| Source | `stream_source.type` | Use case |
|---|---|---|
| Webcam | `webcam` | Lab / demo |
| Video file | `video` | Offline replay |
| MQTT | `mqtt` | Industrial IoT cameras |
| TCP socket | `tcp` | High-throughput line scanners |

```yaml
# stream_config.yml
stream_mode: true
stream_source:
  type: webcam
  camera_id: 0
model: padim_model.onnx
thresh: 13.0
enable_visualization: true
```

```bash
anomavision detect --config stream_config.yml
```

</details>

<details>
<summary><strong>⚙️ Configuration</strong></summary>
<br>

All scripts accept `--config config.yml` and CLI overrides. **CLI always wins.**

```yaml
# Minimal working config.yml
dataset_path:    ./dataset
class_name:      bottle

resize:          [256, 192]
crop_size:       [224, 224]
normalize:       true
norm_mean:       [0.485, 0.456, 0.406]
norm_std:        [0.229, 0.224, 0.225]

backbone:        resnet18
batch_size:      16
feat_dim:        100
layer_indices:   [0, 1, 2]
output_model:    padim_model.pt
run_name:        exp1
model_data_path: ./distributions/anomav_exp

model:           padim_model.onnx
device:          auto        # auto | cpu | cuda
thresh:          13.0

log_level:       INFO
```

Full key reference: [`docs/config.md`](docs/config.md)

</details>

<details>
<summary><strong>🔌 Integrations</strong></summary>
<br>

| Integration | Description |
|---|---|
| **FastAPI** | REST API — `/predict`, `/predict/batch`, Swagger UI at `/docs` |
| **Streamlit** | Browser demo — heatmap overlay, threshold slider, batch upload |
| **Gradio** | [Live HuggingFace Space](https://huggingface.co/spaces/DeepKnowledge1/mvtec-anomaly-detection) — try it instantly |
| **C++ Runtime** | ONNX + OpenCV, no Python required — see [`docs/cpp/`](docs/cpp/README.md) |
| **OpenVINO** | Intel CPU/VPU edge optimization |
| **TensorRT** | NVIDIA GPU maximum throughput |
| **INT8 Quantization** | Dynamic + static INT8 via ONNX Runtime |

```bash
# Terminal 1 — backend
uvicorn apps.api.fastapi_app:app --host 0.0.0.0 --port 8000

# Terminal 2 — UI
streamlit run apps/ui/streamlit_app.py -- --port 8000
```

Open **http://localhost:8501**

</details>

<details>
<summary><strong>📂 Dataset Format</strong></summary>
<br>

AnomaVision uses [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) layout. Custom datasets work with the same structure:

```
dataset/
└── <class_name>/
    ├── train/
    │   └── good/          ← normal images only (no anomalies needed)
    └── test/
        ├── good/          ← normal test images
        └── <defect_name>/ ← anomalous test images (any subfolder name)
```

</details>

---

## 🏗️ Architecture

<img src="docs/images/archti.png" width="100%" alt="AnomaVision architecture"/>

**Key design decisions:**

**PaDiM needs no gradient training.** Features are extracted once with a frozen ResNet. The model fits a multivariate Gaussian at each spatial location — training is a matrix decomposition, not backprop. That's why it finishes in ~8 seconds.

**`ModelWrapper` makes the backend transparent.** The same `predict(batch) → (scores, maps)` call works whether you loaded `.pt`, `.onnx`, `.engine`, or an OpenVINO directory. Every downstream caller — CLI, FastAPI, Streamlit, eval loop — uses the same interface.

**Adaptive Gaussian post-processing** is applied to score maps after inference. The kernel is sized relative to the image resolution, which is a key factor behind the Pixel AUROC gain over baseline.

---

## 🛠️ Development

```bash
# Clone and create environment
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision

uv venv --python 3.11 .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

# Install with dev dependencies
uv sync --extra cpu              # or --extra cu121 for GPU

# Install the package in editable mode
uv pip install -e .

# Verify CLI is working
anomavision --help

# Test
pytest tests/

# Format + lint
black . && isort . && flake8 .
```

**Commit convention:**

```
feat(export):  add TensorRT INT8 calibration
fix(detect):   handle empty directories
docs(api):     improve ModelWrapper examples
```

Types: `feat` · `fix` · `docs` · `refactor` · `test` · `chore`

PRs must pass `pytest` + `flake8` and include doc updates if behavior changes. See [`docs/contributing.md`](docs/contributing.md).

---

## 🚢 Deploy

<details>
<summary><strong>Docker</strong></summary>

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git git-lfs libsm6 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir uv && \
    uv pip install --system "anomavision[cpu]"

USER user
ENV PATH="/home/user/.local/bin:/usr/local/bin:$PATH"

WORKDIR /home/user/app
COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]
```

```bash
docker build -t anomavision .
docker run -p 7860:7860 -v $(pwd)/distributions:/home/user/app/distributions anomavision
```

</details>

<details>
<summary><strong>Production (Gunicorn + Uvicorn)</strong></summary>

```bash
gunicorn apps.api.fastapi_app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

</details>

> **Production tip:** Serve ONNX or TensorRT models — `.pt` inference is 2–3× slower than ONNX Runtime at batch size 1.

---

## ❓ FAQ

<details>
<summary><strong>Training is slow on CPU</strong></summary>

Lower `resize` (e.g. `[128, 128]`), reduce `batch_size`, or use `--device cuda`. PaDiM training is a single forward pass — it should finish in under 30 s for most datasets even on CPU.

</details>

<details>
<summary><strong>All anomaly scores are low / nothing detected</strong></summary>

Run `anomavision eval --config config.yml` first to see the score distribution histogram. Set `--thresh` just above the peak of the normal score distribution. Typical values: 10–20 for ResNet18 with default preprocessing.

</details>

<details>
<summary><strong>RuntimeError: Input size mismatch during inference</strong></summary>

Your `resize` / `crop_size` must match what was used at training time. Load the config saved alongside the model: `--config ./distributions/anomav_exp/exp1/config.yml`.

</details>

<details>
<summary><strong>CUDA version mismatch</strong></summary>

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with your actual CUDA version (`cu118`, `cu124`, etc.).

</details>

<details>
<summary><strong>Unsupported operator during ONNX export</strong></summary>

Try `--opset 16`. If it still fails, use `--format torchscript` — TorchScript has no ONNX operator constraints.

</details>

<details>
<summary><strong>Can I use my own dataset without MVTec structure?</strong></summary>

Yes. Put your normal training images in `<any_root>/train/good/`. For evaluation, add test images under `<root>/test/<defect_name>/`. No anomalous images are needed at training time.

</details>

More: [`docs/troubleshooting.md`](docs/troubleshooting.md)

---

## 🗺️ Roadmap

- [ ] Pre-trained model zoo for all 15 MVTec classes
- [ ] Few-shot adaptation (5–10 anomalous examples)
- [ ] Native TensorRT export in `export.py`
- [ ] Pixel-level mask in REST `/predict` response
- [ ] ONNX Runtime Web (browser inference via WASM)
- [ ] Helm chart for Kubernetes deployment

[Request a feature →](https://github.com/DeepKnowledge1/AnomaVision/discussions)

---

## 📚 Documentation

| | |
|---|---|
| [Quick Start](docs/quickstart.md) | Train → detect → eval → export in 5 minutes |
| [CLI Reference](docs/cli.md) | All arguments for all `anomavision` subcommands |
| [Python API](docs/api.md) | Library usage and class reference |
| [Config Guide](docs/config.md) | Every YAML key explained |
| [Benchmarks](docs/benchmark.md) | Full per-class results vs Anomalib |
| [FastAPI Backend](docs/fastapi_backend.md) | REST API setup and endpoints |
| [C++ Inference](docs/cpp/README.md) | Deploy without Python |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and fixes |
| [Contributing](docs/contributing.md) | Development workflow |

---

## 💬 Community

- 🐛 [Issues](https://github.com/DeepKnowledge1/AnomaVision/issues) — bug reports
- 💡 [Discussions](https://github.com/DeepKnowledge1/AnomaVision/discussions) — questions, ideas, show & tell
- 🤗 [Live Demo](https://huggingface.co/spaces/DeepKnowledge1/mvtec-anomaly-detection) — try it in your browser
- 📧 [deepp.knowledge@gmail.com](mailto:deepp.knowledge@gmail.com) — direct contact

---

## Citation

```bibtex
@software{anomavision2025,
  title   = {AnomaVision: Edge-Ready Visual Anomaly Detection},
  author  = {DeepKnowledge Contributors},
  year    = {2025},
  url     = {https://github.com/DeepKnowledge1/AnomaVision},
}
```

---

## License

Released under the [MIT License](LICENSE).
Built on [Anodet](https://github.com/OpenAOI/anodet) — thanks to the original authors.
