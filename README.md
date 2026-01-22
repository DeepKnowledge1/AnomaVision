<img src="docs/images/banner.png" width="100%" alt="AnomaVision banner"/>

# AnomaVision: Edge-Ready Visual Anomaly Detection
AnomaVision is a production-ready **visual anomaly detection framework** built on **PaDiM** with **real-time inference (CPU/GPU)** and multiple deployment backends like **PyTorch, ONNX, and TensorRT**.

<!-- Row 1: Language + Frameworks -->
[![Python 3.9–3.12](https://img.shields.io/badge/python-3.9–3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org)
[![ONNX Ready](https://img.shields.io/badge/ONNX-Export%20Ready-orange.svg)](https://onnx.ai/)
[![OpenVINO Ready](https://img.shields.io/badge/OpenVINO-Ready-blue.svg)](https://docs.openvino.ai/)
[![TorchScript](https://img.shields.io/badge/Export-TorchScript-red.svg)](https://pytorch.org/docs/stable/jit.html)
[![TensorRT](https://img.shields.io/badge/Deploy-TensorRT-green.svg)](https://developer.nvidia.com/tensorrt)
[![Quantization](https://img.shields.io/badge/Optimized-Quantization-purple.svg)](https://onnxruntime.ai/docs/performance/quantization.html)

<!-- Row 2: Distribution + License -->
[![PyPI Version](https://img.shields.io/pypi/v/anomavision?label=PyPI%20version)](https://pypi.org/project/anomavision/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/anomavision?label=PyPI%20downloads&color=blue)](https://pypi.org/project/anomavision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)


**Lightweight, fast, and production-ready anomaly detection powered by PaDiM.**
*Deploy anywhere: edge devices, servers, or the cloud.*

---

## Overview

AnomaVision delivers state-of-the-art visual anomaly detection optimized for real-world deployment. Built for speed and efficiency, it outperforms existing solutions while maintaining a small footprint perfect for edge devices.

### Key Features

- 🎯 **Superior Performance** — Higher AUROC across MVTec AD and Visa datasets
- ⚡ **3× Faster Inference** — Optimized for both CPU and GPU deployment
- 📦 **Smaller Models** — 30MB models with lower memory footprint
- 🌐 **Multi-Backend Export** — PyTorch, ONNX, TorchScript, OpenVINO, TensorRT, INT8 Quantization
- 🖥️ **Production Ready** — Python API, CLI tools, C++ runtime, and REST API
- 🎨 **Rich Visualizations** — Heatmaps, bounding boxes, and ROC curves
- 🎮 **Interactive Demo** — Streamlit web interface for instant testing

---

## Why Choose AnomaVision?

<h3 style="color:red;">Performance Advantages Over Anomalib</h3>

**CPU Inference:**

| Metric | AnomaVision | Anomalib | Improvement |
|:--|--:|--:|--:|
| Training Time (s) | **8.38** | 13.07 | **-35.9%** |
| Inference FPS | **43.41** | 13.03 | **+233%** |
| ms / image | **23.0** | 76.7 | **-70%** |

**GPU Inference (CUDA):**

| Metric | AnomaVision | Anomalib | Improvement |
|:--|--:|--:|--:|
| Training Time (s) | **8.38** | 13.07 | **-35.9%** |
| Inference FPS | **547.46** | 355.72 | **+53.9%** |
| ms / image | **1.83** | 2.81 | **-35.0%** |

**Accuracy:**

- **MVTec AD:** Image AUROC 0.85 vs 0.81 | Pixel AUROC 0.96 vs 0.94
- **Visa:** Image AUROC 0.81 vs 0.78 | Pixel AUROC 0.96 vs 0.95

💡 [Download Full Performance Analysis (PDF)](docs/AnomaVision_vs_Anomalib.pdf) | [Detailed Benchmarks](docs/benchmark.md)

---

## Quick Start

### Installation

**Using Poetry (Recommended):**
```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision
poetry install
poetry shell
```

**Using pip:**
```bash
pip install AnomaVision
```

📖 [Detailed Installation Guide](docs/installation.md)

### Basic Usage

**Training:**
```bash
python train.py --config config.yml
# Outputs: padim_model.pt, padim_model.pth, config.yml
```

**Detection:**
```bash
python detect.py --config config.yml
```

**Evaluation:**
```bash
python eval.py --config config.yml
```

**Export:**
```bash
python export.py --config export_config.yml
```

📖 [Complete Quick Start Guide](docs/quickstart.md)

---

## Interactive Demo

<div align="left">
  <img src="docs/images/streamlit.png" alt="AnomaVision Streamlit Demo" width="50%">
  <p><em>Real-time anomaly detection with explainable heatmaps</em></p>
</div>

Experience AnomaVision through our intuitive web interface:

- ⚡ Real-time anomaly detection
- 🎯 Explainable AI with visual heatmaps
- 📊 Interactive threshold controls
- 🔍 Batch processing support
- 💾 Export results as JSON
- 🎮 No coding required

### Launch the Demo

```bash
# Start FastAPI backend
uvicorn apps.api.fastapi_app:app --host 0.0.0.0 --port 8000

# Launch Streamlit demo (new terminal)
streamlit run apps/ui/streamlit_app.py -- --port 8000
```

Open `http://localhost:8501` in your browser.

📖 [Streamlit Demo Guide](docs/streamlit_demo.md)

---

## Deployment Options

| Method | Best For | Key Benefits |
|:-------|:---------|:-------------|
| 🎨 **Streamlit Demo** | Testing, demonstrations | Zero-code UI, instant feedback |
| 📌 **FastAPI Backend** | Production APIs | REST endpoints, scalable |
| 🖥️ **C++ Runtime** | Edge devices | No Python dependency, ultra-fast |
| 💻 **Python CLI** | Batch processing | Scriptable, configurable |
| 📦 **PyPI Package** | Custom integration | Import as library |

### REST API Example

```python
import requests

with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
        params={"include_visualizations": True}
    )
    result = response.json()
    print(f"Anomaly Score: {result['anomaly_score']}")
    print(f"Is Anomaly: {result['is_anomaly']}")
```

📖 [FastAPI Setup Guide](docs/fastapi_backend.md)

### C++ Inference

Deploy without Python using our ONNX Runtime + OpenCV implementation:

- 🖼️ Complete pipeline from preprocessing to visualization
- 📦 Modular architecture
- 🌐 Perfect for edge devices

📖 [C++ Inference Guide](docs/cpp/README.md)

---

## Use Cases

- 🏭 **Manufacturing QC** — Real-time defect detection on production lines
- 🔬 **Medical Imaging** — Anomaly identification in X-rays, MRIs, microscopy
- 🏗️ **Infrastructure** — Crack and corrosion detection
- 📱 **PCB Inspection** — Soldering defects and component issues
- 🌾 **Agriculture** — Plant disease and crop monitoring
- 🚗 **Automotive** — Paint defects and assembly quality

---

## Documentation

- 📖 [Installation](docs/installation.md)
- 🚀 [Quick Start](docs/quickstart.md)
- 🎨 [Streamlit Demo](docs/streamlit_demo.md)
- 📌 [FastAPI Backend](docs/fastapi_backend.md)
- 💻 [CLI Reference](docs/cli.md)
- 📚 [API Reference](docs/api.md)
- ⚙️ [Configuration](docs/config.md)
- 📊 [Benchmarks](docs/benchmark.md)
- 🔧 [Troubleshooting](docs/troubleshooting.md)
- 🤝 [Contributing](docs/contributing.md)

---

## Community & Support

- 💬 [GitHub Discussions](https://github.com/DeepKnowledge1/AnomaVision/discussions)
- 🐛 [Issue Tracker](https://github.com/DeepKnowledge1/AnomaVision/issues)
- 📧 [Email Support](mailto:deepp.knowledge@gmail.com)

---

## Citation

```bibtex
@software{anomavision2025,
  title={AnomaVision: Edge-Ready Visual Anomaly Detection},
  author={DeepKnowledge Contributors},
  year={2025},
  url={https://github.com/DeepKnowledge1/AnomaVision},
}
```

---

## Acknowledgments

Built on the foundation of [Anodet](https://github.com/OpenAOI/anodet). We thank the original authors for their contributions to open-source anomaly detection research.

---

## License

Released under the [MIT License](LICENSE).

---

**Ready to get started?** Follow our [Quick Start Guide](docs/quickstart.md) and build your first anomaly detection pipeline in 5 minutes!
