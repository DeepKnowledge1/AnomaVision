# 🔍 AnomaVision: Production-Ready Visual Anomaly Detection

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![PyTorch 1.13+](https://img.shields.io/badge/pytorch-1.13+-red.svg)](https://pytorch.org)
[![CUDA 11.7](https://img.shields.io/badge/CUDA-11.7-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ONNX Ready](https://img.shields.io/badge/ONNX-Export%20Ready-orange.svg)](https://onnx.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**State-of-the-art anomaly detection powered by PaDiM algorithm**  
*From research to production in minutes*

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-api-reference) • [🎯 Examples](#-examples) • [🔧 Installation](#-installation)

---

</div>

## ✨ What is AnomaVision?

AnomaVision is a **high-performance deep learning library** for visual anomaly detection, built on the proven PaDiM (Patch Distribution Modeling) algorithm. Whether you're detecting defects in manufacturing, monitoring infrastructure, or ensuring quality control, AnomaVision delivers enterprise-grade performance with research-level accuracy.

> **🎯 Built for Production:** Optimized for real-world deployment with ONNX export, memory efficiency, and multi-format model support.

### 🏆 Key Features

- **⚡ Lightning Fast**: 40-60% less memory usage, 20-30% faster inference
- **🎛️ Multiple Backends**: PyTorch, ONNX, TensorRT (coming soon), OpenVINO (coming soon)
- **🔧 Production Ready**: One-click ONNX export for edge deployment
- **🎨 Rich Visualizations**: Built-in heatmaps, boundary detection, and highlighting
- **📊 Memory Efficient**: Process large datasets without OOM errors
- **🛡️ Robust**: Mixed precision support (FP16/FP32) with automatic fallbacks

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision

# Install with Poetry (recommended)
poetry install && poetry shell

# Or with pip
pip install -r requirements.txt
```

### 2️⃣ Train Your First Model

```python
import anodet
import torch
from torch.utils.data import DataLoader

# 📁 Load your "good" training images
dataset = anodet.AnodetDataset("path/to/train/good")
dataloader = DataLoader(dataset, batch_size=2)

# 🧠 Initialize and train PaDiM model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = anodet.Padim(
    backbone='resnet18',
    device=device,
    layer_indices=[0, 1],
    feat_dim=50
)

# 🎯 Train the model
model.fit(dataloader)

# 💾 Save for later use
torch.save(model, "my_anomaly_model.pt")
```

### 3️⃣ Detect Anomalies

```python
# 🔍 Load test images and detect anomalies
test_dataset = anodet.AnodetDataset("path/to/test/images")
test_dataloader = DataLoader(test_dataset, batch_size=2)

for batch, images, _, _ in test_dataloader:
    # Get anomaly scores and heatmaps
    image_scores, score_maps = model.predict(batch)
    
    # Classify as anomalous (threshold = 13)
    predictions = anodet.classification(image_scores, threshold=13)
    print(f"Anomaly scores: {image_scores}")
    print(f"Predictions: {predictions}")
    break
```

### 4️⃣ Export for Production

```python
from export import export_onnx, _ExportWrapper

# 🚀 Export to ONNX for deployment
wrapper = _ExportWrapper(model)
export_onnx(
    wrapper, 
    filepath="anomaly_detector.onnx",
    input_shape=(1, 3, 224, 224)
)
```

---

## 🔧 Installation

### Prerequisites
- Python 3.9+
- CUDA 11.7+ (for GPU acceleration)
- PyTorch 1.13+

### Method 1: Poetry (Recommended)
```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision
poetry install
poetry shell
```

### Method 2: pip
```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision
pip install -r requirements.txt
```

### Verify Installation
```python
python -c "import anodet; print('✅ AnomaVision installed successfully!')"
```

---

## 🎯 Examples

### 📋 Command Line Usage

#### Train a Model
```bash
python train.py \
  --dataset_path "data/bottle/train/good" \
  --model_data_path "./models/" \
  --backbone resnet18 \
  --batch_size 4 \
  --layer_indices 0 1 \
  --feat_dim 100
```

#### Run Inference
```bash
python detect.py \
  --dataset_path "data/bottle/test" \
  --model_data_path "./models/" \
  --model "padim_model.onnx" \
  --batch_size 4 \
  --thresh 15 \
  --enable_visualization
```

#### Evaluate Performance
```bash
python eval.py \
  --dataset_path "data/mvtec" \
  --model_data_path "./models/" \
  --model_name "padim_model.pt"
```

### 🎨 Visualization Examples

```python
import anodet.visualization as viz

# Create boundary visualizations
boundary_images = viz.framed_boundary_images(
    images, classifications, padding=40
)

# Generate anomaly heatmaps
heatmap_images = viz.heatmap_images(
    images, score_maps, alpha=0.6
)

# Highlight anomalous regions
highlighted = viz.highlighted_images(
    images, classifications, color=(255, 0, 0)
)
```

### 🔄 Multi-Format Model Support

```python
from anodet.inference.model_wrapper import ModelWrapper

# Automatically detect and load any supported format
model = ModelWrapper("model.onnx", device='cuda')  # ONNX
model = ModelWrapper("model.pt", device='cuda')    # PyTorch
# model = ModelWrapper("model.engine", device='cuda')  # TensorRT (coming soon)

# Unified prediction interface
scores, maps = model.predict(batch)
```

---

## 📖 API Reference

### Core Classes

#### `anodet.Padim`
Main PaDiM model class for training and inference.

```python
model = anodet.Padim(
    backbone='resnet18',           # 'resnet18' or 'wide_resnet50'
    device=torch.device('cuda'),   # Target device
    layer_indices=[0, 1, 2],       # ResNet layers to use
    feat_dim=100,                  # Feature dimensions
    channel_indices=None           # Optional channel selection
)
```

**Methods:**
- `fit(dataloader)` - Train the model on normal images
- `predict(batch)` - Detect anomalies in test images
- `evaluate(dataloader)` - Full evaluation with metrics

#### `anodet.AnodetDataset`
Dataset loader for anomaly detection.

```python
dataset = anodet.AnodetDataset(
    "path/to/images",        # Path to image folder
    transforms=None          # Optional transformations
)
```

#### `ModelWrapper`
Unified interface for multiple model formats.

```python
wrapper = ModelWrapper(
    model_path="model.onnx",    # Path to model file
    device='cuda'               # Target device
)
```

### Utility Functions

```python
# Classification with threshold
predictions = anodet.classification(scores, threshold=15)

# Export to ONNX
export_onnx(model, "output.onnx", input_shape=(1, 3, 224, 224))
```

---

## 🏗️ Project Structure

```
AnomaVision/
├── 📁 anodet/                     # Core library
│   ├── 📄 padim.py               # PaDiM implementation
│   ├── 📄 feature_extraction.py  # Feature extraction
│   ├── 📄 mahalanobis.py         # Distance computation
│   ├── 📁 datasets/              # Dataset loaders
│   ├── 📁 visualization/         # Visualization tools
│   ├── 📁 inference/             # Multi-format inference
│   └── 📄 utils.py               # Utility functions
├── 📄 train.py                   # Training script
├── 📄 detect.py                  # Inference script
├── 📄 eval.py                    # Evaluation script
├── 📄 export.py                  # ONNX export utilities
└── 📁 notebooks/                 # Example notebooks
```

---

## 🎛️ Configuration

### Training Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `backbone` | Feature extractor | `resnet18` | `resnet18`, `wide_resnet50` |
| `layer_indices` | ResNet layers to use | `[0]` | `[0, 1, 2, 3]` |
| `feat_dim` | Feature dimensions | `50` | `1-2048` |
| `batch_size` | Training batch size | `2` | `1+` |

### Inference Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `thresh` | Anomaly threshold | `13` |
| `gaussian_blur` | Apply blur to score maps | `True` |
| `enable_visualization` | Show results | `False` |

---

## 🚀 Performance Benchmarks

| Metric | Original Implementation | AnomaVision | Improvement |
|--------|------------------------|-------------|-------------|
| **Memory Usage** | Baseline | 40-60% less | ⬇️ 40-60% |
| **Training Speed** | Baseline | 15-25% faster | ⬆️ 15-25% |
| **Inference Speed** | Baseline | 20-30% faster | ⬆️ 20-30% |
| **Batch Size** | Limited | 2x larger | ⬆️ 100% |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | GTX 1060 (6GB) | RTX 3080+ (10GB+) |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 5GB | 10GB+ |

---

## 🔬 Advanced Usage

### Custom Datasets

```python
# For MVTec AD format
dataset = anodet.MVTecDataset("path/to/mvtec", "bottle", is_train=True)

# For custom folder structure
dataset = anodet.AnodetDataset("path/to/custom/good_images")
```

### Memory-Efficient Evaluation

```python
# For large datasets
results = model.evaluate_memory_efficient(test_dataloader)
images, targets, masks, scores, maps = results
```

### Advanced Visualization

```python
# Save visualizations
python detect.py \
  --save_visualizations \
  --viz_output_dir "./results/" \
  --viz_alpha 0.7 \
  --viz_color "255,0,128"
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Setup
```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision
poetry install --dev
pre-commit install
```

---

## 📚 Citation & References

If you use AnomaVision in your research, please cite:

```bibtex
@misc{anomavision2024,
  title={AnomaVision: Production-Ready Visual Anomaly Detection},
  author={Deep Knowledge},
  year={2024},
  url={https://github.com/DeepKnowledge1/AnomaVision}
}
```

### Original PaDiM Paper
```bibtex
@article{defard2021padim,
  title={PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization},
  author={Defard, Thomas and Setkov, Aleksandr and Loesch, Angelique and Audigier, Romaric},
  journal={arXiv preprint arXiv:2011.08785},
  year={2021}
}
```

---

## 📞 Support & Community

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/DeepKnowledge1/AnomaVision/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/DeepKnowledge1/AnomaVision/discussions)
- 📧 **Email**: [Deepp.Knowledge@gmail.com](mailto:Deepp.Knowledge@gmail.com)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Original Implementation**: [OpenAOI/anodet](https://github.com/OpenAOI/anodet)
- **PaDiM Algorithm**: Thomas Defard et al.
- **PyTorch Team**: For the amazing deep learning framework
- **Community**: All contributors and users who make this project better

---

<div align="center">

**⭐ Star this repo if it helped you! ⭐**

Made with ❤️ by [Deep Knowledge](https://github.com/DeepKnowledge1)

</div>