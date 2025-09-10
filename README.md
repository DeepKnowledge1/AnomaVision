# 🚀 AnomaVision: Edge-Ready Visual Anomaly Detection

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org)
[![CUDA 11.7+](https://img.shields.io/badge/CUDA-11.7+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ONNX Ready](https://img.shields.io/badge/ONNX-Export%20Ready-orange.svg)](https://onnx.ai/)
[![OpenVINO Ready](https://img.shields.io/badge/OpenVINO-Ready-blue.svg)](https://docs.openvino.ai/)
[![TorchScript Ready](https://img.shields.io/badge/TorchScript-Ready-red.svg)](https://pytorch.org/docs/stable/jit.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

<img src="docs/images/AnomaVision_banner.png" alt="bg" width="100%" style="border-radius: 15px;"/>

**🔥 Production-ready anomaly detection powered by state-of-the-art PaDiM algorithm**
*Deploy anywhere, run everywhere - from edge devices to cloud infrastructure*


<details open>
<summary>✨ Supported Export Formats</summary>

| Format | Status | Use Case | Language Support |
|--------|--------|----------|------------------|
| **PyTorch** | ✅ Ready | Development & Research | Python |
| **Statistics (.pth)** | ✅ Ready | Ultra-compact deployment (2-4x smaller) | Python |
| **ONNX** | ✅ Ready | Cross-platform deployment | Python, C++ |
| **TorchScript** | ✅ Ready | Production Python deployment | Python |
| **OpenVINO** | ✅ Ready | Intel hardware optimization | Python|
| **TensorRT** | 🚧 Coming Soon | NVIDIA GPU acceleration | Python|

</details>

</div>


---

<details open>
<summary>✨ What's New (September 2025)</summary>

- **Slim artifacts (`.pth`)**: Save only PaDiM statistics (mean, cov_inv, channel indices, layer indices, backbone) for **2–4× smaller files** vs. full `.pt` checkpoints
- **Plug-and-play loading**: `.pth` loads seamlessly through `TorchBackend` and exporter via lightweight runtime (`PadimLite`) with same `.predict(...)` interface
- **CPU-first pipeline**: Everything works on machines **without a GPU**. FP16 used only for storage; compute happens in FP32 on CPU
- **Export from `.pth`**: ONNX/TorchScript/OpenVINO export now accepts stats-only `.pth` directly
- **Test coverage**: New pytest cases validate saving stats, loading via `PadimLite`, CPU inference, and exporter compatibility

</details>

---

<details open>
<summary>✨ Why Choose AnomaVision?</summary>

**🎯 Unmatched Performance** • **🔄 Multi-Format Support** • **📦 Production Ready** • **🎨 Rich Visualizations** • **📏 Flexible Image Dimensions**

AnomaVision transforms the cutting-edge **PaDiM (Patch Distribution Modeling)** algorithm into a production-ready powerhouse for visual anomaly detection. Whether you're detecting manufacturing defects, monitoring infrastructure, or ensuring quality control, AnomaVision delivers enterprise-grade performance with research-level accuracy.

</details>

---
<details>
<summary>✨ Benchmark Results: AnomaVision vs Anomalib (MVTec Bottle, CPU-only)</summary>

<img src="docs/images/av_al.png" alt="bg" width="50%" style="border-radius: 15px;"/>

</details>

---

<details >
<summary>✨ Installation</summary>

### 📋 Prerequisites
- **Python**: 3.9+
- **CUDA**: 11.7+ for GPU acceleration
- **PyTorch**: 2.0+ (automatically installed)

### 🎯 Method 1: Poetry (Recommended)
```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision
poetry install
poetry shell
```

### 🎯 Method 2: pip
```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision
pip install -r requirements.txt
```

### ✅ Verify Installation
```python
python -c "import anodet; print('🎉 AnomaVision installed successfully!')"
```

### 🐳 Docker Support
```bash
# Build Docker image (coming soon)
docker build -t anomavision:latest .
docker run --gpus all -v $(pwd):/workspace anomavision:latest
```

</details>

---

<details >
<summary>✨ Quick Start</summary>

### 🎯 Train Your First Model (2 minutes)

```python
import anodet
import torch
from torch.utils.data import DataLoader

# 📂 Load your "good" training images
dataset = anodet.AnodetDataset(
    "path/to/train/good",
    resize=[256, 192],          # Flexible width/height
    crop_size=[224, 224],       # Final crop size
    normalize=True              # ImageNet normalization
)
dataloader = DataLoader(dataset, batch_size=4)

# 🧠 Initialize PaDiM with optimal settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = anodet.Padim(
    backbone='resnet18',           # Fast and accurate
    device=device,
    layer_indices=[0, 1],          # Multi-scale features
    feat_dim=100                   # Optimal feature dimension
)

# 🔥 Train the model (surprisingly fast!)
print("🚀 Training model...")
model.fit(dataloader)

# 💾 Save for production deployment
torch.save(model, "anomaly_detector.pt")
model.save_statistics("compact_model.pth", half=True)  # 4x smaller!
print("✅ Model trained and saved!")
```

### 🔍 Detect Anomalies Instantly

```python
# 📊 Load test data and detect anomalies (uses same preprocessing as training)
test_dataset = anodet.AnodetDataset("path/to/test/images")
test_dataloader = DataLoader(test_dataset, batch_size=4)

for batch, images, _, _ in test_dataloader:
    # 🎯 Get anomaly scores and detailed heatmaps
    image_scores, score_maps = model.predict(batch)

    # 🏷️ Classify anomalies (threshold=13 works great for most cases)
    predictions = anodet.classification(image_scores, threshold=13)

    print(f"🔥 Anomaly scores: {image_scores.tolist()}")
    print(f"📋 Predictions: {predictions.tolist()}")
    break
```

### 🚀 Export for Production Deployment

```python
# 📦 Export to ONNX for universal deployment
python export.py \
  --model_data_path "./models/" \
  --model "padim_model.pt" \
  --format onnx \
  --opset 17

print("✅ ONNX model ready for deployment!")
```

</details>

---

<details >
<summary>✨ Real-World Examples</summary>

### 🖥️ Command Line Interface

#### 📚 Train a High-Performance Model
```bash
# Using command line arguments
python train.py \
  --dataset_path "data/bottle" \
  --class_name "bottle" \
  --model_data_path "./models/" \
  --backbone resnet18 \
  --batch_size 8 \
  --layer_indices 0 1 2 \
  --feat_dim 200 \
  --resize 256 224 \
  --crop_size 224 224 \
  --normalize

# Or using config file (recommended)
python train.py --config config.yml
```

**Sample config.yml:**
```yaml
# Dataset configuration
dataset_path: "D:/01-DATA"
class_name: "bottle"
resize: [256, 224]        # Width, Height - flexible dimensions!
crop_size: [224, 224]     # Final square crop
normalize: true
norm_mean: [0.485, 0.456, 0.406]
norm_std: [0.229, 0.224, 0.225]

# Model configuration
backbone: "resnet18"
feat_dim: 100
layer_indices: [0, 1]
batch_size: 8

# Output configuration
model_data_path: "./distributions/bottle_exp"
output_model: "padim_model.pt"
run_name: "bottle_experiment"
```

#### 🔍 Run Lightning-Fast Inference
```bash
# Automatically uses training configuration
python detect.py \
  --model_data_path "./distributions/bottle_exp" \
  --model "padim_model.pt" \
  --img_path "data/bottle/test/broken_large" \
  --batch_size 16 \
  --thresh 13 \
  --enable_visualization \
  --save_visualizations

# Multi-format support
python detect.py --model padim_model.pt          # PyTorch
python detect.py --model padim_model.torchscript # TorchScript
python detect.py --model padim_model.onnx        # ONNX Runtime
python detect.py --model padim_model_openvino    # OpenVINO

# Or using config file (recommended)
python train.py --config config.yml

```

#### 📊 Comprehensive Model Evaluation
```bash
# Uses saved configuration automatically
python eval.py \
  --model_data_path "./distributions/bottle_exp" \
  --model "padim_model.pt" \
  --dataset_path "data/mvtec" \
  --class_name "bottle" \
  --batch_size 8

# Or using config file (recommended)
python eval.py --config config.yml

```

#### 🔄 Export to Multiple Formats
```bash
# Export to all formats
python export.py \
  --model_data_path "./distributions/bottle_exp" \
  --model "padim_model.pt" \
  --format all

# Or using config file (recommended)
python export.py --config config.yml
```

### 🔄 Universal Model Format Support

```python
from anodet.inference.model.wrapper import ModelWrapper

# 🎯 Automatically detect and load ANY supported format
pytorch_model = ModelWrapper("model.pt", device='cuda')        # PyTorch
onnx_model = ModelWrapper("model.onnx", device='cuda')         # ONNX Runtime
torchscript_model = ModelWrapper("model.torchscript", device='cuda')  # TorchScript
openvino_model = ModelWrapper("model_openvino/model.xml", device='cpu')  # OpenVINO

# 🚀 Unified prediction interface - same API for all formats!
scores, maps = pytorch_model.predict(batch)
scores, maps = onnx_model.predict(batch)

# 🧹 Always clean up resources
pytorch_model.close()
onnx_model.close()
```

### 🔧 C++ ONNX Integration

```cpp
// C++ ONNX Runtime integration example

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>

.
.
.
.

```

</details>

---

<details >
<summary>✨ Configuration Guide</summary>

### 🎯 Training Parameters

| Parameter | Description | Default | Range | Pro Tip |
|-----------|-------------|---------|-------|---------|
| `backbone` | Feature extractor | `resnet18` | `resnet18`, `wide_resnet50` | Use ResNet18 for speed, Wide-ResNet50 for accuracy |
| `layer_indices` | ResNet layers | `[0]` | `[0, 1, 2, 3]` | `[0, 1]` gives best speed/accuracy balance |
| `feat_dim` | Feature dimensions | `50` | `1-2048` | Higher = more accurate but slower |
| `batch_size` | Training batch size | `2` | `1-64` | Use largest size that fits in memory |

### 📏 Image Processing Parameters

| Parameter | Description | Default | Example | Pro Tip |
|-----------|-------------|---------|---------|---------|
| `resize` | Initial resize | `[224, 224]` | `[256, 192]` | Flexible width/height, maintains aspect ratio |
| `crop_size` | Final crop size | `None` | `[224, 224]` | Square crops often work best for CNN models |
| `normalize` | ImageNet normalization | `true` | `true/false` | Usually improves performance with pretrained models |
| `norm_mean` | RGB mean values | `[0.485, 0.456, 0.406]` | Custom values | Use ImageNet stats for pretrained backbones |
| `norm_std` | RGB std values | `[0.229, 0.224, 0.225]` | Custom values | Match your training data distribution |

### 🔍 Inference Parameters

| Parameter | Description | Default | Range | Pro Tip |
|-----------|-------------|---------|-------|---------|
| `thresh` | Anomaly threshold | `13` | `1-100` | Start with 13, tune based on your data |
| `enable_visualization` | Show results | `false` | `true/false` | Great for debugging and demos |
| `save_visualizations` | Save images | `false` | `true/false` | Essential for production monitoring |

### 📄 Configuration File Structure

```yaml
# =========================
# Dataset / preprocessing (shared by train, detect, eval)
# =========================
dataset_path: "D:/01-DATA"               # Root dataset folder
class_name: "bottle"                     # Class name for MVTec dataset
resize: [224, 224]                       # Resize dimensions [width, height]
crop_size: [224, 224]                    # Final crop size [width, height]
normalize: true                          # Whether to normalize images
norm_mean: [0.485, 0.456, 0.406]         # ImageNet normalization mean
norm_std: [0.229, 0.224, 0.225]          # ImageNet normalization std

# =========================
# Model / training
# =========================
backbone: "resnet18"                     # Backbone CNN architecture
feat_dim: 50                             # Feature dimension size
layer_indices: [0]                       # Which backbone layers to use
model_data_path: "./distributions/exp"   # Path to store model data
output_model: "padim_model.pt"           # Saved model filename
batch_size: 2                            # Training/inference batch size
device: "auto"                           # Device: "cpu", "cuda", or "auto"

# =========================
# Inference (detect.py)
# =========================
img_path: "D:/01-DATA/bottle/test/broken_large"  # Test images path
thresh: 13.0                            # Anomaly detection threshold
enable_visualization: true               # Enable visualizations
save_visualizations: true                # Save visualization results
viz_output_dir: "./visualizations/"      # Visualization output directory

# =========================
# Export (export.py)
# =========================
format: "all"                           # Export format: onnx, torchscript, openvino, all
opset: 17                               # ONNX opset version
dynamic_batch: true                     # Allow dynamic batch size
fp32: false                             # Export precision (false = FP16 for OpenVINO)
```

</details>

---

<details >
<summary>✨ Complete API Reference</summary>

### 🧠 Core Classes

#### `anodet.Padim` - The Heart of AnomaVision
```python
model = anodet.Padim(
    backbone='resnet18',              # 'resnet18' | 'wide_resnet50'
    device=torch.device('cuda'),      # Target device
    layer_indices=[0, 1, 2],          # ResNet layers [0-3]
    feat_dim=100,                     # Feature dimensions (1-2048)
    channel_indices=None              # Optional channel selection
)
```

**🔥 Methods:**
- `fit(dataloader, extractions=1)` - Train on normal images
- `predict(batch, gaussian_blur=True)` - Detect anomalies
- `evaluate(dataloader)` - Full evaluation with metrics
- `evaluate_memory_efficient(dataloader)` - For large datasets
- `save_statistics(path, half=False)` - Save compact statistics
- `load_statistics(path, device, force_fp32=True)` - Load statistics

#### `anodet.AnodetDataset` - Smart Data Loading with Flexible Sizing
```python
dataset = anodet.AnodetDataset(
    "path/to/images",               # Image directory
    resize=[256, 192],              # Flexible width/height resize
    crop_size=[224, 224],           # Final crop dimensions
    normalize=True,                 # ImageNet normalization
    mean=[0.485, 0.456, 0.406],     # Custom mean values
    std=[0.229, 0.224, 0.225]       # Custom std values
)

# For MVTec format with same flexibility
mvtec_dataset = anodet.MVTecDataset(
    "path/to/mvtec",
    class_name="bottle",
    is_train=True,
    resize=[300, 300],              # Square resize
    crop_size=[224, 224],           # Final crop
    normalize=True
)
```

#### `ModelWrapper` - Universal Model Interface
```python
wrapper = ModelWrapper(
    model_path="model.onnx",        # Any supported format (.pt, .onnx, .torchscript, etc.)
    device='cuda'                   # Target device
)

# 🎯 Unified API for all formats
scores, maps = wrapper.predict(batch)
wrapper.close()  # Always clean up!
```

### 🛠️ Utility Functions

```python
# 🏷️ Smart classification with optimal thresholds
predictions = anodet.classification(scores, threshold=15)

# 📊 Comprehensive evaluation metrics
images, targets, masks, scores, maps = model.evaluate(dataloader)

# 🎨 Rich visualization functions
boundary_images = anodet.visualization.framed_boundary_images(images, classifications)
heatmap_images = anodet.visualization.heatmap_images(images, score_maps)
highlighted_images = anodet.visualization.highlighted_images(images, classifications)
```

### ⚙️ Configuration Management

```python
from anodet.config import load_config
from anodet.utils import merge_config

# Load configuration from file
config = load_config("config.yml")

# Merge with command line arguments
final_config = merge_config(args, config)

# Image processing with automatic parameter application
dataset = anodet.AnodetDataset(
    image_path,
    resize=config.resize,           # From config: [256, 224]
    crop_size=config.crop_size,     # From config: [224, 224]
    normalize=config.normalize,     # From config: true
    mean=config.norm_mean,          # From config: ImageNet values
    std=config.norm_std             # From config: ImageNet values
)
```

</details>

---


<details >
<summary>✨ Architecture Overview</summary>

```
AnomaVision/
├── 🧠 anodet/                      # Core AI library
│   ├── 📄 padim.py                 # PaDiM implementation
│   ├── 📄 padim_lite.py            # Lightweight runtime module
│   ├── 📄 feature_extraction.py    # ResNet feature extraction
│   ├── 📄 mahalanobis.py          # Distance computation
│   ├── 📁 datasets/               # Dataset loaders with flexible sizing
│   ├── 📁 visualization/          # Rich visualization tools
│   ├── 📁 inference/              # Multi-format inference engine
│   │   ├── 📄 wrapper.py          # Universal model wrapper
│   │   ├── 📄 modelType.py        # Format detection
│   │   └── 📁 backends/           # Format-specific backends
│   │       ├── 📄 base.py         # Backend interface
│   │       ├── 📄 torch_backend.py    # PyTorch support
│   │       ├── 📄 onnx_backend.py     # ONNX Runtime support
│   │       ├── 📄 torchscript_backend.py # TorchScript support
│   │       ├── 📄 tensorrt_backend.py # TensorRT (coming soon)
│   │       └── 📄 openvino_backend.py # OpenVINO support
│   ├── 📁 config/                 # Configuration management
│   └── 📄 utils.py                # Utility functions
├── 📄 train.py                    # Training script with config support
├── 📄 detect.py                   # Inference script
├── 📄 eval.py                     # Evaluation script
├── 📄 export.py                   # Multi-format export utilities
├── 📄 config.yml                  # Default configuration
└── 📁 notebooks/                  # Interactive examples
```

</details>

---

<details >
<summary>✨ Contributing</summary>

We love contributions! Here's how to make AnomaVision even better:

### 🚀 Quick Start for Contributors
```bash
# 🔥 Fork and clone
git clone https://github.com/yourusername/AnomaVision.git
cd AnomaVision

# 🔧 Setup development environment
poetry install --dev
pre-commit install

# 🌿 Create feature branch
git checkout -b feature/awesome-improvement

# 🔨 Make your changes
# ... code, test, commit ...

# 🚀 Submit pull request
git push origin feature/awesome-improvement
```

### 📝 Development Guidelines

- **Code Style**: Follow PEP 8 with 88-character line limit (Black formatting)
- **Type Hints**: Add type hints to all new functions and methods
- **Docstrings**: Use Google-style docstrings for all public functions
- **Tests**: Add pytest tests for new functionality
- **Documentation**: Update README and docstrings as needed

### 🐛 Bug Reports & Feature Requests

- **Bug Reports**: Use the [bug report template](.github/ISSUE_TEMPLATE/bug-report.yml)
- **Feature Requests**: Use the [feature request template](.github/ISSUE_TEMPLATE/feature-request.yml)
- **Questions**: Use [GitHub Discussions](https://github.com/DeepKnowledge1/AnomaVision/discussions)

</details>

---

<details >
<summary>✨ Support & Community</summary>

### 🤝 Getting Help

1. **📖 Documentation**: Check this README and code documentation
2. **🔍 Search Issues**: Someone might have had the same question
3. **💬 Discussions**: Use GitHub Discussions for questions
4. **🐛 Bug Reports**: Create detailed issue reports with examples

### 👥 Maintainers

- **Core Team**: [@DeepKnowledge1](https://github.com/DeepKnowledge1)
- **Contributors**: See [CONTRIBUTORS.md](CONTRIBUTORS.md)

### 🌟 Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- GitHub contributors page

</details>

---
---

<details >
<summary>✨ Roadmap</summary>

### 📅 Q4 2025
- **🚀 TensorRT Backend**: NVIDIA GPU acceleration
- **📱 Mobile Export**: CoreML and TensorFlow Lite support
- **🔧 C++ API**: Native C++ library with Python bindings
- **🎯 AutoML**: Automatic hyperparameter optimization

### 📅 Q1 2026
- **🧠 Transformer Models**: Vision Transformer (ViT) backbone support
- **🔄 Online Learning**: Continuous model updates
- **📊 MLOps Integration**: MLflow, Weights & Biases support
- **🌐 Web Interface**: Browser-based inference and visualization

### 📅 Q2 2026
- **🎥 Video Anomaly Detection**: Temporal anomaly detection
- **🔍 Multi-Class Support**: Beyond binary anomaly detection
- **⚡ Quantization**: INT8 optimization for edge devices
- **🔗 Integration**: Kubernetes operators and Helm charts

</details>

---

<details >
<summary>✨ License & Citation</summary>

### 📜 MIT License

AnomaVision is released under the **MIT License** - see [LICENSE](LICENSE) for details.

### 📖 Citation

If AnomaVision helps your research or project, we'd appreciate a citation:

```bibtex
@software{anomavision2025,
  title={AnomaVision: Edge-Ready Visual Anomaly Detection},
  author={DeepKnowledge Contributors},
  year={2025},
  url={https://github.com/DeepKnowledge1/AnomaVision},
  version={2.0.46},
  note={High-performance anomaly detection library optimized for edge deployment}
}
```

### 🙏 Acknowledgments

AnomaVision builds upon the excellent work of:
- **PaDiM**: Original algorithm by Defard et al.
- **PyTorch**: Deep learning framework
- **ONNX**: Open Neural Network Exchange
- **OpenVINO**: Intel's inference optimization toolkit
- **Anomalib**: Intel's anomaly detection library (for inspiration)

</details>

---

<details >
<summary>✨ Related Projects</summary>

- **[Anodet](https://github.com/OpenAOI/anodet)**:  anomaly detection

</details>

---

<details >
<summary>✨ Contact & Support</summary>

### 🤝 Community Channels

- **💬 GitHub Discussions**: [Community Forum](https://github.com/DeepKnowledge1/AnomaVision/discussions)
- **🐛 Issues**: [Bug Reports & Features](https://github.com/DeepKnowledge1/AnomaVision/issues)
- **📧 Email**: [deepp.knowledge@gmail.com](mailto:deepp.knowledge@gmail.com)
- **📖 Documentation**: [Wiki](https://github.com/DeepKnowledge1/AnomaVision/wiki)

### 💼 Enterprise Support

For enterprise deployments, custom integrations, or commercial support:
- **🏢 Enterprise Consulting**: Available upon request
- **🎓 Training Workshops**: Custom training for your team
- **🔧 Custom Development**: Tailored solutions for your use case

</details>

---

<div align="center">

## 🚀 Ready to Transform Your Anomaly Detection?

**Stop settling for slow, bloated solutions. Experience the future of edge-ready anomaly detection.**

[![Get Started](https://img.shields.io/badge/Get%20Started-Now-brightgreen?style=for-the-badge&logo=rocket)](https://github.com/DeepKnowledge1/AnomaVision)
[![Run Benchmark](https://img.shields.io/badge/Run%20Benchmark-Compare-orange?style=for-the-badge&logo=speedtest)](compare_with_anomalib.py)
[![Documentation](https://img.shields.io/badge/Read%20Docs-Here-blue?style=for-the-badge&logo=book)](docs/)
[![Star Us](https://img.shields.io/badge/⭐%20Star%20Us-GitHub-yellow?style=for-the-badge&logo=github)](https://github.com/DeepKnowledge1/AnomaVision)

---

**🏆 Benchmark Results Don't Lie: AnomaVision Wins 10/10 Metrics**
*Deploy fast. Detect better. AnomaVision.*

**Made with ❤️ for the edge AI community**

</div>
