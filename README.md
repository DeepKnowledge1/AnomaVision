# 🚀 AnomaVision: Next-Gen Visual Anomaly Detection

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org)
[![CUDA 11.7+](https://img.shields.io/badge/CUDA-11.7+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ONNX Ready](https://img.shields.io/badge/ONNX-Export%20Ready-orange.svg)](https://onnx.ai/)
[![OpenVINO Ready](https://img.shields.io/badge/OpenVINO-Ready-blue.svg)](https://docs.openvino.ai/)
[![TorchScript Ready](https://img.shields.io/badge/TorchScript-Ready-red.svg)](https://pytorch.org/docs/stable/jit.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
<img src="notebooks/example_images/banner.png" alt="bg" width="85%" style="border-radius: 15px;"/>

**🔥 Production-ready anomaly detection powered by state-of-the-art PaDiM algorithm**
*Deploy anywhere, run everywhere - from edge devices to cloud infrastructure*

### 🚀 Supported Export Formats

| Format  | Status | Use Case | Backend |
|--------|--------|----------|---------|
| **PyTorch**  | ✅ <span style="color: green;"> **Ready**</span>| Development & Research | TorchBackend |
| **TorchScript**  | ✅ <span style="color: green;"> **Ready**</span> | Production Deployment | TorchScriptBackend |
| **ONNX**  | ✅ <span style="color: green;"> **Ready**</span> | Cross-platform Deployment | OnnxBackend |
| **OpenVINO**  | ✅ <span style="color: green;"> **Ready**</span> | Intel Hardware Optimization | OpenVinoBackend |
| **TensorRT**  | 🚧 Coming Soon | NVIDIA GPU Acceleration | TensorRTBackend |

### 🎯 Format Recommendations

| Use Case | Recommended Format | Reason |
|----------|-------------------|---------|
| **Development** | **PyTorch** (.pt) | Training and experimentation |
| **Production** | **TorchScript** (.torchscript) | Python deployment |
| **Cross-platform** | **ONNX** (.onnx) | Maximum compatibility |
| **Intel Hardware** | **OpenVINO** | CPUs, iGPUs, and VPUs |
| **NVIDIA GPUs** | **TensorRT** | Maximum GPU performance (coming soon) |

[⚡ Quick Start](#-quick-start) • [📚 Documentation](#-complete-api-reference) • [🎯 Examples](#-real-world-examples) • [🔧 Installation](#-installation)

---

### 🌟 Why Choose AnomaVision?

**🎯 Unmatched Performance** • **🔄 Multi-Format Support** • **📦 Production Ready** • **🎨 Rich Visualizations** • **📐 Flexible Image Dimensions**

</div>

<details open>
<summary>✨ What Makes AnomaVision Special?</summary>

AnomaVision transforms the cutting-edge **PaDiM (Patch Distribution Modeling)** algorithm into a production-ready powerhouse for visual anomaly detection. Whether you're detecting manufacturing defects, monitoring infrastructure, or ensuring quality control, AnomaVision delivers enterprise-grade performance with research-level accuracy.

### 🏆 Key Highlights

| Feature | Benefit | Impact |
|---------|---------|--------|
| **⚡ Lightning Fast** | 40-60% less memory usage, 20-30% faster inference | Deploy on resource-constrained devices |
| **🔄 Multi-Format Backend** | PyTorch, ONNX, TensorRT*, OpenVINO* support | One model, multiple deployment targets |
| **🎛️ Production Ready** | One-click ONNX export, memory optimization | From prototype to production in minutes |
| **🎨 Rich Visualizations** | Built-in heatmaps, boundary detection, highlighting | Instant insights for decision making |
| **🧠 Smart Memory Management** | Process datasets 2x larger without OOM | Scale to enterprise workloads |
| **📐 Flexible Image Dimensions** | Support for non-square images, configurable sizing | Work with real-world image formats |
| **⚙️ Unified Configuration** | CLI args or config file, persistent settings | Streamlined workflow and reproducibility |
| **🛡️ Robust & Reliable** | Mixed precision (FP16/FP32), automatic fallbacks | Consistent performance across hardware |

*\*Coming soon*

</details>

<details open>
<summary>🔧 Installation</summary>

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

<details>
<summary>🚀 Quick Start</summary>

### 🏃‍♂️ 30-Second Setup

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

<details>
<summary>🎯 Real-World Examples</summary>

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
```

#### 🔄 Export to Multiple Formats
```bash
# Export to all formats
python export.py \
  --model_data_path "./distributions/bottle_exp" \
  --model "padim_model.pt" \
  --format all

# Export specific format with options
python export.py \
  --model_data_path "./distributions/bottle_exp" \
  --model "padim_model.pt" \
  --format onnx \
  --opset 17 \
  --dynamic_batch
```

### 🎨 Advanced Visualization Magic

```python
import anodet.visualization as viz

# 🖼️ Create stunning boundary visualizations
boundary_images = viz.framed_boundary_images(
    images, classifications,
    padding=50,                    # Generous padding
    color=(255, 64, 64)           # Eye-catching red
)

# 🔥 Generate professional anomaly heatmaps
heatmap_images = viz.heatmap_images(
    images, score_maps,
    alpha=0.7,                     # Perfect transparency
    colormap='hot'                 # Heat-style colormap
)

# ✨ Highlight anomalous regions with precision
highlighted = viz.highlighted_images(
    images, classifications,
    color=(255, 255, 0),          # Bright yellow highlight
    thickness=3                    # Bold boundaries
)
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

</details>

<details>
<summary>⚙️ Configuration Guide</summary>

### 🎯 Training Parameters

| Parameter | Description | Default | Range | Pro Tip |
|-----------|-------------|---------|-------|---------|
| `backbone` | Feature extractor | `resnet18` | `resnet18`, `wide_resnet50` | Use ResNet18 for speed, Wide-ResNet50 for accuracy |
| `layer_indices` | ResNet layers | `[0]` | `[0, 1, 2, 3]` | `[0, 1]` gives best speed/accuracy balance |
| `feat_dim` | Feature dimensions | `50` | `1-2048` | Higher = more accurate but slower |
| `batch_size` | Training batch size | `2` | `1-64` | Use largest size that fits in memory |

### 📐 Image Processing Parameters

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

### 📁 Configuration File Structure

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

<details>
<summary>📚 Complete API Reference</summary>

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

<details>
<summary>🚀 Performance Benchmarks</summary>

### 📊 Speed & Memory Comparison

| Metric | Baseline | AnomaVision | Improvement |
|--------|----------|-------------|-------------|
| **🧠 Memory Usage** | 100% | **40-60%** | 🔥 40-60% reduction |
| **⚡ Training Speed** | 100% | **125-140%** | 🚀 25-40% faster |
| **🔍 Inference Speed** | 100% | **120-130%** | ⚡ 20-30% faster |
| **📐 Image Flexibility** | Square only | **Any dimensions** | 🎯 Real-world compatibility |

### 🖥️ Hardware Requirements

| Use Case | Minimum | Recommended | Enterprise |
|----------|---------|-------------|------------|
| **GPU** | GTX 1060 6GB | RTX 3070 8GB | RTX 4090 24GB |
| **RAM** | 8GB | 16GB | 32GB+ |
| **Storage** | 5GB | 10GB | 50GB+ |
| **Throughput** | 10 FPS | 50 FPS | 200+ FPS |

### 📐 Image Size Performance

| Input Size | Resize | Final Size | Training Time | Inference FPS |
|------------|--------|------------|---------------|---------------|
| 1024x768 | [256, 256] | [224, 224] | 100% | 50 FPS |
| 1920x1080 | [256, 192] | [224, 224] | 95% | 52 FPS |
| 640x480 | [224, 224] | [224, 224] | 105% | 48 FPS |
| Various | Auto-scale | [224, 224] | 98% | 50 FPS |

</details>

<details>
<summary>🗃️ Architecture Overview</summary>

```
AnomaVision/
├── 🧠 anodet/                      # Core AI library
│   ├── 🔄 padim.py                 # PaDiM implementation
│   ├── 🔄 feature_extraction.py    # ResNet feature extraction
│   ├── 🔄 mahalanobis.py          # Distance computation
│   ├── 📁 datasets/               # Dataset loaders with flexible sizing
│   ├── 📁 visualization/          # Rich visualization tools
│   ├── 📁 inference/              # Multi-format inference engine
│   │   ├── 🔄 wrapper.py          # Universal model wrapper
│   │   ├── 🔄 modelType.py        # Format detection
│   │   └── 📁 backends/           # Format-specific backends
│   │       ├── 🔄 base.py         # Backend interface
│   │       ├── 🔄 torch_backend.py    # PyTorch support
│   │       ├── 🔄 onnx_backend.py     # ONNX Runtime support
│   │       ├── 🔄 torchscript_backend.py # TorchScript support
│   │       ├── 🔄 tensorrt_backend.py # TensorRT (coming soon)
│   │       └── 🔄 openvino_backend.py # OpenVINO support
│   ├── 📁 config/                 # Configuration management
│   └── 🔄 utils.py                # Utility functions
├── 🔄 train.py                    # Training script with config support
├── 🔄 detect.py                   # Inference script
├── 🔄 eval.py                     # Evaluation script
├── 🔄 export.py                   # Multi-format export utilities
├── 🔄 config.yml                  # Default configuration
└── 📁 notebooks/                  # Interactive examples
```

</details>

<details>
<summary>🔬 Advanced Features</summary>

### 💾 Memory-Optimized Evaluation
```python
# 🚀 Process massive datasets without OOM
results = model.evaluate_memory_efficient(huge_dataloader)
images, targets, masks, scores, maps = results

# 📊 Pre-allocated arrays for maximum efficiency
# Handles datasets 10x larger than traditional methods
```

### 📐 Flexible Image Processing Pipeline
```python
# 🎯 Handle any image dimensions
dataset = anodet.AnodetDataset(
    image_path,
    resize=[512, 384],              # Custom width/height
    crop_size=[448, 224],           # Non-square crop
    normalize=True                  # Automatic normalization
)

# 🔄 Automatic configuration persistence
# Training settings automatically used in detect.py and eval.py
```

### ⚙️ Unified Configuration System
```python
# 📁 Create config.yml for your project
config = {
    'dataset_path': 'data/custom',
    'resize': [320, 240],           # Custom dimensions
    'crop_size': [224, 224],        # Standard CNN input
    'backbone': 'resnet18',
    'feat_dim': 150,
    'layer_indices': [0, 1, 2]
}

# 🚀 Use consistently across all scripts
python train.py --config config.yml
python detect.py --config config.yml    # Uses same preprocessing
python eval.py --config config.yml      # Consistent evaluation
python export.py --config config.yml    # Same model configuration
```

### 🎨 Professional Visualizations
```python
# 💫 Create publication-ready visualizations
python detect.py \
  --save_visualizations \
  --viz_output_dir "./results/" \
  --viz_alpha 0.8 \
  --viz_color "255,64,128" \
  --viz_padding 60
```

### 🔧 Custom Layer Hooks
```python
# 🎯 Advanced feature engineering
def custom_hook(layer_output):
    # Apply custom transformations
    return F.normalize(layer_output, p=2, dim=1)

model = anodet.Padim(
    backbone='resnet18',
    layer_hook=custom_hook,    # 👈 Custom processing
    layer_indices=[1, 2, 3]
)
```

</details>

<details>
<summary><span style="color: red;">🔧 Adding New Model Formats</span></summary>

AnomaVision's architecture makes it incredibly easy to add support for new model formats. Here's how to integrate a new backend:

### 🎯 Step 1: Add Model Type

```python
# In modelType.py
class ModelType(Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    YOUR_NEW_FORMAT = "your_format"  # 👈 Add your format

    @classmethod
    def from_extension(cls, model_path):
        extension_map = {
            '.pt': cls.PYTORCH,
            '.onnx': cls.ONNX,
            '.torchscript': cls.TORCHSCRIPT,
            '.engine': cls.TENSORRT,
            '.xml': cls.OPENVINO,
            '.your_ext': cls.YOUR_NEW_FORMAT,  # 👈 Add file extension
        }
        # ... rest of method
```

### 🎯 Step 2: Create Backend Implementation

```python
# In backends/your_backend.py
from .base import Batch, ScoresMaps, InferenceBackend
from anodet.utils import get_logger

logger = get_logger(__name__)

class YourBackend(InferenceBackend):
    """Your custom backend implementation."""

    def __init__(self, model_path: str, device: str = "cuda"):
        logger.info(f"Initializing YourBackend with {model_path}")

        # 🔧 Initialize your model format here
        self.model = your_framework.load_model(model_path)
        self.device = device

        # 🎯 Any format-specific setup
        self.setup_optimizations()

    def predict(self, batch: Batch) -> ScoresMaps:
        """Run inference using your framework."""
        logger.debug(f"Running inference on batch shape: {batch.shape}")

        # 🔄 Convert input format if needed
        if isinstance(batch, torch.Tensor):
            input_data = batch.numpy()
        else:
            input_data = batch

        # 🚀 Run inference with your framework
        outputs = self.model.run(input_data)

        # 📊 Extract scores and maps
        scores, maps = outputs[0], outputs[1]

        logger.debug(f"Inference complete. Output shapes: {scores.shape}, {maps.shape}")
        return scores, maps

    def close(self) -> None:
        """Release resources."""
        logger.info("Closing YourBackend resources")
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()
        self.model = None
```

### 🎯 Step 3: Register in Factory

```python
# In wrapper.py - add to make_backend function
def make_backend(model_path: str, device: str) -> InferenceBackend:
    model_type = ModelType.from_extension(model_path)

    # ... existing backends ...

    if model_type == ModelType.YOUR_NEW_FORMAT:
        from .backends.your_backend import YourBackend
        logger.debug("Selected YourBackend for %s", model_path)
        return YourBackend(model_path, device)  # 👈 Add your backend

    raise NotImplementedError(f"ModelType {model_type} is not supported.")
```

### ✅ That's It!

Your new format is now fully integrated with AnomaVision's unified API:

```python
# 🎉 Use your new format just like any other!
model = ModelWrapper("model.your_ext", device='cuda')
scores, maps = model.predict(batch)
model.close()
```

</details>

<details>
<summary>🤝 Contributing</summary>

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
