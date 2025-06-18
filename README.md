
# 🚀 AnomaVision: State-of-the-Art Visual Anomaly Detection with PaDiM

[![Version](https://img.shields.io/badge/version-2.0.35-blue.svg)](https://github.com/your-repo/AnomaVision)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-1.13.1-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.7-yellow.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> # **Notice:**  
> This project is a highly optimized and extended fork of [OpenAOI/anodet](https://github.com/OpenAOI/anodet).  
> All core algorithms and designs are adapted and enhanced from the original anodet repository.

---

### 🔥 Production-Ready Deep Learning Library for Anomaly Detection

AnomaVision brings **cutting-edge PaDiM-based anomaly detection** to your projects, optimized for both research and deployment. Whether you work in manufacturing, quality control, or research, AnomaVision offers blazing-fast inference, easy ONNX export, and a flexible, modern API.

---

### ✨ Why AnomaVision?

- **Lightning Fast & Memory Efficient**: Train and infer faster with up to 60% less memory usage.
- **ONNX Deployment Out-of-the-Box**: Go from training to production in minutes—on the cloud or at the edge.
- **Mixed Precision Power**: Supports FP16/FP32 automatically for peak GPU/CPU performance.
- **Flexible & Modular**: Customize everything—backbone, feature layers, dimensions—no code rewrites needed.
- **Zero-Frustration Integration**: Train, export, and predict via Python or CLI—one codebase, infinite workflows.

---

#### 📸 Example: Detecting Anomalies on MVTec AD
![Example](notebooks/example_images/padim_example_image.png)

---

## 🚀 Get Started in Minutes

## 🛠️ 1. Installation

```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision

# Install with Poetry (recommended)
poetry shell
poetry install
````

---

## ⚡ 2. Quick Usage Examples

### Python API

```python
import anodet
import torch
from torch.utils.data import DataLoader

# Load dataset
dataset = anodet.AnodetDataset("path/to/train/good")
dataloader = DataLoader(dataset, batch_size=2)

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build and train PaDiM model
model = anodet.Padim(
    backbone='resnet18',
    device=device,
    layer_indices=[0, 1],
    feat_dim=50
)
model.fit(dataloader)

# Export to ONNX for production deployment
from anodet.export import export_onnx
export_onnx(model, "padim_model.onnx", input_shape=(1, 3, 224, 224))

# Predict anomalies on new data
test_batch = next(iter(dataloader))[0]
image_scores, score_map = model.predict(test_batch)
```




#### Command-Line Power (CLI)

Train and export in a single command:

```bash
python main.py \
  --dataset_path "/path/to/dataset" \           # Path to the dataset folder (should contain 'train/good' subfolder)
  --model_data_path "./model_dir" \             # Directory to save trained model/distribution files and ONNX output
  --backbone resnet18 \                         # Backbone network to use ('resnet18' or 'wide_resnet50')
  --layer_indices 0 1 \                         # Indices of backbone layers to extract features from (space separated)
  --feat_dim 50 \                               # Number of random feature dimensions to select for training
  --batch_size 2 \                              # Batch size for training
  --output_model "padim_model.pt"               # Output filename for PT model
```

*Show all CLI options:*

```bash
python main.py --help
```

---

## 🗂️ Project Structure

```
AnomaVision/
├── anodet/
│   ├── feature_extraction.py   # ResNet feature extraction utilities
│   ├── mahalanobis.py          # Mahalanobis distance (fast, ONNX-friendly)
│   ├── padim.py                # PaDiM main model class
│   ├── export.py               # ONNX export helper
│   └── ... (other utilities)
├── main.py                     # CLI for training/export
├── pyproject.toml              # Dependencies and build settings
├── README.md                   # This file!
```

---

## 🛠️ Powerful, Intuitive API

**Model Instantiation**

```python
Padim(
    backbone='resnet18',         # 'resnet18' or 'wide_resnet50'
    device=torch.device('cuda'), # Target device
    layer_indices=[0, 1],        # List of ResNet layers (0: shallowest)
    feat_dim=50,                 # Number of random feature dims (see code)
    channel_indices=None         # Optional custom channel indices
)
```

**Training**

```python
model.fit(
    dataloader,      # torch DataLoader of "good" images
    extractions=1    # Optional: repeat count for augmentation
)
```

**Inference**

```python
image_scores, score_map = model.predict(
    batch,            # Input tensor (B, 3, H, W)
    gaussian_blur=True   # Apply Gaussian blur (default: True)
)
```

**ONNX Export**

```python
from anodet.export import export_onnx
export_onnx(
    model,
    "padim_model.onnx",
    input_shape=(1, 3, 224, 224) # (batch, channels, height, width)
)
```

---

## 🏆 Performance at a Glance

| Metric         | Original  | AnomaVision | Improvement   |
| -------------- | --------- | ----------- | ------------- |
| Memory Usage   | High      | Low         | 40-60% ↓      |
| Training Speed | Baseline  | Faster      | 15-25% ↑      |
| Inference      | Baseline  | Faster      | 20-30% ↑      |
| Precision      | FP32 only | Mixed       | 2x batch size |

* **ONNX Export**: Deploy anywhere—cloud, edge, production.
* **Scalable**: Large batches on the same hardware.
* **Hybrid Precision**: FP16/FP32 auto on GPU, safe fallback on CPU.
* **Versatile**: Python & CLI—your workflow, your way.

---

## 🧩 Architecture Highlights

* **`ResnetEmbeddingsExtractor`**: Feature extraction from any ResNet backbone, optimized for GPU/CPU.
* **`MahalanobisDistance`**: Fast, ONNX-exportable anomaly scoring module.
* **`Padim`**: Fit, feature extraction, scoring, ONNX export, and inference—all-in-one.
* **`export_onnx`**: Seamless export for fast, portable inference.

---

## 🔗 References & Acknowledgments

* **PaDiM Paper**: [arxiv.org/abs/2011.08785](https://arxiv.org/abs/2011.08785)
* **TorchVision**: [pytorch.org/vision](https://pytorch.org/vision/)
* **Example Data**: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
* **Original Codebase**: [OpenAOI/anodet](https://github.com/OpenAOI/anodet)

*Special thanks to all original authors and contributors for their outstanding work.*

---

## 🤝 Contributing

1. **Fork** this repo
2. **Create** a feature branch
3. **Commit & push** your changes
4. **Open a Pull Request**—collaboration welcome!

---

## 📬 Contact

Questions? Feature requests?
**Deep Knowledge** – [Deepp.Knowledge@gmail.com](mailto:Deepp.Knowledge@gmail.com)

---

⭐ **If this project helps you, please star the repo and share it!** ⭐
