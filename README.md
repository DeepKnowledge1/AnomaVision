<img src="docs/images/banner.png" width="100%" alt="AnomaVision banner"/>

# 🚀 AnomaVision: Edge-Ready Visual Anomaly Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org)
[![ONNX Ready](https://img.shields.io/badge/ONNX-Export%20Ready-orange.svg)](https://onnx.ai/)
[![OpenVINO Ready](https://img.shields.io/badge/OpenVINO-Ready-blue.svg)](https://docs.openvino.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)


**🔥 Lightweight, fast, and production-ready anomaly detection — powered by PaDiM.**
*Deploy anywhere: edge devices, servers, or the cloud.*


---

## ✨ Features

* 🎯 **State-of-the-art PaDiM** algorithm
* ⚡ **3× faster inference** than Anomalib (CPU benchmarks)
* 📦 **Multi-backend exports**: PyTorch, ONNX, TorchScript, OpenVINO, TorchRT
* 🎨 **Visualizations**: anomaly heatmaps, bounding boxes, ROC curves
* 🖥️ **Unified Python + CLI workflows**
* 🌍 **Edge-first design** with compact `.pth` models

---

## 🆚 Why AnomaVision over Anomalib?

* ⚡ **3× faster inference** on CPU (MVTec & Visa benchmarks)
* 📦 **Smaller models** (30 MB vs 40 MB) with **lower memory usage**
* 🎯 **Higher AUROC** across most classes on **both MVTec AD and Visa datasets**
* 🌍 **Edge-first design** → optimized for ONNX, TorchScript, and OpenVINO
* 🛡️ **Production-ready** with clean API, CLI, and deployment options

👉 See detailed results in [Benchmarks](docs/benchmarks.md).

👉 [Download: AnomaVision vs Anomalib — A Comprehensive Performance Analysis (PDF)](docs/AnomaVision_vs_Anomalib.pdf)



---

## 📚 Documentation

📖 Full docs are available in the [`/docs`](docs/index.md) folder.

* [Installation](docs/installation.md)
* [Quick Start](docs/quickstart.md)
* [CLI Reference](docs/cli.md)
* [API Reference](docs/api.md)
* [Configuration Guide](docs/config.md)
* [Benchmarks](docs/benchmark.md)
* [Contributing](CONTRIBUTING.md)

---

## ⚡ Quick Example

Train with a config file:

```bash
python train.py --config config.yml

## Save:
# Full model → padim_model.pt
# Compact stats-only model → padim_model.pth
# Config snapshot → config.yml

```

Run detection:

```bash
python detect.py --config config.yml
```

Evaluate performance:

```bash
python eval.py --config config.yml
```

Export to ONNX:

```bash
python export.py --config export_config.yml
```

➡ For more examples, see [Quick Start](docs/quickstart.md).

---

## 📊 Benchmarks (Summary)

### MVTec AD (15 classes)

* **Image AUROC**: AV 0.85 ↑ vs AL 0.81
* **Pixel AUROC**: AV 0.96 ↑ vs AL 0.94
* **FPS**: AV 43 ↑ vs AL 13

### Visa (12 classes)

* **Image AUROC**: AV 0.81 ↑ vs AL 0.78
* **Pixel AUROC**: AV 0.96 ↑ vs AL 0.95
* **FPS**: AV 45 ↑ vs AL 13

📊 Full tables & plots → [Benchmarks](docs/benchmarks.md)

---

## 🤝 Contributing

We welcome contributions!

* See [Contributing Guide](CONTRIBUTING.md) for high-level steps

---

## 📜 Citation

If you use AnomaVision in your research, please cite:

```bibtex
@software{anomavision2025,
  title={AnomaVision: Edge-Ready Visual Anomaly Detection},
  author={DeepKnowledge Contributors},
  year={2025},
  url={https://github.com/DeepKnowledge1/AnomaVision},
}
```

---

## 💬 Community & Support

* 📢 [Discussions](https://github.com/DeepKnowledge1/AnomaVision/discussions)
* 🐛 [Issues](https://github.com/DeepKnowledge1/AnomaVision/issues)
* 📧 [deepp.knowledge@gmail.com](mailto:deepp.knowledge@gmail.com)

---

👉 Start with [Quick Start](docs/quickstart.md) and build your first anomaly detection pipeline in **5 minutes**!

---
