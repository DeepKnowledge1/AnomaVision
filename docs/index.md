
# 📖 AnomaVision Documentation

Welcome to the official documentation for **AnomaVision** —
an **edge-ready visual anomaly detection framework** powered by **PaDiM**.

AnomaVision is designed for **fast, lightweight, and accurate anomaly detection** across **manufacturing, inspection, and industrial applications**.

---

## ✨ Key Features

* ⚡ **3× faster** inference than Anomalib, with **smaller models**
* 🖥️ **Multi-backend support**: PyTorch, ONNX, TorchScript, OpenVINO
* 📦 **Production-ready CLI and API**
* 🎨 **Rich visualization tools** for debugging and monitoring
* 🚀 **Edge-first design** for deployment on constrained devices

---

## 📚 Documentation Index

* [Installation](installation.md) → How to install AnomaVision
* [Quick Start](quickstart.md) → Train, detect, evaluate, and export in minutes
* [CLI Reference](cli.md) → Full command-line options for all scripts
* [API Reference](api.md) → Use AnomaVision directly in Python
* [Configuration Guide](config.md) → Explanation of all `config.yml` fields
* [Benchmarks](benchmark.md) → MVTec & Visa results vs Anomalib
* [Troubleshooting & FAQ](troubleshooting.md) → Common issues and fixes
* [Contributing](contributing.md) → How to contribute to AnomaVision

---

## 🚀 Get Started

Install with **Poetry (recommended)**:

```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision
poetry install --extras "full"
poetry shell
```

Train your first model:

```bash
python train.py --config config.yml
```

Run inference:

```bash
python detect.py --config detect_config.yml
```

---

## 📊 Benchmarks at a Glance

* **MVTec AD (15 classes)**

  * AV Image AUROC ↑ **0.85** vs AL 0.81
  * AV Pixel AUROC ↑ **0.96** vs AL 0.94
  * AV FPS ↑ **43.4** vs AL 13.0

* **Visa (12 classes)**

  * AV Image AUROC ↑ **0.81** vs AL 0.78
  * AV Pixel AUROC ↑ **0.96** vs AL 0.95
  * AV FPS ↑ **44.7** vs AL 13.5

✅ AnomaVision = **Faster, smaller, more accurate**

---

## 📜 License & Citation

Released under the **MIT License**.

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

👉 Start with [Quick Start](quickstart.md) and build your first anomaly detection pipeline in **5 minutes**!
