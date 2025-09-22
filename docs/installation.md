
---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision
```

## 2. Install Dependencies

### Poetry (Recommended)

**CPU-only (default):**

```bash
poetry install
```

**GPU (choose one backend):**

```bash
poetry install --extras "cu118"   # CUDA 11.8
poetry install --extras "cu121"   # CUDA 12.1
poetry install --extras "cu124"   # CUDA 12.4
```

**Full install (CPU + GPU support):**

```bash
poetry install --extras "full"
```

Then activate:

```bash
poetry shell
```

---

### pip (Basic CPU-only)

```bash
pip install git+https://github.com/DeepKnowledge1/AnomaVision.git
```

### Development Mode

For contributors:

```bash
pip install -e .[dev]
```

---

## 3. Optional Backends

* **ONNX Runtime**

  ```bash
  pip install onnxruntime onnxruntime-tools
  ```

* **OpenVINO**

  ```bash
  pip install openvino
  ```

* **TensorRT** (requires NVIDIA setup)
  Follow [NVIDIA TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

* **Visualization (Matplotlib)**

  ```bash
  pip install matplotlib
  ```

---

## 4. Verify Installation

Run a quick test to confirm everything is available:

```bash
python train.py --help
python detect.py --help
python eval.py --help
python export.py --help
```

---
