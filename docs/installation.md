---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision
```

## 2. Install Dependencies

### uv (Recommended)

**Create and activate a virtual environment:**

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
```

**CPU-only:**

```bash
uv sync --extra cpu
```

**GPU (choose your CUDA version):**

```bash
uv sync --extra cu118   # CUDA 11.8
uv sync --extra cu121   # CUDA 12.1
uv sync --extra cu124   # CUDA 12.4
```

**Install the package in editable mode** (registers the `anomavision` CLI command):

```bash
uv pip install -e .
```

---

### pip (Basic CPU-only)

```bash
pip install anomavision
```

### Development Mode

For contributors:

```bash
uv pip install -e ".[dev]"
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
python -c "import anomavision, torch; print('✅ Ready —', torch.__version__)"

# Confirm the CLI is registered
anomavision --help
anomavision train --help
anomavision detect --help
anomavision eval --help
anomavision export --help
```

---
