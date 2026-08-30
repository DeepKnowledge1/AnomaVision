# AnomaVision

<p align="center">
  <img src="docs/images/banner.png" width="100%" alt="AnomaVision banner"/>
</p>

<p align="center">
  <strong>Simple visual anomaly detection from normal images.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/anomavision/"><img src="https://img.shields.io/pypi/v/anomavision?label=PyPI&color=blue" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/anomavision/"><img src="https://img.shields.io/pypi/dm/anomavision?color=blue" alt="PyPI downloads"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10--3.12-blue" alt="Python 3.10 to 3.12"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-red" alt="PyTorch 2.0 or newer"/></a>
  <a href="https://onnx.ai/"><img src="https://img.shields.io/badge/ONNX-Export%20Ready-orange" alt="ONNX export ready"/></a>
  <a href="https://developer.nvidia.com/tensorrt"><img src="https://img.shields.io/badge/TensorRT-Supported-76b900" alt="TensorRT supported"/></a>
  <a href="https://docs.openvino.ai/"><img src="https://img.shields.io/badge/OpenVINO-Supported-0071C5" alt="OpenVINO supported"/></a>
  <a href="https://github.com/DeepKnowledge1/AnomaVision/actions/workflows/ci.yml"><img src="https://github.com/DeepKnowledge1/AnomaVision/actions/workflows/ci.yml/badge.svg" alt="CI status"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="MIT license"/></a>
  <a href="docs/kv260_xmodel.md"><img src="https://img.shields.io/badge/KV260-DPU-blue" alt="KV260 DPU support"/></a>
</p>

AnomaVision is a production-oriented computer vision toolkit for detecting **defects and unusual patterns** from normal images.

It supports three anomaly detection methods:

- **PaDiM** — a simple, fast feature-distribution baseline.
- **PatchCore** — a lightweight memory-based method designed for efficient inference.
- **EfficientAD** — a lightweight student-teacher method designed for fast industrial anomaly detection.

Training requires only **normal (`good`) images**. Labeled test images can then be used for evaluation, threshold calibration, and production model selection.

<p align="center">
  <a href="https://huggingface.co/spaces/DeepKnowledge1/mvtec-anomaly-detection"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-xl-dark.svg" alt="Open the AnomaVision live demo"/></a>
</p>

**New here?** Start with the [five-minute quickstart](docs/quickstart.md).

## What can AnomaVision do?

- Train anomaly detection models using normal images.
- Detect image-level anomalies and generate anomaly heatmaps.
- Evaluate anomaly detection and localization performance.
- Calibrate anomaly thresholds from validation data.
- Export models to **ONNX, OpenVINO, and TensorRT** where supported.
- Run production model selection with **Production Autopilot**.
- Export and compile **PaDiM and PatchCore to XModel for the AMD/Xilinx Kria KV260**.

## Quick start

### 1. Install

#### Option A — From Source

```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision

uv venv --python 3.11 .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

uv sync --extra cpu              # CPU
uv sync --extra cu121            # CUDA 12.1
```

#### Option B — From PyPI

```bash
uv pip install "anomavision[cpu]"

# NVIDIA GPU
uv pip install "anomavision[cu118]"
uv pip install "anomavision[cu121]"
uv pip install "anomavision[cu124]"
```

For other environments, see [Installation](docs/installation.md).

### 2. Prepare your dataset

Use an MVTec-style structure:

```text
dataset/
└── bottle/
    ├── ground_truth/
    │   ├── broken_large/
    │   ├── broken_small/
    │   └── contamination/
    ├── test/
    │   ├── broken_large/
    │   ├── broken_small/
    │   ├── contamination/
    │   └── good/
    └── train/
        └── good/
```

Training uses only `train/good`. Test images may contain defects.

### 3. Train

Create or edit `config.yml` and set `dataset_path` to your dataset.

Select the algorithm in the configuration:

```yaml
algorithm: padim       # padim | patchcore | efficientad
```

```bash
anomavision train --config config.yml
```


### 4. Detect

```bash
anomavision detect --config config.yml --img_path ./dataset/bottle/test
```

### 5. Export

```bash
anomavision export --config config.yml --format onnx
```

See [Export and deployment](docs/production_deployment.md) for deployment-specific options.

## Production Autopilot

Production Autopilot compares trained anomaly models on the **same validation data**, measures their performance and latency on the target device, calibrates thresholds, and selects the best candidate for deployment.

PaDiM, PatchCore, and EfficientAD can be supplied as independent candidate models. The model paths are provided directly through the CLI:

```bash
anomavision autopilot \
  --config config.yml \
  --padim_model ./distributions/padim/bottle/anomav_exp/model.pt \
  --patchcore_model ./distributions/patchcore/bottle/anomav_exp/model.pt \
  --efficientad_model ./distributions/efficientad/bottle/anomav_exp/model.pt \
  --device cpu \
  --validation_split 1.0 \
  --target_latency_ms 50 \
  --output_dir ./production_package
```

### How selection works

For every supplied model, Autopilot:

1. Evaluates the model on the validation split.
2. Calibrates an image-level anomaly threshold.
3. Measures inference latency on the selected device.
4. Calculates image-level and pixel-level metrics when localization maps are available.
5. Checks localization quality and false-positive behavior.
6. Applies the target latency constraint when selecting the production candidate.
7. Packages the selected model and writes a deployment manifest.

If multiple models satisfy the latency target, the model with the strongest image-level AUROC is preferred, with latency used as a tie-breaker.

### Output

Autopilot creates a production package containing:

```text
production_package/
├── model.pt
├── deployment_manifest.json
├── localization_report.md
└── production_autopilot_report.html
```

The HTML report is a self-contained dashboard showing the candidate comparison, selected model, AUROC, calibrated threshold, latency, localization diagnostics, and deployment recommendation.


## KV260 support

AnomaVision supports a **Vitis AI workflow for PaDiM and PatchCore on the AMD/Xilinx Kria KV260**.

```text
PyTorch → INT8 quantization → XModel → KV260 DPU compilation
```

Both PaDiM and PatchCore currently compile with **1 DPU subgraph** in the KV260 compiler.

See the complete [KV260 XModel Guide](docs/kv260_xmodel.md).

> XModel compilation has been validated in the Vitis AI environment. Final on-device KV260 validation requires the physical hardware.

## Documentation

| Topic | Guide |
|---|---|
| Quick start | [`docs/quickstart.md`](docs/quickstart.md) |
| Installation | [`docs/installation.md`](docs/installation.md) |
| CLI and configuration | [`docs/cli.md`](docs/cli.md), [`docs/config.md`](docs/config.md) |
| Python API | [`docs/api.md`](docs/api.md) |
| KV260 / XModel | [`docs/kv260_xmodel.md`](docs/kv260_xmodel.md) |
| Production deployment | [`docs/production_deployment.md`](docs/production_deployment.md) |
| Benchmarks | [`docs/benchmark.md`](docs/benchmark.md) |
| Troubleshooting | [`docs/troubleshooting.md`](docs/troubleshooting.md) |
| Examples | [`examples/README.md`](examples/README.md) |
| Contributing | [`docs/contributing.md`](docs/contributing.md) |

## Python example

```python
import torch
from torch.utils.data import DataLoader
import anomavision

train_set = anomavision.AnodetDataset("./dataset/bottle/train/good")
train_loader = DataLoader(train_set, batch_size=16, shuffle=False)

model = anomavision.Padim(backbone="resnet18", device=torch.device("cpu"))
model.fit(train_loader)

batch = next(iter(train_loader))
if isinstance(batch, (tuple, list)):
    batch = batch[0]

scores, maps = model.predict(batch)
```

## License

AnomaVision is released under the **MIT License**. See [`LICENSE`](LICENSE).

## Questions and contributions

Found a problem or have an idea? Feel free to open an issue or contribute to the project.
