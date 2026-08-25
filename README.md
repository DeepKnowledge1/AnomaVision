# AnomaVision

<p align="center">
  <img src="docs/images/banner.png" width="100%" alt="AnomaVision banner"/>
</p>

<p align="center">
  <strong>Simple visual anomaly detection from normal images.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/anomavision/"><img src="https://img.shields.io/pypi/v/anomavision?label=PyPI" alt="PyPI version"/></a>
  <a href="https://github.com/DeepKnowledge1/AnomaVision/actions/workflows/ci.yml"><img src="https://github.com/DeepKnowledge1/AnomaVision/actions/workflows/ci.yml/badge.svg" alt="CI status"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="MIT license"/></a>
  <a href="docs/kv260_xmodel.md"><img src="https://img.shields.io/badge/KV260-DPU-blue" alt="KV260 DPU support"/></a>
</p>

AnomaVision is a computer vision project for finding **defects and unusual patterns** in images.

It supports two anomaly detection methods:

- **PaDiM** — a simple and fast baseline.
- **PatchCore** — a lightweight memory-based method.

You only need **normal (`good`) images** to train the anomaly detector.

<p align="center">
  <a href="https://huggingface.co/spaces/DeepKnowledge1/mvtec-anomaly-detection"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-xl-dark.svg" alt="Open the AnomaVision live demo"/></a>
</p>

**New here?** Start with the [five-minute quickstart](docs/quickstart.md).

## What can AnomaVision do?

- Train anomaly detection models using normal images.
- Detect image-level anomalies.
- Create anomaly heatmaps showing where the problem is.
- Export models to **ONNX, OpenVINO, and TensorRT**.
- Export and compile **PaDiM and PatchCore to XModel for the AMD/Xilinx Kria KV260**.

## Quick start

### 1. Install

```bash
pip install uv
uv pip install "anomavision[cpu]"
```

For other environments, see [Installation](docs/installation.md).

### 2. Prepare your images

Use a simple MVTec-style folder structure:

```text
dataset/
└── bottle/
    ├── train/
    │   └── good/
    └── test/
        ├── good/
        └── scratch/
```

Training uses the **good** images. Test images can contain defects.

### 3. Train

Create or edit `config.yml` and point `dataset_path` to your dataset.

Then run:

```bash
anomavision train --config config.yml
```

PaDiM is the default model. For PatchCore, set `algorithm: patchcore` in the configuration.

### 4. Detect

```bash
anomavision detect --config config.yml --img_path ./dataset/bottle/test
```

### 5. Export

For a portable model, ONNX is a good place to start:

```bash
anomavision export --config config.yml --format onnx
```

For more export options, see [Export and deployment](docs/production_deployment.md).

## KV260 support

AnomaVision also supports a **Vitis AI workflow for PaDiM and PatchCore on the AMD/Xilinx Kria KV260**.

The workflow is:

```text
PyTorch → INT8 quantization → XModel → KV260 DPU compilation
```

Both PaDiM and PatchCore currently compile with **1 DPU subgraph** in the KV260 compiler.

The complete setup and commands are in:

**[KV260 XModel Guide](docs/kv260_xmodel.md)**

> XModel compilation has been validated in the Vitis AI environment. Final on-device KV260 validation requires the physical hardware.

## Which model should I try?

| Model | Good starting point |
|---|---|
| **PaDiM** | Simple and fast baseline |
| **PatchCore** | Lightweight memory-based anomaly detection |

If you are new to anomaly detection, **start with PaDiM**.

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

You can also use AnomaVision directly from Python:

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
