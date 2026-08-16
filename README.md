# AnomaVision

AnomaVision is a production-oriented library for **visual anomaly detection from normal images**. It supports PaDiM and lightweight PatchCore, image-level scores, pixel-level maps, and deployment exports.

[![PyPI](https://img.shields.io/pypi/v/anomavision?label=PyPI)](https://pypi.org/project/anomavision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Why use it?

- Train with normal images only; anomaly labels are not required for training.
- Use **PaDiM** for the default fast baseline or **PatchCore** for a compact nearest-neighbor memory bank.
- Run inference through PyTorch, ONNX Runtime, OpenVINO, or native TensorRT.
- Export FP16 or calibrated INT8 TensorRT engines for NVIDIA production deployments.

Benchmark results and reproduction details are documented in [`docs/benchmark.md`](docs/benchmark.md). Treat benchmark numbers as workload-specific, and reproduce them on your own hardware before making production claims.

## Quickstart

### 1. Install

```bash
pip install uv
uv pip install "anomavision[cpu]"
```

For NVIDIA GPUs, choose the matching extra such as `anomavision[cu121]`. Source installation and environment setup are described in [`docs/installation.md`](docs/installation.md).

### 2. Prepare data

Use an MVTec-style directory. Training uses only the `good` images:

```text
dataset/
└── bottle/
    ├── train/good/
    └── test/
        ├── good/
        └── scratch/
```

### 3. Train

```bash
anomavision train --config config.yml
```

The default configuration trains PaDiM. To train lightweight PatchCore, set `algorithm: patchcore` in `config.yml` or pass the corresponding CLI option. The model and compact deployment artifact are saved under `model_data_path`.

### 4. Detect and evaluate

```bash
anomavision detect --config config.yml --img_path ./dataset/bottle/test
anomavision eval --config config.yml
```

### 5. Export

```bash
# Portable ONNX export
anomavision export --config config.yml --format onnx

# Native TensorRT FP16 export
anomavision export --config config.yml --format tensorrt \
  --device cuda --tensorrt-precision fp16

# Native TensorRT calibrated INT8 export
anomavision export --config config.yml --format tensorrt \
  --device cuda --tensorrt-precision int8 \
  --calib-dir ./dataset/bottle/train/good --calib-samples 100
```

Every command provides help:

```bash
anomavision --help
anomavision train --help
anomavision export --help
```

## Choosing a model

| Model | Best starting point | Memory use | Production note |
|---|---|---:|---|
| PaDiM | Fast, simple baseline | Low | Recommended first experiment |
| Lightweight PatchCore | Higher-fidelity patch retrieval with bounded memory | Configurable | Use `coreset_ratio` and `max_memory_patches` to control latency |

## Documentation

| Topic | Guide |
|---|---|
| Installation | [`docs/installation.md`](docs/installation.md) |
| Five-minute workflow | [`docs/quickstart.md`](docs/quickstart.md) |
| CLI and configuration | [`docs/cli.md`](docs/cli.md), [`docs/config.md`](docs/config.md) |
| Python API | [`docs/api.md`](docs/api.md) |
| PatchCore and TensorRT deployment | [`docs/production_deployment.md`](docs/production_deployment.md) |
| Benchmark methodology | [`docs/benchmark.md`](docs/benchmark.md) |
| Troubleshooting | [`docs/troubleshooting.md`](docs/troubleshooting.md) |
| Contributing | [`docs/contributing.md`](docs/contributing.md) |

## Python API

```python
import torch
from torch.utils.data import DataLoader
import anomavision

train_set = anomavision.AnodetDataset("./dataset/bottle/train/good")
train_loader = DataLoader(train_set, batch_size=16, shuffle=False)

model = anomavision.Padim(backbone="resnet18", device=torch.device("cpu"))
model.fit(train_loader)
scores, maps = model.predict(batch)
```

## Community and adoption

The most useful path to adoption is a small, reproducible example rather than more README text: publish one benchmark script, one production export example, a model card with hardware and preprocessing details, and a short comparison against Anomalib. Invite users to reproduce the result, report failures, and contribute adapters for their own datasets. See [`docs/production_deployment.md`](docs/production_deployment.md) for the project’s recommended release checklist.

## License

AnomaVision is released under the MIT License. See [`LICENSE`](LICENSE).
