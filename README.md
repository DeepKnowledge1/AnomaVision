# AnomaVision

<p align="center">
  <img src="docs/images/banner.png" width="100%" alt="AnomaVision banner"/>
</p>

<p align="center">
  <strong>Production-oriented anomaly detection and industrial computer vision.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/anomavision/"><img src="https://img.shields.io/pypi/v/anomavision?label=PyPI&color=blue" alt="PyPI version"/></a>
  <a href="https://img.shields.io/badge/Python-3.10--3.12-blue"><img src="https://img.shields.io/badge/Python-3.10--3.12-blue" alt="Python 3.10 to 3.12"/></a>
  <a href="https://github.com/DeepKnowledge1/AnomaVision/actions/workflows/ci.yml"><img src="https://github.com/DeepKnowledge1/AnomaVision/actions/workflows/ci.yml/badge.svg" alt="CI status"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="MIT license"/></a>
</p>

AnomaVision is an open-source computer vision toolkit for detecting **defects and unusual patterns** from normal images. It is designed to support the full journey from experimentation to edge and industrial deployment.

## Highlights

- Train using only normal (`good`) images.
- Detect image-level anomalies and generate anomaly heatmaps.
- Evaluate models and calibrate anomaly thresholds.
- Choose between **PaDiM, PatchCore, and EfficientAD**.
- Run from files, cameras, video, MQTT, TCP, and other streaming sources.
- Export to **ONNX, OpenVINO, and TensorRT** where supported.
- Deploy PaDiM and PatchCore on the **AMD/Xilinx Kria KV260**.
- Connect detection results to industrial systems through optional **Industrial Actions**.

**New here?** Start with the [five-minute quickstart](docs/quickstart.md).

## Quick start

### 1. Install

```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision
uv venv --python 3.11 .venv
```

Activate the environment and install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
uv sync --extra cpu
```

For other environments, see [Installation](docs/installation.md).

### 2. Prepare your dataset

Use an MVTec-style structure:

```text
dataset/
└── bottle/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   └── defect_type/
    └── ground_truth/
```

Training uses only `train/good`. Test images may contain defects.

### 3. Train

Select an algorithm in `config.yml`:

```yaml
algorithm: padim  # padim | patchcore | efficientad
```

Then run:

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

See [Production deployment](docs/production_deployment.md) for deployment options.

## Industrial Actions

Industrial Actions allow AnomaVision to send the **existing detection result** to external systems after inference.

```text
Image or Stream
      ↓
AnomaVision inference
      ↓
Existing anomaly decision
      ↓
Industrial Action
 ├── MQTT
 ├── OPC UA
 └── Evidence storage
```

Industrial Actions do **not** introduce a second inference or decision pipeline. AnomaVision continues to use its existing anomaly score and classification logic, while Actions consume the resulting event.

### Disabled by default

Industrial Actions are explicitly disabled in the default configuration:

```yaml
actions_enabled: false
actions_fail_fast: false
actions: []
```

This means existing AnomaVision workflows work exactly as before. No MQTT, OPC UA, or external connection is attempted unless you explicitly enable Actions.

### Enable MQTT

For example:

```yaml
actions_enabled: true
actions_fail_fast: false

actions:
  - type: mqtt
    broker: localhost
    port: 1883
    topic: factory/anomavision/results
```

When MQTT is unavailable and `actions_fail_fast` is `false`, AnomaVision logs the action failure and continues inference. This is useful for development and non-critical integrations.

Set `actions_fail_fast: true` only when an external action is required for your deployment and an action failure should stop execution.

> **Note:** Enabling MQTT requires an MQTT broker to be running and reachable at the configured address.

See [`docs/config.md`](docs/config.md) for configuration details and action-specific options.

## Production Autopilot

Production Autopilot compares candidate models on the same validation data, measures performance and latency, calibrates thresholds, and selects a suitable candidate for deployment.

```bash
anomavision autopilot \
  --config config.yml \
  --padim_model ./padim/model.pt \
  --patchcore_model ./patchcore/model.pt \
  --efficientad_model ./efficientad/model.pt \
  --device cpu \
  --target_latency_ms 50 \
  --output_dir ./production_package
```

The generated production package can include the selected model, deployment manifest, localization report, and an HTML comparison report.

## KV260 support

AnomaVision supports a Vitis AI workflow for PaDiM and PatchCore on the AMD/Xilinx Kria KV260:

```text
PyTorch → INT8 quantization → XModel → KV260 DPU compilation
```

See the complete [KV260 XModel Guide](docs/kv260_xmodel.md).

## Documentation

| Topic | Guide |
|---|---|
| Quick start | [`docs/quickstart.md`](docs/quickstart.md) |
| Installation | [`docs/installation.md`](docs/installation.md) |
| CLI and configuration | [`docs/cli.md`](docs/cli.md), [`docs/config.md`](docs/config.md) |
| Python API | [`docs/api.md`](docs/api.md) |
| Production deployment | [`docs/production_deployment.md`](docs/production_deployment.md) |
| KV260 / XModel | [`docs/kv260_xmodel.md`](docs/kv260_xmodel.md) |
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

model = anomavision.Padim(
    backbone="resnet18",
    device=torch.device("cpu"),
)
model.fit(train_loader)

batch = next(iter(train_loader))
if isinstance(batch, (tuple, list)):
    batch = batch[0]

scores, maps = model.predict(batch)
```

## Questions and contributions

Found a problem or have an idea? Feel free to open an issue or contribute to the project. AnomaVision welcomes contributions around industrial computer vision, edge deployment, data pipelines, integrations, and anomaly detection.

## License

AnomaVision is released under the **MIT License**. See [`LICENSE`](LICENSE).
