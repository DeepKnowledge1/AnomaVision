# AnomaVision

<p align="center">
  <img src="docs/images/banner.png" width="100%" alt="AnomaVision banner"/>
</p>

<p align="center"><strong>Simple visual anomaly detection from normal images.</strong></p>

AnomaVision is a production-oriented computer vision toolkit for detecting **defects and unusual patterns** from normal images.

Supported methods:
- **PaDiM**
- **PatchCore**
- **EfficientAD**

It supports training, evaluation, threshold calibration, ONNX/OpenVINO/TensorRT export where supported, KV260/XModel deployment, **production data-drift monitoring**, and **non-invasive deployment validation**.

## Quick start

### Install

```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision
uv venv --python 3.11 .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\Activate.ps1
uv sync --extra cpu
```

### Train

```bash
anomavision train --config config.yml
```

### Detect

```bash
anomavision detect --config config.yml --img_path ./dataset/bottle/test
```

### Export

```bash
anomavision export --config config.yml --format onnx
```

---

## Deployment validation

Before moving an exported model into production, AnomaVision can validate the deployment artifact without changing the underlying anomaly-detection algorithm.

It checks model integrity, inference, performance, supported backend availability, and—when a reference model is supplied—output consistency.

### Quick example

```powershell
anomavision validate `
  --model distributions\\padim\\bottle\\anomav_exp\\model.onnx `
  --reference-model distributions\\padim\\bottle\\anomav_exp\\model.pt `
  --config config.yml `
  --runs 20
```

Typical output includes:

```text
Output consistency
  Score max abs diff:  ...
  Score mean abs diff: ...
  Map max abs diff:    ...
  Map mean abs diff:   ...
  Tolerance:           0.0001

Backend compatibility
  pytorch: available
  torchscript: available
  onnxruntime: available
  openvino: available
  tensorrt: not installed
  hailo: not installed
  vitis_ai_vart: available
  vitis_ai_xir: available
  vitis_ai_library: not installed
  kv260: available
```

Supported deployment artifacts include PyTorch, TorchScript, ONNX, TensorRT, OpenVINO, Hailo HEF, and Vitis AI/KV260 XModel.

**Note:** backend availability means the required runtime components are installed. It does not by itself prove that a specific model has been tested on physical target hardware.

See the [Deployment Validation guide](docs/deployment_validation.md) for all options, consistency validation, JSON output, backend details, and recommended production usage.

## Production data drift

Production data-drift monitoring is available as an optional observer around anomaly detection.

See the [Production Data Drift guide](docs/anomaly_detection_production_data_drift.md) for setup, reference generation, metrics, dashboard, and troubleshooting.

---

## KV260 support

AnomaVision supports a **Vitis AI workflow for PaDiM and PatchCore on the AMD/Xilinx Kria KV260**. XModel compilation has been validated in the Vitis AI environment; final on-device validation requires the physical hardware.

## Documentation

| Topic | Guide |
|---|---|
| Quick start | [docs/quickstart.md](docs/quickstart.md) |
| Installation | [docs/installation.md](docs/installation.md) |
| CLI / configuration | [docs/cli.md](docs/cli.md), [docs/config.md](docs/config.md) |
| Python API | [docs/api.md](docs/api.md) |
| **Deployment validation** | [docs/deployment_validation.md](docs/deployment_validation.md) |
| **Production data drift** | [docs/anomaly_detection_production_data_drift.md](docs/anomaly_detection_production_data_drift.md) |
| KV260 / XModel | [docs/kv260_xmodel.md](docs/kv260_xmodel.md) |
| Production deployment | [docs/production_deployment.md](docs/production_deployment.md) |
| Benchmarks | [docs/benchmark.md](docs/benchmark.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |

## License

AnomaVision is released under the **MIT License**. See [LICENSE](LICENSE).
