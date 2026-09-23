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

It supports training, evaluation, threshold calibration, ONNX/OpenVINO/TensorRT export where supported, KV260/XModel deployment, and **production data-drift monitoring**.\n\nIt also includes **non-invasive deployment validation** for exported ONNX models, reporting model validity, tensor shapes, CPU inference latency, and available deployment backends without changing anomaly-detection logic.

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
| **Production data drift** | [docs/anomaly_detection_production_data_drift.md](docs/anomaly_detection_production_data_drift.md) |
| KV260 / XModel | [docs/kv260_xmodel.md](docs/kv260_xmodel.md) |
| Production deployment | [docs/production_deployment.md](docs/production_deployment.md) |
| Benchmarks | [docs/benchmark.md](docs/benchmark.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |

## License

AnomaVision is released under the **MIT License**. See [LICENSE](LICENSE).
