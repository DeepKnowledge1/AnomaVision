# AnomaVision

<p align="center">
  <img src="docs/images/banner.png" width="100%" alt="AnomaVision banner"/>
</p>

<p align="center"><strong>Simple visual anomaly detection from normal images.</strong></p>

AnomaVision is a production-oriented computer vision toolkit for detecting **defects and unusual patterns** from normal images.

It supports three anomaly detection methods:

- **PaDiM** — a simple, fast feature-distribution baseline.
- **PatchCore** — a lightweight memory-based method designed for efficient inference.
- **EfficientAD** — a lightweight student-teacher method designed for fast industrial anomaly detection.

Training requires only **normal (`good`) images**. Labeled test images can then be used for evaluation, threshold calibration, and production model selection.

## What can AnomaVision do?

- Train anomaly detection models using normal images.
- Detect image-level anomalies and generate anomaly heatmaps.
- Evaluate anomaly detection and localization performance.
- Calibrate anomaly thresholds from validation data.
- Export models to **ONNX, OpenVINO, and TensorRT** where supported.
- Run production model selection with **Production Autopilot**.
- Export and compile **PaDiM and PatchCore to XModel for the AMD/Xilinx Kria KV260**.
- Monitor production feature distributions for **data drift** without changing anomaly scores.
- Launch a **live, customer-facing production health dashboard** while drift monitoring is enabled.

## Quick start

### 1. Install

```bash
git clone https://github.com/DeepKnowledge1/AnomaVision.git
cd AnomaVision
uv venv --python 3.11 .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\Activate.ps1
uv sync --extra cpu
```

### 2. Prepare your dataset

Use an MVTec-style structure with normal images under `train/good` and test images under `test`.

### 3. Train

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

## Anomaly Detection: Production Data Drift

Production images can change over time even while the anomaly detection pipeline continues to run. AnomaVision can monitor these changes by comparing production model representations with a trusted reference distribution, without changing the existing anomaly detection path.

With drift monitoring enabled, AnomaVision can also start a **live production health dashboard** showing the current monitoring status, recent production images, and a simple explanation of detected changes for operators.

Example:

```powershell
python -m anomavision.cli detect `
  --config config.yml `
  --model "model.pt" `
  --enable-drift-monitoring `
  --drift-reference ".\\drift\\reference_embeddings.npy"
```

The dashboard is available at `http://127.0.0.1:7860` while the monitoring process is running.

> **Data drift is an early-warning signal.** It indicates that production data has changed relative to the reference data; it does not by itself prove that model accuracy has degraded or that an image is defective.

For the complete implementation and usage details, see **[`docs/anomaly_detection_production_data_drift.md`](docs/anomaly_detection_production_data_drift.md)**.

## Production Autopilot

See `docs/production_deployment.md` for deployment-specific options.

## KV260 support

AnomaVision supports a **Vitis AI workflow for PaDiM and PatchCore on the AMD/Xilinx Kria KV260**. XModel compilation has been validated in the Vitis AI environment; final on-device validation requires the physical hardware.

## Documentation

| Topic | Guide |
|---|---|
| Quick start | [`docs/quickstart.md`](docs/quickstart.md) |
| Installation | [`docs/installation.md`](docs/installation.md) |
| CLI and configuration | [`docs/cli.md`](docs/cli.md), [`docs/config.md`](docs/config.md) |
| Python API | [`docs/api.md`](docs/api.md) |
| **Production data drift** | [`docs/anomaly_detection_production_data_drift.md`](docs/anomaly_detection_production_data_drift.md) |
| KV260 / XModel | [`docs/kv260_xmodel.md`](docs/kv260_xmodel.md) |
| Production deployment | [`docs/production_deployment.md`](docs/production_deployment.md) |
| Benchmarks | [`docs/benchmark.md`](docs/benchmark.md) |
| Troubleshooting | [`docs/troubleshooting.md`](docs/troubleshooting.md) |
| Examples | [`examples/README.md`](examples/README.md) |
| Contributing | [`docs/contributing.md`](docs/contributing.md) |

## License

AnomaVision is released under the **MIT License**. See [`LICENSE`](LICENSE).
