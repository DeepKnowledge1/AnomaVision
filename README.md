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

Production anomaly detection can fail silently when incoming images change even though the inference pipeline is still running. A camera, lighting setup, product variant, acquisition process, or other operating condition can change the feature distribution seen by a deployed model.

AnomaVision provides **production data-drift monitoring as an additive observer**. It compares model representations from incoming production data against a trusted reference population while leaving the anomaly detection algorithm and scoring path unchanged.

### Live customer dashboard

When `--enable-drift-monitoring` is used with `anomavision detect`, AnomaVision automatically starts the live production dashboard at:

```text
http://127.0.0.1:7860
```

The dashboard refreshes automatically and is designed for both operators and technical users. The main view shows:

- **Production status** — collecting data, stable, or data change detected.
- **Images checked** — how many production samples have been observed.
- **Recent monitoring window** — how much of the rolling window is populated.
- **Change level** — a normalized drift score shown as a simple percentage.
- **Recent production images** — representative source images for investigating what changed.
- **What changed?** — plain-language interpretation of distribution, mean, and variation changes.
- **What should I do?** — practical investigation steps for an operator.
- **Technical details** — PSI, mean shift, standard-deviation shift, cosine shift, and threshold for advanced users.

The dashboard is read-only. It does not modify model predictions, anomaly thresholds, localization, or hardware inference.

### 1. Generate a trusted reference embedding set

Use healthy/reference images from the same operating population used to train or validate the deployed model:

```bash
python -m anomavision.drift_reference \
  --img_path ./dataset/bottle/train/good \
  --model_data_path ./distributions \
  --algorithm patchcore \
  --class_name bottle \
  --run_name anomav_exp \
  --model model.pt \
  --device cpu \
  --batch_size 8 \
  --max_samples 500 \
  --output ./drift/reference_embeddings.npy
```

The command writes the embedding matrix and a JSON metadata sidecar. Reference embeddings should come from trusted, representative normal data; do not build the baseline from already-drifted production data.

### 2. Run detection with live monitoring

If `img_path`, algorithm, class name, run name, and other model settings are already defined in `config.yml`, the command can be:

```bash
python -m anomavision.cli detect \
  --config config.yml \
  --model model.pt \
  --enable-drift-monitoring \
  --drift-reference ./drift/reference_embeddings.npy
```

On Windows PowerShell:

```powershell
python -m anomavision.cli detect `
  --config config.yml `
  --model "model.pt" `
  --enable-drift-monitoring `
  --drift-reference ".\\drift\\reference_embeddings.npy"
```

With monitoring enabled, the CLI starts the dashboard automatically and continues with the normal anomaly detection pipeline. Open `http://127.0.0.1:7860` to view the live production health page.

### 3. Configure the monitoring window

Optional monitoring parameters are:

```bash
anomavision detect \
  --config config.yml \
  --model model.pt \
  --enable-drift-monitoring \
  --drift-reference ./drift/reference_embeddings.npy \
  --drift-window 500 \
  --drift-min-samples 100 \
  --drift-threshold 0.20 \
  --drift-evaluation-interval 25 \
  --drift-output ./drift/drift_status.json
```

The monitor keeps only a bounded rolling production window. Before enough samples arrive it reports `warming_up`; once ready it reports `stable` or `drift` and writes the latest machine-readable status to the configured JSON file.

### 4. Understand the dashboard

The dashboard intentionally avoids presenting raw ML metrics as the primary user experience.

For example, a production result such as:

```json
{
  "samples_seen": 900,
  "window_size": 500,
  "window_fill": 500,
  "ready": true,
  "status": "drift",
  "drift_score": 0.7909,
  "psi": 16.6859,
  "mean_shift": 0.2238,
  "std_shift": 0.8994,
  "cosine_shift": 0.0240,
  "threshold": 0.20,
  "warnings": ["feature_distribution_shift"]
}
```

is presented to an operator as **Data Change Detected — recent production images look different from the approved normal pattern**, together with recent images and recommended investigation steps. Advanced users can expand **Technical details** to inspect the underlying metrics.

A drift alert is an **early-warning signal**. It does not by itself prove that model accuracy has degraded or that any particular image is defective.

### 5. Production monitoring architecture

```text
                 Trusted normal data
                         │
                         ▼
              Reference embeddings
                         │
                         │
Production images ──► Normal inference ──► Anomaly result
                         │
                         └───────────────► Drift observer
                                                │
                                                ▼
                                       Rolling comparison
                                                │
                                  ┌─────────────┴─────────────┐
                                  ▼                           ▼
                              Stable                    Drift detected
                                  │                           │
                                  └─────────────┬─────────────┘
                                                ▼
                                      Live customer dashboard
```

For supported PyTorch representations such as PatchCore, the production monitor reuses the representation captured during normal inference rather than running the backbone a second time. If a backend does not expose a drift representation, monitoring is skipped safely and anomaly inference continues.

The existing anomaly algorithms and hardware paths remain independent of the monitoring observer. Enabling drift monitoring does not alter PatchCore/PaDiM scoring, EfficientAD inference, localization, Hailo inference, or KV260/XModel execution.

### 6. Machine-readable status

The monitor writes a JSON status file, by default:

```text
./drift/drift_status.json
```

This file is intended for integration with external dashboards, alerting systems, APIs, or deployment infrastructure. The customer dashboard consumes the same status information.

### 7. Recommended operational workflow

1. Build the reference from trusted normal data.
2. Deploy the anomaly model normally.
3. Enable drift monitoring as an observer.
4. Let the rolling window collect enough production samples.
5. When drift is detected, inspect the recent production images.
6. Check whether the change is expected, such as a new product, camera, lighting, or process condition.
7. Check model quality using appropriate production labels or quality measurements.
8. Only then decide whether the reference data or model needs to be updated.

For the complete architecture, metrics, lifecycle, dashboard behavior, and troubleshooting guidance, see [`docs/anomaly_detection_production_data_drift.md`](docs/anomaly_detection_production_data_drift.md).

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