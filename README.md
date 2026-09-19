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

It supports training, evaluation, threshold calibration, ONNX/OpenVINO/TensorRT export where supported, KV260/XModel deployment, and **production data-drift monitoring**.

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

# Production Data Drift Monitoring

AnomaVision can detect when production images become different from a trusted normal population.

**The anomaly detector keeps running normally. Drift monitoring is an additional observer.**

### Workflow

```text
Trusted normal images
        ↓
Generate reference .npy
        ↓
Run detection + drift monitoring
        ↓
Live dashboard + drift_status.json
```

## 1. Generate a reference

Use **normal/good images** that represent an approved production condition.

### Windows PowerShell — PatchCore + PyTorch

```powershell
python -m anomavision.drift_reference `
  --config config.yml `
  --img_path "D:\\01-DATA\\VisA_pytorch\\candle\\train\\good" `
  --model "model.pt" `
  --model_data_path ".\\distributions" `
  --algorithm patchcore `
  --class_name candle `
  --run_name anomav_exp `
  --device cpu `
  --batch_size 8 `
  --max_samples 500 `
  --output ".\\drift\\reference_embeddings_patchcore.npy"
```

Verify it:

```powershell
python -c "import numpy as np; x=np.load('./drift/reference_embeddings_patchcore.npy'); print(x.shape)"
```

Example:

```text
(500, 15)
```

The exact feature dimension depends on the model and representation.

## 2. Start detection with drift monitoring

Use the **same algorithm/model representation** used to create the reference.

### PatchCore + ONNX

```powershell
python -m anomavision.cli detect `
  --config config.yml `
  --model "model.onnx" `
  --enable-drift-monitoring `
  --algorithm patchcore `
  --batch_size 1 `
  --drift-reference ".\\drift\\reference_embeddings_patchcore.npy"
```

### PatchCore + PyTorch

```powershell
python -m anomavision.cli detect `
  --config config.yml `
  --model "model.pt" `
  --enable-drift-monitoring `
  --algorithm patchcore `
  --drift-reference ".\\drift\\reference_embeddings_patchcore.npy"
```

## Important: reference must match the model

Do **not** mix references between algorithms or models.

For example:

```text
PaDiM reference      → PaDiM production model
PatchCore reference  → PatchCore production model
```

If the reference is `(500, 64)` but production embeddings are `(1, 15)`, monitoring will be skipped because the dimensions do not match.

Keep separate files when needed:

```text
drift/
├── reference_embeddings_padim.npy
└── reference_embeddings_patchcore.npy
```

## 3. Open the dashboard

When detection starts with drift monitoring:

```text
http://127.0.0.1:7860/anomavision_dashboard.html
```

The dashboard shows:
- production monitoring status
- samples and rolling window
- drift level
- recent production images
- technical metrics

The machine-readable status is written to:

```text
./drift/drift_status.json
```

## What does drift mean?

Drift means that recent production data is different from the trusted reference data.

It **does not automatically mean**:
- an image is defective
- model accuracy has decreased
- the model needs retraining

Use the dashboard and recent images to investigate the cause.

## Useful options

```text
--enable-drift-monitoring       Enable monitoring
--drift-reference <file>       Trusted .npy reference
--drift-window 500              Rolling production window
--drift-min-samples 100        Samples before evaluation
--drift-threshold 0.20         PSI drift threshold
--drift-evaluation-interval 25 Evaluation frequency
--drift-output <file>          JSON status output
```

Full technical documentation:

**[Production Data Drift](docs/anomaly_detection_production_data_drift.md)**

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
