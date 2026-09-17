# Anomaly Detection: Production Data Drift


<p align="center">
  <img src="images/Production_Data_Drift.png" width="50%" alt="Production Data Drift"/>
</p>

## Overview

AnomaVision can monitor whether production images have changed relative to a trusted reference population while leaving the existing anomaly detection path unchanged.

Data drift can be caused by a new camera or lens, different illumination, camera position or focus changes, a new product variant, manufacturing-process changes, image acquisition or preprocessing changes, environmental changes, or a production population that differs from the approved reference population.

> **Important:** data drift is an early-warning signal. It does not by itself mean that an image is defective or that model accuracy has degraded. It means that the production population should be investigated.

---

## Architecture

Drift monitoring is an additive observer around the existing anomaly detector:

```text
                         ┌──────────────► Anomaly detection
                         │                score / localization
Production image ────────┤
                         │
                         └──────────────► Drift monitoring
                                          reference vs production
                                                   │
                                                   ▼
                                            Production health
                                                   │
                                                   ▼
                                          Live customer dashboard
```

For supported PyTorch models, the representation captured during normal inference can be reused by the drift observer, avoiding an unnecessary second backbone pass where possible.

Existing PatchCore, PaDiM, EfficientAD, Hailo, KV260/XModel, ONNX, TensorRT, and OpenVINO inference paths are not replaced by drift monitoring.

---

# End-to-end workflow

There are two main steps:

1. Generate a trusted `.npy` reference embedding file.
2. Start normal detection with drift monitoring enabled.

```text
Trusted normal images
        │
        ▼
Generate reference_embeddings.npy
        │
        ▼
Run normal AnomaVision detection
        │
        ├────────► anomaly results
        │
        └────────► drift monitoring
                         │
                         ▼
                  drift_status.json
                         │
                         ▼
                Live customer dashboard
```

---

# 1. Generate the reference `.npy` file

The `.npy` file is the **trusted reference population** used by the production drift monitor.

It is not a copy of the input images. It contains the model representations extracted from trusted normal images.

```text
Trusted normal images
        │
        ▼
     model.pt
        │
        ▼
model representation
        │
        ▼
reference_embeddings.npy
```

## What images should be used?

Use normal/good images representing the production conditions that you want to approve as the baseline.

For an MVTec/VisA-style dataset, a typical source is:

```text
D:\01-DATA\VisA_pytorch\candle\train\good
```

or:

```text
./dataset/candle/train/good
```

The reference should come from a trusted operating condition. For example, if the camera, lighting, product, and preprocessing were known to be correct during a validation period, that population is a suitable candidate.

Avoid creating the reference from a population that is already known to contain an unwanted production change.

---

## Model data required by reference generation

The reference builder needs the same model information used by the anomaly detector.

Typical trained-model structure:

```text
<model_data_path>/
└── <algorithm>/
    └── <class_name>/
        └── <run_name>/
            └── model.pt
```

For example:

```text
./distributions/
└── patchcore/
    └── bottle/
        └── anomav_exp/
            └── model.pt
```

Use the algorithm, class name, run name, model path, and model-data path corresponding to the trained model that will run in production.

---

## Generate the `.npy` reference

Use `drift_reference`:

```bash
python -m anomavision.drift_reference \
  --config config.yml \
  --img_path ./dataset/bottle/train/good \
  --model model.pt \
  --model_data_path ./distributions \
  --algorithm patchcore \
  --class_name bottle \
  --run_name anomav_exp \
  --device cpu \
  --batch_size 8 \
  --max_samples 500 \
  --output ./drift/reference_embeddings.npy
```

### Windows PowerShell

```powershell
python -m anomavision.drift_reference `
  --config config.yml `
  --img_path "D:\01-DATA\VisA_pytorch\candle\train\good" `
  --model "model.pt" `
  --model_data_path ".\distributions" `
  --algorithm patchcore `
  --class_name candle `
  --run_name anomav_exp `
  --device cpu `
  --batch_size 8 `
  --max_samples 500 `
  --output ".\drift\reference_embeddings.npy"
```

Adjust the paths and model identifiers to match your trained model.

### What happens during generation?

The reference builder:

1. Loads the configured image dataset.
2. Selects up to `--max_samples` trusted images.
3. Runs the normal model inference path.
4. Extracts the representation used for drift monitoring.
5. Validates that the resulting embeddings are finite and compatible.
6. Saves the embedding matrix to the requested `.npy` path.
7. Writes a JSON metadata sidecar.

This means the reference is generated using the same model representation that production monitoring expects.

---

## Output files

After successful generation:

```text
AnomaVision/
├── drift/
│   ├── reference_embeddings.npy
│   └── reference_embeddings.npy.json
```

The important file is:

```text
reference_embeddings.npy
```

The JSON sidecar stores metadata associated with reference generation.

You can verify the `.npy` file with Python:

```python
import numpy as np

x = np.load("./drift/reference_embeddings.npy")
print("shape:", x.shape)
print("dtype:", x.dtype)
print("finite:", np.isfinite(x).all())
```

A valid reference is expected to be a two-dimensional finite numeric matrix:

```text
(number_of_reference_samples, feature_dimensions)
```

For example:

```text
(500, 1024)
```

The exact feature dimension depends on the selected model and representation.

Do not manually edit the `.npy` file.

---

# 2. Run production detection with drift monitoring

Once the reference exists, start the normal detection command and add:

```text
--enable-drift-monitoring
--drift-reference <path-to-reference.npy>
```

### Linux/macOS

```bash
python -m anomavision.cli detect \
  --config config.yml \
  --model model.pt \
  --enable-drift-monitoring \
  --drift-reference ./drift/reference_embeddings.npy
```

### Windows PowerShell

```powershell
python -m anomavision.cli detect `
  --config config.yml `
  --model "model.pt" `
  --enable-drift-monitoring `
  --drift-reference ".\drift\reference_embeddings.npy"
```

This is the normal AnomaVision detection command with the drift observer enabled.

The existing anomaly detection pipeline continues to process production images.

---

# 3. Live customer dashboard

When drift monitoring is enabled, the CLI automatically starts the live production health dashboard.

Open:

```text
http://127.0.0.1:7860
```

The dashboard refreshes automatically while the monitoring process is running.

The dashboard is designed for operators and non-technical users. The primary interface does not require knowledge of embeddings, PSI, or statistical terminology.

It shows:

- **Production status** — collecting data, stable, or data change detected.
- **Images checked** — production samples observed.
- **Recent window** — current rolling monitoring window.
- **Change level** — normalized drift score presented as a percentage.
- **Recent production images** — visual context for investigating changes.
- **What changed?** — plain-language interpretation.
- **What should I do?** — practical investigation steps.
- **Technical details** — underlying metrics for engineers and researchers.

The dashboard is read-only and must not become a dependency of anomaly inference.

---

# 4. How the `.npy` is used in production

The reference file is loaded as the baseline. Production embeddings are collected from the normal inference path and compared with the reference.

```text
reference_embeddings.npy
          │
          │ trusted baseline
          ▼
     DriftMonitor
          ▲
          │ current rolling window
          │
Production images
          │
          ▼
     ModelWrapper
          │
          ├────────► anomaly score / localization
          │
          └────────► drift representation
```

For supported PyTorch paths, the representation generated during normal prediction can be reused rather than running the backbone twice.

The production window is bounded, so embeddings are not accumulated indefinitely.

---

# 5. Rolling monitoring window

The production monitor maintains a rolling window.

Example:

```text
Window size              500 samples
Minimum samples          100 samples
Evaluation interval       25 samples
PSI threshold              0.20
```

Conceptually:

```text
Production stream

1  2  3  4  5  ...  99  100  101 ... 500  501  502 ...
│                       │                    │
│                       └─ first evaluation  └─ oldest samples leave
│
└─ monitor warming up
```

Only the most recent `window_size` embeddings are retained.

This makes the monitor suitable for long-running production workloads without unbounded memory growth.

---

# 6. Monitoring states

## Collecting data

The monitor has not collected enough production samples for the configured comparison.

Example:

```text
COLLECTING DATA

We are learning what normal production data looks like.
```

This is a normal startup state, not a model failure.

## Production looks stable

The monitor has enough samples and the measured PSI is below the configured drift threshold.

Example:

```text
PRODUCTION LOOKS STABLE

Recent production data is consistent with the approved normal pattern.
```

## Data change detected

The monitor has enough samples and the measured PSI has reached or exceeded the configured threshold.

Example:

```text
DATA CHANGE DETECTED

Recent production images look different from the approved normal pattern.
```

This should trigger investigation rather than automatic model replacement.

## Monitoring unavailable

The dashboard cannot obtain a valid monitoring status.

This is different from `stable`. “No monitoring data” must not be interpreted as “production is healthy.”

---

# 7. Dashboard and recent images

The dashboard is intentionally operator-first.

The **Recent production images** area shows source images from the configured production image directory when available.

These images help answer:

- Did the lighting change?
- Did the camera move?
- Is a new product variant being inspected?
- Are images cropped differently?
- Did the production environment change?

When an external image directory is used, such as:

```text
D:\01-DATA\VisA_pytorch\candle\train\good
```

the dashboard launch configuration allows Gradio to serve that directory. The original dataset is not modified merely to display samples.

---

# 8. What should the operator do?

When drift is detected:

### 1. Look at recent images

Start with the actual production data shown by the dashboard.

### 2. Identify the operational change

Check camera, illumination, product type, acquisition process, preprocessing, and production configuration.

### 3. Check model quality

Use available labels, quality inspections, human review, or other appropriate production measurements.

### 4. Decide whether the change is expected

An expected production change may require a new approved reference population.

An unexpected change may require investigation before changing the model or reference.

### 5. Update deliberately

Do not automatically replace the model or reference solely because drift was detected.

---

# 9. Technical metrics

Advanced users can inspect the underlying metrics.

## PSI

Population Stability Index measures how the reference and production distributions differ across bins derived from the reference population.

The implementation uses reference-derived quantile bins, smoothing, and handling of values outside the reference range.

The configured PSI threshold determines the explicit `stable` / `drift` state.

For example:

```text
threshold = 0.20
```

means the monitor reports `drift` when PSI is at least `0.20`.

PSI is a monitoring statistic, not a probability that the model is wrong.

## Mean shift

Measures relative movement of the feature mean between the reference and current production populations.

## Standard-deviation shift

Measures relative changes in feature spread.

## Cosine shift

Measures change in embedding direction.

## Normalized drift score

The dashboard's overall change level combines normalized PSI, mean shift, and standard-deviation shift:

```text
score = min(
    1,
    0.60 * min(PSI, 1)
  + 0.25 * min(mean_shift, 1)
  + 0.15 * min(std_shift, 1)
)
```

The score is bounded to `[0, 1]`.

The explicit `stable` / `drift` state is based on the configured PSI threshold.

Therefore:

```text
Drift score != model accuracy
Drift score != percentage of defective images
Drift score != probability of failure
```

---

# 10. Machine-readable status

By default the monitor writes:

```text
./drift/drift_status.json
```

Example:

```json
{
  "samples_seen": 900,
  "window_size": 500,
  "window_fill": 500,
  "ready": true,
  "status": "drift",
  "drift_score": 0.7908556872,
  "psi": 16.6859248018,
  "mean_shift": 0.2237742298,
  "std_shift": 0.8994141984,
  "cosine_shift": 0.0240436164,
  "threshold": 0.2,
  "warnings": [
    "feature_distribution_shift"
  ]
}
```

The dashboard reads this monitoring state and refreshes automatically.

The same JSON can be consumed by another application, API, alerting service, or deployment platform.

---

# 11. Complete CLI example

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

### Options

| Option | Purpose | Default |
|---|---|---:|
| `--enable-drift-monitoring` | Enable production monitoring | disabled |
| `--drift-reference` | Trusted `.npy` / `.npz` embeddings | required when enabled |
| `--drift-window` | Maximum production embeddings retained | 500 |
| `--drift-min-samples` | Minimum samples before evaluation | 100 |
| `--drift-threshold` | PSI threshold for `drift` | 0.20 |
| `--drift-evaluation-interval` | Evaluate every N new samples | 25 |
| `--drift-output` | JSON status output path | `./drift/drift_status.json` |

---

# 12. Reference generation options

```bash
python -m anomavision.drift_reference \
  --config config.yml \
  --img_path ./dataset/bottle/train/good \
  --model model.pt \
  --model_data_path ./distributions \
  --algorithm patchcore \
  --class_name bottle \
  --run_name anomav_exp \
  --device cpu \
  --batch_size 8 \
  --max_samples 500 \
  --output ./drift/reference_embeddings.npy
```

Important inputs:

| Option | Meaning |
|---|---|
| `--config` | AnomaVision configuration file |
| `--img_path` | Trusted normal/reference images |
| `--model` | Model filename |
| `--model_data_path` | Directory containing model artifacts |
| `--algorithm` | Anomaly algorithm, such as PatchCore |
| `--class_name` | Model/dataset class |
| `--run_name` | Trained model run name |
| `--device` | `cpu` or `cuda` |
| `--batch_size` | Reference extraction batch size |
| `--max_samples` | Maximum reference samples |
| `--output` | Destination `.npy` file |

---

# 13. Compatibility requirements

The reference and production path must use compatible representations.

They should use the same:

- model
- representation
- feature dimensions
- preprocessing assumptions
- algorithm/model configuration

If the reference has shape:

```text
(500, 1024)
```

production drift embeddings must have feature dimension `1024`.

A feature-dimension mismatch must be reported explicitly rather than silently accepted.

---

# 14. Integration with existing inference

The production path is intentionally structured as:

```text
DataLoader / stream
       │
       ▼
     batch
       │
       ▼
  ModelWrapper
       │
       ├──────────────► anomaly scores / maps
       │
       └──────────────► drift representation when supported
                              │
                              ▼
                     InferenceDriftRuntime
                              │
                              ▼
                    ProductionDriftMonitor
                              │
                              ▼
                      drift_status.json
                              │
                              ▼
                     Live dashboard
```

The drift observer is isolated from the anomaly result path.

If drift processing fails, anomaly inference should continue.

This is important for industrial deployment: monitoring should not become a new single point of failure for the detector.

---

# 15. Backend behavior

## PyTorch

For supported PyTorch models, the backend can cache the representation generated during normal prediction so the monitoring runtime can reuse it.

## Other backends

The existing model wrapper supports multiple inference backends. Drift monitoring depends on a backend exposing a suitable representation.

If a backend cannot provide one, monitoring should fail safely without changing normal anomaly inference.

## Hailo / KV260

Drift monitoring does not replace or modify the existing Hailo or KV260/XModel inference path.

The monitoring feature is an observer rather than a new hardware inference backend.

---

# 16. Performance considerations

The main monitoring cost is comparing the reference population with the bounded current window.

For supported PyTorch paths, representation extraction can be reused from normal inference where possible.

The rolling window limits memory usage.

For production deployment, monitor:

- inference latency
- throughput
- memory usage
- evaluation frequency
- rolling-window size

The evaluation interval can be increased when drift checks do not need to run after every small batch.

---

# 17. Re-create the reference

A new reference can be generated when the organization intentionally approves a new normal operating population.

For example:

```text
Old approved population
        │
        ▼
reference_v1.npy

Production changes intentionally
        │
        ▼
New approved normal population
        │
        ▼
reference_v2.npy
```

Do not continuously replace the reference with the latest production data without an approval process. Otherwise, a gradual production problem can become part of the baseline and become harder to detect.

---

# 18. Troubleshooting

## The `.npy` file is not created

Check:

1. The reference image directory exists.
2. The directory contains supported images.
3. The model path is correct.
4. `--algorithm`, `--class_name`, and `--run_name` match the trained model.
5. The model-data directory contains the required artifacts.
6. The output directory can be created.

PowerShell example:

```powershell
Test-Path ".\drift"
Test-Path ".\drift\reference_embeddings.npy"
Test-Path ".\distributions"
```

## Reference and production dimensions do not match

Inspect the reference:

```python
import numpy as np
x = np.load("./drift/reference_embeddings.npy")
print(x.shape)
```

Then verify that the production representation has the same feature dimension.

## Dashboard says `MONITORING UNAVAILABLE`

Check:

```text
./drift/drift_status.json
```

and make sure the dashboard uses the same status path as the detection process.

## Dashboard does not show images

The dashboard needs access to the configured production image directory. If the directory is outside the project directory, it must be included in the Gradio allowed paths used by the dashboard.

For example:

```text
D:\01-DATA\VisA_pytorch\candle\train\good
```

The source dataset should remain untouched.

## Dashboard starts but detection should continue

The dashboard is launched as a separate process. A dashboard startup problem should not prevent normal anomaly detection.

---

# 19. Complete Windows example

Assume:

```text
Dataset:
D:\01-DATA\VisA_pytorch\candle\train\good

Model:
model.pt

Reference:
.\drift\reference_embeddings.npy
```

### Step 1 — Generate the reference

```powershell
python -m anomavision.drift_reference `
  --config config.yml `
  --img_path "D:\01-DATA\VisA_pytorch\candle\train\good" `
  --model "model.pt" `
  --model_data_path ".\distributions" `
  --algorithm patchcore `
  --class_name candle `
  --run_name anomav_exp `
  --device cpu `
  --batch_size 8 `
  --max_samples 500 `
  --output ".\drift\reference_embeddings.npy"
```

Verify:

```powershell
Test-Path ".\drift\reference_embeddings.npy"
```

Expected:

```text
True
```

### Step 2 — Start detection and monitoring

```powershell
python -m anomavision.cli detect `
  --config config.yml `
  --model "model.pt" `
  --enable-drift-monitoring `
  --drift-reference ".\drift\reference_embeddings.npy"
```

### Step 3 — Open the dashboard

```text
http://127.0.0.1:7860
```

### Step 4 — Watch the production status

The dashboard starts with:

```text
COLLECTING DATA
```

After enough samples are collected, it changes to either:

```text
PRODUCTION LOOKS STABLE
```

or:

```text
DATA CHANGE DETECTED
```

Recent production images are shown alongside the status when the configured image directory is accessible.

---

# 20. Operational recommendations

For production deployments:

1. Generate the reference from trusted normal data.
2. Store the reference as a versioned artifact.
3. Record the model and preprocessing configuration associated with it.
4. Deploy anomaly detection normally.
5. Enable drift monitoring as an observer.
6. Keep a bounded production window.
7. Review recent images when drift is detected.
8. Correlate drift with actual quality measurements where available.
9. Establish a controlled process for approving a new reference.
10. Do not treat drift alone as proof of model degradation.

---

# Summary

The `.npy` reference is the trusted baseline. Production embeddings are collected from the normal inference pipeline and compared against that baseline in a bounded rolling window.

The practical workflow is:

```text
1. Trusted normal images
          │
          ▼
2. Generate reference_embeddings.npy
          │
          ▼
3. Start AnomaVision detect
          │
          ├────────► anomaly detection continues normally
          │
          └────────► drift monitoring
                           │
                           ▼
                    drift_status.json
                           │
                           ▼
                  live customer dashboard
```

The dashboard provides the operator with the production status and recent images, while the technical metrics remain available for engineers and researchers.
