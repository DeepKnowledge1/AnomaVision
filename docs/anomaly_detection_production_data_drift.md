# Anomaly Detection: Production Data Drift

## Overview

Anomaly detection models are usually trained and validated on a known population of images. Production data can change after deployment while the inference service continues to run normally.

Examples include:

- a new camera or lens
- different illumination
- camera position or focus changes
- a new product variant
- changes in the manufacturing process
- changes in image acquisition or preprocessing
- seasonal or environmental changes
- a production population that differs from the reference population

These changes are called **data drift** when the distribution of production inputs or their model representations changes relative to a trusted reference population.

AnomaVision monitors this change without replacing or modifying the anomaly detector.

> **Important:** data drift is an early-warning signal. Drift does not automatically mean that an image is defective or that model accuracy has degraded. It tells the team that the production population should be investigated.

---

## What problem does this solve?

A production anomaly system can look healthy from an infrastructure perspective:

```text
Model is running       ✓
Images are arriving    ✓
Predictions are returned ✓
No exceptions          ✓
```

while the input population has changed:

```text
Training / reference data
          │
          │  distribution changes
          ▼
Production data ───────────────► model still returns predictions
                                  but the operating conditions changed
```

Without drift monitoring, this type of change can remain invisible until a user notices a quality problem.

AnomaVision adds a second observation path:

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
```

The two paths are intentionally separated.

---

## Design principles

### 1. Additive observer

Drift monitoring observes the representation used by the production pipeline. It does not replace the anomaly detector.

### 2. Preserve existing algorithms

PatchCore, PaDiM, and EfficientAD remain responsible for anomaly detection. Enabling drift monitoring does not intentionally change their scoring or localization logic.

### 3. Preserve hardware paths

The monitoring feature is designed so existing Hailo, KV260/XModel, ONNX, TensorRT, OpenVINO, and other inference paths can continue independently. If a backend cannot provide a monitoring representation, drift monitoring must fail safely rather than stop anomaly inference.

### 4. Reuse representations when possible

For supported PyTorch models, the representation captured during normal inference can be reused by the drift observer. This avoids unnecessarily running a second backbone inference pass.

### 5. Bounded memory

Production embeddings are stored in a bounded rolling window. The monitor does not accumulate production data indefinitely in memory.

### 6. Operator-first presentation

The dashboard presents a simple operational message first. Technical metrics are available as secondary details for engineers and researchers.

---

# End-to-end workflow

## Step 1 — Create a trusted reference

A reference population describes what the system considers normal production behavior.

The reference should be:

- representative of the approved operating conditions
- predominantly normal/good data
- collected from a trusted period
- processed using the same model representation as production
- large enough to represent normal variation

Generate the reference embeddings with:

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

The reference builder runs normal model inference and extracts the representation used for drift monitoring. It writes:

```text
./drift/reference_embeddings.npy
./drift/reference_embeddings.npy.json
```

The JSON sidecar contains metadata about the generated reference.

### Reference data recommendations

Do not create the reference from data that is already known to represent an abnormal operating condition.

For example, if a factory camera was correctly aligned for January production, use that trusted population as the reference. If February production contains a known camera replacement and you use February as the reference, the monitor will no longer treat the camera change as drift.

The reference should represent the operating state that the team has approved as the baseline.

---

# Step 2 — Start production inference and monitoring

The normal detection command can enable monitoring with two additional arguments:

```powershell
python -m anomavision.cli detect `
  --config config.yml `
  --model "model.pt" `
  --enable-drift-monitoring `
  --drift-reference ".\drift\reference_embeddings.npy"
```

When monitoring is enabled, the CLI starts the live production dashboard automatically.

Open:

```text
http://127.0.0.1:7860
```

The anomaly detection process continues normally.

The dashboard is a separate read-only process. A dashboard startup problem must not stop the anomaly detection pipeline.

---

# Step 3 — Rolling production monitoring

The production monitor maintains a rolling window.

Example configuration:

```text
Window size             500 samples
Minimum samples         100 samples
Evaluation interval      25 new samples
PSI threshold            0.20
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

This makes the monitor suitable for long-running production workloads without allowing the monitoring window to grow indefinitely.

---

# Monitoring states

The dashboard simplifies the underlying monitor state into operator-friendly messages.

## Collecting data

The monitor has not collected enough production samples to make a reliable comparison.

Example:

```text
COLLECTING DATA

We are learning what normal production data looks like.
```

This is not an error.

## Production looks stable

The current production window is sufficiently populated and the measured drift is below the configured alert condition.

Example:

```text
PRODUCTION LOOKS STABLE

Recent production data is consistent with the approved normal pattern.
```

## Data change detected

The monitor has detected a distribution change according to the configured drift criteria.

Example:

```text
DATA CHANGE DETECTED

Recent production images look different from the approved normal pattern.
```

This should trigger investigation rather than an automatic model replacement.

## Monitoring unavailable

The dashboard cannot currently obtain a valid monitoring status.

This is different from `stable`.

A system should not interpret “no monitoring data” as “production is healthy.”

---

# Dashboard

The dashboard is designed for a production operator who may not know machine learning terminology.

## Main status

The first thing the operator sees is the current production condition.

Examples:

```text
PRODUCTION LOOKS STABLE
```

or:

```text
DATA CHANGE DETECTED
```

The message explains what the status means in ordinary language.

## Images checked

Shows how many production samples have been observed.

Example:

```text
900
production images seen
```

## Recent window

Shows how much of the configured monitoring window is currently populated.

Example:

```text
500 / 500
100% of the monitoring window
```

## Change level

The normalized drift score is presented as a percentage to make the signal easier to understand.

For example:

```text
79%
```

means the current monitor score is approximately 0.79 on the monitor's normalized scale.

It is **not** an accuracy percentage and should not be interpreted as “79% of images are bad.”

## Recent production images

The dashboard can display source images from the configured production image directory. These images provide the most useful context when a drift alert occurs.

The operator can visually investigate questions such as:

- Did the lighting change?
- Did the camera move?
- Is a new product variant being inspected?
- Are images now cropped differently?
- Did the production environment change?

This visual context is intentionally placed near the drift status.

---

# What changed?

The dashboard translates the technical monitoring signals into three simple concepts.

### Overall change

The normalized drift score summarizes the measured distribution change.

### Typical appearance

Mean shift indicates how much the typical representation has moved relative to the reference.

### Variation

Standard-deviation shift indicates whether the spread of the production representation has changed.

These are monitoring signals, not defect classifications.

---

# What should the operator do?

When drift is detected, the recommended workflow is:

### 1. Look at recent images

Start with the actual production data.

### 2. Identify the operational change

Check camera, illumination, product type, acquisition process, preprocessing, and production configuration.

### 3. Check model quality

Use available labels, quality inspections, human review, or other appropriate production measurements.

### 4. Decide whether the change is expected

An expected production change may simply require a new approved reference population.

An unexpected change may require investigation before changing the model or reference.

### 5. Update deliberately

Do not automatically replace the model or reference solely because drift was detected.

---

# Technical metrics

Advanced users can inspect the underlying metrics.

## PSI

Population Stability Index (PSI) measures how the reference and production distributions differ across bins derived from the reference population.

In this implementation, PSI is calculated from reference-derived quantile bins with smoothing and explicit handling of values outside the reference range.

The configured PSI threshold determines the `drift` state.

A threshold of:

```text
0.20
```

means that the configured monitor reports drift when the calculated PSI reaches or exceeds that threshold.

PSI is a monitoring statistic, not a probability that the model is wrong.

## Mean shift

Mean shift measures relative movement of the feature mean between the reference and current production populations.

Higher values indicate a larger change in the typical representation.

## Standard-deviation shift

Standard-deviation shift measures relative changes in feature spread.

A high value can indicate that production inputs have become more or less variable.

## Cosine shift

Cosine shift measures a change in embedding direction.

It can reveal directional changes even when magnitude-based statistics are less informative.

## Normalized drift score

The current implementation combines the normalized PSI, mean shift, and standard-deviation shift into a bounded score:

```text
score = min(
    1,
    0.60 * min(PSI, 1)
  + 0.25 * min(mean_shift, 1)
  + 0.15 * min(std_shift, 1)
)
```

The score is bounded to `[0, 1]`.

The score is intended for dashboard presentation and overall signal strength. The configured `threshold` is applied to PSI for the explicit `stable` / `drift` status.

Therefore:

```text
Drift score != model accuracy
Drift score != percentage of defective images
Drift score != probability of failure
```

---

# Example status

The monitor can write a machine-readable status similar to:

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

The customer dashboard does not expose this raw JSON as the primary interface. Instead, it turns the result into an operator-facing status and provides the technical values under **Technical details**.

---

# Status file

By default the live status is written to:

```text
./drift/drift_status.json
```

You can change it with:

```bash
--drift-output ./drift/drift_status.json
```

The status file can also be consumed by another application, service, API, alerting system, or deployment platform.

The dashboard reads this same status file and refreshes automatically.

---

# Complete CLI reference

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
| `--drift-reference` | Reference `.npy` / `.npz` embeddings | required when enabled |
| `--drift-window` | Maximum production embeddings retained | 500 |
| `--drift-min-samples` | Minimum samples before evaluation | 100 |
| `--drift-threshold` | PSI threshold for `drift` | 0.20 |
| `--drift-evaluation-interval` | Evaluate every N new samples | 25 |
| `--drift-output` | JSON status output path | `./drift/drift_status.json` |

---

# Reference generation CLI

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

Important reference-generation inputs are:

- `img_path`: trusted normal images
- `model`: model filename
- `algorithm`: anomaly algorithm
- `class_name`: model class
- `run_name`: trained model run
- `device`: CPU or CUDA
- `max_samples`: maximum reference samples
- `output`: reference embedding file

---

# Integration with existing inference

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

If drift processing raises an exception, the production inference loop catches the monitoring error and continues anomaly inference.

This property is important for industrial deployment: monitoring should not become a new single point of failure for the detector.

---

# Backend behavior

## PyTorch

PyTorch models can expose the representation needed for drift monitoring. For supported models, the backend caches the representation generated during normal prediction and the monitoring runtime reuses it.

This avoids an unnecessary second model/backbone pass where possible.

## Other model formats

The existing model wrapper supports multiple backends. Drift monitoring depends on a backend being able to provide a suitable representation.

If a backend does not expose one, the monitoring layer should report the limitation without changing normal anomaly inference.

This is intentional: the drift feature must not force unsupported behavior into hardware or optimized inference backends.

---

# Performance considerations

The main production cost of drift monitoring is the comparison of the reference population with the bounded current window.

For supported PyTorch paths, the most expensive representation extraction is reused from normal inference where possible.

The rolling window also limits memory consumption.

For industrial deployment, monitor:

- inference latency
- throughput
- memory usage
- evaluation frequency
- rolling-window size

The monitoring evaluation interval can be increased when drift checks do not need to run after every small batch.

---

# Troubleshooting

## Dashboard says `MONITORING UNAVAILABLE`

Check:

```text
./drift/drift_status.json
```

and verify that the detection process has drift monitoring enabled.

The default dashboard refresh interval is a few seconds.

## Dashboard says `COLLECTING DATA`

This normally means fewer than `--drift-min-samples` production samples have been observed.

Wait until enough samples have entered the rolling window.

## No production images appear

The dashboard reads the configured image source. Verify that `img_path` in `config.yml` points to an existing image directory, or set the dashboard image path explicitly with:

```text
ANOMAVISION_DRIFT_IMAGE_PATH
```

When images are stored outside the application directory, the dashboard registers the source directory with Gradio as an allowed path.

## Drift monitoring does not stop inference

This is intentional. Monitoring failures are isolated from the anomaly inference path.

Inspect the application log for the monitoring warning.

## Drift is detected after a known production change

This is expected if the reference represents the previous approved operating condition.

The correct next step is investigation and deliberate reference management, not automatic suppression of the alert.

## PSI is much larger than 1

That is possible. PSI itself is not bounded to `[0, 1]` in this implementation.

The dashboard therefore presents PSI as a technical metric and uses the bounded normalized drift score for its simple percentage visualization.

---

# Operational guidance

A good production monitoring strategy separates three questions:

### Question 1 — Did the input population change?

**Data drift monitoring** answers this.

### Question 2 — Did the model's predictions change in a problematic way?

**Anomaly results and production quality monitoring** answer this.

### Question 3 — What should we change?

This requires human/engineering investigation using production context.

These questions should not be collapsed into one metric.

---

# Research and industrial use

The monitoring architecture can support further extensions without changing the anomaly detector, including:

- historical drift timelines
- per-class drift monitoring
- per-camera drift monitoring
- production quality correlation
- alerting and webhooks
- email/Slack/Teams notifications
- drift snapshots
- reference-version management
- automated report generation
- model-version comparison
- production segmentation by shift or line

These should remain separate monitoring capabilities layered around the detector.

---

# Summary

AnomaVision's production data-drift feature provides a second layer of observability around anomaly detection:

```text
              ANOMALY DETECTION
                     │
                     │ Is this image unusual?
                     ▼
                 Prediction

                     +

                DATA DRIFT
                     │
                     │ Has production changed?
                     ▼
              Production health
```

The goal is not to replace anomaly detection with a drift metric. The goal is to detect changes in the production population early, provide the operator with the actual images needed to investigate them, and preserve the existing anomaly detection and deployment paths.
