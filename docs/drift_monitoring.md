# Data Drift Monitoring

AnomaVision detects silent changes in production visual data by comparing a trusted reference embedding distribution with a rolling production window. The monitor is label-free and operates on the model representation rather than raw pixels.

## Why it matters

A deployed anomaly detector can continue returning predictions while the camera, lighting, material, process, or image distribution has changed. Monitoring the feature distribution provides an early warning before defect-detection quality degrades.

## Offline CLI

Create two NumPy matrices with shape `(samples, features)`:

```text
reference.npy   # trusted normal/training population
production.npy  # recent production window
```

Run:

```bash
anomavision drift \
  --reference reference.npy \
  --current production.npy \
  --output drift.json
```

The report contains PSI, mean shift, standard-deviation shift, cosine shift, a combined drift score, and actionable warnings. The default PSI threshold is `0.20`.

## Detect-time monitoring

Drift monitoring can now run directly inside the normal `detect` pipeline:

```bash
python detect.py \
  --algorithm patchcore \
  --model_data_path ./distributions \
  --model model.pth \
  --img_path ./production_images \
  --enable-drift-monitoring \
  --drift-reference ./reference.npy \
  --drift-window 500 \
  --drift-min-samples 100 \
  --drift-threshold 0.20 \
  --drift-evaluation-interval 25 \
  --drift-output ./drift/drift_status.json
```

The existing anomaly scores and visualizations continue to run normally. Drift monitoring is best-effort: an embedding/monitoring failure is logged as a warning and does not stop anomaly detection.

The JSON status is updated during the run and saved again when detection finishes. Once the rolling window is ready, the console reports `stable` or emits a `DRIFT ALERT` with PSI, drift score, and warnings.

## Rolling production monitoring

For long-running camera processes, use `ProductionDriftMonitor`. It keeps a bounded rolling window and does not retain the complete production history in memory.

```python
from anomavision.production_monitor import ProductionDriftMonitor
from anomavision.drift_runtime import InferenceDriftRuntime

monitor = ProductionDriftMonitor(
    reference_embeddings,
    window_size=500,
    min_samples=100,
    threshold=0.20,
    evaluation_interval=25,
)

runtime = InferenceDriftRuntime(monitor, model)

# Inside the existing inference loop:
result = runtime.update(batch)
if result and result["status"] == "drift":
    print("DRIFT ALERT", result)
```

The runtime extracts the same model representation used by PatchCore when `_extract()` is available, pools patch embeddings per image, and feeds them to the rolling monitor. Current detect-time integration performs embedding extraction after anomaly inference; optimized single-pass feature reuse can be added later for deployments where the extra feature-extraction cost matters.

### Lifecycle

```text
Camera / image stream
        |
        v
  preprocessing
        |
        v
   model inference ------> anomaly score / heatmap
        |
        +-----------------> model embeddings
                              |
                              v
                     rolling drift window
                              |
                              v
                       PSI + shift metrics
                              |
                    +---------+---------+
                    |                   |
                  stable              drift
                    |                   |
                 continue          alert / inspect
```

Keep the reference distribution immutable for each model version. Reset the rolling window after a validated model/data refresh rather than silently replacing the reference.

## Operational recommendation

For a factory deployment, persist the status periodically and visualize drift alongside anomaly rate, throughput, latency, and image-quality metrics. A drift alert is an investigation signal; it should not automatically retrain or replace a production model without validation.
