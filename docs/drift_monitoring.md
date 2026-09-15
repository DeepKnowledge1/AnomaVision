# Data Drift Monitoring

AnomaVision can detect silent changes in production visual data by comparing a trusted reference embedding distribution with a current production window. This is intentionally label-free and model-agnostic.

## Why it matters

A deployed anomaly detector can continue returning predictions while the camera, lighting, material, process, or image distribution has changed. Monitoring the feature distribution provides an early warning before defect-detection quality degrades.

## CLI

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

The report contains:

- **PSI (Population Stability Index)** — primary distribution-shift metric.
- **Mean shift** — relative movement of feature centroids.
- **Std shift** — relative change in feature scale.
- **Cosine shift** — change in embedding direction.
- **Drift score** — combined operational signal.
- **Warnings** — actionable indicators such as small windows or feature-distribution shift.

The default PSI threshold is `0.20`. Override it with `--threshold` when your application has a validated operating point.

## Python API

```python
from anomavision.drift import DriftMonitor

monitor = DriftMonitor(reference_embeddings, threshold=0.20)
report = monitor.compare(production_embeddings)

if report.status == "drift":
    print("Investigate production data before trusting model quality.")
```

## Production integration

For a real deployment, feed the monitor with the same stable embedding representation used by the anomaly model. Keep the reference distribution immutable for a model version and compare it against rolling production windows (for example, hourly or per shift).

A recommended next step is to persist reports and visualize PSI, drift score, image quality, throughput, and anomaly-rate changes together in a production dashboard.
