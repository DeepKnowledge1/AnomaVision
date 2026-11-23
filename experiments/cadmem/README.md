# CAD-Mem controlled benchmark (exploratory)

This benchmark compares frozen PatchCore, a bounded naive append-only memory
baseline, and CAD-Mem using a common trained checkpoint. It runs multiple
contamination ratios, with optional synthetic brightness/contrast drift.

## Run (PowerShell, uv)

```powershell
git pull origin feat/cadmem-patchcore-integration

uv run python experiments/cadmem/run_controlled_benchmark.py `
  --dataset-path "D:/01-DATA/mvtec" `
  --class-name cable `
  --model-path "distributions/patchcore/cable/anomav_exp/model.pt" `
  --output-dir "results/cadmem/cable-controlled" `
  --drift brightness `
  --strength 0.15 `
  --seed 42
```

Run `--drift none` for a no-shift control, or `--drift contrast`.
Default contamination rates: 0%, 1%, 5%, 10%, 20%. Small datasets may
round low contamination rates to zero; the report records *realized* rates.

Outputs: `metrics.json`, `metrics.csv`, `predictions.csv`.
AUROC and AP are threshold-free. Recall and FPR use a single frozen
threshold calibrated on a separate normal validation subset, with target
validation FPR 5%. The threshold is **not** recalibrated under drift.

## Protocol and limitations

- Calibration and threshold-validation subsets are disjoint from each other.
- Adaptation and evaluation subsets of the test set are disjoint.
- Adaptation-stream composition is label-controlled for the experiment, but
  the algorithms are not given those labels.
- A single trained checkpoint may have seen the training images later used
  for calibration and threshold validation. **This is not a publication-grade
  independent split.** For a publishable protocol, split train/good first,
  retrain PatchCore on train-only, and use independent calibration/validation.
- The naive baseline simply appends patches from the stream until its fixed
  256-patch capacity is reached; it is not a canonical online PatchCore.
- Drift is deterministic synthetic brightness or contrast, not a validated
  industrial domain shift. Anomalous images receive the same transform.
- Report multiple seeds, categories, confidence intervals, memory and
  inference overhead before making research claims.
- This is an offline experiment, not a causal production-stream simulation.
- The implementation is not executed against your local MVTec dataset here.
