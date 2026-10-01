# AnomaVision Core Safety Gate

The Core Safety Gate protects the validated detector behavior while new
features—especially Studio/API/UI work—are being developed.

## What it protects

A feature can be fully functional while accidentally changing the detector.
The gate therefore checks the core inference contract separately:

- model loads successfully;
- the expected anomaly/normal classification is preserved;
- the anomaly score remains inside a validated range;
- an anomaly map is returned;
- the anomaly map has a valid spatial shape.

## 1. Fast deterministic gate

Run the contract tests before committing:

```powershell
uv run pytest tests/test_core_safety_contract.py -q
```

## 2. Real-model regression gate

Use a small set of validated images from your real dataset and a validated
model. Do not commit proprietary images or model artifacts.

Example:

```powershell
uv run python scripts/regression/core_safety_gate.py `
  --model "C:\path\to\model.pt" `
  --image "D:\01-DATA\mvtec\cable\test\bent_wire\000.png" `
  --threshold 0.25 `
  --expected anomaly `
  --min-score 0.25
```

Run the same command for several known-good and known-defect images.

For each case, use a score range that was measured from a known-good commit.
Do not use a range so wide that a real regression can pass.

## 3. Before changing core behavior

If a Studio or deployment feature requires a core change:

1. Add or update the regression case first.
2. Run the real-model gate on the current known-good code.
3. Make the core change.
4. Run the same cases again.
5. Investigate any changed score, prediction, or map.
6. Only then commit the change.

This makes an intentional core change explicit instead of allowing a silent
ML regression.

## 4. CI

The deterministic contract is executed in GitHub Actions. It does not require
a model or private dataset, so it is safe to run on every pull request.

The real-model gate remains a local release/validation gate because model and
dataset paths are machine-specific.

## 5. Golden baseline

Keep the validated baseline associated with a Git tag, for example:

```text
core-stable-YYYY-MM
```

When a result looks suspicious, compare the current branch with that tag and
rerun exactly the same regression cases.
