# Deployment Validation

AnomaVision includes a non-invasive deployment validation step for checking a trained or exported anomaly-detection model before it is moved into production.

The goal is simple:

> **Validate the deployment artifact without changing the anomaly-detection algorithm or inference behavior.**

Deployment validation is especially useful when the same model moves through multiple representations such as PyTorch, ONNX, TensorRT, OpenVINO, Hailo, or Vitis AI/KV260.

## Why this exists

A model can work correctly during development and still fail later because of:

- an invalid or corrupted export
- an unsupported runtime
- unexpected input shapes
- inference failures in the deployment format
- performance that is too slow for the target use case
- output differences between a reference model and an exported model

Deployment validation catches these issues before deployment.

It is an additional production-safety layer around the existing AnomaVision inference stack.

## Supported model formats

The validator recognizes the model formats already supported by AnomaVision:

| Format | Extension / representation |
|---|---|
| PyTorch | `.pt`, `.pth` |
| TorchScript | `.torchscript` |
| ONNX | `.onnx` |
| TensorRT | `.engine`, `.trt` |
| OpenVINO | `.xml`, `.bin`, or OpenVINO model directory |
| Hailo | `.hef` |
| Vitis AI / KV260 | `.xmodel` |

The validator reuses AnomaVision's existing `ModelWrapper` and backend implementations wherever possible. It does not introduce a second inference implementation for the anomaly-detection algorithms.

## Basic validation

Validate a model:

```bash
anomavision validate --model distributions/padim/bottle/anomav_exp/model.pt
```

For configuration-dependent validation, provide the existing configuration:

```bash
anomavision validate \
  --model distributions/padim/bottle/anom_exp/model.pt \
  --config config.yml
```

On PowerShell:

```powershell
anomavision validate `
  --model distributions\padim\bottle\anomav_exp\model.pt `
  --config config.yml
```

## Reference-model consistency

An exported model can also be compared with a reference model.

For example, compare ONNX against the original PyTorch model:

```powershell
anomavision validate `
  --model distributions\padim\bottle\anomav_exp\model.onnx `
  --reference-model distributions\padim\bottle\anomav_exp\model.pt `
  --config config.yml `
  --runs 20
```

The validator compares the AnomaVision inference outputs:

- anomaly scores
- anomaly maps

It reports maximum and mean absolute differences for both.

Example:

```text
Output consistency
  Score max abs diff:  ...
  Score mean abs diff: ...
  Map max abs diff:    ...
  Map mean abs diff:   ...
  Tolerance:           0.0001
```

The tolerance can be changed when small numerical differences are expected:

```bash
anomavision validate --model model.onnx --reference-model model.pt --consistency-tolerance 1e-3
```

## Validation checks

Depending on the model format, validation can report:

### Model integrity

- file exists
- model loads successfully
- deployment format is structurally valid where supported

### Inference

- model can execute inference
- output is compatible with the AnomaVision inference interface

### Input shape

For ONNX models, static input shape information is checked where available.

### Performance

The validator performs inference runs and reports:

- latency
- FPS

Use `--warmup-runs` to control warmup iterations and `--runs` to control measured iterations.

Example:

```bash
anomavision validate --model model.onnx --warmup-runs 5 --runs 20
```

### Output consistency

When `--reference-model` is supplied, the candidate model is compared with the reference model using the existing AnomaVision output representation.

### Backend availability

The report also shows whether supported deployment runtimes are installed.

Example:

```text
Backend compatibility
  pytorch: available
  torchscript: available
  onnxruntime: available
  openvino: available
  tensorrt: not installed
  hailo: not installed
  vitis_ai_vart: available
  vitis_ai_xir: available
  vitis_ai_library: not installed
  kv260: available
```

## Important: runtime availability vs hardware validation

A backend marked `available` means that the required runtime components are detected in the current environment.

For example:

```text
kv260: available
```

does **not** mean that the specific model has been physically tested on a KV260 board.

Actual target-device validation still requires the corresponding hardware and deployment environment.

The same principle applies to Hailo and other hardware-specific targets.

## JSON output

For automation and CI/CD workflows, use:

```bash
anomavision validate --model model.onnx --json
```

This makes the validation result easier to consume from scripts and deployment pipelines.

## CLI options

| Option | Purpose |
|---|---|
| `--model` | Model/deployment artifact to validate |
| `--config` | Existing AnomaVision configuration |
| `--reference-model` | Optional reference model for output comparison |
| `--consistency-tolerance` | Maximum allowed output difference; default `1e-4` |
| `--runs` | Number of measured inference runs |
| `--warmup-runs` | Number of warmup runs |
| `--json` | Print the validation report as JSON |

Run:

```bash
anomavision validate --help
```

for the current CLI help.

## Production workflow

Deployment validation fits between export and deployment:

```text
Train
  ↓
Evaluate
  ↓
Compare experiments
  ↓
Export
  ↓
Deployment Validation
  ├── Model integrity
  ├── Inference
  ├── Input/output checks
  ├── Performance
  ├── Reference consistency
  └── Backend availability
  ↓
Deploy
  ↓
Production monitoring
  ↓
Data-drift monitoring
```

This makes deployment validation a gate around the deployment artifact rather than a modification to the anomaly-detection algorithm.

## Design principles

Deployment validation is intentionally non-invasive.

It does **not**:

- modify PaDiM
- modify PatchCore
- modify EfficientAD
- change preprocessing
- change anomaly scoring
- retrain the model
- alter localization behavior
- change production inference behavior

It observes and validates the existing model/backend behavior.

## Recommended usage

For an exported model, a practical validation command is:

```powershell
anomavision validate `
  --model distributions\padim\bottle\anomav_exp\model.onnx `
  --reference-model distributions\padim\bottle\anomav_exp\model.pt `
  --config config.yml `
  --runs 20 `
  --warmup-runs 5
```

Review:

1. model validity
2. inference success
3. latency/FPS
4. output consistency
5. backend availability
6. target hardware requirements

Only the runtime/backend availability is automatically inferred from the current environment. Physical target-device validation remains a separate deployment test.
