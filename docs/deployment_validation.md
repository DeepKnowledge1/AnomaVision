# Deployment Validation

AnomaVision includes a non-invasive deployment validation step for checking a trained or exported anomaly-detection model before it is moved into production.

> **Validate the deployment artifact without changing the anomaly-detection algorithm or inference behavior.**

## CLI

Run `anomavision validate --help` to see the current command-line interface.

| Parameter | Type | Default | Required | Description |
|---|---|---:|---|---|
| `--model` | path | — | **Yes** | Path to the trained/exported model or deployment artifact to validate. |
| `--config` | path | `None` | No | Existing AnomaVision configuration used to build the validation input shape and run runtime inference. |
| `--reference-model` | path | `None` | No | Existing reference model used for output-consistency validation. |
| `--consistency-tolerance` | float | `1e-4` | No | Maximum allowed absolute difference for score and anomaly-map consistency checks. |
| `--runs` | int | `10` | No | Number of measured inference runs used for latency/FPS. Must be at least 1. |
| `--warmup-runs` | int | `2` | No | Number of warm-up inference runs excluded from latency measurement. Must be 0 or greater. |
| `--json` | flag | off | No | Print the complete validation report as JSON instead of the human-readable report. |

## `--model`

The only required parameter. It accepts the model formats already supported by AnomaVision: `.pt`, `.pth`, `.torchscript`, `.onnx`, `.engine`, `.trt`, `.xml`, `.bin`, `.hef`, and `.xmodel`. OpenVINO model directories are also accepted.

```bash
anomavision validate --model distributions/padim/bottle/anomav_exp/model.pt
```

## `--config`

Use the existing AnomaVision configuration to build the validation input from its resize/crop settings.

```powershell
anomavision validate `
  --model distributions\padim\bottle\anomav_exp\model.pt `
  --config config.yml
```

For non-ONNX runtime validation, `--config` enables runtime inference measurement. Without it, latency is not measured for those formats.

## `--reference-model`

Compares the candidate deployment model against an existing reference model using the existing AnomaVision inference wrapper. `--config` is required when this option is used.

```powershell
anomavision validate `
  --model distributions\padim\bottle\anomav_exp\model.onnx `
  --reference-model distributions\padim\bottle\anomav_exp\model.pt `
  --config config.yml
```

The comparison checks both anomaly scores and anomaly maps and reports maximum and mean absolute differences. Shape mismatches are rejected.

## `--consistency-tolerance`

Default: `1e-4`. Both score and map maximum absolute differences must be less than or equal to this value. Negative values are rejected.

```bash
anomavision validate --model model.onnx --reference-model model.pt --config config.yml --consistency-tolerance 1e-3
```

## `--runs`

Controls the number of measured inference iterations. Default: `10`. The value must be at least `1`.

```bash
anomavision validate --model model.onnx --runs 50
```

## `--warmup-runs`

Controls warm-up inference iterations excluded from latency measurement. Default: `2`. The value may be `0`.

```bash
anomavision validate --model model.onnx --warmup-runs 5 --runs 20
```

## `--json`

Prints the complete validation report as JSON. This is useful for CI/CD, deployment gates, scripts, and storing validation results.

```bash
anomavision validate --model model.onnx --json
```

The JSON report contains model path, detected format, inputs, outputs, performance, consistency, backend availability, validation checks, deployment readiness, and the validation note.

## Supported formats

| Format | Extension / representation |
|---|---|
| PyTorch | `.pt`, `.pth` |
| TorchScript | `.torchscript` |
| ONNX | `.onnx` |
| TensorRT | `.engine`, `.trt` |
| OpenVINO | `.xml`, `.bin`, or model directory |
| Hailo | `.hef` |
| Vitis AI / KV260 | `.xmodel` |

## Validation checks

- Model/file existence and loading
- ONNX structural validity where applicable
- ONNX input shape information where applicable
- Runtime inference
- Latency and FPS
- Reference-model output consistency
- Deployment-backend availability
- Final `ready_for_deployment` result

## Backend compatibility

The report checks availability of `pytorch`, `torchscript`, `onnxruntime`, `openvino`, `tensorrt`, `hailo`, `vitis_ai_vart`, `vitis_ai_xir`, `vitis_ai_library`, and `kv260`.

**Important:** `available` means the required runtime components are detected in the current environment. It does not prove that a specific model has been physically tested on the target hardware. Actual KV260, Hailo, or other hardware validation requires the corresponding deployment environment.

## Production workflow

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

## Design principles

Deployment validation is observational and non-invasive. It does not modify PaDiM, PatchCore, or EfficientAD; change preprocessing, anomaly scoring, or localization; retrain models; or change production inference behavior. It reuses the existing AnomaVision inference/backends wherever possible.

## Recommended command

```powershell
anomavision validate `
  --model distributions\padim\bottle\anomav_exp\model.onnx `
  --reference-model distributions\padim\bottle\anomav_exp\model.pt `
  --config config.yml `
  --runs 20 `
  --warmup-runs 5
```

Review model validity, inference success, latency/FPS, output consistency, backend availability, and target hardware requirements.