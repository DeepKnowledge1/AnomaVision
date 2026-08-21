# AnomaVision on AMD Kria K26 with Hailo

This branch targets deployment of AnomaVision **PaDiM** and **ultra-light PatchCore** on a Kria K26 system with a Hailo accelerator. The first requirement is strict: the accelerator graph must contain the entire anomaly algorithm, not only the ResNet feature extractor.

> **End-to-end quantization scope:** RGB preprocessing contract, ResNet feature extraction, multi-scale feature fusion, channel selection or embedding normalization, PaDiM Mahalanobis distance or PatchCore memory-bank distance, score-map reconstruction, and image-score reduction.

## Architecture

The export path produces a fixed-resolution ONNX graph with two outputs:

| Output | Meaning |
|---|---|
| `image_scores` | One anomaly score per input image. |
| `score_map` | Full-resolution anomaly localization map. |

The Hailo Dataflow Compiler must parse, optimize with representative calibration images, quantize, and compile this complete graph into a HEF. The runtime does not execute a CPU-side distance calculation after Hailo feature extraction. It only preprocesses the image, submits the complete graph, and reads the two final outputs.

## Export artifacts

A PaDiM artifact must contain `backbone`, `layer_indices`, `channel_indices`, `mean`, and `cov_inv`. A PatchCore artifact must contain `backbone`, `layer_indices`, `memory_bank`, and, when applicable, `patch_grid`.

Export a complete graph on a host with the project environment:

```powershell
uv run python -m anomavision.hailo_export `
  --algorithm padim `
  --artifact .\model_stats.pt `
  --calibration-dir .\normal_calibration `
  --output-dir .\hailo\padim_k26
```

or:

```powershell
uv run python -m anomavision.hailo_export `
  --algorithm patchcore `
  --artifact .\patchcore_artifact.pt `
  --calibration-dir .\normal_calibration `
  --output-dir .\hailo\patchcore_k26
```

This creates the complete ONNX graph and a calibration manifest. It does not claim to create a HEF unless the Hailo SDK is installed and an explicit compiler command is supplied:

```powershell
uv run python -m anomavision.hailo_export `
  --algorithm patchcore `
  --artifact .\patchcore_artifact.pt `
  --calibration-dir .\normal_calibration `
  --output-dir .\hailo\patchcore_k26 `
  --hailo-command "<Hailo SDK parse/optimize/compile command using {onnx} and {output}>"
```

The exact compiler command depends on the installed Hailo Dataflow Compiler release and target accelerator. It should include representative-image calibration and must produce a HEF whose outputs are exactly `image_scores` and `score_map`.

## Kria runtime

Copy the generated HEF and the runtime package to the Linux image running on the Kria K26. The runtime adapter is:

```python
from anomavision.hailo_runtime import HailoAnomalyRuntime

with HailoAnomalyRuntime("/opt/models/padim_k26.hef") as detector:
    result = detector.predict("/opt/images/part.png")
    image_score = result["image_scores"]
    heatmap = result["score_map"]
```

The Kria image must provide the HailoRT Python package and a working Hailo device driver. The runtime deliberately fails if the HEF does not expose both final AnomaVision outputs; this prevents accidentally deploying a feature-only HEF while believing the whole algorithm is quantized.

## Operator audit and support boundary

The host export audit currently reports these operators in the complete graphs:

| Graph | Emitted operations requiring Hailo-8 compiler verification |
|---|---|
| PaDiM | `AveragePool`, `Clip`, `Concat`, `Gather`, `MatMul`, `ReduceMax`, `ReduceMean`, `Resize`, `Sqrt`, `Sub`, `Transpose`, and shape/constant plumbing. |
| PatchCore | `AveragePool`, `Concat`, `Div`, `MatMul`, `ReduceL2`, `ReduceMax`, `ReduceMean`, `Resize`, `Sqrt`, `Sub`, `Transpose`, and shape/constant plumbing. |

The repository does not include the Hailo Dataflow Compiler, so these graphs are **not marked as compiled or hardware-verified**. The audit script is:

```powershell
uv run python scripts/audit_hailo_ops.py
```

If the Hailo parser rejects any operation, the graph must be rewritten before deployment. Moving the rejected distance or reduction operation to ONNX Runtime would improve compatibility but would violate the requirement that the whole algorithm be quantized on Hailo. The implementation therefore fails closed: it requires final `image_scores` and `score_map` outputs and does not silently fall back to CPU distance calculation.

## Validation requirements

Before hardware deployment, compare the complete Hailo graph against the original PyTorch model on a held-out normal and anomalous set. Record image-score correlation, classification agreement at the selected threshold, pixel-map correlation, pixel AUROC, and P95 latency. Any calibration or compiler change requires repeating this parity check.

The current host test suite validates graph construction, full score/map output shapes, ONNX export, artifact validation, and calibration-manifest generation. Actual HEF compilation and Kria latency measurements require the Hailo SDK and physical hardware; they cannot be honestly claimed from a CPU-only development host.

## Important deployment constraints

The graph uses fixed input dimensions and fixed patch grids. PaDiM covariance tensors and PatchCore memory banks are embedded as constants, so artifact size and compiler memory must be checked for each selected backbone, feature dimension, and coreset size. Ultra-light PatchCore should use a bounded memory bank and reduced feature dimension for K26 deployment. PaDiM should use channel selection and a compact patch grid before compilation.

The Hailo toolchain performs the device-side quantization from representative data. A `float32` ONNX export is an input to that process, not evidence that the model has already been quantized. A CPU `int8` cast of outputs is not an acceptable substitute for compiling the distance and reduction operations into the HEF.

## References

[1]: https://hailo.ai/products/hailo-software/hailo-ai-software-suite/ "Hailo AI Software Suite"
[2]: https://github.com/hailo-ai/hailo-apps/tree/main/hailo_apps/cpp/onnxrt_hailo_pipeline "Hailo ONNX Runtime pipeline example"
[3]: https://www.amd.com/en/products/system-on-modules/kria/k26.html "AMD Kria K26 SOM"
