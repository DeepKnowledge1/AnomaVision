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
uv run python -m anomavision.quantize.model.backends.hef.exporter `
  --algorithm padim `
  --artifact .\model_stats.pt `
  --calibration-dir .\normal_calibration `
  --output-dir .\hailo\padim_k26
```

or:

```powershell
uv run python -m anomavision.quantize.model.backends.hef.exporter `
  --algorithm patchcore `
  --artifact .\patchcore_artifact.pt `
  --calibration-dir .\normal_calibration `
  --output-dir .\hailo\patchcore_k26
```

This creates the complete ONNX graph and a calibration manifest. It does not claim to create a HEF unless the Hailo SDK is installed and an explicit compiler command is supplied:

```powershell
uv run python -m anomavision.quantize.model.backends.hef.exporter `
  --algorithm patchcore `
  --artifact .\patchcore_artifact.pt `
  --calibration-dir .\normal_calibration `
  --output-dir .\hailo\patchcore_k26 `
  --hailo-command "<Hailo SDK parse/optimize/compile command using {onnx} and {output}>"
```

The exact compiler command depends on the installed Hailo Dataflow Compiler release and target accelerator. It should include representative-image calibration and must produce a HEF whose outputs are exactly `image_scores` and `score_map`.

## Compile both models with the Hailo Dataflow Compiler

Run `scripts/compile_hailo8.py` inside the Hailo Dataflow Compiler environment. It compiles both exported complete graphs independently and writes one HEF per algorithm:

```powershell
uv run python scripts/compile_hailo8.py `
  --padim-onnx .\hailo\padim_k26\anomavision_padim_k26_end_to_end.onnx `
  --patchcore-onnx .\hailo\patchcore_k26\anomavision_patchcore_k26_end_to_end.onnx `
  --calibration-dir .\normal_calibration `
  --output-dir .\hailo8_compile `
  --hw-arch hailo8 `
  --calibration-limit 1024
```

Expected outputs are:

```text
hailo8_compile\padim\anomavision_padim_k26_hailo8.hef
hailo8_compile\patchcore\anomavision_patchcore_k26_hailo8.hef
hailo8_compile\compile_all_manifest.json
```

The script uses the Hailo `ClientRunner` flow: ONNX translation, calibration/optimization, compilation, and HEF creation. It fails if the Hailo SDK is missing, calibration images are absent, the ONNX input is not fixed NCHW RGB, or the compiler returns no HEF bytes. It does not silently replace distance calculation with CPU or ONNX Runtime postprocessing.

## XModel path for the K26 DPU

The AMD DPU path is separate from Hailo-8. Hailo uses `.hef`; the K26 DPU uses `.xmodel`. Start from a Vitis AI XIR graph and compile it with:

```powershell
uv run python -m anomavision.quantize.model.backends.xmodel.compiler `
  --xir .\xir\anomavision_k26.xir `
  --arch .\arch\DPUCZDX8G_ISA1_B4096.json `
  --output-dir .\xmodel\k26
```

The compiler requires `vai_c_xir` from the Vitis AI toolchain and fails if no `.xmodel` is produced. It does not convert a Hailo HEF into an XModel. XModel inference is intentionally not selected by `ModelWrapper` until the AMD Vitis AI runtime is installed on the board.

## Verify supported layers and fallback status

Run the verification script after compilation:

```powershell
uv run python scripts/verify_hailo8_graph.py `
  --padim-onnx .\hailo\padim_k26\anomavision_padim_k26_end_to_end.onnx `
  --patchcore-onnx .\hailo\patchcore_k26\anomavision_patchcore_k26_end_to_end.onnx `
  --padim-hef .\hailo8_compile\padim\anomavision_padim_k26_hailo8.hef `
  --patchcore-hef .\hailo8_compile\patchcore\anomavision_patchcore_k26_hailo8.hef `
  --padim-har .\hailo8_compile\padim\anomavision_padim_k26_hailo8.har `
  --patchcore-har .\hailo8_compile\patchcore\anomavision_patchcore_k26_hailo8.har `
  --compiler-log .\hailo8_compile\compiler.log `
  --output .\hailo8_compile\verification.json
```

The verifier checks that both ONNX graphs expose the complete `image_scores` and `score_map` outputs, lists all graph operators, requires a non-empty HEF, and scans the HAR/compiler evidence for CPU, ONNX Runtime, host-postprocess, or fallback markers. A HEF without a same-build HAR or compiler log is reported as **not proven fallback-free** rather than being accepted automatically.

Without the Hailo compiler artifacts, the only valid result is `onnx_only_not_hardware_verified`. A successful ONNX export is not evidence that Hailo-8 supports every operation.

## Package locations

The implementation is organized with the existing backend conventions:

```text
anomavision/inference/model/backends/hailo_backend.py
anomavision/inference/model/backends/k260_backend.py
anomavision/quantize/model/backends/hef/exporter.py
anomavision/quantize/model/backends/hef/verifier.py
anomavision/quantize/model/backends/hef/audit.py
anomavision/quantize/model/backends/xmodel/compiler.py
```

## Kria runtime

Copy the generated HEF and the runtime package to the Linux image running on the Kria K26. The runtime adapter is:

```python
from anomavision.inference.model.backends.hailo_backend import HailoAnomalyRuntime

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
