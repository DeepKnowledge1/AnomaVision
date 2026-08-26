# Hailo Quantization

This guide shows how to export an AnomaVision model, parse it with the Hailo Dataflow Compiler (DFC), optimize it with representative images, and compile it to a HEF.

The workflow is intended for **end-to-end anomaly detection**. The Hailo graph should contain the feature extraction and the final anomaly score/map calculation. Do not move the distance calculation back to CPU after Hailo feature extraction.

## Requirements

- AnomaVision installed from source
- Hailo Dataflow Compiler / Hailo SDK
- Python environment compatible with the installed Hailo SDK
- Representative **normal/good images** for calibration
- A fixed input model, normally `224 x 224 RGB` for the current PatchCore workflow

Check the Hailo installation:

```bash
hailo --help
hailo parser --help
hailo optimize --help
hailo compiler --help
```

The compiler target must match the Hailo device:

```text
hailo8   -> Hailo-8
hailo8l  -> Hailo-8L
hailo8r  -> Hailo-8R
```

## 1. Export the complete Hailo ONNX graph

Use the trained AnomaVision artifact and a directory containing normal calibration images.

### PatchCore

```bash
anomavision quantize \
  --algorithm patchcore \
  --artifact distributions/patchcore/bottle/anomav_exp/model.pt \
  --calibration-dir /root/dataset/bottle/train/good \
  --output-dir distributions/patchcore/bottle/hailo
```

The export should create:

```text
distributions/patchcore/bottle/hailo/
└── anomavision_patchcore_k26_end_to_end.onnx
```

The exporter creates the ONNX graph and calibration manifest. It does **not** compile a HEF unless the Hailo compiler is explicitly configured.

### PaDiM

```bash
anomavision quantize \
  --algorithm padim \
  --artifact distributions/padim/bottle/anomav_exp/model.pt \
  --calibration-dir /root/dataset/bottle/train/good \
  --output-dir distributions/padim/bottle/hailo
```

## 2. Parse the ONNX model

Start with the normal parser command:

```bash
hailo parser onnx distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.onnx
```

If the parser reports recommended end nodes, use the exact names printed by Hailo. For example, during the current PatchCore Hailo workflow the parser eventually succeeded with:

```bash
hailo parser onnx \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.onnx \
  --end-node-names "/MaxPool" "/Squeeze"
```

The successful parse produces a HAR file, for example:

```text
anomavision_patchcore_k26_end_to_end.har
```

Do not assume that the first recommended end nodes are the final solution. If Hailo reports an unsupported reduction or reshape, adjust the graph/end nodes and parse again. The goal is a successfully translated graph containing the required final outputs.

## 3. Prepare calibration data

Hailo optimization requires representative calibration data.

For the current PatchCore export, the network input is:

```text
224 x 224 x 3
```

Calibration samples must match the network input exactly.

For example, if the Hailo model expects:

```text
(224, 224, 3)
```

do not provide samples shaped as:

```text
(1, 224, 224, 3)
```

The extra batch dimension can cause:

```text
BadInputsShape: Data shape (1, 224, 224, 3)
doesn't match network's input shape (224, 224, 3)
```

Before optimization, inspect generated `.npy` calibration files if necessary:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

files = list(Path("distributions/patchcore/bottle/hailo").rglob("*.npy"))
print(f"Found {len(files)} npy files")

for f in files[:10]:
    print(f, np.load(f).shape)
PY
```

For the current graph, the expected shape is:

```text
(224, 224, 3)
```

## 4. Optimize / quantize the HAR

Run Hailo optimization using the HAR produced by the parser and the representative calibration set.

A typical Hailo DFC command is:

```bash
hailo optimize anomavision_patchcore_k26_end_to_end.har \
  --hw-arch hailo8 \
  --calib-set-path /root/dataset/bottle/train/good \
  --output-har-path anomavision_patchcore_k26_end_to_end_optimized.har
```

Use the exact option names exposed by your installed Hailo DFC version:

```bash
hailo optimize --help
```

A successful optimization ends with messages similar to:

```text
Model Optimization is done
Saved HAR to: ..._optimized.har
```

This is the actual Hailo optimization/quantization stage. The ONNX file itself is not a Hailo-quantized model.

## 5. Compile the optimized HAR to HEF

Compile the optimized HAR for the actual target device.

For Hailo-8:

```bash
hailo compiler anomavision_patchcore_k26_end_to_end_optimized.har \
  --hw-arch hailo8
```

For Hailo-8L:

```bash
hailo compiler anomavision_patchcore_k26_end_to_end_optimized.har \
  --hw-arch hailo8l
```

The compiler should produce a `.hef` file. The HEF is the deployable Hailo model.

## 6. Verify the result

Check that the HEF exists and is non-empty:

```bash
ls -lh *.hef
```

Keep the following artifacts together when debugging or validating a build:

```text
model.pt
       ↓
complete ONNX
       ↓
parsed HAR
       ↓
optimized HAR
       ↓
HEF
```

The optimized HAR and compiler output are important evidence that the graph was actually processed by Hailo. An ONNX export alone is not proof of Hailo hardware support.

## Troubleshooting

### `UnsupportedShuffleLayerError` / `Reshape`

If parsing fails around `Reshape` and Hailo recommends end nodes, try the recommended end-node names first. If necessary, simplify the exported graph rather than blindly accepting a feature-only graph.

### `UnsupportedReduceMaxLayerError`

Hailo may require `ReduceMax` to operate on a supported axis with `keepdim=True`. If the final anomaly-score reduction is not accepted, the ONNX graph must be rewritten or the graph boundaries must be changed while preserving the required end-to-end outputs.

### `StopIteration` during optimization

This usually means the calibration path configured for Hailo contains no files in the format Hailo expects. Check the calibration directory and make sure the generated `.npy` files are actually present when using an `.npy` calibration dataset.

### `BadInputsShape`

Check the calibration sample shape against the network input. For the current PatchCore graph:

```text
network:    (224, 224, 3)
calibration: (224, 224, 3)
```

Remove an unwanted leading batch dimension from calibration samples when required.

## Important: Hailo-8 vs KV260 DPU

Hailo deployment and AMD/Xilinx KV260 DPU deployment are different paths:

```text
Hailo      -> ONNX -> HAR -> optimized HAR -> HEF
KV260 DPU  -> ONNX/INT8 -> XModel -> vai_c_xir -> XModel
```

A Hailo `.hef` cannot be used as a KV260 `.xmodel`, and an XModel is not a Hailo model.

For the KV260/XModel workflow, see [`kv260_xmodel.md`](kv260_xmodel.md).

## End-to-end requirement

For AnomaVision PatchCore and PaDiM, the intended Hailo graph includes the anomaly calculation, not only the backbone. The final deployment should therefore expose the model's anomaly outputs without requiring a CPU-side PatchCore nearest-neighbour calculation or PaDiM Mahalanobis calculation after Hailo inference.

If the Hailo compiler cannot support an operation, do not silently describe a feature-extractor-only HEF as a fully quantized AnomaVision model. Either adapt the graph and validate numerical parity, or clearly document the remaining host-side operation.
