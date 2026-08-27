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

Export a complete graph:

```bash
python -m anomavision.quantize.model.backends.hef.exporter \
  --algorithm patchcore \
  --artifact distributions/patchcore/bottle/anomav_exp/model.pt \
  --calibration-dir /root/dataset/bottle/train/good \
  --output-dir distributions/patchcore/bottle/hailo
```

For PaDiM:

```bash
python -m anomavision.quantize.model.backends.hef.exporter \
  --algorithm padim \
  --artifact distributions/padim/bottle/model_stats.pt \
  --calibration-dir /root/dataset/bottle/train/good \
  --output-dir distributions/padim/bottle/hailo
```

This creates the complete ONNX graph and calibration manifest. It does not claim that a HEF exists until the Hailo compiler has actually produced one.

## Compile ONNX -> HAR -> optimized HAR -> HEF

Use the repository script from the Hailo DFC environment:

```bash
bash scripts/hailo_compile.sh \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.onnx \
  /root/dataset/bottle/train/good \
  hailo8
```

The script performs all required steps:

```text
1. Convert representative calibration images to HxWx3 .npy samples
2. hailo parser       ONNX -> HAR
3. hailo optimize     HAR -> INT8 optimized HAR
4. hailo compiler     optimized HAR -> HEF
```

**Important:** the ONNX model, HAR files, and HEF are kept in the **same directory as the ONNX model**. The script never puts the HEF/HAR in the parent directory.

For the PatchCore example the directory becomes:

```text
distributions/patchcore/bottle/hailo/
├── anomavision_patchcore_k26_end_to_end.onnx
├── anomavision_patchcore_k26_end_to_end.har
├── anomavision_patchcore_k26_end_to_end_optimized.har
├── anomavision_patchcore_k26_end_to_end.hef
└── calibration_npy/
```

The same layout is used for PaDiM. This makes the model path the single source of truth for all deployment artifacts.

The script defaults to `hailo8`. For another supported target, pass the architecture as the third argument:

```bash
bash scripts/hailo_compile.sh \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.onnx \
  /root/dataset/bottle/train/good \
  hailo8l
```

## Manual commands

If you do not want to use the script, run the same operations from the model directory so every generated artifact stays beside the model:

```bash
cd distributions/patchcore/bottle/hailo

hailo parser onnx anomavision_patchcore_k26_end_to_end.onnx --hw-arch hailo8

hailo optimize \
  anomavision_patchcore_k26_end_to_end.har \
  --hw-arch hailo8 \
  --calib-set-path ./calibration_npy

hailo compiler \
  anomavision_patchcore_k26_end_to_end_optimized.har \
  --hw-arch hailo8
```

After the final command, the `.hef` must also be in this same directory.

## Validate the generated artifacts

Before deploying, verify that the expected files exist:

```bash
MODEL_DIR=distributions/patchcore/bottle/hailo
MODEL_NAME=anomavision_patchcore_k26_end_to_end

ls -lh "$MODEL_DIR/$MODEL_NAME.onnx"
ls -lh "$MODEL_DIR/$MODEL_NAME.har"
ls -lh "$MODEL_DIR/${MODEL_NAME}_optimized.har"
ls -lh "$MODEL_DIR/$MODEL_NAME.hef"
```

You can also inspect the HEF with Hailo tools available in your DFC/HailoRT environment.

## Runtime command

Once a physical Hailo device is available, the normal AnomaVision command should use the HEF directly:

```bash
anomavision detect \
  --config config.yml \
  --model distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.hef \
  --img_path ./test_images
```

The corresponding ONNX validation command is:

```bash
anomavision detect \
  --config config.yml \
  --model distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.onnx \
  --img_path ./test_images
```

The goal is that both commands produce nearly identical anomaly scores and heatmaps. Hailo-specific runtime execution cannot be validated on a host without a Hailo device; compilation success alone does not prove numerical parity.

## XModel path for the K26 DPU

The AMD DPU path is separate from Hailo-8. Hailo uses `.hef`; the K26 DPU uses `.xmodel`. Start from a Vitis AI XIR graph and compile it with the Vitis AI toolchain.

## Package locations

Both accelerators follow the same `InferenceBackend` protocol. The implementation is organized with the existing backend conventions:

```text
anomavision/inference/model/backends/hailo_backend.py
anomavision/inference/model/backends/k260_backend.py
anomavision/quantize/model/backends/hef/exporter.py
anomavision/quantize/model/backends/hef/verifier.py
anomavision/quantize/model/backends/hef/audit.py
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

## Validation requirements

Before hardware deployment, compare the complete Hailo graph against the original PyTorch and ONNX models on a held-out normal and anomalous set. Record image-score correlation, classification agreement at the selected threshold, pixel-map correlation, pixel AUROC, and P95 latency. Any calibration or compiler change requires repeating this parity check.

The repository can validate graph construction and ONNX behavior without a Hailo device. Actual HEF inference and Kria latency measurements require the Hailo SDK/HailoRT and physical hardware.

## Important deployment constraints

The graph uses fixed input dimensions and fixed patch grids. PaDiM covariance tensors and PatchCore memory banks are embedded as constants, so artifact size and compiler memory must be checked for each selected backbone, feature dimension, and coreset size. Ultra-light PatchCore should use a bounded memory bank and reduced feature dimension for K26 deployment. PaDiM should use channel selection and a compact patch grid before compilation.

The Hailo toolchain performs device-side quantization from representative data. A float32 ONNX export is an input to that process, not evidence that the model has already been quantized. A CPU int8 cast of outputs is not an acceptable substitute for compiling the distance and reduction operations into the HEF.
