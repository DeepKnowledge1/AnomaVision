
# Hailo Quantization

<p align="center">

  <img src="./images/hailo8.png" width="30%" alt="Hailo8"/>

</p>
# Hailo Quantization

This guide shows how to export an AnomaVision model, parse it with the Hailo Dataflow Compiler (DFC), optimize it with representative images, and compile it to a HEF.

The workflow is intended for **end-to-end anomaly detection**. The Hailo graph should contain the feature extraction and the final anomaly score/map calculation.

## Requirements

* AnomaVision installed from source
* Hailo Dataflow Compiler / Hailo SDK
* Python environment compatible with the installed Hailo SDK
* Representative **normal/good images** for calibration
* A fixed input model, normally `224 x 224 RGB` for the current PatchCore workflow

Check the Hailo installation:

```bash
hailo --help
```

## 1. Export the complete Hailo ONNX graph

### PatchCore

```
python -m anomavision.quantize.model.backends.hef.exporter
--algorithm patchcore
--artifact distributions/patchcore/bottle/anomav_exp/model.pt
--calibration-dir /root/dataset/bottle/train/good
--output-dir distributions/patchcore/bottle/hailo
```
# Hailo Quantization

This guide shows how to export an AnomaVision model, parse it with the Hailo Dataflow Compiler (DFC), optimize it with representative images, and compile it to a HEF.

The workflow is intended for **end-to-end anomaly detection**. The Hailo graph should contain the feature extraction and the final anomaly score/map calculation.

## Requirements

* AnomaVision installed from source
* Hailo Dataflow Compiler / Hailo SDK
* Python environment compatible with the installed Hailo SDK
* Representative **normal/good images** for calibration
* A fixed input model, normally `224 x 224 RGB` for the current PatchCore workflow

Check the Hailo installation:

```bash
hailo --help
```

## 1. Export the complete Hailo ONNX graph

### PatchCore

python -m anomavision.quantize.model.backends.hef.exporter
--algorithm patchcore
--artifact distributions/patchcore/bottle/anomav_exp/model.pt
--calibration-dir /root/dataset/bottle/train/good
--output-dir distributions/patchcore/bottle/hailo

The exporter creates the ONNX graph and calibration manifest. It does **not** compile a HEF unless the Hailo compiler is explicitly configured.

## 2. Parse the ONNX model

For PatchCore:

```bash
hailo parser onnx \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.onnx
```

If the parser reports recommended end nodes, use the exact names printed by Hailo.

For the current PatchCore workflow, the successful parse used:

```bash
hailo parser onnx \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.onnx \
  --end-node-names "/MaxPool" "/Squeeze"
```

A successful parse produces:

```text
anomavision_patchcore_k26_end_to_end.har
```

## 3. Prepare calibration data

For the current PatchCore export, the network input is:

```text
224 x 224 x 3
```

Calibration samples must match the network input exactly.

Expected shape:

```text
(224, 224, 3)
```

Not:

```text
(1, 224, 224, 3)
```

Create `.npy` calibration files from normal images:

```python
from pathlib import Path

import numpy as np
from PIL import Image

src = Path("/root/dataset/bottle/train/good")
dst = Path("distributions/patchcore/bottle/hailo/calibration_npy")

dst.mkdir(parents=True, exist_ok=True)

paths = sorted(
    p
    for p in src.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

print(f"Found {len(paths)} calibration images")

for i, path in enumerate(paths):
    with Image.open(path) as im:
        im = im.convert("RGB").resize((224, 224))
        arr = np.asarray(im, dtype=np.float32)

    np.save(dst / f"{i:05d}.npy", arr)

print(f"Created {len(paths)} calibration tensors")
```

## 4. Optimize / quantize the HAR

Run Hailo optimization:

```bash
hailo optimize \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.har \
  --hw-arch hailo8 \
  --calib-set-path distributions/patchcore/bottle/hailo/calibration_npy
```

A successful optimization ends with:

```text
Model Optimization is done
```

## 5. Compile the optimized HAR to HEF

For Hailo-8:

```bash
hailo compiler \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end_optimized.har \
  --hw-arch hailo8
```

The compiler produces a `.hef` file.

## Hailo architecture

Use the architecture corresponding to the target device:

```text
hailo8  -> Hailo-8
hailo8l -> Hailo-8L
hailo8r -> Hailo-8R
```

## Hailo vs KV260

Hailo deployment and AMD/Xilinx KV260 DPU deployment are different paths:

```text
Hailo:
ONNX -> HAR -> optimized HAR -> HEF

KV260 DPU:
ONNX/INT8 -> XModel -> vai_c_xir -> XModel
```

A Hailo `.hef` cannot be used as a KV260 `.xmodel`, and an XModel is not a Hailo model.

For the KV260/XModel workflow, see `kv260_xmodel.md`.

## End-to-end requirement

For AnomaVision PatchCore, the intended Hailo graph includes the anomaly calculation, not only the backbone.

If the Hailo compiler cannot support an operation, do not describe a feature-extractor-only HEF as a fully quantized AnomaVision model. Either adapt the graph and validate numerical parity, or clearly document the remaining host-side operation.

The exporter creates the ONNX graph and calibration manifest. It does **not** compile a HEF unless the Hailo compiler is explicitly configured.

## 2. Parse the ONNX model

For PatchCore:

```bash
hailo parser onnx \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.onnx
```

If the parser reports recommended end nodes, use the exact names printed by Hailo.

For the current PatchCore workflow, the successful parse used:

```bash
hailo parser onnx \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.onnx \
  --end-node-names "/MaxPool" "/Squeeze"
```

A successful parse produces:

```text
anomavision_patchcore_k26_end_to_end.har
```

## 3. Prepare calibration data

For the current PatchCore export, the network input is:

```text
224 x 224 x 3
```

Calibration samples must match the network input exactly.

Expected shape:

```text
(224, 224, 3)
```

Not:

```text
(1, 224, 224, 3)
```

Create `.npy` calibration files from normal images:

```python
from pathlib import Path

import numpy as np
from PIL import Image

src = Path("/root/dataset/bottle/train/good")
dst = Path("distributions/patchcore/bottle/hailo/calibration_npy")

dst.mkdir(parents=True, exist_ok=True)

paths = sorted(
    p
    for p in src.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

print(f"Found {len(paths)} calibration images")

for i, path in enumerate(paths):
    with Image.open(path) as im:
        im = im.convert("RGB").resize((224, 224))
        arr = np.asarray(im, dtype=np.float32)

    np.save(dst / f"{i:05d}.npy", arr)

print(f"Created {len(paths)} calibration tensors")
```

## 4. Optimize / quantize the HAR

Run Hailo optimization:

```bash
hailo optimize \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.har \
  --hw-arch hailo8 \
  --calib-set-path distributions/patchcore/bottle/hailo/calibration_npy
```

A successful optimization ends with:

```text
Model Optimization is done
```

## 5. Compile the optimized HAR to HEF

For Hailo-8:

```bash
hailo compiler \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end_optimized.har \
  --hw-arch hailo8
```

The compiler produces a `.hef` file.

## Hailo architecture

Use the architecture corresponding to the target device:

```text
hailo8  -> Hailo-8
hailo8l -> Hailo-8L
hailo8r -> Hailo-8R
```

## Hailo vs KV260

Hailo deployment and AMD/Xilinx KV260 DPU deployment are different paths:

```text
Hailo:
ONNX -> HAR -> optimized HAR -> HEF

KV260 DPU:
ONNX/INT8 -> XModel -> vai_c_xir -> XModel
```

A Hailo `.hef` cannot be used as a KV260 `.xmodel`, and an XModel is not a Hailo model.

For the KV260/XModel workflow, see `kv260_xmodel.md`.

## End-to-end requirement

For AnomaVision PatchCore and PaDiM, the intended Hailo graph includes the anomaly calculation, not only the backbone.

If the Hailo compiler cannot support an operation, do not describe a feature-extractor-only HEF as a fully quantized AnomaVision model. Either adapt the graph and validate numerical parity, or clearly document the remaining host-side operation.




```
if you have no permission to write file:

sudo chown -R vitis-ai-user:vitis-ai-group /workspace

docker run --rm -it   -v ~/Vitis-AI/AnomaVision:/workspace/AnomaVision   -v /root/dataset:/workspace/dataset   xilinx/vitis-ai-pytorch-cpu:latest   bash
activate vitis-ai-pytorch


python -m anomavision.quantize.model.backends.hef.exporter \
  --algorithm patchcore \
  --artifact distributions/patchcore/bottle/anomav_exp/model.pt \
  --calibration-dir /root/dataset/bottle/train/good \
  --output-dir distributions/patchcore/bottle/hailo

hailo parser onnx \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.onnx \
  --end-node-names "/MaxPool" "/Squeeze"

hailo optimize \
  anomavision_patchcore_k26_end_to_end.har \
  --hw-arch hailo8 \
  --calib-set-path distributions/patchcore/bottle/hailo/calibration_npy

hailo compiler \
  anomavision_patchcore_k26_end_to_end_optimized.har \
  --hw-arch hailo8
```
