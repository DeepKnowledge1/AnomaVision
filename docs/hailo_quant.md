# Hailo Quantization

<p align="center">
  <img src="./images/hailo8.png" width="30%" alt="Hailo-8"/>
</p>

This guide shows how to convert an AnomaVision model into a **Hailo HEF** for Hailo-8.

The workflow is:

```text
AnomaVision Model
       ↓
     ONNX
       ↓
      HAR
       ↓
Optimized HAR
       ↓
      HEF
```

The goal is **end-to-end anomaly detection**, including feature extraction and anomaly score/map calculation.

## Requirements

* AnomaVision installed from source
* Hailo Dataflow Compiler (DFC)
* HailoRT 5.3.0
* Python 3.10 environment
* Normal/good images for calibration
* Hailo-8 target

Check the Hailo installation:

```bash
hailo --help
```

---

## 1. Install HailoRT 5.3.0

HailoRT requires both the native runtime and Python bindings.

### Native Runtime

Download the HailoRT 5.3.0 Ubuntu `.deb` from the [HailoRT documentation](https://hailo.ai/developer-zone/documentation/hailort-v5-3-0/).

Install it:

```bash
cd /root
dpkg -i hailort_5.3.0_amd64.deb
```

Verify:

```bash
find /usr /lib -name 'libhailort.so.5.3.0' 2>/dev/null
```

### Python Package

Install the Python wheel:

```bash
uv pip install /root/hailort-5.3.0-cp310-cp310-linux_x86_64.whl
```

Verify:

```bash
python -c "from hailo_platform import HEF; print('HailoRT OK')"
```

Expected:

```text
HailoRT OK
```

> **Note:** The Python `.whl` alone is not enough. The native `.deb` provides `libhailort.so.5.3.0`.

---

## 2. Export the AnomaVision Model

For PatchCore:

```bash
python -m anomavision.quantize.model.backends.hef.exporter \
  --algorithm patchcore \
  --artifact distributions/patchcore/bottle/anomav_exp/model.pt \
  --calibration-dir /root/dataset/bottle/train/good \
  --output-dir distributions/patchcore/bottle/hailo
```

This creates the complete PatchCore ONNX graph.

---

## 3. Parse the ONNX Model

```bash
hailo parser onnx \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_end_to_end.onnx \
  --end-node-names "/MaxPool" "/Squeeze"
```

A successful parse creates:

```text
anomavision_patchcore_end_to_end.har
```

---

## 4. Prepare Calibration Data

PatchCore currently uses:

```text
224 × 224 × 3
```

Calibration data must have this shape:

```text
(224, 224, 3)
```

Create calibration `.npy` files from normal images:

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

Use **normal/good images only** for calibration.

---

## 5. Optimize and Quantize

```bash
hailo optimize \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_end_to_end.har \
  --hw-arch hailo8 \
  --calib-set-path distributions/patchcore/bottle/hailo/calibration_npy
```

Successful optimization ends with:

```text
Model Optimization is done
```

---

## 6. Compile to HEF

```bash
hailo compiler \
  distributions/patchcore/bottle/hailo/anomavision_patchcore_end_to_end_optimized.har \
  --hw-arch hailo8
```

The compiler produces the final `.hef` model.

---

## Hailo Architecture

Use the architecture corresponding to your Hailo device:

```text
hailo8   → Hailo-8
hailo8l  → Hailo-8L
hailo8r  → Hailo-8R
```

For this guide:

```text
--hw-arch hailo8
```

## End-to-End Anomaly Detection

For AnomaVision PatchCore and PaDiM, the Hailo graph should contain the complete anomaly detection pipeline:

```text
Input
  ↓
Feature Extraction
  ↓
Anomaly Calculation
  ↓
Anomaly Score + Anomaly Map
```

If an operation is not supported by Hailo, the graph must be adapted and its numerical results validated before calling it a complete end-to-end model.


## 7. Run the HEF

After compiling the model, run the generated `.hef` with AnomaVision:

```bash
anomavision detect \
  --config config.yml \
  --model model.hef
```

For example:

```bash
anomavision detect \
  --config config.yml \
  --model distributions/patchcore/bottle/hailo/anomavision_patchcore_end_to_end.hef
```

> **Note:** Running a HEF requires a connected and accessible Hailo device.



python -m anomavision.quantize.model.backends.hef.exporter   --algorithm efficientad   --artifact distributions/efficientad/bottle/anomav_exp/model.pt   --calibration-dir /root/dataset/bottle/train/good   --output-dir distributions/efficientad/bottle/hailo


hailo parser onnx \
  distributions/efficientad/bottle/hailo/anomavision_efficientad_k26_end_to_end.onnx \
  --end-node-names "/MaxPool" "/Squeeze"


hailo optimize \
  anomavision_efficientad_k26_end_to_end.har \
  --hw-arch hailo8 \
  --calib-set-path distributions/patchcore/bottle/hailo/calibration_npy


hailo compiler \
  anomavision_efficientad_k26_end_to_end.har \
  --hw-arch hailo8

