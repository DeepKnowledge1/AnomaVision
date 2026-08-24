# PatchCore → KV260 XModel

Simple beginner guide for quantizing **AnomaVision PatchCore** with **Vitis AI 3.5** and generating an **XModel for KV260**.

## 1. Activate Vitis AI

```bash
conda activate vitis-ai-pytorch
```

Check:

```bash
vai_c_xir -h
```

---

## 2. Go to AnomaVision

```bash
cd /workspace/AnomaVision
```

---

## 3. Check the model

Our PatchCore model:

```text
distributions/patchcore/bottle/anomav_exp/model.pt
```

Calibration images:

```text
/workspace/dataset/bottle/train/good
```

---

## 4. Create calibration data

PatchCore needs **normal/good images** for calibration.

Example:

```bash
ls /workspace/dataset/bottle/train/good | head
```

You should see images such as:

```text
000.png
001.png
002.png
...
```

---

## 5. Run INT8 calibration

Run:

```bash
python quantize_patchcore_kv260.py \
  --model distributions/patchcore/bottle/anomav_exp/model.pt \
  --calibration-dir /workspace/dataset/bottle/train/good \
  --output-dir compiled_patchcore_kv260 \
  --quant_mode calib
```

Successful calibration should finish with:

```text
Calibration finished.
Quant config exported.
```

---

## 6. Generate the XModel

Run:

```bash
python quantize_patchcore_kv260.py \
  --model distributions/patchcore/bottle/anomav_exp/model.pt \
  --calibration-dir /workspace/dataset/bottle/train/good \
  --output-dir compiled_patchcore_kv260 \
  --quant_mode test
```

The important output is:

```text
compiled_patchcore_kv260/PatchCoreKV260_int.xmodel
```

---

## 7. Check the XModel

```bash
ls -lh compiled_patchcore_kv260/*.xmodel
```

Then:

```bash
python -c "import xir; g=xir.Graph.deserialize('compiled_patchcore_kv260/PatchCoreKV260_int.xmodel'); print('XModel OK:', g.get_name()); print('Ops:', len(g.get_ops()))"
```

Expected:

```text
XModel OK: PatchCoreKV260
Ops: 98
```

---

## 8. Compile for KV260

Use the KV260 architecture file:

```bash
vai_c_xir \
  -x compiled_patchcore_kv260/PatchCoreKV260_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  -o compiled_patchcore_kv260/compiled
```

---

## Final result

You want:

```text
compiled_patchcore_kv260/
├── PatchCoreKV260_int.xmodel
└── compiled/
    └── PatchCoreKV260_int.xmodel
```

The final XModel is intended for the **AMD/Xilinx KV260 DPU**.
