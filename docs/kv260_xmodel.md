# AnomaVision → KV260 XModel

This guide explains how to quantize **AnomaVision PaDiM and PatchCore** with **Vitis AI 3.5**, generate an INT8 XModel, validate it, and compile it for the **AMD/Xilinx Kria KV260 DPU**.

The workflow is intended for a Linux environment with the Vitis AI tools available.

## 1. Workflow

The deployment flow is:

```text
AnomaVision PyTorch model
        ↓
Vitis AI INT8 calibration
        ↓
INT8 XModel
        ↓
XModel validation
        ↓
vai_c_xir + KV260 arch.json
        ↓
KV260 DPU compiled model
```

Both PaDiM and PatchCore use the same basic workflow.

## 2. Activate Vitis AI

Activate the Vitis AI PyTorch environment:

```bash
conda activate vitis-ai-pytorch
```

Check that the compiler is available:

```bash
vai_c_xir -h
```

You should be able to run the command without a `command not found` error.

## 3. Go to AnomaVision

```bash
cd /workspace/AnomaVision
```

Check the repository:

```bash
ls
```

The KV260 quantization scripts are:

```text
quantize_padim_kv260.py
quantize_patchcore_kv260.py
```

## 4. Prepare calibration images

Use **normal/good training images** for INT8 calibration.

For the MVTec bottle example:

```text
/workspace/dataset/bottle/train/good
```

Check the directory:

```bash
ls /workspace/dataset/bottle/train/good | head
```

The calibration set should contain representative normal images. It should normally come from the same type of data used to train the anomaly detector.

---

# PaDiM → KV260 XModel

## 5. PaDiM model

The example PaDiM model is:

```text
distributions/padim/bottle/anomav_exp/model.pt
```

The output directory used below is:

```text
compiled_padim_kv260
```

## 6. Run PaDiM INT8 calibration

Run the calibration phase:

```bash
python quantize_padim_kv260.py \
  --model distributions/padim/bottle/anomav_exp/model.pt \
  --calibration-dir /workspace/dataset/bottle/train/good \
  --output-dir compiled_padim_kv260 \
  --quant_mode calib
```

The calibration phase collects activation statistics and prepares the quantization configuration.

Check the generated files:

```bash
find compiled_padim_kv260 -maxdepth 2 -type f
```

## 7. Generate the PaDiM INT8 XModel

Run the script in test mode:

```bash
python quantize_padim_kv260.py \
  --model distributions/padim/bottle/anomav_exp/model.pt \
  --calibration-dir /workspace/dataset/bottle/train/good \
  --output-dir compiled_padim_kv260 \
  --quant_mode test
```

The important output is:

```text
compiled_padim_kv260/PadimKV260_int.xmodel
```

## 8. Validate the PaDiM XModel

Check that the XModel exists:

```bash
ls -lh compiled_padim_kv260/*.xmodel
```

You can also inspect it with XIR:

```bash
python -c "import xir; g=xir.Graph.deserialize('compiled_padim_kv260/PadimKV260_int.xmodel'); print('XModel OK:', g.get_name()); print('Ops:', len(g.get_ops()))"
```

If the graph loads successfully, the generated XModel is readable by XIR.

## 9. Compile PaDiM for KV260

Use the KV260 DPU architecture file:

```bash
vai_c_xir \
  -x compiled_padim_kv260/PadimKV260_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  -o compiled_padim_kv260/compiled \
  -n PadimKV260
```

The compiled output is placed under:

```text
compiled_padim_kv260/compiled
```

---

# PatchCore → KV260 XModel

## 10. PatchCore model

The example PatchCore model is:

```text
distributions/patchcore/bottle/anomav_exp/model.pt
```

Calibration images are the same normal training images:

```text
/workspace/dataset/bottle/train/good
```

## 11. Run PatchCore INT8 calibration

```bash
python quantize_patchcore_kv260.py \
  --model distributions/patchcore/bottle/anomav_exp/model.pt \
  --calibration-dir /workspace/dataset/bottle/train/good \
  --output-dir compiled_patchcore_kv260 \
  --quant_mode calib
```

Check the output:

```bash
find compiled_patchcore_kv260 -maxdepth 2 -type f
```

## 12. Generate the PatchCore INT8 XModel

Run test mode:

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

## 13. Validate the PatchCore XModel

```bash
ls -lh compiled_patchcore_kv260/*.xmodel
```

Inspect it with XIR:

```bash
python -c "import xir; g=xir.Graph.deserialize('compiled_patchcore_kv260/PatchCoreKV260_int.xmodel'); print('XModel OK:', g.get_name()); print('Ops:', len(g.get_ops()))"
```

A successful graph load confirms that the XModel can be deserialized by XIR.

## 14. Compile PatchCore for KV260

```bash
vai_c_xir \
  -x compiled_patchcore_kv260/PatchCoreKV260_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  -o compiled_patchcore_kv260/compiled \
  -n PatchCoreKV260
```

The compiled output is placed under:

```text
compiled_patchcore_kv260/compiled
```

---

# Expected output

After completing both workflows, the directories should contain the generated INT8 XModels and compiler output.

```text
compiled_padim_kv260/
├── PadimKV260_int.xmodel
└── compiled/
    └── ...

compiled_patchcore_kv260/
├── PatchCoreKV260_int.xmodel
└── compiled/
    └── ...
```

The exact files inside `compiled/` can vary with the Vitis AI compiler output.

## 15. Important notes

### Calibration vs. test mode

Use the two modes in this order:

```text
calib → test → vai_c_xir
```

Do not skip calibration when generating a calibrated INT8 model.

### Calibration data

Use representative **normal/good images**. Poor calibration data can reduce INT8 accuracy.

### KV260 architecture

The compiler must use the architecture file for the target KV260 DPU:

```text
/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
```

Do not use an architecture file for another DPU target.

### Linux/Vitis AI requirement

The XModel workflow depends on Vitis AI tooling and is intended to run in the supported Linux/Vitis AI environment. Windows-only execution is not expected to provide the required Vitis AI compiler commands.

### XModel generation vs. device validation

A successfully generated and compiled XModel does **not by itself prove end-to-end KV260 application correctness**. The final step is to deploy the compiled model on the KV260 and validate preprocessing, tensor layout, postprocessing, anomaly scores, and latency on the target hardware.

## 16. Quick reference

### PaDiM

```bash
# Calibration
python quantize_padim_kv260.py \
  --model distributions/padim/bottle/anomav_exp/model.pt \
  --calibration-dir /workspace/dataset/bottle/train/good \
  --output-dir compiled_padim_kv260 \
  --quant_mode calib

# XModel generation
python quantize_padim_kv260.py \
  --model distributions/padim/bottle/anomav_exp/model.pt \
  --calibration-dir /workspace/dataset/bottle/train/good \
  --output-dir compiled_padim_kv260 \
  --quant_mode test

# KV260 compilation
vai_c_xir \
  -x compiled_padim_kv260/PadimKV260_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  -o compiled_padim_kv260/compiled \
  -n PadimKV260
```

### PatchCore

```bash
# Calibration
python quantize_patchcore_kv260.py \
  --model distributions/patchcore/bottle/anomav_exp/model.pt \
  --calibration-dir /workspace/dataset/bottle/train/good \
  --output-dir compiled_patchcore_kv260 \
  --quant_mode calib

# XModel generation
python quantize_patchcore_kv260.py \
  --model distributions/patchcore/bottle/anomav_exp/model.pt \
  --calibration-dir /workspace/dataset/bottle/train/good \
  --output-dir compiled_patchcore_kv260 \
  --quant_mode test

# KV260 compilation
vai_c_xir \
  -x compiled_patchcore_kv260/PatchCoreKV260_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  -o compiled_patchcore_kv260/compiled \
  -n PatchCoreKV260
```

## 17. Troubleshooting

### `vai_c_xir: command not found`

The Vitis AI environment is probably not activated. Check:

```bash
conda activate vitis-ai-pytorch
vai_c_xir -h
```

### `arch.json` not found

Verify the KV260 architecture path:

```bash
ls -lh /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
```

If the file is not present, use the architecture path provided by your Vitis AI installation, but make sure it corresponds to the KV260 DPU.

### XModel does not exist after test mode

Check the script output and the contents of the selected output directory:

```bash
find compiled_padim_kv260 -maxdepth 2 -type f
```

or:

```bash
find compiled_patchcore_kv260 -maxdepth 2 -type f
```

Also confirm that the calibration phase completed successfully before running test mode.

### XModel cannot be deserialized

Check that the XModel was generated by the Vitis AI tooling in the active environment and that `xir` is available:

```bash
python -c "import xir; print('XIR OK')"
```

## 18. Final deployment step

Once `vai_c_xir` completes successfully, copy the compiled deployment artifacts to the KV260 application environment and run the corresponding AnomaVision inference pipeline.

For production deployment, validate the complete pipeline on the actual KV260 hardware rather than relying only on host-side XModel compilation.
