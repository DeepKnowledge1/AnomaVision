# 🛠️ Troubleshooting & FAQ

This guide covers **common issues** when using AnomaVision and how to fix them.

---

## 1. Installation Issues

### ❌ Problem: CUDA mismatch (`torch not compiled with CUDA enabled`)

**Cause:** Installed PyTorch version doesn’t match your CUDA toolkit.
**Fix:** Reinstall PyTorch with the correct CUDA version:

```bash
# Example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Or use Poetry:

```bash
poetry install --extras "cu121"
```

---

### ❌ Problem: `onnxruntime` not found

**Cause:** ONNX Runtime not installed.
**Fix:**

```bash
pip install onnxruntime-gpu onnxruntime-tools
```

---

### ❌ Problem: OpenVINO not working on Windows

**Cause:** Missing Intel OpenVINO dependencies.
**Fix:**

```bash
pip install openvino
```

Check [OpenVINO installation guide](https://docs.openvino.ai/) if errors persist.

---

## 2. Dataset Issues

### ❌ Problem: `FileNotFoundError: train/good not found`

**Cause:** Dataset not in MVTec structure.
**Expected Layout:**

```
dataset/
└── bottle/
    ├── train/good/
    └── test/broken_large/
```

**Fix:** Reorganize folders into the correct structure.

---

### ❌ Problem: Some images are ignored

**Cause:** Non-image files (e.g., `.txt`, `.DS_Store`) in dataset.
**Fix:** Remove or filter out invalid files.

---

## 3. Training Issues

### ❌ Problem: Training is very slow on CPU

**Fixes:**

* Use smaller image size (`resize: [128,128]`)
* Reduce `batch_size`
* Switch to GPU (`--device cuda`) if available

---

### ❌ Problem: `CUDA out of memory` during training

**Fixes:**

* Reduce `batch_size`
* Use smaller backbone (`resnet18` instead of `wide_resnet50`)
* Enable memory-efficient mode (where supported)

---

## 4. Inference Issues

### ❌ Problem: `RuntimeError: Input size mismatch`

**Cause:** Model expects a fixed input size.
**Fix:** Resize input images to the same size used during training (`resize` / `crop_size`).

---

### ❌ Problem: No anomalies detected (all scores low)

**Causes & Fixes:**

* Threshold too high → lower `--thresh`
* Wrong normalization → ensure dataset uses same `mean/std` as training
* Wrong class name → check `--class_name` or `config.yml`

---

## 5. Export Issues

### ❌ Problem: `Unsupported operator` during ONNX export

**Fixes:**

* Upgrade to latest PyTorch and ONNX
* Try lower opset (e.g., `--opset 16`)
* If still failing → use TorchScript export

---

### ❌ Problem: Quantized ONNX too large / inaccurate

**Fixes:**

* Use `--quantize-dynamic` for lightweight INT8
* If accuracy drops, stick to `fp16`

---

## 6. Visualization Issues

### ❌ Problem: Heatmaps look empty

**Causes & Fixes:**

* Wrong normalization → check `norm_mean` & `norm_std`
* `viz_alpha` too low → increase to `0.7`
* Threshold too high → lower `--thresh`

---

## 7. General FAQ

**Q: Which backbones are supported?**
A: Currently `resnet18` and `wide_resnet50`. More will be added in future.

**Q: How to deploy on edge devices without Python?**
A: Use the **C++ ONNX runtime** provided in `/docs/cpp/`.

**Q: Can I train on custom datasets?**
A: Yes, as long as the dataset follows **MVTec-style folder structure**.

---

✅ With this guide, you should be able to quickly solve most common problems.
If an issue persists, please [open a GitHub Issue](https://github.com/DeepKnowledge1/AnomaVision/issues) with details.

---

## 8. Asset Download Issues

### ❌ Problem: Red banner — `MISSING ASSET` with hard error

```
╔══════════════════════════════════════════════════════════╗
║  ✖  AnomaVision – MISSING ASSET                         ║
╠══════════════════════════════════════════════════════════╣
║  Model path is not set or does not exist.
║  Given    : ./distributions
║  Run with : --download_model_from_github
╚══════════════════════════════════════════════════════════╝
```

**Cause:** Model file or image path is missing and no download flag was passed.
**Fix:** Add the appropriate flag:

```bash
# Missing model
anomavision detect --config config.yml --model model.onnx --download_model_from_github

# Missing images
anomavision detect --config config.yml --model model.onnx --download_images_from_github

# Both missing
anomavision detect --config config.yml --model model.onnx \
  --download_model_from_github --download_images_from_github
```

Assets are downloaded from the `assets-stable` GitHub release and cached in `~/.anomavision/`.
Subsequent runs use the cache automatically — no re-download unless the cache is deleted.

---

### ❌ Problem: Yellow banner — auto-fallback to cache, but wrong model loaded

```
╔══════════════════════════════════════════════════════════╗
║  ⚠  AnomaVision – PATH NOT FOUND                        ║
╠══════════════════════════════════════════════════════════╣
║  Model path is not set or does not exist.
║  Given    : ./distributions
║  Action   : downloading / using cached assets …
║  Using    : C:\Users\you\.anomavision\models
╚══════════════════════════════════════════════════════════╝
```

**Cause:** `model_data_path` in config points to a non-existent directory.
**Fix:** Either correct `model_data_path` in `config.yml` to point to your trained model, or use `--download_model_from_github` to use the cached sample model.

---

### ❌ Problem: Cache exists but model still not found

**Cause:** The zip extracted into an unexpected subfolder.
**Fix:** Check what's in the cache:

```bash
# Windows
dir %USERPROFILE%\.anomavision\models /s

# Linux / macOS
find ~/.anomavision/models -type f
```

Then set `model_data_path` in your config to the folder containing the model file.
