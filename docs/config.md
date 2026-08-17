

# ⚙️ Configuration Guide

AnomaVision scripts (`train.py`, `detect.py`, `eval.py`, `export.py`) all accept a **YAML/JSON config file**.
You can override any field via CLI arguments.

Example:

```bash
python train.py --config config.yml
```

---

## 1. Dataset & Preprocessing

| Key            | Type       | Default                | Description                                                                                  |
| -------------- | ---------- | ---------------------- | -------------------------------------------------------------------------------------------- |
| `dataset_path` | str        | None                   | Root dataset folder containing MVTec-style structure (`class/train/good`, `class/test/...`). |
| `class_name`   | str        | None                   | Target class name (e.g. `bottle`, `cable`).                                                  |
| `resize`       | \[int,int] | None                   | Resize before processing (e.g. `[256,192]`). One value applies square resize.                |
| `crop_size`    | \[int,int] | None                   | Center crop size (e.g. `[224,224]`). One value = square crop.                                |
| `normalize`    | bool       | True                   | Apply input normalization.                                                                   |
| `norm_mean`    | \[float]   | \[0.485, 0.456, 0.406] | Mean values (RGB) if normalize is enabled.                                                   |
| `norm_std`     | \[float]   | \[0.229, 0.224, 0.225] | Std values (RGB) if normalize is enabled.                                                    |

---

## 2. Training

| Key               | Type | Default         | Description                                      |
| ----------------- | ---- | --------------- | ------------------------------------------------ |
| `backbone`        | str  | resnet18        | Feature extractor (`resnet18`, `wide_resnet50`). |
| `batch_size`      | int  | 16              | Training batch size.                             |
| `feat_dim`        | int  | 100             | Number of random feature dimensions kept.        |
| `layer_indices`   | list | \[0,1,2]        | Backbone layers used for features.               |
| `run_name`        | str  | exp             | Name of training run.                            |
| `model_data_path` | str  | ./distributions | Where trained models/configs are stored.         |
| `output_model`    | str  | padim\_model.pt | Name of saved model.                             |

---

## 3. Detection

| Key                    | Type  | Default     | Description                          |
| ---------------------- | ----- | ----------- | ------------------------------------ |
| `img_path`             | str   | None        | Path to test images or folder.       |
| `model`                | str   | None        | Model file (`.pt`, `.pth`, `.onnx`). |
| `device`               | str   | auto        | Device (`cpu`, `cuda`, or `auto`).   |
| `batch_size`           | int   | 1           | Batch size for inference.            |
| `thresh`               | float | None        | Legacy global anomaly threshold fallback. |
| `thresh_padim`         | float | None        | PaDiM-specific threshold; takes precedence over `thresh`. |
| `thresh_patchcore`     | float | None        | PatchCore-specific threshold; takes precedence over `thresh`. |
| `enable_visualization` | bool  | False       | Enable heatmap overlays.             |
| `save_visualizations`  | bool  | False       | Save visualization images.           |
| `viz_output_dir`       | str   | ./results/  | Directory to save images.            |
| `viz_alpha`            | float | 0.5         | Heatmap transparency.                |
| `viz_padding`          | int   | 40          | Padding around bounding boxes.       |
| `viz_color`            | str   | "128,0,128" | RGB highlight color.                 |

---

## 4. PatchCore

| Key | Type | Default | Description |
|---|---|---:|---|
| `coreset_ratio` | float | 0.02 | Fraction of normal patches retained. |
| `max_memory_patches` | int | 2048 | Hard memory-bank cap. |
| `patch_grid` | int | 14 | Spatial pooling grid for lightweight localization. |
| `search_chunk_size` | int | 1024 | Chunk size for bounded nearest-neighbor search. |
| `coreset_method` | str | kcenter | `kcenter` for deterministic diverse selection or `random`. |
| `coreset_seed` | int | 42 | Reproducibility seed for coreset selection. |

---

## 5. Evaluation

| Key                | Type | Default | Description                      |
| ------------------ | ---- | ------- | -------------------------------- |
| `memory_efficient` | bool | True    | Use memory-efficient evaluation. |
| `detailed_timing`  | bool | False   | Log per-image timings.           |

(Other keys mirror **Detection** and **Training**.)

---

## 6. Export

| Key                | Type | Default | Description                                               |
| ------------------ | ---- | ------- | --------------------------------------------------------- |
| `format`           | str  | onnx    | Export target (`onnx`, `torchscript`, `openvino`, `all`). |
| `precision`        | str  | auto    | Precision (`fp32`, `fp16`, or `auto`).                    |
| `opset`            | int  | 17      | ONNX opset version.                                       |
| `static_batch`     | bool | False   | Disable dynamic batch.                                    |
| `optimize`         | bool | False   | TorchScript mobile optimization.                          |
| `quantize_dynamic` | bool | False   | Export dynamic INT8 ONNX.                                 |
| `quantize_static`  | bool | False   | Export static INT8 ONNX (requires calibration).           |
| `calib_samples`    | int  | 100     | Calibration samples for static quantization.              |
| `tensorrt_precision` | str | fp16 | TensorRT precision (`fp32`, `fp16`, or `int8`). |
| `workspace_gb`     | float | 2.0 | TensorRT builder workspace limit in GB. |
| `min_batch`        | int | 1 | Minimum TensorRT dynamic batch size. |
| `opt_batch`        | int | 1 | Optimized TensorRT dynamic batch size. |
| `max_batch`        | int | 4 | Maximum TensorRT dynamic batch size. |
| `calib_dir`        | str/null | null | Real-image directory for TensorRT INT8 calibration. |

---

## 7. Production Autopilot

Autopilot reads `dataset_path` and `class_name`, uses the complete labeled `test` split by default, and writes `deployment_manifest.json`, `production_autopilot_report.html`, and a Markdown fallback. Its HTML report distinguishes image AUROC, pixel AUROC, anomaly localization coverage, normal-image false-positive localization, and anomaly mean mask area.

---

## 8. Logging

| Key         | Type | Default | Description                                          |
| ----------- | ---- | ------- | ---------------------------------------------------- |
| `log_level` | str  | INFO    | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## Example Config

```yaml
dataset_path: ./dataset
class_name: bottle
resize: [256, 192]
crop_size: [224, 224]
normalize: true
norm_mean: [0.485, 0.456, 0.406]
norm_std: [0.229, 0.224, 0.225]

backbone: resnet18
batch_size: 16
feat_dim: 100
layer_indices: [0, 1, 2]
output_model: model.pt
run_name: exp1
model_data_path: ./distributions/padim/bottle/anomav_exp

model: model.onnx
device: auto
enable_visualization: true
save_visualizations: true
viz_output_dir: ./results/

format: onnx
precision: fp16
quantize_dynamic: true

# Algorithm-specific thresholds
thresh: null
thresh_padim: null
thresh_patchcore: null

# Ultra-light PatchCore
algorithm: patchcore
coreset_method: kcenter
coreset_seed: 42
coreset_ratio: 0.02
max_memory_patches: 2048
patch_grid: 14
search_chunk_size: 1024
```

---
