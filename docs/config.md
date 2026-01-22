

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
| `thresh`               | float | None        | Anomaly threshold.                   |
| `enable_visualization` | bool  | False       | Enable heatmap overlays.             |
| `save_visualizations`  | bool  | False       | Save visualization images.           |
| `viz_output_dir`       | str   | ./results/  | Directory to save images.            |
| `viz_alpha`            | float | 0.5         | Heatmap transparency.                |
| `viz_padding`          | int   | 40          | Padding around bounding boxes.       |
| `viz_color`            | str   | "128,0,128" | RGB highlight color.                 |

---

## 4. Evaluation

| Key                | Type | Default | Description                      |
| ------------------ | ---- | ------- | -------------------------------- |
| `memory_efficient` | bool | True    | Use memory-efficient evaluation. |
| `detailed_timing`  | bool | False   | Log per-image timings.           |

(Other keys mirror **Detection** and **Training**.)

---

## 5. Export

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

---

## 6. Logging

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
output_model: padim_model.pt
run_name: exp1
model_data_path: ./distributions/anomav_exp

model: padim_model.onnx
device: auto
enable_visualization: true
save_visualizations: true
viz_output_dir: ./results/

format: onnx
precision: fp16
quantize_dynamic: true
```

---
