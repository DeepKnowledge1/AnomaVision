
# 🛠️ CLI Reference

AnomaVision provides four main entry points:

* `train.py` → Train a PaDiM anomaly detection model
* `detect.py` → Run inference on test images
* `eval.py` → Evaluate performance on MVTec-style datasets
* `export.py` → Export models to ONNX, TorchScript, or OpenVINO

Each script accepts both **CLI arguments** and **config files** (`--config config.yml`).

---

## 1. Training — `train.py`

```bash
python train.py [options]
```

| Argument            | Type     | Default    | Description                                          |
| ------------------- | -------- | ---------- | ---------------------------------------------------- |
| `--config`          | str      | config.yml | Path to config file                                  |
| `--dataset_path`    | str      | None       | Dataset root containing `train/good`                 |
| `--resize`          | int(s)   | None       | Resize images (one value = square, two values = W H) |
| `--crop_size`       | int(s)   | None       | Crop size (one = square, two = W H)                  |
| `--normalize`       | flag     | None       | Enable normalization                                 |
| `--no_normalize`    | flag     | None       | Disable normalization (overrides `--normalize`)      |
| `--norm_mean`       | float(3) | None       | Normalization mean (RGB)                             |
| `--norm_std`        | float(3) | None       | Normalization std (RGB)                              |
| `--backbone`        | str      | resnet18   | Feature extractor (`resnet18`, `wide_resnet50`)      |
| `--batch_size`      | int      | 16         | Batch size                                           |
| `--feat_dim`        | int      | None       | Number of random features                            |
| `--layer_indices`   | int list | None       | Backbone layer indices                               |
| `--output_model`    | str      | None       | Model filename (`.pt`)                               |
| `--run_name`        | str      | None       | Experiment name                                      |
| `--model_data_path` | str      | None       | Output directory                                     |
| `--log_level`       | str      | INFO       | Logging level                                        |

---

## 2. Detection — `detect.py`

```bash
python detect.py [options]
```

| Argument                 | Type  | Default                     | Description                            |
| ------------------------ | ----- | --------------------------- | -------------------------------------- |
| `--config`               | str   | None                        | Path to config file                    |
| `--img_path`             | str   | None                        | Path to test images                    |
| `--model_data_path`      | str   | ./distributions/anomav\_exp | Directory with model files             |
| `--model`                | str   | padim\_model.onnx           | Model file (`.pt`, `.onnx`, `.engine`) |
| `--device`               | str   | auto                        | Device (`cpu`, `cuda`, `auto`)         |
| `--batch_size`           | int   | 1                           | Batch size                             |
| `--thresh`               | float | None                        | Anomaly threshold                      |
| `--num_workers`          | int   | 1                           | Data loader workers                    |
| `--pin_memory`           | flag  | False                       | Use pinned memory (GPU transfer)       |
| `--enable_visualization` | flag  | False                       | Show anomaly maps                      |
| `--save_visualizations`  | flag  | False                       | Save images to disk                    |
| `--viz_output_dir`       | str   | ./visualizations/           | Save path                              |
| `--run_name`             | str   | detect\_exp                 | Experiment name                        |
| `--overwrite`            | flag  | False                       | Overwrite run dir                      |
| `--viz_alpha`            | float | 0.5                         | Heatmap transparency                   |
| `--viz_padding`          | int   | 40                          | Boundary padding                       |
| `--viz_color`            | str   | 128,0,128                   | RGB anomaly highlight                  |
| `--log_level`            | str   | INFO                        | Logging level                          |
| `--detailed_timing`      | flag  | False                       | Log detailed timings                   |

---

## 3. Evaluation — `eval.py`

```bash
python eval.py [options]
```

| Argument                 | Type | Default                     | Description                  |
| ------------------------ | ---- | --------------------------- | ---------------------------- |
| `--config`               | str  | None                        | Path to config file          |
| `--dataset_path`         | str  | ./data                      | Root dataset path            |
| `--class_name`           | str  | bottle                      | Class name (MVTec style)     |
| `--model_data_path`      | str  | ./distributions/anomav\_exp | Directory with model files   |
| `--model`                | str  | padim\_model.onnx           | Model file                   |
| `--device`               | str  | auto                        | Device (`cpu`, `cuda`)       |
| `--batch_size`           | int  | 1                           | Batch size                   |
| `--num_workers`          | int  | 1                           | Data loader workers          |
| `--pin_memory`           | flag | False                       | Use pinned memory            |
| `--memory_efficient`     | flag | True                        | Enable memory-efficient eval |
| `--enable_visualization` | flag | False                       | Show plots                   |
| `--save_visualizations`  | flag | False                       | Save plots                   |
| `--viz_output_dir`       | str  | ./eval\_visualizations/     | Output path                  |
| `--log_level`            | str  | INFO                        | Logging level                |
| `--detailed_timing`      | flag | False                       | Log detailed timings         |

---

## 4. Export — `export.py`

```bash
python export.py [options]
```

| Argument             | Type | Default                     | Description                                              |
| -------------------- | ---- | --------------------------- | -------------------------------------------------------- |
| `--config`           | str  | None                        | Path to config file                                      |
| `--model_data_path`  | str  | ./distributions/anomav\_exp | Directory with model & outputs                           |
| `--model`            | str  | None                        | Model file (`.pt`)                                       |
| `--output_path`      | str  | None                        | Output filename                                          |
| `--format`           | str  | all                         | Export format (`onnx`, `torchscript`, `openvino`, `all`) |
| `--device`           | str  | auto                        | Export device                                            |
| `--precision`        | str  | auto                        | Precision (`fp32`, `fp16`, `auto`)                       |
| `--opset`            | int  | 17                          | ONNX opset version                                       |
| `--static-batch`     | flag | False                       | Disable dynamic batch                                    |
| `--optimize`         | flag | False                       | Optimize TorchScript for mobile                          |
| `--quantize-dynamic` | flag | False                       | Export dynamic INT8 ONNX                                 |
| `--quantize-static`  | flag | False                       | Export static INT8 ONNX (needs calibration)              |
| `--calib-samples`    | int  | 100                         | Calibration samples                                      |
| `--log-level`        | str  | INFO                        | Logging level                                            |

---
