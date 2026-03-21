# 🛠️ CLI Reference

AnomaVision provides a unified `anomavision` command with four subcommands:

```bash
anomavision train    # Train a PaDiM anomaly detection model
anomavision detect   # Run inference on test images
anomavision eval     # Evaluate performance on MVTec-style datasets
anomavision export   # Export models to ONNX, TorchScript, or OpenVINO
```

Each subcommand accepts both **CLI arguments** and **config files** (`--config config.yml`).
**CLI arguments always override config file values.**

Get help at any level:

```bash
anomavision --help
anomavision train --help
anomavision detect --help
anomavision eval --help
anomavision export --help
```

---

## 1. Training — `anomavision train`

```bash
anomavision train [options]
```

| Argument            | Type     | Default         | Description                                          |
| ------------------- | -------- | --------------- | ---------------------------------------------------- |
| `--config`          | str      | config.yml      | Path to config file                                  |
| `--dataset_path`    | str      | None            | Dataset root containing `train/good`                 |
| `--resize`          | int(s)   | None            | Resize images (one value = square, two values = W H) |
| `--crop_size`       | int(s)   | None            | Crop size (one = square, two = W H)                  |
| `--normalize`       | flag     | False           | Enable normalization                                 |
| `--no_normalize`    | flag     | False           | Disable normalization (overrides `--normalize`)      |
| `--norm_mean`       | float(3) | None            | Normalization mean (RGB)                             |
| `--norm_std`        | float(3) | None            | Normalization std (RGB)                              |
| `--backbone`        | str      | resnet18        | Feature extractor (`resnet18`, `wide_resnet50`)      |
| `--batch_size`      | int      | 2               | Batch size                                           |
| `--feat_dim`        | int      | 50              | Number of random features                            |
| `--layer_indices`   | int list | [0]             | Backbone layer indices                               |
| `--output_model`    | str      | model.pt  | Model filename (`.pt`)                               |
| `--run_name`        | str      | anomav_exp       | Experiment name                                      |
| `--model_data_path` | str      | ./distributions | Output directory                                     |
| `--log_level`       | str      | INFO            | Logging level                                        |

**Example:**

```bash
anomavision train \
  --config config.yml \
  --dataset_path ./dataset \
  --backbone resnet18 \
  --batch_size 16
```

---

## 2. Detection — `anomavision detect`

```bash
anomavision detect [options]
```

| Argument                 | Type  | Default                   | Description                            |
| ------------------------ | ----- | ------------------------- | -------------------------------------- |
| `--config`               | str   | None                      | Path to config file                    |
| `--img_path`             | str   | None                      | Path to test images                    |
| `--model_data_path`      | str   | ./distributions/anomav_exp | Directory with model files             |
| `--model`                | str   | model.pt            | Model file (`.pt`, `.onnx`, `.engine`) |
| `--device`               | str   | auto                      | Device (`cpu`, `cuda`, `auto`)         |
| `--batch_size`           | int   | 1                         | Batch size                             |
| `--thresh`               | float | None                      | Anomaly threshold                      |
| `--num_workers`          | int   | 1                         | Data loader workers                    |
| `--pin_memory`           | flag  | False                     | Use pinned memory (GPU transfer)       |
| `--enable_visualization` | flag  | False                     | Show anomaly maps                      |
| `--save_visualizations`  | flag  | False                     | Save images to disk                    |
| `--viz_output_dir`       | str   | ./visualizations          | Save path                              |
| `--run_name`             | str   | detect_exp                | Experiment name                        |
| `--overwrite`            | flag  | False                     | Overwrite run dir                      |
| `--log_level`            | str   | INFO                      | Logging level                          |
| `--detailed_timing`                 | flag  | False                     | Log detailed timings                                                                     |
| `--download_model_from_github`      | flag  | False                     | Download `model_bottle.zip` from GitHub (`assets-stable`) if model is missing locally   |
| `--download_images_from_github`     | flag  | False                     | Download `sample_bottle_images.zip` from GitHub (`assets-stable`) if images are missing |

> ⚠️ Without these flags, a missing model or image path raises a hard error with clear instructions on what to run.

**Example — standard run:**

```bash
anomavision detect \
  --config config.yml \
  --img_path ./dataset/bottle/test \
  --thresh 13.0 \
  --enable_visualization \
  --save_visualizations
```

**Example — first run, no local assets:**

```bash
anomavision detect \
  --config config.yml \
  --model model.onnx \
  --download_model_from_github \
  --download_images_from_github
```


---

## 3. Evaluation — `anomavision eval`

```bash
anomavision eval [options]
```

| Argument                 | Type | Default                   | Description                |
| ------------------------ | ---- | ------------------------- | -------------------------- |
| `--config`               | str  | None                      | Path to config file        |
| `--dataset_path`         | str  | None                      | Root dataset path          |
| `--class_name`           | str  | bottle                    | Class name (MVTec style)   |
| `--model_data_path`      | str  | ./distributions/anomav_exp | Directory with model files |
| `--model`                | str  | model.onnx                | Model file                 |
| `--device`               | str  | auto                      | Device (`cpu`, `cuda`)     |
| `--batch_size`           | int  | 32                        | Batch size                 |
| `--num_workers`          | int  | 1                         | Data loader workers        |
| `--pin_memory`           | flag | False                     | Use pinned memory          |
| `--thresh`               | float| None                      | Threshold (auto-computed if not set) |
| `--enable_visualization` | flag | False                     | Show plots                 |
| `--save_visualizations`  | flag | False                     | Save plots                 |
| `--viz_output_dir`       | str  | ./eval_visualizations     | Output path                |
| `--log_level`            | str  | INFO                      | Logging level              |
| `--detailed_timing`      | flag | False                     | Log detailed timings       |

**Example:**

```bash
anomavision eval \
  --config config.yml \
  --dataset_path ./dataset \
  --class_name bottle \
  --enable_visualization
```

---

## 4. Export — `anomavision export`

```bash
anomavision export [options]
```

| Argument             | Type | Default                   | Description                                              |
| -------------------- | ---- | ------------------------- | -------------------------------------------------------- |
| `--config`           | str  | None                      | Path to config file                                      |
| `--model_data_path`  | str  | ./distributions/anomav_exp | Directory with model & outputs                           |
| `--model`            | str  | *(required)*              | Model file (`.pt`)                                       |
| `--format`           | str  | *(required)*              | Export format (`onnx`, `torchscript`, `openvino`, `all`) |
| `--device`           | str  | auto                      | Export device                                            |
| `--precision`        | str  | auto                      | Precision (`fp32`, `fp16`, `auto`)                       |
| `--opset`            | int  | 17                        | ONNX opset version                                       |
| `--static-batch`     | flag | False                     | Disable dynamic batch                                    |
| `--optimize`         | flag | False                     | Optimize TorchScript for mobile                          |
| `--quantize-dynamic` | flag | False                     | Export dynamic INT8 ONNX                                 |
| `--quantize-static`  | flag | False                     | Export static INT8 ONNX (needs calibration)              |
| `--calib-samples`    | int  | 100                       | Calibration samples                                      |
| `--log_level`        | str  | INFO                      | Logging level                                            |

**Example:**

```bash
anomavision export \
  --model_data_path ./distributions/anomav_exp \
  --model model.pt \
  --format onnx \
  --precision fp16 \
  --quantize-dynamic
```

---

## Config File + CLI Override Pattern

All subcommands follow the same priority: **CLI args > config file > defaults**.

```bash
# Config sets backbone=resnet18, CLI overrides to wide_resnet50
anomavision train --config config.yml --backbone wide_resnet50
```

This makes it easy to run sweeps or one-off experiments without editing config files.
