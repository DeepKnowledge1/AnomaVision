
# 🚀 Quick Start

This guide shows how to **train, detect, evaluate, and export** with AnomaVision in just a few steps.

---

## 1. Prepare Dataset

AnomaVision supports **MVTec AD** and custom datasets.
The dataset folder should look like:

```
dataset/
└── bottle/
    ├── train/
    │   └── good/
    │       ├── 000.png
    │       ├── 001.png
    │       └── ...
    └── test/
        ├── good/
        │   ├── 100.png
        │   └── ...
        └── broken_large/
            ├── 200.png
            └── ...
```

---



## 2. Train a Model

### Option A — CLI Arguments

```bash
python train.py \
  --dataset_path ./dataset \
  --class_name bottle \
  --backbone resnet18 \
  --batch_size 16 \
  --feat_dim 100 \
  --layer_indices 0 1 2 \
  --output_model padim_model.pt \
  --run_name exp1 \
  --model_data_path ./distributions/anomav_exp
```

---

### Option B — Config File

Create a **`config.yml`** or use the default one:

```yaml
dataset_path: ./dataset
class_name: bottle
backbone: resnet18
batch_size: 16
feat_dim: 100
layer_indices: [0, 1, 2]
output_model: padim_model.pt
run_name: exp1
model_data_path: ./distributions/anomav_exp
resize: [256, 192]
crop_size: [224, 224]
normalize: true
norm_mean: [0.485, 0.456, 0.406]
norm_std: [0.229, 0.224, 0.225]
log_level: INFO
```

Then run:

```bash
python train.py --config config.yml
```

---

✅ Both approaches will:

* Train PaDiM on `dataset/bottle/train/good`
* Save:

  * Full model → `padim_model.pt`
  * Compact stats-only model → `padim_model.pth`
  * Config snapshot → `config.yml`

---


---

## 3. Run Detection

### Option A — CLI Arguments

```bash
python detect.py \
  --img_path ./dataset/bottle/test \
  --model_data_path ./distributions/anomav_exp \
  --model padim_model.onnx \
  --device auto \
  --batch_size 8 \
  --enable_visualization \
  --save_visualizations \
  --viz_output_dir ./results/
```

---

### Option B — Config File

Create a **`config.yml`** or use the one saved in the model’s directory.:

```yaml
img_path: ./dataset/bottle/test
model_data_path: ./distributions/anomav_exp
model: padim_model.onnx
device: auto
batch_size: 8
enable_visualization: true
save_visualizations: true
viz_output_dir: ./results/
viz_alpha: 0.5
viz_padding: 40
viz_color: "128,0,128"
log_level: INFO
```

Run:

```bash
python detect.py --config config.yml
```

---

➡ Both options will:

* Run inference on the **test dataset**
* Log anomaly scores and classifications
* Save **visualization images** (boundaries, heatmaps, highlighted anomalies) in `./visualization/`


---

## 4. Evaluate Performance

### Option A — CLI Arguments

```bash
python eval.py \
  --dataset_path ./dataset \
  --class_name bottle \
  --model_data_path ./distributions/anomav_exp \
  --model padim_model.onnx \
  --batch_size 8 \
  --enable_visualization \
  --save_visualizations \
  --viz_output_dir ./eval_results/
```

---

### Option B — Config File

Create a **`config.yml`** or use the one saved in the model’s directory.:

```yaml
dataset_path: ./dataset
class_name: bottle
model_data_path: ./distributions/anomav_exp
model: padim_model.onnx
batch_size: 8
enable_visualization: true
save_visualizations: true
viz_output_dir: ./eval_results/
log_level: INFO
normalize: true
resize: [256, 192]
crop_size: [224, 224]
```

Run:

```bash
python eval.py --config config.yml
```

---

➡ Both methods will:

* Evaluate the model on **MVTec test set**
* Report **AUC, FPS, avg inference time, throughput**
* Save **evaluation plots** (ROC, PR, histograms, anomaly maps) to `./eval_results/`

---

## 5. Export Model

You can export trained models to **ONNX**, **TorchScript**, or **OpenVINO**.
Quantization (INT8) is also supported.

---

### Option A — CLI Arguments

```bash
python export.py \
  --model_data_path ./distributions/anomav_exp \
  --model padim_model.pt \
  --format onnx \
  --precision fp16 \
  --quantize-dynamic
```

---

### Option B — Config File

Create a **`config.yml`** or use the one saved in the model’s directory.:

```yaml
model_data_path: ./distributions/anomav_exp
model: padim_model.pt
format: onnx          # choices: onnx | torchscript | openvino | all
precision: fp16       # fp32 | fp16 | auto
opset: 17
static_batch: false
quantize_dynamic: true
quantize_static: false
calib_samples: 100
dataset_path: ./dataset
class_name: bottle
log_level: INFO
```

Run:

```bash
python export.py --config config.yml
```

---

➡ Both methods will:

* Export the model in the selected format
* Save artifacts in `./distributions/anomav_exp`
* Optionally produce **quantized ONNX models** (dynamic or static INT8)

---
