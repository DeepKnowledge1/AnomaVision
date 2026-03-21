# 🚀 Quick Start

This guide shows how to **train, detect, evaluate, export, and stream** with AnomaVision in just a few steps.

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
anomavision train \
  --config config.yml
  --dataset_path ./dataset \
  --class_name bottle \
  --backbone resnet18 \
  --batch_size 16 \
  --feat_dim 100 \
  --layer_indices 0 1 2 \
  --output_model model.pt \
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
output_model: model.pt
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
anomavision train --config config.yml
```

---

✅ Both approaches will:

* Train PaDiM on `dataset/bottle/train/good`
* Save:
  * Full model → `model.pt`
  * Compact stats-only model → `model.pth`
  * Config snapshot → `config.yml`

---

## 3. Run Detection

> 💡 **First time?** If you don't have model files or test images locally, use the auto-download flags:
> ```bash
> anomavision detect --config config.yml --model model.onnx \
>   --download_model_from_github --download_images_from_github
> ```
> This downloads `model_bottle.zip` and `sample_bottle_images.zip` from the GitHub `assets-stable` release
> into `~/.anomavision/` and uses them automatically. Without these flags, missing assets raise a hard error.

### Offline Detection (Static Images)

#### Option A — CLI Arguments

```bash
anomavision detect \
  --img_path ./dataset/bottle/test \
  --model_data_path ./distributions/anomav_exp \
  --model model.onnx \
  --device auto \
  --batch_size 8 \
  --thresh 13.0 \
  --enable_visualization \
  --save_visualizations \
  --viz_output_dir ./results/
```

---

#### Option B — Config File

Create a **`config.yml`** or use the one saved in the model's directory:

```yaml
stream_mode: false
img_path: ./dataset/bottle/test
model_data_path: ./distributions/anomav_exp
model: model.onnx
device: auto
batch_size: 8
thresh: 13.0
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
anomavision detect --config config.yml
```

---

### Real-Time Streaming Detection

AnomaVision supports multiple streaming sources: **webcam, video files, MQTT, and TCP**.

#### Webcam Streaming

```yaml
stream_mode: true
stream_source:
  type: webcam
  camera_id: 0

model_data_path: ./distributions/anomav_exp
model: model.onnx
batch_size: 1
thresh: 13.0
enable_visualization: true
save_visualizations: true
stream_max_frames: null  # null = infinite
stream_display_fps: true
stream_save_detections: true
stream_detection_dir: ./stream_detections/
```

#### Video File Streaming

```yaml
stream_mode: true
stream_source:
  type: video
  video_path: path/to/video.mp4
  loop: false

model_data_path: ./distributions/anomav_exp
model: model.onnx
batch_size: 1
thresh: 13.0
```

#### MQTT Streaming

```yaml
stream_mode: true
stream_source:
  type: mqtt
  broker: localhost
  port: 1883
  topic: camera/frames
  client_id: anomavision_client
  keepalive: 60
  qos: 0
  max_queue_size: 10
  read_timeout: 1.0

model_data_path: ./distributions/anomav_exp
model: model.onnx
batch_size: 1
```

#### TCP Streaming

```yaml
stream_mode: true
stream_source:
  type: tcp
  host: 192.168.1.100
  port: 8080
  recv_timeout: 1.0
  header_size: 4
  max_message_size: 10485760

model_data_path: ./distributions/anomav_exp
model: model.onnx
batch_size: 1
```

Run streaming detection:

```bash
anomavision detect --config config.yml
```

---

➡ Detection will:

* Run inference on the **test dataset** or **streaming source**
* Log anomaly scores and classifications
* Save **visualization images** (boundaries, heatmaps, highlighted anomalies) in `./visualizations/`
* For streaming: continuously process frames until interrupted (Ctrl+C) or max frames reached

---

## 4. Evaluate Performance

### Option A — CLI Arguments

```bash
anomavision eval \
  --dataset_path ./dataset \
  --class_name bottle \
  --model_data_path ./distributions/anomav_exp \
  --model model.onnx \
  --batch_size 8 \
  --enable_visualization \
  --save_visualizations \
  --viz_output_dir ./eval_results/
```

---

### Option B — Config File

Create a **`config.yml`** or use the one saved in the model's directory:

```yaml
dataset_path: ./dataset
class_name: bottle
model_data_path: ./distributions/anomav_exp
model: model.onnx
batch_size: 8
enable_visualization: true
save_visualizations: true
viz_output_dir: ./eval_results/
log_level: INFO
normalize: true
resize: [256, 192]
crop_size: [224, 224]
memory_efficient: true
```

Run:

```bash
anomavision eval --config config.yml
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
anomavision export \
  --model_data_path ./distributions/anomav_exp \
  --model model.pt \
  --format onnx \
  --precision fp16 \
  --quantize-dynamic
```

---

### Option B — Config File

Create a **`config.yml`** or use the one saved in the model's directory:

```yaml
model_data_path: ./distributions/anomav_exp
model: model.pt
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
anomavision export --config config.yml
```

---

➡ Both methods will:

* Export the model in the selected format
* Save artifacts in `./distributions/anomav_exp`
* Optionally produce **quantized ONNX models** (dynamic or static INT8)

---

## Key Configuration Parameters

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Batch size for processing | 2 |
| `device` | Device to run on | auto |
| `thresh` | Anomaly detection threshold | 13.0 |
| `normalize` | Apply normalization | true |
| `resize` | Image resize dimensions [W, H] | [224, 224] |
| `crop_size` | Crop dimensions [W, H] | null |

### Streaming Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `stream_mode` | Enable streaming mode | false |
| `stream_max_frames` | Max frames to process | null (infinite) |
| `stream_display_fps` | Show FPS counter | true |
| `stream_save_detections` | Save anomalies | true |
| `stream_detection_dir` | Output directory | ./stream_detections/ |

### Visualization Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enable_visualization` | Enable visualizations | false |
| `save_visualizations` | Save to disk | false |
| `viz_alpha` | Heatmap transparency | 0.5 |
| `viz_padding` | Border padding | 40 |
| `viz_color` | Highlight color (RGB) | 128,0,128 |

---

## Important Notes

⚠️ **Streaming Mode Requirements:**
- `num_workers` must be 0 for streaming sources
- `pin_memory` must be false for streaming sources
- Batch size should typically be 1 for real-time processing
- The dataloader cannot prefetch for streaming sources

⚠️ **Performance Tips:**
- Use ONNX or TensorRT models for faster inference
- Enable GPU acceleration with `device: cuda`
- Adjust batch size based on available memory
- Use FP16 precision for better performance on GPUs

---
