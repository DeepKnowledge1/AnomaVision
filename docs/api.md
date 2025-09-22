

# 📚 API Reference

AnomaVision provides a clean **Python API** for training, inference, evaluation, and exporting models.
This guide covers the main classes and functions.

---

## 1. Dataset

### `AnodetDataset`

Loads MVTec-style anomaly datasets with preprocessing.

```python
from anomavision import AnodetDataset
from torch.utils.data import DataLoader

dataset = AnodetDataset(
    root="./dataset/bottle/train/good",
    resize=(256, 192),
    crop_size=(224, 224),
    normalize=True,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

loader = DataLoader(dataset, batch_size=8, shuffle=False)
print(len(dataset))  # number of samples
```

---

## 2. Model

### `Padim`

PaDiM anomaly detection model with flexible backbones.

```python
import anomavision

# Initialize model
model = anomavision.Padim(
    backbone="resnet18",        # or wide_resnet50
    device="cuda",              # "cpu" or "cuda"
    layer_indices=[0, 1, 2],    # backbone layers
    feat_dim=100                # random features kept
)

# Train
model.fit(loader)

# Save full model
model.save("padim_model.pt")

# Save compact stats-only artifact
model.save_statistics("padim_model.pth", half=True)
```

---

## 3. Inference

```python
import torch

# Load images (e.g. from dataset or cv2/PIL)
x, _ = next(iter(loader))   # shape [B,C,H,W]

# Run prediction
scores, maps = model.predict(x)

print("Anomaly scores:", scores)
print("Heatmap shape:", maps.shape)
```

* **scores** → 1D anomaly scores per image
* **maps** → anomaly heatmaps per image

---

## 4. Evaluation

```python
from anomavision.eval import evaluate_model

metrics = evaluate_model(
    model=model,
    dataset_root="./dataset",
    class_name="bottle",
    batch_size=8,
    visualize=True,
    save_dir="./eval_results"
)

print(metrics)  # includes AUC, FPS, timings
```

---

## 5. Export

```python
from pathlib import Path
from anomavision.export import ModelExporter
from anomavision.utils import get_logger

logger = get_logger("anomavision.export")

exporter = ModelExporter(
    model_path=Path("./distributions/anomav_exp/padim_model.pt"),
    output_dir=Path("./exports"),
    logger=logger,
    device="cuda"
)

# Export to ONNX
onnx_path = exporter.export_onnx(
    input_shape=(1,3,224,224),
    output_name="padim.onnx",
    opset_version=17,
    quantize_dynamic_flag=True
)

# Export to TorchScript
ts_path = exporter.export_torchscript(
    input_shape=(1,3,224,224),
    output_name="padim.torchscript"
)

# Export to OpenVINO
ov_dir = exporter.export_openvino(
    input_shape=(1,3,224,224),
    output_name="padim_openvino",
    fp16=True
)
```

---

## 6. Utilities

* **Config Loader**

  ```python
  from anomavision.config import load_config
  cfg = load_config("config.yml")
  print(cfg)
  ```

* **Logging**

  ```python
  from anomavision.utils import setup_logging, get_logger
  setup_logging(enabled=True, log_level="INFO", log_to_file=True)
  logger = get_logger("anomavision.demo")
  logger.info("Hello, AnomaVision!")
  ```

---

✅ With this API, you can embed AnomaVision into:

* Research notebooks
* Production inference pipelines
* MLOps training/eval workflows
