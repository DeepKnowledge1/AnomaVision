# ⚙️ Configuration Guide

AnomaVision scripts (`train.py`, `detect.py`, `eval.py`, `export.py`) all accept a **YAML/JSON config file**. You can override any field via CLI arguments.

## 1. Dataset & Preprocessing

| Key | Type | Default | Description |
|---|---|---|---|
| `dataset_path` | str | None | Root dataset folder containing MVTec-style structure. |
| `class_name` | str | None | Target class name. |
| `resize` | [int,int] | None | Resize before processing. |
| `crop_size` | [int,int] | None | Center crop size. |
| `normalize` | bool | True | Apply input normalization. |
| `norm_mean` | [float] | [0.485,0.456,0.406] | RGB mean. |
| `norm_std` | [float] | [0.229,0.224,0.225] | RGB standard deviation. |

## 2. Algorithm selection

The historical selector remains supported:

```yaml
algorithm: padim
```

Available values are `padim`, `patchcore`, and `efficientad`.

A native model selector is also supported:

```yaml
model:
  name: efficientad
```

When `model.name` is used, `load_config()` maps it to the same internal `algorithm` value, so the existing CLI workflow does not change.

## 3. Training

| Key | Type | Default | Description |
|---|---|---:|---|
| `backbone` | str | resnet18 | PaDiM/PatchCore feature extractor. |
| `batch_size` | int | 16 | Training batch size. |
| `feat_dim` | int | 100 | PaDiM feature dimensions. |
| `layer_indices` | list | [0,1,2] | PaDiM/PatchCore feature layers. |
| `run_name` | str | exp | Training run name. |
| `model_data_path` | str | ./distributions | Model artifact root. |
| `output_model` | str | padim_model.pt | Saved model filename. |

### EfficientAD

| Key | Type | Default | Description |
|---|---|---:|---|
| `efficientad_model_size` | str | s | EfficientAD size: `s` or `m`. |
| `efficientad_lr` | float | 0.0001 | Adam learning rate. |
| `efficientad_weight_decay` | float | 0.00001 | Adam weight decay. |
| `efficientad_epochs` | int | 1 | Number of normal-data training epochs. |
| `efficientad_pretrained_teacher` | bool | true | Use ImageNet-pretrained EfficientNet teacher. |

EfficientAD requires `normalize: true` in the current integration because the teacher uses ImageNet preprocessing.

## 4. Detection

| Key | Type | Default | Description |
|---|---|---|---|
| `img_path` | str | None | Path to test images or folder. |
| `model` | str | None | Model file (`.pt`, `.pth`, `.onnx`). |
| `device` | str | auto | Device (`cpu`, `cuda`, or `auto`). |
| `batch_size` | int | 1 | Inference batch size. |
| `thresh` | float | None | Legacy global anomaly threshold. |
| `thresh_padim` | float | None | PaDiM-specific threshold. |
| `thresh_patchcore` | float | None | PatchCore-specific threshold. |
| `thresh_efficientad` | float | None | EfficientAD-specific threshold. |
| `enable_visualization` | bool | False | Enable heatmap overlays. |
| `save_visualizations` | bool | False | Save visualization images. |
| `viz_output_dir` | str | ./results/ | Visualization directory. |

## 5. PatchCore

| Key | Type | Default | Description |
|---|---|---:|---|
| `coreset_ratio` | float | 0.02 | Fraction of normal patches retained. |
| `max_memory_patches` | int | 2048 | Hard memory-bank cap. |
| `patch_grid` | int | 14 | Spatial pooling grid. |
| `search_chunk_size` | int | 1024 | Nearest-neighbor chunk size. |
| `coreset_method` | str | kcenter | `kcenter` or `random`. |
| `coreset_seed` | int | 42 | Coreset reproducibility seed. |

## 6. Evaluation

| Key | Type | Default | Description |
|---|---|---|---|
| `memory_efficient` | bool | True | Use memory-efficient evaluation. |
| `detailed_timing` | bool | False | Log detailed timings. |

Other keys mirror **Detection** and **Training**.

## 7. Export

| Key | Type | Default | Description |
|---|---|---:|---|
| `format` | str | onnx | `onnx`, `torchscript`, `openvino`, `all`. |
| `precision` | str | auto | `fp32`, `fp16`, or `auto`. |
| `opset` | int | 17 | ONNX opset version. |
| `static_batch` | bool | False | Disable dynamic batch. |
| `quantize_dynamic` | bool | False | Export dynamic INT8 ONNX. |
| `quantize_static` | bool | False | Export static INT8 ONNX. |
| `calib_samples` | int | 100 | Static quantization samples. |

## Example: switch PaDiM → EfficientAD

```yaml
# Everything else in the existing config can remain unchanged.
algorithm: efficientad

efficientad_model_size: s
efficientad_lr: 0.0001
efficientad_weight_decay: 0.00001
efficientad_epochs: 1
efficientad_pretrained_teacher: true
```

Then use the same commands:

```bash
anomavision train --config config.yml
anomavision export --config config.yml --model model.pt --format onnx
anomavision detect --config config.yml --model model.onnx --img_path ./test_images
anomavision eval --config config.yml --model model.pt --class_name bottle
```

See [`docs/efficientad.md`](efficientad.md) for the complete EfficientAD guide.
