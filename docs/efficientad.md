# EfficientAD

EfficientAD is available as a native AnomaVision algorithm and uses the same training, model loading, detection, evaluation, and export workflow as PaDiM and PatchCore.

The implementation follows the EfficientAD student/teacher design: a frozen EfficientNet teacher provides normal feature targets, a lightweight student learns those features, and a compact autoencoder adds a global reconstruction signal. The original EfficientAD paper combines local teacher/student discrepancy with global reconstruction discrepancy for fast anomaly detection.

## Select the algorithm

The existing top-level selector remains supported:

```yaml
algorithm: efficientad
```

AnomaVision also accepts the native model selector:

```yaml
model:
  name: efficientad
```

For existing projects, changing only `algorithm` is the safest option because the rest of the current configuration remains unchanged.

## Training

```bash
anomavision train --config config.yml
```

Recommended starting values:

```yaml
algorithm: efficientad
resize: [224, 224]
normalize: true
batch_size: 1
efficientad_model_size: s
efficientad_lr: 0.0001
efficientad_weight_decay: 0.00001
efficientad_epochs: 1
efficientad_pretrained_teacher: true
```

EfficientAD uses ImageNet preprocessing for the teacher, so `normalize: true` is required by the AnomaVision integration.

## Detection

Use the same command as the other algorithms:

```bash
anomavision detect --config config.yml --model model.pt --img_path ./test_images
```

The model returns an image-level anomaly score and a full-resolution anomaly map, so the existing visualization and post-processing pipeline can be reused.

## Export

The trained PyTorch model can be exported through the existing exporter:

```bash
anomavision export --config config.yml --model model.pt --format onnx
```

The ONNX graph contains the EfficientAD inference path, including the teacher, student, autoencoder, score calculation, and anomaly map generation.

## Evaluation

```bash
anomavision eval --config config.yml --model model.pt --class_name bottle
```

EfficientAD has its own score distribution, so thresholds should be calibrated independently from PaDiM. Keep `thresh_efficientad: null` to let evaluation determine an appropriate threshold where the existing evaluation workflow supports automatic threshold selection.

## Model artifacts

Training produces the same primary artifact layout used by the other AnomaVision algorithms:

```text
distributions/
└── efficientad/
    └── bottle/
        └── anomav_exp/
            ├── model.pt
            ├── model.pth
            └── config.yml
```

`model.pt` is the complete PyTorch model used by the normal AnomaVision inference backend. `model.pth` is a self-contained EfficientAD checkpoint artifact containing the model state and metadata.

## Difference from PaDiM

EfficientAD is not numerically interchangeable with PaDiM. PaDiM models feature distributions with Gaussian statistics, while EfficientAD learns a student/teacher representation and reconstruction model. Consequently, scores, thresholds, training time, and localization patterns will differ. What remains intentionally identical is the AnomaVision contract: dataset input, CLI commands, artifact layout, `fit`, `predict`, model loading, and ONNX export.
