# Hailo PaDiM

## Goal

Run the PaDiM pipeline with Hailo acceleration while keeping the existing PaDiM implementation unchanged.

## Initial flow

```
Image
  |
Preprocessing
  |
Hailo INT8 backbone (HEF)
  |
Feature maps
  |
Existing PaDiM embedding and scoring
  |
Anomaly score + heatmap
```

## Hailo workflow

1. Export the PaDiM feature extractor to ONNX.
2. Parse ONNX model to Hailo HAR.
3. Quantize with representative calibration images.
4. Compile HAR to HEF.
5. Run inference on Hailo device.

## Quantization

Calibration images should match the normal training distribution.

Example:

```
dataset/
  bottle/
    train/
      good/
```

The first milestone only moves feature extraction to Hailo. PaDiM post-processing remains unchanged until Hailo compatibility is validated.
