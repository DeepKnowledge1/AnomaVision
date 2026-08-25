# Hailo PaDiM

## Flow

```
padim_backbone.onnx
        |
        v
Hailo parser
        |
        v
padim_backbone.har
        |
        v
INT8 calibration
        |
        v
HEF
```

## Notes

- ONNX export is handled by the existing AnomaVision export flow.
- PaDiM implementation is unchanged.
- This folder contains only Hailo-specific conversion/runtime helpers.

## Next steps

1. Optimize HAR with calibration images.
2. Compile optimized HAR to HEF.
3. Validate feature map output against PyTorch/ONNX.
