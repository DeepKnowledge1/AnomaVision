# Cross-Backend Validation

AnomaVision provides a common validation command for checking that different
model formats produce consistent anomaly scores and anomaly maps for the same
images.

The validation path is backend-agnostic and is intended for **PatchCore** and
**PaDiM**. It currently supports the existing PyTorch and ONNX Runtime
backends and can also run against a Hailo `.hef` when a Hailo device is
available.

## Goal

For the same trained model and the same input image:

```text
PyTorch  ≈  ONNX Runtime  ≈  Hailo HEF
   score       score          score
   heatmap     heatmap        heatmap
```

Compilation of a HEF is not considered validation. A HEF must be executed on
hardware and compared against the software reference.

## PT vs ONNX on a machine without Hailo hardware

```bash
anomavision validate \
  --config config.yml \
  --models distributions/patchcore/bottle/anomav_exp/model.pt \
           distributions/patchcore/bottle/anomav_exp/model.onnx \
  --img_path ./test_images \
  --output_dir ./validation_results \
  --save_visualizations
```

For PaDiM, use the corresponding PT and ONNX artifacts:

```bash
anomavision validate \
  --config config.yml \
  --models distributions/padim/bottle/anomav_exp/model.pt \
           distributions/padim/bottle/anomav_exp/model.onnx \
  --img_path ./test_images \
  --output_dir ./validation_results/padim \
  --save_visualizations
```

The command uses the same dataset preprocessing for every backend. It reports:

- maximum and mean absolute score error
- maximum and mean relative score error
- heatmap MAE
- heatmap RMSE
- maximum heatmap absolute error
- relative heatmap MAE
- PASS/FAIL for every model pair

The machine-readable report is written to:

```text
validation_results/validation_report.json
```

Visual comparisons are written under:

```text
validation_results/visualizations/
```

## Tolerances

Defaults are deliberately configurable because INT8 Hailo inference is not
expected to be bit-identical to floating-point PyTorch/ONNX inference.

```bash
--score_abs_tol 0.001
--score_rel_tol 0.01
--map_mae_tol 0.001
--map_rel_tol 0.01
```

A score comparison passes when every score is within the configured absolute
or relative tolerance. A heatmap comparison passes when its MAE is within the
absolute or relative tolerance.

For an initial software check, keep the defaults. Tighten them when you have
measured the expected numerical differences for a particular model/export.

## Hailo validation

After a Hailo device is available, include the HEF in the same command:

```bash
anomavision validate \
  --config config.yml \
  --models distributions/patchcore/bottle/anomav_exp/model.pt \
           distributions/patchcore/bottle/anomav_exp/model.onnx \
           distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.hef \
  --img_path ./test_images \
  --output_dir ./validation_results/patchcore_hailo \
  --save_visualizations
```

The same command structure applies to PaDiM. Keep the `.hef`, `.har`, and
related Hailo artifacts together with the model artifacts for that model
rather than relying on a parent-directory convention.

## Interpreting results

A successful compilation only proves that the Hailo compiler accepted and
mapped the graph. The deployment is considered validated only after the same
images have been executed and the score and heatmap differences have been
measured.

The final target is:

```text
PT score      ~= ONNX score      ~= HEF score
PT heatmap    ~= ONNX heatmap    ~= HEF heatmap
```

This validation command intentionally does not require HailoRT when only PT
and ONNX models are supplied, so software-side validation can be performed on
a normal CPU machine before hardware is available.
