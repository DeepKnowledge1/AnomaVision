# Production deployment

This page contains the details that are useful after the first successful AnomaVision run. The main README intentionally keeps the beginner workflow short.

## Lightweight PatchCore

PatchCore stores normal training patches in a memory bank and scores each test patch by its nearest stored feature. This implementation follows the PaDiM design contract: training is `model.fit(dataloader)`, inference is `model.predict(batch)`, and both image scores and spatial maps are returned.

Use the configuration below when inference latency or memory is more important than retaining every training patch:

```yaml
algorithm: patchcore
backbone: resnet18
layer_indices: [0, 1]
coreset_ratio: 0.02
max_memory_patches: 2048
patch_grid: 14
search_chunk_size: 1024
```

`coreset_ratio` controls the fraction of extracted normal patches retained. `max_memory_patches` provides a hard upper bound. `patch_grid` pools the native feature map to a small spatial grid, and `search_chunk_size` prevents a full query-by-memory distance matrix from being allocated. Start with `resnet18`, `[0]`, a 14x14 grid, and the defaults above; increase the memory cap only after measuring validation quality and latency on the target device.

The implementation uses normalized embeddings and chunked matrix multiplication rather than `torch.cdist` over the entire batch. This keeps the working memory bounded while preserving a nearest-normal-patch score. It is intentionally an ultra-light approximation of full PatchCore: the lower memory and patch count can reduce accuracy, so validate the trade-off on the target defect classes.

## TensorRT export

TensorRT is an optional NVIDIA deployment dependency. The exporter first creates the model graph, then uses the native TensorRT builder and ONNX parser to produce a serialized `.engine` file. TensorRT is imported lazily, so CPU installations continue to support the other formats.

```bash
# FP16
anomavision export --config config.yml --format tensorrt \
  --device cuda --tensorrt-precision fp16

# Calibrated INT8
anomavision export --config config.yml --format tensorrt \
  --device cuda --tensorrt-precision int8 \
  --calib-dir ./dataset/bottle/train/good \
  --calib-samples 100 --workspace-gb 4
```

INT8 calibration images should represent the normal production input distribution and use the same resize, crop, and normalization settings as training. The calibration cache is written beside the engine and can be reused when the graph and preprocessing remain unchanged. TensorRT export must be performed on a machine with a compatible CUDA/TensorRT/PyCUDA installation; an engine is generally tied to the TensorRT and GPU environment in which it was built.

## Verification checklist

| Check | Recommended evidence |
|---|---|
| Accuracy | Report image and pixel AUROC with the exact dataset split. |
| Latency | Report warm-up policy, batch size, input shape, device, and percentile latency. |
| Memory | Record peak GPU memory and PatchCore memory-bank size. |
| Export parity | Compare PyTorch scores/maps with ONNX or TensorRT outputs on the same images. |
| Reproducibility | Save the effective config, dependency versions, commit SHA, and calibration directory description. |

## Making the project popular

Adoption is more likely when a project is easy to verify and easy to integrate. The recommended sequence is to publish a minimal reproducible benchmark command, a short model card, one production export example, and a small demo using a user-provided image. Keep claims tied to the benchmark script rather than to a single headline number.

A useful release should include a comparison table with dataset split, preprocessing, backbone, hardware, batch size, and latency methodology. Link the raw results and invite independent reproduction. Then publish a short example showing how to load the exported artifact in an existing service. This creates three entry points: researchers can inspect the benchmark, engineers can copy the deployment path, and beginners can run the quickstart.

For discoverability, use a clear repository description, topic tags such as `anomaly-detection`, `computer-vision`, `patchcore`, and `tensorrt`, a small release note for each version, and issue templates for bug reports and benchmark reproduction. Avoid unsupported claims such as “best” unless the comparison protocol is public and repeatable.

## Automatic TensorRT conversion

Use `scripts/convert_to_tensorrt.py` when you already have a compact PaDiM statistics artifact or an ultra-light PatchCore memory-bank artifact. The utility detects the artifact type and delegates model loading and TensorRT construction to the shared export pipeline.

```bash
python scripts/convert_to_tensorrt.py `
  --model ./model_data/patchcore/bottle/run/model.pth `
  --output-dir ./engines/bottle `
  --precision int8 `
  --device cuda `
  --calib-dir ./dataset/bottle/train/good `
  --calib-samples 100 `
  --min-batch 1 --opt-batch 1 --max-batch 4
```

For FP16, omit the calibration directory and change `--precision int8` to `--precision fp16`. INT8 conversion requires real normal calibration images; the TensorRT path no longer falls back to random calibration data. The utility accepts PNG, JPEG, BMP, TIFF, and nested calibration-image directories, writes a reusable calibration cache beside the engine, and deserializes the generated engine for validation unless `--skip-validation` is supplied.

The input artifact may be either a PaDiM `.pth` statistics file or a PatchCore artifact containing its memory bank and feature settings. The default dynamic profile is batch 1/1/4; override it when the deployment workload has a different batch distribution.

## Algorithm-specific thresholds and PatchCore coreset selection

PaDiM and PatchCore produce different score scales. PaDiM uses Mahalanobis distances, while PatchCore uses bounded normalized nearest-neighbor distances, so a single threshold should not be shared between them.

Use separate values in `config.yml`:

```yaml
thresh: null
thresh_padim: null
thresh_patchcore: null
```

With an algorithm-specific value set to `null`, `eval` selects a threshold from the evaluation labels and logs the selected value. For production `detect`, copy the threshold selected on a separate validation set into the corresponding field, for example `thresh_patchcore: 0.35`. A threshold of `0.0` is valid and remains active.

PatchCore now uses deterministic greedy k-center selection by default instead of random memory-bank sampling. This improves coverage of normal feature space while retaining the configured `coreset_ratio` and `max_memory_patches` limits. Set `coreset_method: random` only when you explicitly prefer faster training over representative memory-bank coverage.

## Production Autopilot

Production Autopilot evaluates available PaDiM and ultra-light PatchCore artifacts on the same validation data, calibrates a separate threshold for each algorithm, measures median and p95 latency on the selected hardware, checks whether localization maps are non-empty, and packages the selected artifact with a deployment manifest and report.

Run it on CPU with:

```powershell
anomavision autopilot `
  --config config.yml `
  --padim_model .\distributions\padim\bottle\run\model.pt `
  --patchcore_model .\distributions\patchcore\bottle\run\model.pt `
  --device cpu `
  --target_latency_ms 50 `
  --output_dir .\production_package
```

The output contains `model.*`, `deployment_manifest.json`, `production_autopilot_report.html`, and `localization_report.md`. Open `production_autopilot_report.html` in any browser for the polished dashboard; it is self-contained and needs no internet connection or additional assets. The manifest records preprocessing, calibrated thresholds, metrics, latency, localization sanity checks, selected model, and runtime environment. Recheck the selected threshold on a production validation set before release.
