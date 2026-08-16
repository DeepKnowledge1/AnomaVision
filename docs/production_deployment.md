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
