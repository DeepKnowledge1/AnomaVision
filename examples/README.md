# AnomaVision examples

These examples are intentionally copy-ready. They use an MVTec-style dataset with the following layout:

```text
dataset/
└── bottle/
    ├── train/good/
    └── test/
        ├── good/
        └── scratch/
```

## Five-minute CPU anomaly detection

Start with [`quickstart_cpu.yml`](quickstart_cpu.yml) for a PaDiM CPU baseline:

```bash
anomavision train --config examples/quickstart_cpu.yml
anomavision detect --config examples/quickstart_cpu.yml \
  --img_path ./dataset/bottle/test \
  --device cpu
```

## Ultra-light PatchCore on CPU

Use [`patchcore_cpu.yml`](patchcore_cpu.yml) when bounded memory and low-latency nearest-patch inference are the priority:

```bash
anomavision train --config examples/patchcore_cpu.yml
anomavision detect --config examples/patchcore_cpu.yml \
  --img_path ./dataset/bottle/test \
  --device cpu
```

The configuration uses deterministic k-center coreset selection, a capped memory bank, pooled patch features, and chunked nearest-neighbor search.

## TensorRT INT8 anomaly detection

Use [`tensorrt_int8.yml`](tensorrt_int8.yml) on a compatible NVIDIA machine after training the selected model. The calibration directory must contain real normal production-like images:

```bash
anomavision export --config examples/tensorrt_int8.yml \
  --format tensorrt \
  --device cuda \
  --tensorrt-precision int8 \
  --calib-dir ./dataset/bottle/train/good \
  --calib-samples 100
```

For model-artifact conversion, see [`docs/production_deployment.md`](../docs/production_deployment.md).
