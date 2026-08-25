# TensorRT

AnomaVision TensorRT export builds a native NVIDIA TensorRT engine from the model's FP32 ONNX graph.

## What requires an NVIDIA GPU

A real TensorRT engine cannot be built or executed on a CPU-only machine. The following steps require an NVIDIA Linux environment with CUDA, TensorRT, and PyCUDA:

- TensorRT network parsing and engine building
- FP16/FP32 engine generation
- INT8 calibration
- TensorRT engine deserialization and inference

Do **not** install `nvidia-utils` on a CPU-only machine just to test this path. The NVIDIA GPU and driver must actually be available.

## What can be tested on CPU

The repository now includes CPU-only tests for the TensorRT implementation:

```bash
pytest -q tests/test_tensorrt_cpu_validation.py
```

These tests verify:

- the main `anomavision export` parser accepts all TensorRT arguments;
- the standalone `scripts/convert_to_tensorrt.py` parser accepts TensorRT options;
- `ModelExporter.export_tensorrt()` stops cleanly on CPU before importing TensorRT or PyCUDA;
- no temporary ONNX file is left behind when the CPU guard is triggered.

You can also verify the CLI help without a GPU:

```bash
python -m anomavision.cli export --help
python scripts/convert_to_tensorrt.py --help
```

## CPU guard test

The expected behavior on a CPU-only machine is a clean TensorRT failure with the message:

```text
TensorRT export requires a CUDA device.
```

The export method returns `None`, allowing the CLI to report the failed export instead of crashing during a TensorRT/PyCUDA import.

## Real TensorRT test on NVIDIA Linux

After moving the repository to an NVIDIA Linux machine, verify the environment first:

```bash
nvidia-smi
```

Then test the normal export path. For example:

```bash
anomavision export \
  --config config.yml \
  --format tensorrt \
  --device cuda \
  --tensorrt-precision fp16
```

For INT8:

```bash
anomavision export \
  --config config.yml \
  --format tensorrt \
  --device cuda \
  --tensorrt-precision int8 \
  --calib-dir ./dataset/<class>/train/good \
  --calib-samples 100
```

The standalone converter is also available:

```bash
python scripts/convert_to_tensorrt.py \
  --model ./path/to/model.pth \
  --output-dir ./engines \
  --precision fp16 \
  --device cuda
```

For INT8:

```bash
python scripts/convert_to_tensorrt.py \
  --model ./path/to/model.pth \
  --output-dir ./engines \
  --precision int8 \
  --device cuda \
  --calib-dir ./dataset/<class>/train/good \
  --calib-samples 100
```

## Dynamic batch profiles

TensorRT dynamic export uses:

```text
min_batch <= opt_batch <= max_batch
```

Defaults are:

```text
min_batch = 1
opt_batch = 1
max_batch = 4
```

Use `--static-batch` to disable the optimization profile and build a fixed-batch engine.

## INT8 calibration

TensorRT INT8 calibration intentionally requires real calibration images. Random calibration data is disabled for TensorRT export.

Calibration images are loaded recursively and preprocessed to NCHW FP32 tensors using the same image preprocessing used by AnomaVision.

A reusable TensorRT calibration cache is written next to the engine output.
