#!/usr/bin/env bash
set -euo pipefail

# Compile a complete AnomaVision Hailo ONNX graph into HAR and HEF.
# All generated artifacts are intentionally written next to the ONNX model:
#   <model-dir>/<model>.onnx
#   <model-dir>/<model>.har
#   <model-dir>/<model>.hef
#
# Usage:
#   bash scripts/hailo_compile.sh <model.onnx> <calibration_dir> [hw_arch]
#
# Example:
#   bash scripts/hailo_compile.sh \
#     distributions/patchcore/bottle/hailo/anomavision_patchcore_k26_end_to_end.onnx \
#     /root/dataset/bottle/train/good \
#     hailo8

MODEL="${1:-}"
CALIBRATION_DIR="${2:-}"
HW_ARCH="${3:-hailo8}"

if [[ -z "$MODEL" || -z "$CALIBRATION_DIR" ]]; then
  echo "Usage: $0 <model.onnx> <calibration_dir> [hw_arch]"
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: ONNX model not found: $MODEL"
  exit 1
fi

if [[ ! -d "$CALIBRATION_DIR" ]]; then
  echo "ERROR: calibration directory not found: $CALIBRATION_DIR"
  exit 1
fi

MODEL="$(realpath "$MODEL")"
CALIBRATION_DIR="$(realpath "$CALIBRATION_DIR")"
MODEL_DIR="$(dirname "$MODEL")"
MODEL_NAME="$(basename "$MODEL" .onnx)"

HAR="$MODEL_DIR/$MODEL_NAME.har"
OPT_HAR="$MODEL_DIR/${MODEL_NAME}_optimized.har"
HEF="$MODEL_DIR/$MODEL_NAME.hef"
CALIB_DIR="$MODEL_DIR/calibration_npy"

mkdir -p "$CALIB_DIR"

# Convert representative RGB images to the format expected by Hailo
# optimization for the parsed fixed input shape: one sample, HxWx3.
python - "$CALIBRATION_DIR" "$CALIB_DIR" <<'PY'
import sys
from pathlib import Path
import numpy as np
from PIL import Image

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
images = sorted(p for p in src.rglob("*") if p.suffix.lower() in suffixes)
if not images:
    raise SystemExit(f"ERROR: no calibration images found in {src}")

# Keep calibration bounded and deterministic.
images = images[:1024]

for index, path in enumerate(images):
    out = dst / f"sample_{index:04d}.npy"
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224))
        array = np.asarray(image, dtype=np.float32)
    np.save(out, array)

print(f"Created {len(images)} calibration samples in {dst}")
PY

echo "[1/3] Parsing ONNX -> HAR"
hailo parser onnx "$MODEL" --hw-arch "$HW_ARCH" --output-dir "$MODEL_DIR"

# The parser normally creates <model>.har in the output directory. Some SDK
# releases derive the HAR name from the network name, so locate it explicitly.
PARSED_HAR="$MODEL_DIR/$MODEL_NAME.har"
if [[ ! -f "$PARSED_HAR" ]]; then
  PARSED_HAR="$(find "$MODEL_DIR" -maxdepth 1 -type f -name '*.har' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
fi
if [[ -z "${PARSED_HAR:-}" || ! -f "$PARSED_HAR" ]]; then
  echo "ERROR: parser completed but no HAR was produced in $MODEL_DIR"
  exit 1
fi

if [[ "$PARSED_HAR" != "$HAR" ]]; then
  mv -f "$PARSED_HAR" "$HAR"
fi

echo "[2/3] Optimizing / quantizing HAR"
hailo optimize "$HAR" --hw-arch "$HW_ARCH" --use-random-calib-set --calib-set-path "$CALIB_DIR" --output-dir "$MODEL_DIR"

OPT_FOUND="$(find "$MODEL_DIR" -maxdepth 1 -type f -name '*_optimized.har' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
if [[ -z "${OPT_FOUND:-}" || ! -f "$OPT_FOUND" ]]; then
  echo "ERROR: optimization completed but no optimized HAR was produced in $MODEL_DIR"
  exit 1
fi
if [[ "$OPT_FOUND" != "$OPT_HAR" ]]; then
  mv -f "$OPT_FOUND" "$OPT_HAR"
fi

echo "[3/3] Compiling optimized HAR -> HEF"
hailo compiler "$OPT_HAR" --hw-arch "$HW_ARCH" --output-dir "$MODEL_DIR"

HEF_FOUND="$(find "$MODEL_DIR" -maxdepth 1 -type f -name '*.hef' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
if [[ -z "${HEF_FOUND:-}" || ! -f "$HEF_FOUND" ]]; then
  echo "ERROR: compilation completed but no HEF was produced in $MODEL_DIR"
  exit 1
fi
if [[ "$HEF_FOUND" != "$HEF" ]]; then
  mv -f "$HEF_FOUND" "$HEF"
fi

echo
echo "Hailo compilation completed successfully."
echo "ONNX:       $MODEL"
echo "HAR:        $HAR"
echo "Optimized:  $OPT_HAR"
echo "HEF:        $HEF"
echo "Calibration:$CALIB_DIR"
echo "HW arch:    $HW_ARCH"
