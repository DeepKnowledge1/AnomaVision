#!/usr/bin/env bash
set -euo pipefail

# Compile an AnomaVision Hailo model into HAR and HEF.
# The AnomaVision quantize command creates the end-to-end ONNX graph first.
# All generated model artifacts stay beside the model itself.
#
# Usage:
#   bash scripts/hailo_compile.sh <algorithm> <artifact> <calibration_dir> <output_dir> [hw_arch]
#
# Example (PaDiM):
#   bash scripts/hailo_compile.sh \
#     padim \
#     distributions/padim/bottle/anomav_exp/model.pt \
#     /root/dataset/bottle/train/good \
#     distributions/padim/bottle/hailo \
#     hailo8
#
# Example (PatchCore):
#   bash scripts/hailo_compile.sh \
#     patchcore \
#     distributions/patchcore/bottle/anomav_exp/model.pt \
#     /root/dataset/bottle/train/good \
#     distributions/patchcore/bottle/hailo \
#     hailo8

ALGORITHM="${1:-}"
ARTIFACT="${2:-}"
CALIBRATION_DIR="${3:-}"
OUTPUT_DIR="${4:-}"
HW_ARCH="${5:-hailo8}"

if [[ -z "$ALGORITHM" || -z "$ARTIFACT" || -z "$CALIBRATION_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 <algorithm> <artifact> <calibration_dir> <output_dir> [hw_arch]"
  exit 1
fi

if [[ "$ALGORITHM" != "padim" && "$ALGORITHM" != "patchcore" ]]; then
  echo "ERROR: algorithm must be 'padim' or 'patchcore'"
  exit 1
fi

if [[ ! -f "$ARTIFACT" ]]; then
  echo "ERROR: artifact not found: $ARTIFACT"
  exit 1
fi

if [[ ! -d "$CALIBRATION_DIR" ]]; then
  echo "ERROR: calibration directory not found: $CALIBRATION_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Step 1: use the AnomaVision CLI to create the complete end-to-end ONNX graph.
echo "[1/4] AnomaVision quantize -> ONNX"
anomavision quantize \
  --algorithm "$ALGORITHM" \
  --artifact "$ARTIFACT" \
  --calibration-dir "$CALIBRATION_DIR" \
  --output-dir "$OUTPUT_DIR"

MODEL="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.onnx' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
if [[ -z "${MODEL:-}" || ! -f "$MODEL" ]]; then
  echo "ERROR: anomavision quantize completed but no ONNX model was found in $OUTPUT_DIR"
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

# Hailo DFC optimization expects representative samples as HxWxC for the
# fixed 224x224x3 AnomaVision input. Do not add a batch dimension.
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

images = images[:1024]
for index, path in enumerate(images):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224))
        array = np.asarray(image, dtype=np.float32)
    np.save(dst / f"sample_{index:04d}.npy", array)

print(f"Created {len(images)} calibration samples in {dst}")
PY

echo "[2/4] Parsing ONNX -> HAR"
(
  cd "$MODEL_DIR"
  hailo parser onnx "$MODEL" --hw-arch "$HW_ARCH"
)

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

echo "[3/4] Optimizing / quantizing HAR"
(
  cd "$MODEL_DIR"
  hailo optimize "$HAR" --hw-arch "$HW_ARCH" --calib-set-path "$CALIB_DIR"
)

OPT_FOUND="$(find "$MODEL_DIR" -maxdepth 1 -type f -name '*_optimized.har' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
if [[ -z "${OPT_FOUND:-}" || ! -f "$OPT_FOUND" ]]; then
  echo "ERROR: optimization completed but no optimized HAR was produced in $MODEL_DIR"
  exit 1
fi
if [[ "$OPT_FOUND" != "$OPT_HAR" ]]; then
  mv -f "$OPT_FOUND" "$OPT_HAR"
fi

echo "[4/4] Compiling optimized HAR -> HEF"
(
  cd "$MODEL_DIR"
  hailo compiler "$OPT_HAR" --hw-arch "$HW_ARCH"
)

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
echo "Algorithm:   $ALGORITHM"
echo "Artifact:    $(realpath "$ARTIFACT")"
echo "ONNX:        $MODEL"
echo "HAR:         $HAR"
echo "Optimized:   $OPT_HAR"
echo "HEF:         $HEF"
echo "Calibration: $CALIB_DIR"
echo "HW arch:     $HW_ARCH"
