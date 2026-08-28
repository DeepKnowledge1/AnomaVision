#!/usr/bin/env bash
set -euo pipefail

# Build a complete AnomaVision Hailo model: ONNX -> HAR -> optimized HAR -> HEF.
# All generated artifacts are kept in the same directory as the ONNX model.
#
# Usage:
#   bash scripts/hailo_compile.sh <algorithm> <artifact> <calibration_dir> <output_dir> [hw_arch]
#
# Example:
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

[[ -f "$ARTIFACT" ]] || { echo "ERROR: artifact not found: $ARTIFACT"; exit 1; }
[[ -d "$CALIBRATION_DIR" ]] || { echo "ERROR: calibration directory not found: $CALIBRATION_DIR"; exit 1; }

mkdir -p "$OUTPUT_DIR"

# Step 1: export the complete graph and create Hailo HxWxC calibration tensors.
echo "[1/4] AnomaVision quantize -> ONNX + calibration_npy"
anomavision quantize \
  --algorithm "$ALGORITHM" \
  --artifact "$ARTIFACT" \
  --calibration-dir "$CALIBRATION_DIR" \
  --output-dir "$OUTPUT_DIR"

MODEL="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.onnx' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
[[ -n "${MODEL:-}" && -f "$MODEL" ]] || { echo "ERROR: no ONNX model was produced"; exit 1; }
MODEL="$(realpath "$MODEL")"
MODEL_DIR="$(dirname "$MODEL")"
MODEL_NAME="$(basename "$MODEL" .onnx)"
CALIB_DIR="$MODEL_DIR/calibration_npy"
HAR="$MODEL_DIR/$MODEL_NAME.har"
OPT_HAR="$MODEL_DIR/${MODEL_NAME}_optimized.har"
HEF="$MODEL_DIR/$MODEL_NAME.hef"

[[ -d "$CALIB_DIR" ]] || { echo "ERROR: calibration_npy was not created: $CALIB_DIR"; exit 1; }
COUNT="$(find "$CALIB_DIR" -maxdepth 1 -type f -name '*.npy' | wc -l)"
[[ "$COUNT" -gt 0 ]] || { echo "ERROR: calibration_npy is empty: $CALIB_DIR"; exit 1; }
echo "Calibration samples: $COUNT"

# Step 2: PatchCore's fixed graph is parsed with the end nodes that Hailo DFC
# successfully accepts for the production 224x224 graph. PaDiM keeps the
# normal parser path so its graph can be adapted independently when needed.
echo "[2/4] Parsing ONNX -> HAR"
if [[ "$ALGORITHM" == "patchcore" ]]; then
  (
    cd "$MODEL_DIR"
    hailo parser onnx "$MODEL" --hw-arch "$HW_ARCH" --end-node-names "/MaxPool" "/Squeeze"
  )
else
  (
    cd "$MODEL_DIR"
    hailo parser onnx "$MODEL" --hw-arch "$HW_ARCH"
  )
fi

PARSED_HAR="$MODEL_DIR/$MODEL_NAME.har"
if [[ ! -f "$PARSED_HAR" ]]; then
  PARSED_HAR="$(find "$MODEL_DIR" -maxdepth 1 -type f -name '*.har' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
fi
[[ -n "${PARSED_HAR:-}" && -f "$PARSED_HAR" ]] || { echo "ERROR: no HAR was produced"; exit 1; }
[[ "$PARSED_HAR" == "$HAR" ]] || mv -f "$PARSED_HAR" "$HAR"

# Step 3: INT8 optimization using the unbatched HxWxC calibration tensors.
echo "[3/4] Optimizing / quantizing HAR"
(
  cd "$MODEL_DIR"
  hailo optimize "$HAR" --hw-arch "$HW_ARCH" --calib-set-path "$CALIB_DIR"
)

OPT_FOUND="$(find "$MODEL_DIR" -maxdepth 1 -type f -name '*_optimized.har' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
[[ -n "${OPT_FOUND:-}" && -f "$OPT_FOUND" ]] || { echo "ERROR: no optimized HAR was produced"; exit 1; }
[[ "$OPT_FOUND" == "$OPT_HAR" ]] || mv -f "$OPT_FOUND" "$OPT_HAR"

# Step 4: Compile to HEF.
echo "[4/4] Compiling optimized HAR -> HEF"
(
  cd "$MODEL_DIR"
  hailo compiler "$OPT_HAR" --hw-arch "$HW_ARCH"
)

HEF_FOUND="$(find "$MODEL_DIR" -maxdepth 1 -type f -name '*.hef' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
[[ -n "${HEF_FOUND:-}" && -f "$HEF_FOUND" ]] || { echo "ERROR: no HEF was produced"; exit 1; }
[[ "$HEF_FOUND" == "$HEF" ]] || mv -f "$HEF_FOUND" "$HEF"

echo
echo "Hailo compilation completed successfully."
echo "Algorithm:   $ALGORITHM"
echo "ONNX:        $MODEL"
echo "HAR:         $HAR"
echo "Optimized:   $OPT_HAR"
echo "HEF:         $HEF"
echo "Calibration: $CALIB_DIR"
echo "HW arch:     $HW_ARCH"
