#!/usr/bin/env bash
# Download Google's official hand_landmarker.task, extract the two TFLite
# models and convert them to ONNX with tf2onnx. Run from the repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
PY="${PYTHON312:-/opt/homebrew/bin/python3.12}"
VENV="$ROOT/scripts/.venv"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -r "$ROOT/scripts/requirements.txt"

mkdir -p "$WORK/bundle" "$ROOT/models"
curl -sSfL -o "$WORK/hand_landmarker.task" "$URL"
echo "bundle sha256: $(shasum -a 256 "$WORK/hand_landmarker.task" | cut -d' ' -f1)"
unzip -q -o "$WORK/hand_landmarker.task" -d "$WORK/bundle"

for name in hand_detector hand_landmarks_detector; do
  "$VENV/bin/python" -I -m tf2onnx.convert --opset 13 \
    --tflite "$WORK/bundle/$name.tflite" --output "$ROOT/models/$name.onnx" 2>&1 | grep -v -E '^(W0|I0)' | tail -3
  echo "$name.onnx sha256: $(shasum -a 256 "$ROOT/models/$name.onnx" | cut -d' ' -f1)"
done

"$VENV/bin/python" -I "$ROOT/scripts/check_models.py" "$WORK/bundle" "$ROOT/models"
