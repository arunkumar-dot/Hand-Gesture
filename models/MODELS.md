# Models

Official MediaPipe hand landmarker, converted from TFLite to ONNX. Nothing is
retrained or edited; `scripts/convert_models.sh` reproduces these files.

- **Source:** <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task>
- **Bundle SHA-256** (`hand_landmarker.task`): `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- **License:** Apache License 2.0 (Google LLC, MediaPipe Models)

| File | Made from | SHA-256 |
|---|---|---|
| `hand_detector.onnx` | `hand_detector.tflite` | `92e8bfd6d3eeff87e92b58841fb8c782e52e24e5d46278515c6a0ea5f97447b4` |
| `hand_landmarks_detector.onnx` | `hand_landmarks_detector.tflite` | `7e52b76e7b3254d1d711de5d787078679d5782f24609cf64158d7e5116e4fa36` |

The ONNX hashes depend on the pinned tool versions below; a different
tf2onnx/onnx may produce different bytes with the same behaviour.

## Tool versions (`scripts/requirements.txt`, Python 3.12.15)

tensorflow 2.16.2, tf2onnx 1.17.0 (opset 13), onnx 1.16.1, onnxruntime 1.18.1,
numpy 1.26.4, protobuf 4.25.9, mediapipe 0.10.21 (reference landmarks only).
Python 3.12 because TensorFlow has no wheels for 3.14+.

## Input / output shapes (read from the converted files)

Layout is NHWC, as in the TFLite originals; tf2onnx did not transpose.

### `hand_detector.onnx` (palm detector)

| | name | shape |
|---|---|---|
| in | `input_1` | `[1, 192, 192, 3]` f32 |
| out | `Identity` | `[1, 2016, 18]` f32: per anchor x, y, w, h offsets + 7 keypoints (x, y) |
| out | `Identity_1` | `[1, 2016, 1]` f32: raw score logit |

### `hand_landmarks_detector.onnx` (landmarks)

| | name | shape |
|---|---|---|
| in | `input_1` | `[1, 224, 224, 3]` f32 |
| out | `Identity` | `[1, 63]` 21 landmarks x (x, y, z), x/y in input pixels (0..224) |
| out | `Identity_1` | `[1, 1]` hand presence, already a probability (observed 0.996-0.997 on hands) |
| out | `Identity_2` | `[1, 1]` handedness probability (observed 0.01-0.98) |
| out | `Identity_3` | `[1, 63]` world landmarks (unused in M0) |

`Identity_1` is presence and `Identity_2` is handedness: confirmed by running
the model on the fixtures (presence is ~1.0 on every hand; handedness flips
between the left and right hand in `two_hands.png`).

Numerical check against the TFLite models on random input (`scripts/check_models.py`):
max abs difference 1.3e-4 (detector boxes), 7.9e-5 (landmarks).

## Input normalisation (TFLite metadata)

Both models: `NormalizationOptions` mean 0, std 255, i.e. RGB `x / 255` in `0..1`
(read with `MetadataDisplayer`; the detector's metadata has stats min 0 / max 1).

## Pipeline constants (from the serialized subgraphs in the mediapipe 0.10.21 wheel)

Decoded from the protobuf configs embedded in `_framework_bindings*.so`:

- `PalmDetectionDetectionToRoi`: `DetectionsToRects` keypoints 0 -> 2, target angle
  90 degrees; `RectTransformation` scale 2.6 x 2.6, shift_y -0.5, square_long.
- `HandLandmarkLandmarksToRoi`: `HandLandmarksToRectCalculator`, then
  `RectTransformation` scale 2.0 x 2.0, shift_y -0.1, square_long.
- SSD anchors: 4 layers, strides 8/16/16/16, 192x192, scales 0.1484375..0.75,
  offset 0.5, fixed anchor size, interpolated scale -> 2016 anchors (verified
  against the model output shape).
