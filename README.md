# handwave

Native macOS hand tracking in Rust. It reads your webcam, finds up to two hands
and draws the 21 MediaPipe landmarks per hand over a mirrored video, with a live
performance and jitter readout. This is milestone M0: the tracking pipeline only
(no sound, gestures or Mac control yet).

Everything runs locally. There are no network calls at runtime and no telemetry;
the two ONNX models are embedded in the binary.

## Run

```sh
cargo run --release
```

Always measure performance with `--release`; debug builds are much slower.

Other modes:

```sh
cargo run --release -- --image tests/fixtures/hands/open_palm.png --out out.png   # annotate one photo
cargo run --release -- --bench                                                    # 100 passes over the fixtures
```

`--image` does not mirror (the output lines up with the file as stored); live
mode mirrors the camera frame once, before inference and display. Because of
that, MediaPipe's handedness output is inverted in live mode; M0 does not use it.

## Camera permission

The first run from a terminal or IDE makes macOS ask for camera access for that
app (Terminal, iTerm, VS Code, ...). If you denied it, enable it in
**System Settings > Privacy & Security > Camera**, then run again. handwave waits
up to 60 seconds for you to answer the prompt.

## Controls

| Key | Action |
|---|---|
| `F` | Toggle the One Euro filter (display only; the filter always runs) |
| `J` | Jitter test: hold your hand still for 3 seconds; shows and prints the RMS deviation of the palm centre (mean of landmarks 0, 5, 9, 13, 17), raw and filtered |
| `Esc` | Quit |

HUD (top-left): camera FPS, inference FPS, palm ms, landmark ms, hands tracked,
mode (`detecting` = the palm detector ran this frame, `tracking` = it did not),
filter on/off.

## Models

Google's official MediaPipe `hand_landmarker.task`, converted to ONNX with
tf2onnx. Details, input/output shapes, normalisation and checksums are in
[models/MODELS.md](models/MODELS.md). Licence: Apache 2.0.

| File | SHA-256 |
|---|---|
| `hand_landmarker.task` (source bundle) | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| `models/hand_detector.onnx` | `92e8bfd6d3eeff87e92b58841fb8c782e52e24e5d46278515c6a0ea5f97447b4` |
| `models/hand_landmarks_detector.onnx` | `7e52b76e7b3254d1d711de5d787078679d5782f24609cf64158d7e5116e4fa36` |

Regenerate with `scripts/convert_models.sh` (needs Python 3.12; pinned in
`scripts/requirements.txt`). `scripts/reference_landmarks.py` produces the
reference landmarks in `tests/fixtures/expected/` with the official `mediapipe`
Python package; they are committed so CI never needs Python.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo check --all-targets --all-features
cargo test --all-features
cargo build --release
```

The fixture parity test compares Rust landmarks with the reference and requires a
mean normalised error of at most 0.02 per landmark; see [PROJECT_MAP.md](PROJECT_MAP.md)
for the code layout.

Note: the `ort` crate (ONNX Runtime) is a release candidate; its build step
downloads the ONNX Runtime binaries once.
