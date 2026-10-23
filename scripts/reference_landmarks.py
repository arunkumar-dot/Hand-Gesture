"""Run the official MediaPipe HandLandmarker on every fixture image and write
tests/fixtures/expected/<name>.json (normalised landmarks per hand).

Usage: scripts/.venv/bin/python -I scripts/reference_landmarks.py [hand_landmarker.task]

Without an argument the official bundle is downloaded and its SHA-256 checked.
"""
import hashlib
import tempfile
import urllib.request
import json
import pathlib
import sys

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

root = pathlib.Path(__file__).resolve().parent.parent
BUNDLE_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/1/hand_landmarker.task")
BUNDLE_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

if len(sys.argv) > 1:
    task = sys.argv[1]
else:
    task = str(pathlib.Path(tempfile.mkdtemp()) / "hand_landmarker.task")
    urllib.request.urlretrieve(BUNDLE_URL, task)
    digest = hashlib.sha256(pathlib.Path(task).read_bytes()).hexdigest()
    if digest != BUNDLE_SHA256:
        sys.exit(f"bundle checksum mismatch: {digest}")
out_dir = root / "tests/fixtures/expected"
out_dir.mkdir(parents=True, exist_ok=True)

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=task),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)
with vision.HandLandmarker.create_from_options(options) as lm:
    for img_path in sorted((root / "tests/fixtures/hands").glob("*.png")):
        image = mp.Image.create_from_file(str(img_path))
        res = lm.detect(image)
        hands = []
        for pts, handed in zip(res.hand_landmarks, res.handedness):
            hands.append({
                "handedness": handed[0].category_name,
                "score": handed[0].score,
                "landmarks": [[p.x, p.y, p.z] for p in pts],
            })
        (out_dir / f"{img_path.stem}.json").write_text(
            json.dumps({"image": img_path.name, "width": image.width,
                        "height": image.height, "hands": hands}, indent=1))
        print(img_path.name, "hands:", len(hands))
