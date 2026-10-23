"""Print ONNX I/O names/shapes and check numerical parity with the TFLite originals."""
import sys
import numpy as np
import onnxruntime as ort
import tensorflow as tf

bundle, models = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
for name in ("hand_detector", "hand_landmarks_detector"):
    sess = ort.InferenceSession(f"{models}/{name}.onnx", providers=["CPUExecutionProvider"])
    print(name)
    for i in sess.get_inputs():
        print("  in ", i.name, i.shape, i.type)
    for o in sess.get_outputs():
        print("  out", o.name, o.shape, o.type)
    interp = tf.lite.Interpreter(model_path=f"{bundle}/{name}.tflite")
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    x = rng.random(inp["shape"], dtype=np.float32)
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    feed = {sess.get_inputs()[0].name: x}
    got = sess.run(None, feed)
    by_name = {o.name: g for o, g in zip(sess.get_outputs(), got)}
    for d in interp.get_output_details():
        ref = interp.get_tensor(d["index"])
        g = by_name.get(d["name"])
        if g is None:
            print("  (no ONNX output named", d["name"], ")")
            continue
        print(f"  parity {d['name']}: max|diff|={np.abs(ref - g).max():.2e}")
