"""Export YAMNet (official float TFLite) for the browser: weights as float16 binary.

Reads models/yamnet/yamnet.tflite, walks its ops in order and takes every
CONV_2D / DEPTHWISE_CONV_2D filter + bias (batch-norm already folded in), the
Hann window and the 257x64 mel matrix. Writes
  docs/models/yamnet.bin   concatenated float16 arrays
  docs/models/yamnet.json  manifest: [{name, shape (tf.js layout), offset, length}]
docs/index.html (yamnetEmbed) rebuilds the network with tf.js ops;
tests/test_yamnet_parity.py checks the browser embedding matches TFLite.
"""
import json
import os

import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "models", "yamnet", "yamnet.tflite")
OUT = os.path.join(ROOT, "docs", "models")


def deq(it, det, i):
    """Tensor value as float32. This 'float' TFLite stores most weights as int8
    with per-channel scales (dynamic-range quantisation), so dequantise."""
    v = it.get_tensor(i)
    if v.dtype == np.float32:
        return v
    q = det[i]["quantization_parameters"]
    sc, zp, ax = q["scales"], q["zero_points"], q["quantized_dimension"]
    if len(sc) == 1:
        return (v.astype(np.float32) - zp[0]) * sc[0]
    shape = [1] * v.ndim
    shape[ax] = -1
    return (v.astype(np.float32) - zp.reshape(shape)) * sc.reshape(shape)


def main():
    it = tf.lite.Interpreter(SRC, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    det = {d["index"]: d for d in it.get_tensor_details()}
    arrays, convs = [], []
    window = mel = None
    dft = []
    for op in it._get_ops_details():
        name = op["op_name"]
        ins = list(op["inputs"])
        if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            w, b = deq(it, det, ins[1]), deq(it, det, ins[2])
            if name == "CONV_2D":
                w = np.transpose(w, (1, 2, 3, 0))          # [out,h,w,in] -> [h,w,in,out]
            else:
                w = np.transpose(w, (1, 2, 3, 0))          # [1,h,w,C] -> [h,w,C,1]
            convs.append((name, w, b))
        elif name == "MUL" and window is None:
            for i in ins:
                if i >= 0 and list(det[i]["shape"]) == [1, 400]:
                    try:
                        window = deq(it, det, i).reshape(-1)
                    except ValueError:
                        pass
        elif name == "FULLY_CONNECTED":
            for i in ins[1:2]:
                s = list(det[i]["shape"])
                if s == [257, 512]:                          # real / imaginary DFT rows (int8 in the model)
                    dft.append(deq(it, det, i).T)            # -> [512, 257]
                if s in ([64, 257], [257, 64]):
                    m = deq(it, det, i)
                    mel = m.T if s == [64, 257] else m      # -> [257, 64]
    assert window is not None and mel is not None and len(convs) == 27 and len(dft) == 2, (window is None, mel is None, len(convs), len(dft))
    manifest, off = [], 0
    blobs = [("window", window), ("dft_re", dft[0]), ("dft_im", dft[1]), ("mel", mel)]
    for k, (op, w, b) in enumerate(convs):
        blobs += [(f"l{k}_{'dw' if op.startswith('DEPTH') else 'conv'}_w", w), (f"l{k}_b", b)]
    for name, a in blobs:
        a = np.asarray(a, np.float32)
        manifest.append({"name": name, "shape": list(a.shape), "offset": off, "length": int(a.size)})
        arrays.append(a.astype(np.float16).ravel())
        off += a.size
    os.makedirs(OUT, exist_ok=True)
    np.concatenate(arrays).tofile(os.path.join(OUT, "yamnet.bin"))
    with open(os.path.join(OUT, "yamnet.json"), "w") as f:
        json.dump({"source": "YAMNet (Google, Apache-2.0), tensorflow/models research/audioset/yamnet, float TFLite",
                   "dtype": "float16", "tensors": manifest}, f)
    print("tensors", len(manifest), "values", off, "bytes", off * 2)
    for m in manifest[:6]:
        print(m["name"], m["shape"])


if __name__ == "__main__":
    main()
