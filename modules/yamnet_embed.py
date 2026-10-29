"""YAMNet embeddings (Google, AudioSet-pretrained, 1024-d) for 1 s windows at 16 kHz.

Uses the weights exported for the browser (docs/models/yamnet.{json,bin}:
float16, dequantised from the official TFLite model by tools/export_yamnet_web.py)
and the same maths as docs/index.html (yamnetEmbed): 25 ms Hann frames every
10 ms, the model's own DFT matrices, 64-band mel, log(x + 0.001), 96x64 patch,
27 convolutions with ReLU, global average pool. Training and phone therefore
see identical embeddings (tests/test_yamnet_parity.py).
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 15600
STRIDES = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]


class YamnetEmbedder:
    def __init__(self, models_dir=os.path.join(ROOT, "docs", "models")):
        man = json.load(open(os.path.join(models_dir, "yamnet.json")))
        a = np.fromfile(os.path.join(models_dir, "yamnet.bin"), dtype=np.float16)
        self.W = {t["name"]: a[t["offset"]:t["offset"] + t["length"]].astype(np.float32).reshape(t["shape"])
                  for t in man["tensors"]}

    def logmel(self, x):
        W = self.W
        x = np.asarray(x, np.float32)[:N]
        if len(x) < N:
            x = np.pad(x, (0, N - len(x)))
        idx = np.arange(400)[None, :] + 160 * np.arange(96)[:, None]
        fr = np.pad(x[idx] * W["window"], ((0, 0), (0, 112)))
        mag = np.sqrt((fr @ W["dft_re"]) ** 2 + (fr @ W["dft_im"]) ** 2)
        return np.log(mag @ W["mel"] + 0.001).astype(np.float32)

    def batch(self, windows):
        import tensorflow as tf
        W = self.W
        h = tf.constant(np.stack([self.logmel(w) for w in windows])[..., None])
        h = tf.nn.relu(tf.nn.conv2d(h, W["l0_conv_w"], 2, "SAME") + W["l0_b"])
        for i, s in enumerate(STRIDES):
            h = tf.nn.relu(tf.nn.depthwise_conv2d(h, W[f"l{1 + 2 * i}_dw_w"], [1, s, s, 1], "SAME") + W[f"l{1 + 2 * i}_b"])
            h = tf.nn.relu(tf.nn.conv2d(h, W[f"l{2 + 2 * i}_conv_w"], 1, "SAME") + W[f"l{2 + 2 * i}_b"])
        return tf.reduce_mean(h, [1, 2]).numpy()

    def __call__(self, window):
        return self.batch([window])[0]
