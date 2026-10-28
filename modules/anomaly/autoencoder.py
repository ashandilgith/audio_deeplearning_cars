"""Detector A: conditional log-mel autoencoder (DCASE Task 2 baseline recipe).

Input  : 5 consecutive log-mel frames x 128 bands = 640 values, z-scored per band
         with training statistics, plus the operating-context vector
         (context.CONTEXT_NAMES), which is concatenated to the input AND to the
         bottleneck so the model learns "normal for this rpm / warm-up".
Network: 640 -> 128 x4 -> 8 -> 128 x4 -> 640   (dense + batch-norm + ReLU)
Modes  : "ae"   reconstruct all 5 frames            (DCASE baseline)
         "idnn" predict the centre frame from the other 4 (Suefusa et al. 2020;
                better on non-stationary sounds such as throttle changes)
Score  : mean squared error per 1 s window ("mse"); also a diagonal
         Mahalanobis variant ("maha") that weights each output by the inverse
         of its normal error variance.
Thresholds: gamma distribution fitted per state to the scores of held-out
         normal recordings; warning at the 95th, alarm at the 99.5th percentile.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

from modules.anomaly.context import CONTEXT_DIM
from modules.anomaly.features import stack_frames

N_STACK = 5


def _split_groups(groups, val_frac, seed):
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    if len(uniq) >= 3:
        rng.shuffle(uniq)
        n_val = max(1, int(round(val_frac * len(uniq))))
        val = np.isin(groups, uniq[:n_val])
    else:  # one or two recordings: hold out the last part of each instead
        val = np.zeros(len(groups), bool)
        for g in uniq:
            idx = np.where(groups == g)[0]
            val[idx[int(len(idx) * (1 - val_frac)):]] = True
    return ~val, val


def gamma_thresholds(scores, states, warn_q=0.95, alarm_q=0.995, min_n=30) -> Dict[str, dict]:
    scores = np.asarray(scores, dtype=float)
    states = np.asarray(states)

    def fit(v):
        v = np.clip(v, 1e-9, None)
        a, loc, scale = stats.gamma.fit(v, floc=0)
        return {"warn": float(stats.gamma.ppf(warn_q, a, loc, scale)),
                "alarm": float(stats.gamma.ppf(alarm_q, a, loc, scale)),
                "shape": float(a), "scale": float(scale), "n": int(len(v))}

    out = {"all": fit(scores)}
    for s in sorted(set(states.tolist())):
        v = scores[states == s]
        if len(v) >= min_n:
            out[s] = fit(v)
    return out


def context_dropout(C, rng, p_temp=0.25, p_runtime=0.25, p_rpm=0.1):
    """Randomly hide context variables in training rows, so the model also works
    when a sensor is missing at run time (layout: context.CONTEXT_NAMES)."""
    C = C.copy()
    n = len(C)
    m = rng.random(n) < p_temp
    C[m, 6:8] = 0.0
    m = rng.random(n) < p_runtime
    C[m, 8:10] = 0.0
    m = rng.random(n) < p_rpm
    C[m, 0:6] = 0.0
    C[m, 3] = 1.0          # state -> unknown
    return C


class ConditionalAutoencoder:
    def __init__(self, mode: str = "ae", units: int = 128, bottleneck: int = 8,
                 n_stack: int = N_STACK, n_mels: int = 128):
        assert mode in ("ae", "idnn")
        self.mode, self.units, self.bottleneck = mode, units, bottleneck
        self.n_stack, self.n_mels = n_stack, n_mels
        self.mean = self.std = None
        self.err_var = None
        self.thresholds: Dict[str, dict] = {}
        self.model = None
        self.history = {}

    @property
    def name(self):
        return "autoencoder" if self.mode == "ae" else "idnn"

    # ------------------------------------------------------------------ data
    def _io(self, mel_windows: Sequence[np.ndarray]):
        """List of (T, F) windows -> (inputs, targets, window index per row)."""
        xs, idx = [], []
        for i, m in enumerate(mel_windows):
            s = stack_frames((m - self.mean) / self.std, self.n_stack)
            xs.append(s)
            idx.append(np.full(len(s), i))
        X = np.concatenate(xs).astype(np.float32)
        idx = np.concatenate(idx)
        if self.mode == "ae":
            return X, X, idx
        F, c = self.n_mels, self.n_stack // 2
        centre = X[:, c * F:(c + 1) * F]
        ctxt = np.concatenate([X[:, :c * F], X[:, (c + 1) * F:]], axis=1)
        return ctxt, centre, idx

    def _build(self, d_in, d_out):
        import tensorflow as tf
        L = tf.keras.layers
        x_in = L.Input((d_in,), name="frames")
        c_in = L.Input((CONTEXT_DIM,), name="context")
        h = L.Concatenate()([x_in, c_in])
        for _ in range(4):
            h = L.ReLU()(L.BatchNormalization()(L.Dense(self.units)(h)))
        z = L.ReLU()(L.BatchNormalization()(L.Dense(self.bottleneck)(h)))
        h = L.Concatenate()([z, c_in])
        for _ in range(4):
            h = L.ReLU()(L.BatchNormalization()(L.Dense(self.units)(h)))
        out = L.Dense(d_out, name="recon")(h)
        m = tf.keras.Model([x_in, c_in], out)
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        return m

    # ------------------------------------------------------------------ fit
    def fit(self, mel_windows: List[np.ndarray], ctx_vectors: np.ndarray, states: Sequence[str],
            groups: Sequence = None, epochs: int = 60, batch_size: int = 256, val_frac: float = 0.15,
            seed: int = 0, verbose: int = 0):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        n = len(mel_windows)
        groups = np.asarray(groups if groups is not None else np.arange(n))
        states = np.asarray(states)
        ctx_vectors = np.asarray(ctx_vectors, dtype=np.float32)
        tr, va = _split_groups(groups, val_frac, seed)
        frames = np.concatenate([mel_windows[i] for i in np.where(tr)[0]])
        self.mean = frames.mean(0).astype(np.float32)
        self.std = (frames.std(0) + 1e-3).astype(np.float32)

        Xtr, Ytr, itr = self._io([mel_windows[i] for i in np.where(tr)[0]])
        Ctr = ctx_vectors[np.where(tr)[0]][itr]
        Xva, Yva, iva = self._io([mel_windows[i] for i in np.where(va)[0]])
        Cva = ctx_vectors[np.where(va)[0]][iva]
        Ctr = context_dropout(Ctr, np.random.default_rng(seed))
        self.model = self._build(Xtr.shape[1], Ytr.shape[1])
        cb = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)]
        h = self.model.fit([Xtr, Ctr], Ytr, validation_data=([Xva, Cva], Yva), epochs=epochs,
                           batch_size=batch_size, shuffle=True, callbacks=cb, verbose=verbose)
        self.history = {k: [float(x) for x in v] for k, v in h.history.items()}
        err_tr = (self.model.predict([Xtr, Ctr], batch_size=2048, verbose=0) - Ytr) ** 2
        self.err_var = (err_tr.mean(0) + 1e-6).astype(np.float32)
        # thresholds from held-out normal windows (honest), per state
        va_scores = self._window_scores(Xva, Yva, Cva, iva, len(np.where(va)[0]))
        self.thresholds = {"mse": gamma_thresholds(va_scores["mse"], states[va]),
                           "maha": gamma_thresholds(va_scores["maha"], states[va])}
        return self

    # ------------------------------------------------------------------ score
    def _window_scores(self, X, Y, C, idx, n_windows):
        err = (self.model.predict([X, C], batch_size=2048, verbose=0) - Y) ** 2
        row_mse = err.mean(1)
        row_maha = (err / self.err_var).mean(1)
        mse = np.bincount(idx, row_mse, n_windows) / np.maximum(np.bincount(idx, None, n_windows), 1)
        maha = np.bincount(idx, row_maha, n_windows) / np.maximum(np.bincount(idx, None, n_windows), 1)
        return {"mse": mse, "maha": maha}

    def score(self, mel_windows: List[np.ndarray], ctx_vectors: np.ndarray) -> Dict[str, np.ndarray]:
        X, Y, idx = self._io(mel_windows)
        C = np.asarray(ctx_vectors, dtype=np.float32)[idx]
        return self._window_scores(X, Y, C, idx, len(mel_windows))

    # ------------------------------------------------------------------ io
    def save(self, directory: str, tflite: bool = True):
        os.makedirs(directory, exist_ok=True)
        self.model.save(os.path.join(directory, f"{self.name}.keras"))
        np.savez(os.path.join(directory, f"{self.name}_stats.npz"), mean=self.mean, std=self.std,
                 err_var=self.err_var)
        with open(os.path.join(directory, f"{self.name}.json"), "w") as f:
            json.dump(dict(mode=self.mode, units=self.units, bottleneck=self.bottleneck,
                           n_stack=self.n_stack, n_mels=self.n_mels, thresholds=self.thresholds,
                           final_val_loss=min(self.history.get("val_loss", [float("nan")])),
                           epochs_run=len(self.history.get("loss", []))), f, indent=1)
        if tflite:
            import tensorflow as tf
            conv = tf.lite.TFLiteConverter.from_keras_model(self.model)
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.target_spec.supported_types = [tf.float16]
            with open(os.path.join(directory, f"{self.name}.tflite"), "wb") as f:
                f.write(conv.convert())

    @classmethod
    def load(cls, directory: str, mode: str = "ae") -> "ConditionalAutoencoder":
        import tensorflow as tf
        name = "autoencoder" if mode == "ae" else "idnn"
        with open(os.path.join(directory, f"{name}.json")) as f:
            meta = json.load(f)
        m = cls(meta["mode"], meta["units"], meta["bottleneck"], meta["n_stack"], meta["n_mels"])
        st = np.load(os.path.join(directory, f"{name}_stats.npz"))
        m.mean, m.std, m.err_var = st["mean"], st["std"], st["err_var"]
        m.thresholds = meta["thresholds"]
        m.model = tf.keras.models.load_model(os.path.join(directory, f"{name}.keras"))
        return m

    @staticmethod
    def exists(directory: str, mode: str = "ae") -> bool:
        name = "autoencoder" if mode == "ae" else "idnn"
        return os.path.exists(os.path.join(directory, f"{name}.json"))
