"""Exact numpy port of the phone app's front end (docs/index.html: melWindow).

The fault classifier runs in the browser, so it must be trained on the same
features the browser computes: 16 kHz mono, 1 s windows, 1024-point FFT with
a symmetric Hann window, hop 512 (30 frames), 64 HTK-mel bands 50-8000 Hz,
10*log10(energy + 1e-10), window mean removed. tests/test_web_parity.mjs
checks the two implementations agree.
"""
import subprocess

import numpy as np

SR = 16000
WIN = 16000
NFFT = 1024
HOP = 512
NMEL = 64
NFR = (WIN - NFFT) // HOP + 1   # 30


def _melfb():
    h2m = lambda f: 2595 * np.log10(1 + f / 700)
    m2h = lambda m: 700 * (10 ** (m / 2595) - 1)
    lo, hi = h2m(50), h2m(8000)
    pts = [m2h(lo + (hi - lo) * i / (NMEL + 1)) for i in range(NMEL + 2)]
    fb = np.zeros((NMEL, NFFT // 2 + 1))
    for m in range(NMEL):
        k0, k1 = int(np.floor(pts[m] * NFFT / SR)), int(np.ceil(pts[m + 2] * NFFT / SR))
        for k in range(k0, min(k1, NFFT // 2) + 1):
            f = k * SR / NFFT
            a = (f - pts[m]) / (pts[m + 1] - pts[m])
            b = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1])
            v = max(0.0, min(a, b))
            if v > 0:
                fb[m, k] = v
    return fb


MELFB = _melfb()
HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / (NFFT - 1))


def mel_window(x: np.ndarray) -> np.ndarray:
    """1 s of 16 kHz audio -> (NFR, NMEL) float32, identical to the browser."""
    x = np.asarray(x, dtype=np.float64)[:WIN]
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(NFR)[:, None]
    frames = x[idx] * HANN
    p = np.abs(np.fft.rfft(frames, NFFT, axis=1)) ** 2
    db = 10 * np.log10(p @ MELFB.T + 1e-10)
    return (db - db.mean()).astype(np.float32)


def decode16k(path: str) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def speech_mod(w: np.ndarray) -> float:
    """Syllable-rate (2-8 Hz) modulation of the 300-3400 Hz envelope: high for talking, low for engines."""
    frame = SR // 100
    n = len(w) // frame
    fr = w[: n * frame].reshape(n, frame)
    fs = np.abs(np.fft.rfft(fr * np.hanning(frame), axis=1))
    ff = np.fft.rfftfreq(frame, 1 / SR)
    env = np.log(fs[:, (ff >= 300) & (ff <= 3400)].sum(axis=1) + 1e-9)
    env -= env.mean()
    ms = np.abs(np.fft.rfft(env)) ** 2
    mf = np.fft.rfftfreq(len(env), 1 / 100)
    return float(ms[(mf >= 2) & (mf <= 8)].sum() / (ms[mf > 0.5].sum() + 1e-9))
