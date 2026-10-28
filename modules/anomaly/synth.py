"""Synthetic engine-like sounds for tests and demos (not training data).

A firing pulse train at rpm * firings_per_rev / 60 Hz, each pulse a short
decaying burst of band-limited noise, plus broadband hiss. Optional faults:
dead cylinder (every Nth pulse missing), random misfires, a belt squeal tone,
and slow rpm surging.
"""
import numpy as np
from scipy.signal import butter, lfilter

from modules.audio_features import SAMPLE_RATE
from modules.anomaly.context import EngineProfile


def engine_sound(rpm, profile: EngineProfile, seconds=5.0, seed=0, dead_cylinder=False,
                 misfire_p=0.0, squeal_hz=None, surge=0.0, gain=1.0, sr=SAMPLE_RATE):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    out = 0.02 * rng.standard_normal(n)
    b, a = butter(2, [150 / (sr / 2), 3000 / (sr / 2)], btype="band")
    burst_len = int(0.012 * sr)
    decay = np.exp(-np.arange(burst_len) / (0.003 * sr))
    t, k = 0.0, 0
    cyl = profile.cylinders
    while t < seconds:
        r = rpm * (1 + surge * np.sin(2 * np.pi * 0.7 * t))
        period = 1.0 / profile.rpm_to_firing_hz(r)
        skip = (dead_cylinder and k % cyl == 0) or rng.random() < misfire_p
        if not skip:
            i = int(t * sr)
            burst = lfilter(b, a, rng.standard_normal(burst_len)) * decay
            j = min(n, i + burst_len)
            out[i:j] += burst[: j - i]
        t += period
        k += 1
    if squeal_hz:
        tt = np.arange(n) / sr
        out += 0.08 * np.sin(2 * np.pi * squeal_hz * tt * (1 + 0.002 * np.sin(2 * np.pi * 3 * tt)))
    return (gain * out / (np.abs(out).max() + 1e-9) * 0.5).astype(np.float32)
