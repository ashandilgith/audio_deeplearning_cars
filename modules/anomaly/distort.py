"""Fault-like distortions applied to REAL normal engine audio.

Used to test the anomaly detectors on the same engine and microphone (EXP-006):
the only difference between a clean and a distorted window is the injected
"fault", so detection cannot come from recognising a different recording.

Each distortion takes a 1 s window at 22.05 kHz, a severity in {1, 2, 3} and
the window's firing frequency (Hz, estimated from the clean audio), and
mimics the acoustic signature of a fault class:

    squeal        narrow-band tone 2.5-7 kHz with slight vibrato   (belt / pulley)
    dead_cyl      every second firing interval attenuated          (dead cylinder)
    misfire       random firing intervals dropped, plus pops       (misfire / backfire)
    surge         slow speed + level wobble                        (lean surging)
    knock         short metallic impulse once per revolution       (mechanical knock)
    hiss          added high-frequency noise                       (exhaust / air leak)

NUISANCES are changes that should NOT raise an alarm (mic gain, background
noise, a muffled mic), used to measure false-alarm robustness.

These are approximations of real faults, good for comparing detectors and
settings, not proof of field performance.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter

SR = 22050


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def _band_noise(n, lo, hi, rng):
    b, a = butter(2, [lo / (SR / 2), min(hi, SR / 2 - 100) / (SR / 2)], btype="band")
    return lfilter(b, a, rng.standard_normal(n))


def _smooth_gate(phase_mask, n_smooth=40):
    k = np.hanning(n_smooth)
    k /= k.sum()
    return np.convolve(phase_mask.astype(float), k, mode="same")


def squeal(x, sev, f_fire, rng):
    t = np.arange(len(x)) / SR
    f = rng.uniform(2500, 7000)
    tone = np.sin(2 * np.pi * f * t + 3.0 * np.sin(2 * np.pi * rng.uniform(2, 6) * t))
    level_db = {1: -26, 2: -19, 3: -12}[sev]
    return x + tone * _rms(x) * 10 ** (level_db / 20) * np.sqrt(2)


def dead_cyl(x, sev, f_fire, rng):
    t = np.arange(len(x)) / SR
    phase = (t * f_fire + rng.uniform(0, 2)) % 2.0
    depth = {1: 0.4, 2: 0.7, 3: 0.95}[sev]
    gate = 1.0 - depth * _smooth_gate(phase < 1.0)
    return x * gate


def misfire(x, sev, f_fire, rng):
    p = {1: 0.06, 2: 0.12, 3: 0.25}[sev]
    n_int = int(np.ceil(len(x) / SR * f_fire)) + 1
    t = np.arange(len(x)) / SR
    idx = np.floor(t * f_fire + rng.uniform(0, 1)).astype(int)
    drop = rng.random(n_int) < p
    gate = 1.0 - 0.9 * _smooth_gate(drop[idx])
    y = x * gate
    # backfire pops after some dropped intervals
    peak = np.abs(x).max() + 1e-9
    for k in np.where(drop)[0]:
        if rng.random() < 0.5:
            i = int(k / f_fire * SR)
            n = int(0.004 * SR)
            if i + n < len(y):
                y[i:i + n] += 0.8 * peak * _band_noise(n, 200, 6000, rng) / 3.0 * np.exp(-np.arange(n) / (0.001 * SR))
    return y


def surge(x, sev, f_fire, rng):
    d = {1: 0.02, 2: 0.05, 3: 0.10}[sev]
    n = len(x)
    t = np.arange(n) / SR
    fm = rng.uniform(1.0, 2.0)
    ph0 = rng.uniform(0, 2 * np.pi)
    rate = 1.0 + d * np.sin(2 * np.pi * fm * t + ph0)
    src = np.cumsum(rate)
    src = src / src[-1] * (n - 1)
    y = np.interp(src, np.arange(n), x)
    return y * (1.0 + 2.0 * d * np.sin(2 * np.pi * fm * t + ph0))


def knock(x, sev, f_fire, rng, firings_per_rev=2.0):
    f_rev = f_fire / firings_per_rev
    amp = {1: 0.25, 2: 0.5, 3: 0.9}[sev] * (np.abs(x).max() + 1e-9)
    y = x.copy()
    n = int(0.006 * SR)
    t0 = rng.uniform(0, 1 / max(f_rev, 1))
    ring = np.sin(2 * np.pi * rng.uniform(2500, 4500) * np.arange(n) / SR) * np.exp(-np.arange(n) / (0.0012 * SR))
    for tk in np.arange(t0, len(x) / SR, 1 / max(f_rev, 1)):
        i = int(tk * SR)
        j = min(len(y), i + n)
        y[i:j] += amp * ring[: j - i]
    return y


def hiss(x, sev, f_fire, rng):
    level_db = {1: -24, 2: -17, 3: -10}[sev]
    noise = _band_noise(len(x), 3000, 10000, rng)
    return x + noise / _rms(noise) * _rms(x) * 10 ** (level_db / 20)


# -------------------------------------------------------------- nuisances (should stay NORMAL)
def gain(x, sev, f_fire, rng):
    return x * 10 ** (rng.choice([-1, 1]) * 6 / 20)


def background(x, sev, f_fire, rng):
    # pink-ish noise (wind / water) at 20 dB below the engine
    w = np.cumsum(rng.standard_normal(len(x)))
    w = w - np.convolve(w, np.ones(200) / 200, mode="same")
    return x + w / _rms(w) * _rms(x) * 10 ** (-20 / 20)


def muffled(x, sev, f_fire, rng):
    b, a = butter(2, 6000 / (SR / 2))
    return lfilter(b, a, x)


FAULTS = {"squeal": squeal, "dead_cyl": dead_cyl, "misfire": misfire,
          "surge": surge, "knock": knock, "hiss": hiss}
NUISANCES = {"gain_6db": gain, "background_20db": background, "muffled_mic": muffled}


def apply(name, x, sev, f_fire, seed=0):
    fn = FAULTS.get(name) or NUISANCES[name]
    rng = np.random.default_rng(seed)
    y = fn(np.asarray(x, dtype=np.float64), sev, f_fire, rng)
    return y.astype(np.float32)
