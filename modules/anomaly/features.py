"""Features for the anomaly detectors.

Two families, both computed per 1 s window of 22.05 kHz mono audio (the same
front end as the fault classifier, modules/audio_features.py):

* anomaly_mel(window)   -> (N_FRAMES, N_MELS) log-mel for detector A.
  Unlike the classifier we keep a FIXED dB reference (the classifier's
  per-window max/min-max normalisation would erase level and contrast, which
  is exactly what an anomaly can change). Only the window's mean dB is
  subtracted, to cancel phone-gain / distance differences.

* descriptors(window)   -> dict of named, physically meaningful features for
  detector B (band balance, high-band ratio, firing-rhythm strength,
  uneven-firing ratio, roughness, slow modulation). Names are what the
  regression reports back ("hf_ratio_db +4.1 sigma").

* estimate_firing_hz(window) -> the engine's firing frequency from the
  amplitude envelope, used to estimate rpm when there is no tach.
"""
from __future__ import annotations

from typing import Optional

import librosa
import numpy as np
from scipy.signal import get_window, resample_poly
from scipy.stats import kurtosis

from modules.audio_features import (HOP, N_FFT, N_FRAMES, N_MELS, SAMPLE_RATE, WINDOW_SAMPLES,
                                    iter_windows, to_mono_resampled)
from modules.anomaly.context import EngineProfile

ENV_SR = 1000                      # envelope sample rate (firing up to ~450 Hz)
BAND_EDGES = np.geomspace(60.0, 10000.0, 9)  # 8 log-spaced bands
DESCRIPTOR_NAMES = (
    [f"band{i}_db" for i in range(8)] +
    ["centroid_oct", "rolloff_oct", "flatness_db", "hf_ratio_db",
     "firing_db", "uneven_db", "crest_db", "env_kurtosis", "slow_mod_db"]
)
LEVEL_NAME = "level_db"  # absolute loudness; only meaningful with a fixed mic position

_MEL_FB = None


def _mel_fb():
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmax=SAMPLE_RATE / 2)
    return _MEL_FB


# --------------------------------------------------------------------------- windows
def windows(audio: np.ndarray, sr: int, hop_s: float = 1.0):
    """Raw audio (any rate/channels) -> list of 1 s windows at SAMPLE_RATE."""
    y = to_mono_resampled(audio, sr)
    out = [w for _, w in iter_windows(y, hop_s)]
    if not out and len(y) > WINDOW_SAMPLES // 4:   # short clip: pad once
        out = [np.pad(y, (0, WINDOW_SAMPLES - len(y)))]
    return out


# --------------------------------------------------------------------------- detector A
def anomaly_mel(window: np.ndarray) -> np.ndarray:
    """1 s window -> (N_FRAMES, N_MELS) log-mel dB, fixed reference, window mean removed."""
    spec = np.abs(librosa.stft(window[:WINDOW_SAMPLES], n_fft=N_FFT, hop_length=HOP)) ** 2
    mel = _mel_fb() @ spec
    db = 10.0 * np.log10(mel + 1e-10)[:, :N_FRAMES]
    if db.shape[1] < N_FRAMES:
        db = np.pad(db, ((0, 0), (0, N_FRAMES - db.shape[1])), mode="edge")
    db -= db.mean()
    return db.T.astype(np.float32)


def stack_frames(frames: np.ndarray, n: int = 5) -> np.ndarray:
    """(T, F) -> (T-n+1, n*F): n consecutive frames concatenated (DCASE baseline input)."""
    t, f = frames.shape
    idx = np.arange(n)[None, :] + np.arange(t - n + 1)[:, None]
    return frames[idx].reshape(t - n + 1, n * f)


# --------------------------------------------------------------------------- envelope / rpm
def envelope(window: np.ndarray) -> np.ndarray:
    """Amplitude envelope at ENV_SR (rectify + anti-aliased decimation)."""
    return resample_poly(np.abs(window), ENV_SR, SAMPLE_RATE)


def _env_spectrum(env: np.ndarray, nfft: int = 8192):
    e = (env - env.mean()) * get_window("hann", len(env))
    p = np.abs(np.fft.rfft(e, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / ENV_SR)
    return f, p


def _peak_near(f, p, target, rel=0.04):
    if target <= f[1] or target >= f[-1]:
        return 0.0
    m = (f >= target * (1 - rel)) & (f <= target * (1 + rel))
    return float(p[m].max()) if m.any() else 0.0


def firing_search_range(profile: Optional[EngineProfile]):
    if profile is None:
        return 6.0, 400.0
    lo = profile.rpm_to_firing_hz(0.6 * profile.idle_rpm)
    hi = profile.rpm_to_firing_hz(1.1 * profile.max_rpm)
    return max(lo, 4.0), min(hi, 0.45 * ENV_SR)


def estimate_firing_hz(window: np.ndarray, profile: Optional[EngineProfile] = None,
                       env=None, spec=None):
    """Return (firing_hz, confidence 0..1) from the envelope's harmonic structure."""
    env = envelope(window) if env is None else env
    f, p = _env_spectrum(env) if spec is None else spec
    lo, hi = firing_search_range(profile)
    cand = np.arange(lo, hi, 0.25)
    if len(cand) == 0 or p.sum() <= 0:
        return None, 0.0
    # harmonic sum: firing pulses produce peaks at f, 2f, 3f ...
    score = np.zeros_like(cand)
    for k, w in ((1, 1.0), (2, 0.5), (3, 0.33)):
        score += w * np.interp(k * cand, f, p, right=0.0)
    i = int(np.argmax(score))
    conf = float(1.0 - np.median(score) / (score[i] + 1e-12))  # how far the peak stands out
    return float(cand[i]), conf


def estimate_rpm(audio_windows, profile: EngineProfile, min_conf: float = 0.5):
    """Median rpm over windows whose firing estimate is confident enough (None if none)."""
    est = []
    for w in audio_windows:
        hz, conf = estimate_firing_hz(w, profile)
        if hz is not None and conf >= min_conf:
            est.append(profile.firing_hz_to_rpm(hz))
    return (float(np.median(est)) if est else None), len(est)


# --------------------------------------------------------------------------- detector B
def descriptors(window: np.ndarray, profile: Optional[EngineProfile] = None,
                firing_hz: Optional[float] = None, include_level: bool = False) -> dict:
    """Named descriptor features for one 1 s window."""
    x = window[:WINDOW_SAMPLES].astype(np.float64)
    out = {}
    spec = np.abs(np.fft.rfft(x * get_window("hann", len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / SAMPLE_RATE)
    rng = (freqs >= BAND_EDGES[0]) & (freqs <= BAND_EDGES[-1])
    ps, fs = spec[rng] + 1e-12, freqs[rng]
    total = ps.sum()
    for i in range(8):
        m = (fs >= BAND_EDGES[i]) & (fs < BAND_EDGES[i + 1])
        out[f"band{i}_db"] = 10 * np.log10(ps[m].sum() / total + 1e-12)
    out["centroid_oct"] = float(np.log2((fs * ps).sum() / total))
    cum = np.cumsum(ps)
    out["rolloff_oct"] = float(np.log2(fs[np.searchsorted(cum, 0.85 * total)]))
    out["flatness_db"] = float(10 * np.log10(np.exp(np.mean(np.log(ps))) / np.mean(ps)))
    out["hf_ratio_db"] = float(10 * np.log10(ps[fs >= 4000].sum() / total + 1e-12))

    env = envelope(x)
    f, p = _env_spectrum(env)
    if firing_hz is None:
        firing_hz, _ = estimate_firing_hz(x, profile, env=env, spec=(f, p))
    lo, hi = firing_search_range(profile)
    floor = float(np.median(p[(f >= lo) & (f <= hi)])) + 1e-12
    pf = _peak_near(f, p, firing_hz) if firing_hz else 0.0
    # A weak or dead cylinder makes the pattern repeat once per engine cycle
    # (every `cylinders` firings), adding energy at k * firing / cylinders.
    cyl = profile.cylinders if profile else 4
    sub = max((_peak_near(f, p, firing_hz * k / cyl) for k in range(1, cyl)), default=0.0) if firing_hz else 0.0
    out["firing_db"] = float(10 * np.log10(pf / floor + 1e-12))
    out["uneven_db"] = float(10 * np.log10((sub + 1e-12) / (pf + 1e-12)))
    rms = np.sqrt(np.mean(env ** 2)) + 1e-12
    out["crest_db"] = float(20 * np.log10(env.max() / rms + 1e-12))
    out["env_kurtosis"] = float(np.log1p(max(kurtosis(env, fisher=True) + 2.0, 0.0)))
    slow = p[(f >= 0.5) & (f <= 8.0)].sum()
    allm = p[(f >= 0.5) & (f <= 450.0)].sum() + 1e-12
    out["slow_mod_db"] = float(10 * np.log10(slow / allm + 1e-12))
    if include_level:
        out[LEVEL_NAME] = float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))
    return out
