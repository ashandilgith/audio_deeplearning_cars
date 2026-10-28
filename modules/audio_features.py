"""Shared audio preprocessing for training and inference.

Every audio source (phone recordings, uploaded files) must go
through the same steps, otherwise the model sees different "images" for the
same sound:

1. mono, resampled to SAMPLE_RATE
2. cut into WINDOW_S second windows
3. 128-band log-Mel spectrogram, fixed to N_FRAMES time frames
4. per-window min-max normalisation to [0, 1]
"""
import numpy as np
import librosa

SAMPLE_RATE = 22050
WINDOW_S = 1.0
N_MELS = 128
N_FFT = 2048
HOP = 512
N_FRAMES = 44  # 1 s at 22050 Hz with hop 512 -> 44 frames

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_S)


def to_mono_resampled(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def iter_windows(audio: np.ndarray, hop_s: float = WINDOW_S):
    """Yield (start_seconds, window) for full-length windows of resampled audio."""
    hop = int(SAMPLE_RATE * hop_s)
    for start in range(0, len(audio) - WINDOW_SAMPLES + 1, hop):
        yield start / SAMPLE_RATE, audio[start:start + WINDOW_SAMPLES]


def mel_window(window: np.ndarray) -> np.ndarray:
    """One WINDOW_S window of resampled audio -> (N_MELS, N_FRAMES) in [0, 1]."""
    if len(window) < WINDOW_SAMPLES:
        window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))
    mel = librosa.feature.melspectrogram(
        y=window[:WINDOW_SAMPLES], sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmax=SAMPLE_RATE / 2,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)[:, :N_FRAMES]
    if mel_db.shape[1] < N_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, N_FRAMES - mel_db.shape[1])),
                        constant_values=mel_db.min())
    lo, hi = mel_db.min(), mel_db.max()
    return ((mel_db - lo) / (hi - lo + 1e-6)).astype(np.float32)


def features_for_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Raw audio of any length/rate -> batch (n_windows, N_MELS, N_FRAMES, 1)."""
    audio = to_mono_resampled(audio, sr)
    wins = [mel_window(w) for _, w in iter_windows(audio)]
    if not wins:  # shorter than one window: pad it
        wins = [mel_window(audio)]
    return np.stack(wins)[..., None]
