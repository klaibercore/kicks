"""Waveform-domain analysis primitives.

numpy / scipy / soundfile only — deliberately no torch, so `kicks eval` and
`kicks clean` start in a fraction of a second instead of waiting on torch and
BigVGAN to import.

Everything instrument-dependent (how wide to smooth the envelope, how far apart
two hits must be, which band the instrument lives in) arrives as an
:class:`~kicks.instruments.OnsetSpec` or a band tuple rather than being baked in.
"""

from __future__ import annotations

import numpy as np
import scipy.signal
import soundfile as sf

from ..instruments.profile import Band, OnsetSpec
from .constants import AUDIO_LENGTH, SAMPLE_RATE

EPS = 1e-12


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_audio(
    path: str,
    sample_rate: int = SAMPLE_RATE,
    length: int = AUDIO_LENGTH,
    normalize: bool = True,
) -> np.ndarray | None:
    """Load a wav as mono float32 at ``sample_rate``, fixed length.

    Peak-normalizes by default, which is what the metric analysis wants: every
    measurement is a ratio or a time, so absolute level only adds variance.
    Returns None for unreadable or silent files.
    """
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return None
    x = data.mean(axis=1)
    if sr != sample_rate:
        x = resample(x, sr, sample_rate)
    if length:
        x = fit_length(x, length)
    peak = np.abs(x).max()
    if peak < 1e-4:
        return None
    if normalize:
        x = x / peak
    return x.astype(np.float32)


def resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resample to ``target_sr``."""
    if sr == target_sr:
        return x
    g = np.gcd(int(sr), int(target_sr))
    return scipy.signal.resample_poly(x, target_sr // g, sr // g)


def fit_length(x: np.ndarray, length: int) -> np.ndarray:
    """Truncate or zero-pad to exactly ``length`` samples."""
    if len(x) > length:
        return x[:length]
    if len(x) < length:
        return np.pad(x, (0, length - len(x)))
    return x


# ---------------------------------------------------------------------------
# Envelopes and onsets
# ---------------------------------------------------------------------------

def rms_envelope(x: np.ndarray, win: int = 512) -> np.ndarray:
    """Sample-resolution RMS envelope via Hann-windowed moving average of x²."""
    w = scipy.signal.windows.hann(win)
    w /= w.sum()
    return np.sqrt(np.convolve(x ** 2, w, mode="same") + EPS)


def bandpass(x: np.ndarray, band: Band, sr: int = SAMPLE_RATE, order: int = 4) -> np.ndarray:
    """Filter to ``band = (lo, hi)`` in Hz; ``None`` leaves that side open.

    ``(None, 200.0)`` is a lowpass, ``(3000.0, None)`` a highpass,
    ``(120.0, 800.0)`` a bandpass, ``(None, None)`` a no-op.
    """
    lo, hi = band
    nyq = sr / 2.0
    lo = None if lo is None or lo <= 0 else min(lo, nyq * 0.999)
    hi = None if hi is None or hi >= nyq else hi
    if lo is None and hi is None:
        return x
    if lo is None:
        sos = scipy.signal.butter(order, hi / nyq, btype="lowpass", output="sos")
    elif hi is None:
        sos = scipy.signal.butter(order, lo / nyq, btype="highpass", output="sos")
    else:
        sos = scipy.signal.butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return scipy.signal.sosfilt(sos, x)


def band_envelope(
    x: np.ndarray,
    band: Band,
    sr: int = SAMPLE_RATE,
    smooth_ms: float = 5.0,
) -> np.ndarray:
    """Smoothed rectified envelope of ``x`` restricted to ``band``."""
    env = np.abs(bandpass(x, band, sr))
    win = max(1, int(sr * smooth_ms / 1000))
    return np.convolve(env, np.ones(win) / win, mode="same")


def detect_onsets(
    x: np.ndarray,
    onsets: OnsetSpec,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Distinct hit times (ms) in a waveform.

    The envelope is smoothed hard before peak picking. For a bass-heavy
    instrument this is essential — a 25-30 Hz fundamental has a 33-40 ms period
    and ripples straight through a short RMS window, making every cycle look
    like a separate hit. ``OnsetSpec.min_distance_ms`` then decides what counts
    as a second hit rather than a beater bounce or an envelope wobble.
    """
    env = rms_envelope(x, win=onsets.envelope_win)
    peak = env.max()
    if peak <= 0:
        return np.empty(0)
    peaks, _ = scipy.signal.find_peaks(
        env,
        height=onsets.height_frac * peak,
        prominence=onsets.prominence_frac * peak,
        distance=max(1, int(onsets.min_distance_ms / 1000 * sr)),
    )
    return peaks / sr * 1000.0


def is_loop(
    x: np.ndarray,
    sr: int = SAMPLE_RATE,
    min_lag_ms: float = 100.0,
    acf_threshold: float = 0.3,
    frame_ms: float = 10.0,
) -> bool:
    """Detect cyclical content via energy-envelope autocorrelation.

    A single hit decays monotonically — its autocorrelation falls off and stays
    low. A loop has periodic energy peaks, producing a strong ACF peak at the
    loop period. ``min_lag_ms`` skips past the hit's own body.
    """
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(x) // frame_len
    if n_frames < 4:
        return False
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    env = np.sqrt(np.mean(frames ** 2, axis=1))

    env = env - env.mean()
    norm = float(np.dot(env, env))
    if norm < 1e-10:
        return False

    n_fft = 1
    while n_fft < 2 * n_frames:
        n_fft *= 2
    spec = np.fft.rfft(env, n=n_fft)
    acf = np.fft.irfft(spec * np.conj(spec))[:n_frames] / norm

    min_lag_frames = max(1, int(min_lag_ms / frame_ms))
    if min_lag_frames >= n_frames:
        return False
    return bool(np.max(acf[min_lag_frames:]) > acf_threshold)


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

def stft_power(
    x: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = 1024,
    hop: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(freqs, times_s, power) for a magnitude-squared STFT."""
    f, t, Z = scipy.signal.stft(
        x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
        boundary=None, padded=False,
    )
    return f, t, np.abs(Z) ** 2


def band_power(P: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Per-frame power inside [lo, hi) Hz."""
    return P[(f >= lo) & (f < hi)].sum(axis=0)


def apply_fade_out(x: np.ndarray, end: int, fade_samples: int) -> np.ndarray:
    """Cosine fade-out ending at ``end``, silence after."""
    out = x.copy()
    fade_start = max(0, end - fade_samples)
    fade_len = end - fade_start
    if fade_len > 0:
        out[fade_start:end] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
    out[end:] = 0.0
    return out
