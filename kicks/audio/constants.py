"""Audio and spectrogram constants shared by every instrument.

These are fixed across drum types on purpose: one mel representation means one
vocoder and one VAE architecture serve kicks, snares and hi-hats alike. They
must stay in sync with BigVGAN's config (``n_fft`` 1024, 128 bands, hop 256).

What *does* vary per instrument (which bands and frames carry the transient,
how long the tail runs, where the fundamental sits) lives in
:mod:`kicks.instruments`, expressed in terms of the indices these constants
define.
"""

SAMPLE_RATE = 44100
AUDIO_LENGTH = 65536      # ~1.486 s at 44100 Hz
N_FFT = 1024              # Must match BigVGAN (was 2048)
HOP_LENGTH = 256
WIN_SIZE = 1024           # Must match BigVGAN
N_MELS = 128
N_FRAMES = 256            # AUDIO_LENGTH // HOP_LENGTH
FMIN = 0
FMAX = None               # Nyquist

# Spectrogram contract: (B, 1, N_MELS, N_FRAMES), values in [0, 1]
SPEC_SHAPE = (N_MELS, N_FRAMES)

# Fixed bounds for normalization — derived from BigVGAN's mel_spectrogram output.
# ln(1e-5) = -11.5129 is the silence floor (BigVGAN clamps magnitudes to 1e-5).
# 3.0 provides comfortable headroom above the observed corpus max (~2.23) so that
# loud transient peaks are not hard-clipped (clipping degrades the punchiest part
# of the hit). Changing this requires retraining — model output scale is tied to
# these bounds via ``denormalize()``.
LOG_MEL_MIN = -11.5129
LOG_MEL_MAX = 3.0

# Target integrated loudness for LUFS normalization
TARGET_LUFS = -14.0

# Derived resolutions, useful when reading the window indices in a profile.
MS_PER_FRAME = 1000.0 * HOP_LENGTH / SAMPLE_RATE   # ~5.805 ms


def frames_to_ms(frames: float) -> float:
    """Spectrogram frame count -> milliseconds."""
    return frames * MS_PER_FRAME


def ms_to_frames(ms: float) -> int:
    """Milliseconds -> spectrogram frame count (rounded down)."""
    return int(ms / MS_PER_FRAME)
