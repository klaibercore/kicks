"""Kick profile with gain-invariant, audible-band control measurements."""

from .metrics import standard_metrics
from .profile import (
    DescriptorSpec,
    InstrumentProfile,
    OnsetSpec,
    PathSpec,
    Region,
    StripSpec,
    TransientLossSpec,
)

# BigVGAN's Slaney mel centres: bands 0:6 = 31-186 Hz, 6:32 =
# 217-992 Hz, 53:97 = 2.00-7.93 kHz, 53:118 = 2.00-15.53 kHz.
# Four frames are 23.2 ms; frame 14 is 81.3 ms. Fixed short windows keep
# level controls from implicitly measuring the length of the entire hit.
DESCRIPTORS = (
    DescriptorSpec(
        "sub", "Sub", "power_db_ratio",
        region=Region(bands=(0, 6), frames=(4, 14)),
        reference=Region(bands=(6, 32), frames=(4, 14)),
        doc="Low-end weight below 200 Hz relative to the 200 Hz–1 kHz body.",
    ),
    DescriptorSpec(
        "punch", "Punch", "power_db_ratio",
        region=Region(bands=(0, 32), frames=(0, 4)),
        reference=Region(bands=(0, 32), frames=(4, 14)),
        doc="Low-frequency attack-to-body contrast. Higher means a harder initial hit.",
    ),
    DescriptorSpec(
        "click", "Click", "power_db_ratio",
        region=Region(bands=(53, 118), frames=(0, 4)),
        reference=Region(bands=(0, 32), frames=(0, 4)),
        doc="Beater click at 2–16 kHz relative to the low-frequency attack.",
    ),
    DescriptorSpec(
        "bright", "Bright", "power_db_ratio",
        region=Region(bands=(53, 97), frames=(4, 14)),
        reference=Region(bands=(6, 32), frames=(4, 14)),
        doc="Body brightness at 2–8 kHz, after the initial beater click.",
    ),
    DescriptorSpec(
        "decay", "Decay", "centroid_ms",
        region=Region(bands=(0, 6), frames=(0, None)),
        doc="Energy-weighted low-end duration in milliseconds. Higher means longer sustain.",
    ),
)

PROFILE = InstrumentProfile(
    name="kick",
    display_name="Kick",
    noun="kick",
    plural="kicks",
    description="Bass drum one-shots — sub-dominant, short beater click, long decay.",
    paths=PathSpec(corpus_name="kicks", subdir=""),
    descriptors=DESCRIPTORS,
    metrics=standard_metrics("kick", "kicks"),
    onsets=OnsetSpec(
        # A 25-30 Hz fundamental ripples through a short RMS window, so the
        # envelope is smoothed over ~93 ms before peak picking.
        envelope_win=4096,
        min_distance_ms=150.0,
    ),
    strip=StripSpec(
        onset_band=(None, 200.0),        # the kick lives under 200 Hz
        interrupt_band=(2000.0, None),   # a hat or snare landing later shows up in the top end
        min_duration_ms=50.0,
        max_duration_ms=1200.0,
    ),
    transient_loss=TransientLossSpec(
        band=50,          # ~1.8 kHz and up
        click_frames=6,   # ~35 ms
        tail_start=14,    # ~80 ms, matching the hf_tail_ratio_db eval window
        weight=0.5,
    ),
    hf_band=(2000.0, 16000.0),
    fundamental_band=(25.0, 150.0),
    latent_dim=32,
    waveform_controls=True,
)
