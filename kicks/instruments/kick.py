"""Kick drum profile.

The reference instrument: the descriptor windows, strip heuristics and metric
weights here are the ones the project's corpus and trained checkpoint were
tuned against, so they are kept exactly as they were before the pipeline became
instrument-agnostic. Its paths use the original flat layout
(``data/kicks``, ``models/vae_best.pth``) so existing artefacts keep working.
"""

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

# Frames are ~5.8 ms. Frames 0-3 are the transient; most descriptors start at
# frame 3 because the transient's frame-to-frame variance is large enough to
# swamp a body measurement.
DESCRIPTORS = (
    DescriptorSpec(
        "sub", "Sub", "mean",
        region=Region(bands=(0, 12), frames=(3, None)),
        doc="Sustained low-end weight (below ~400 Hz), transient excluded.",
    ),
    DescriptorSpec(
        "punch", "Punch", "log_ratio",
        region=Region(bands=(10, 30), frames=(0, 3)),
        reference=Region(bands=(10, 30), frames=(3, 30)),
        scale=2.0,
        doc="Attack-to-body contrast in the 345 Hz - 950 Hz range.",
    ),
    DescriptorSpec(
        "click", "Click", "mean",
        region=Region(bands=(40, 100), frames=(0, 3)),
        doc="High-frequency energy in the first ~17 ms — the beater click.",
    ),
    DescriptorSpec(
        "bright", "Bright", "fraction",
        region=Region(bands=(50, None), frames=(3, None)),
        reference=Region(bands=(0, 30), frames=(3, None)),
        doc="Share of body energy above ~1.8 kHz.",
    ),
    DescriptorSpec(
        "decay", "Decay", "inverse_ratio",
        region=Region(bands=(0, 40), frames=(30, 120)),
        reference=Region(bands=(0, 40), frames=(3, 30)),
        doc="Broadband early-to-late energy ratio. Higher = faster decay "
            "(acoustic thump), lower = long sustain (808-style).",
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
)
