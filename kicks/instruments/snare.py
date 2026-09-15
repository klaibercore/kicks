"""Snare drum profile.

A snare is two sounds at once: a pitched shell tone around 150-400 Hz and a
broadband rattle from the wires stretched across the bottom head. The
descriptors split those apart — ``body`` and ``tone`` track the shell, ``crack``
and ``snap`` track the wires — because the useful creative axis on a snare is
the balance between them, not the sub weight that dominates a kick.

Compared to the kick profile: the fundamental search moves up an octave and a
half, the decay windows are shorter (a snare resolves in 150-400 ms rather than
ringing for a second), and the strip detector listens to a band around the
shell rather than the bottom two octaves.
"""

from .metrics import standard_metrics
from .profile import (
    DescriptorSpec,
    EvalWindows,
    InstrumentProfile,
    OnsetSpec,
    PathSpec,
    Region,
    StripSpec,
    TransientLossSpec,
)

# BigVGAN's Slaney mel centres: bands 5:16 = 186-527 Hz, 16:52 =
# 527 Hz-1.94 kHz, 56:96 = 2.21-7.93 kHz, 72:112 = 3.68-13.2 kHz.
# Four frames are 23.2 ms; frame 14 is 81.3 ms, frame 24 is 139 ms.
# Fixed short windows keep level controls from implicitly measuring the
# length of the entire hit; the wire tail legitimately runs to ~140 ms.
DESCRIPTORS = (
    DescriptorSpec(
        "body", "Body", "power_db_ratio",
        region=Region(bands=(5, 16), frames=(4, 14)),
        reference=Region(bands=(16, 52), frames=(4, 14)),
        doc="Shell weight at 190-530 Hz relative to the 0.5-2 kHz mids.",
    ),
    DescriptorSpec(
        "crack", "Crack", "power_db_ratio",
        region=Region(bands=(56, 96), frames=(0, 4)),
        reference=Region(bands=(56, 96), frames=(4, 14)),
        doc="Wire-band attack-to-body contrast at 2-8 kHz. Higher means a harder initial snap.",
    ),
    DescriptorSpec(
        "snap", "Snap", "power_db_ratio",
        region=Region(bands=(56, 96), frames=(4, 24)),
        reference=Region(bands=(5, 16), frames=(4, 24)),
        doc="Wire rattle against shell tone over the first ~140 ms. "
            "High = wires wide open, low = muted / tuned-down thud.",
    ),
    DescriptorSpec(
        "bright", "Bright", "power_db_ratio",
        region=Region(bands=(72, 112), frames=(4, 24)),
        reference=Region(bands=(16, 52), frames=(4, 24)),
        doc="Wire-body brightness at 3.7-13 kHz, after the initial stick crack.",
    ),
    DescriptorSpec(
        "decay", "Decay", "centroid_ms",
        region=Region(bands=(5, 96), frames=(0, None)),
        doc="Energy-weighted duration in milliseconds across shell and wires. "
            "Higher means longer sustain.",
    ),
)

PROFILE = InstrumentProfile(
    name="snare",
    display_name="Snare",
    noun="snare",
    plural="snares",
    description="Snare one-shots — shell tone plus wire rattle, medium decay.",
    paths=PathSpec(corpus_name="snares", subdir="snare"),
    descriptors=DESCRIPTORS,
    metrics=standard_metrics(
        "snare", "snares",
        overrides={
            # The wires are the point: a snare that loses its HF tail early
            # sounds gated, so HF tail energy is judged less harshly than on a
            # kick, while the fundamental matters more (tuning is audible).
            "hf_tail_ratio_db": {"weight": 1.0},
            "hf_decay_ms": {"weight": 1.0},
            "sub_hz": {
                "weight": 1.5,
                "good": "fundamental in the snare tuning range",
                "low": "fundamental too low — reads as a tom or a muted thud",
                "high": "fundamental too high — sounds like a rim or a timbale",
            },
            "pitch_glide": {
                "weight": 0.5,
                "good": "stable pitch through the hit",
                "low": "pitch rises over time — unnatural for a snare",
                "high": "strong downward sweep — sounds like a tuned tom, not a snare",
            },
            "centroid_drop": {
                "good": "brightness falls from crack to shell tone as expected",
            },
            "decay_ms": {
                "high": "rings far longer than real snares — washy / unresolved tail",
            },
        },
    ),
    onsets=OnsetSpec(
        # No sub fundamental to ripple through the envelope, so a shorter
        # window resolves the hit; snare rolls can still be tighter than kicks.
        envelope_win=2048,
        min_distance_ms=100.0,
        double_hit_ms=100.0,
    ),
    strip=StripSpec(
        onset_band=(120.0, 800.0),       # shell fundamental and its first harmonics
        interrupt_band=(6000.0, None),   # a hi-hat landing later sits above the wires
        min_duration_ms=40.0,
        max_duration_ms=900.0,
        transient_skip_ms=25.0,
    ),
    transient_loss=TransientLossSpec(
        band=60,          # ~2.5 kHz and up — the wire band
        click_frames=5,   # ~29 ms
        tail_start=24,    # ~140 ms; wires legitimately ring longer than a beater click
        weight=0.5,
    ),
    eval_windows=EvalWindows(
        click_ms=15.0,               # the stick hit is shorter than a beater thump
        tail_split_ms=140.0,         # wires legitimately ring past a kick's 80 ms
        tail_flatness_ms=(80.0, 300.0),
        body_ms=(15.0, 180.0),
        glide_early_ms=(8.0, 45.0),
        glide_late_ms=(70.0, 180.0),
        centroid_late_ms=(50.0, 150.0),
        noise_floor_ms=300.0,
    ),
    hf_band=(2000.0, 16000.0),
    fundamental_band=(120.0, 400.0),
    latent_dim=32,
    waveform_controls=True,
)
