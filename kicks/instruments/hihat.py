"""Hi-hat profile.

Hi-hats break most of the assumptions the kick pipeline was built on: there is
no fundamental to track, nothing meaningful below ~500 Hz, and the whole event
is often over in 60 ms. So this profile drops the pitch metrics entirely
(``sub_hz`` and ``pitch_glide`` measure noise on an unpitched source), moves
every descriptor window up into the top three octaves, and shortens the decay
windows by roughly a factor of four.

The one axis worth more here than anywhere else is decay: the closed/open
distinction is the single most important thing about a hi-hat, and it is a pure
decay-length difference (~60 ms versus ~300 ms and up).
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

DESCRIPTORS = (
    DescriptorSpec(
        "attack", "Attack", "mean",
        region=Region(bands=(60, 110), frames=(0, 2)),
        doc="Stick contact energy, ~2.5-12 kHz, first ~12 ms.",
    ),
    DescriptorSpec(
        "body", "Body", "mean",
        region=Region(bands=(30, 70), frames=(2, None)),
        doc="Lower shimmer around ~950 Hz - 3.5 kHz — how much metal is heard "
            "rather than pure hiss.",
    ),
    DescriptorSpec(
        "sizzle", "Sizzle", "mean",
        region=Region(bands=(95, None), frames=(2, None)),
        doc="Air and sizzle above ~7.7 kHz, transient excluded.",
    ),
    DescriptorSpec(
        "bright", "Bright", "fraction",
        region=Region(bands=(90, None), frames=(2, None)),
        reference=Region(bands=(30, 70), frames=(2, None)),
        doc="Balance of top-end sizzle against lower shimmer.",
    ),
    DescriptorSpec(
        "decay", "Decay", "inverse_ratio",
        region=Region(bands=(40, 115), frames=(10, 44)),
        reference=Region(bands=(40, 115), frames=(2, 10)),
        doc="Early-to-late energy ratio over ~12-255 ms — the closed/open axis. "
            "Higher = tightly closed, lower = open or pedal-splashed.",
    ),
)

PROFILE = InstrumentProfile(
    name="hihat",
    display_name="Hi-Hat",
    noun="hi-hat",
    plural="hi-hats",
    description="Hi-hat one-shots — unpitched, top-end dominant, very short decay.",
    paths=PathSpec(corpus_name="hihats", subdir="hihat"),
    descriptors=DESCRIPTORS,
    metrics=standard_metrics(
        "hi-hat", "hi-hats",
        # Unpitched: a spectral peak in the low band and its drift over time are
        # both measuring noise, so neither is scored.
        drop=("sub_hz", "pitch_glide"),
        overrides={
            # The entire sound is high frequency, so "HF rings on" is not an
            # artefact here — it is the instrument. Decay length carries the
            # weight instead, since closed vs open is the defining distinction.
            "hf_decay_ms": {
                "weight": 1.0,
                "good": "top end decays like a real hi-hat",
                "low": "top end vanishes instantly — sounds clipped, not closed",
                "high": "top end rings on — sounds like a crash or an open hat",
            },
            "hf_tail_ratio_db": {
                "weight": 0.5,
                "good": "high-end energy distributed like a real hi-hat",
                "low": "",
                "high": "high end sustains far past the hit — reads as a cymbal wash",
            },
            "decay_ms": {
                "weight": 2.5,
                "good": "decay length reads as a real hi-hat",
                "low": "dies off abruptly — sounds gated or truncated",
                "high": "rings far longer than real hi-hats — reads as a crash",
            },
            "hf_click_db": {
                "good": "crisp stick attack",
                "low": "no stick attack — sounds like a wash, not a hit",
                "high": "transient is all high end — sounds like a tick",
            },
            "tail_flatness": {
                "good": "tail spectrum looks like real hi-hat noise",
                "low": "tonal ringing in the tail — metallic vocoder artefact",
                "high": "",  # a noise-like tail is correct for a hi-hat
            },
            "centroid_drop": {
                "weight": 0.5,
                "good": "brightness holds through the decay as expected",
                "low": "spectrum stays static — sounds like a filtered blob",
                "high": "top end collapses immediately — sounds muffled after the attack",
            },
            "crest_db": {"weight": 1.0},
        },
    ),
    onsets=OnsetSpec(
        # Nothing low-frequency to smooth away, but a ringing hat re-peaks the
        # envelope dozens of times: tuned against the corpus, which showed
        # shimmer spikes clearing a 45 ms / 0.25-prominence threshold every
        # ~100-300 ms. Wider spacing and higher prominence filter those; what
        # still flags (multi-peak pedal chicks, bounce-and-return takes) are
        # genuinely multi-event recordings. A fast roll is still caught — 16ths
        # at 180 BPM are ~83 ms apart with far clearer peaks than shimmer.
        envelope_win=512,
        min_distance_ms=90.0,
        height_frac=0.30,
        prominence_frac=0.35,
        double_hit_ms=90.0,
        loop_onsets=4,
    ),
    strip=StripSpec(
        onset_band=(3000.0, None),      # the hat lives in the top three octaves
        interrupt_band=(None, 200.0),   # a kick landing later shows up underneath
        min_duration_ms=20.0,
        max_duration_ms=600.0,
        threshold=0.02,
        fade_ms=5.0,
        transient_skip_ms=10.0,
        loop_min_lag_ms=60.0,
    ),
    transient_loss=TransientLossSpec(
        band=70,          # ~3.5 kHz and up
        click_frames=3,   # ~17 ms
        tail_start=44,    # ~255 ms — well past even an open hat's useful tail
        weight=0.3,       # lighter: over-constraining HF would flatten the sizzle
    ),
    eval_windows=EvalWindows(
        # A closed hat is over in ~60 ms, so every window shrinks. The pitch
        # windows are still declared because the fields exist, but the metrics
        # that read them are dropped above.
        click_ms=8.0,
        tail_split_ms=45.0,
        tail_flatness_ms=(40.0, 200.0),
        body_ms=(5.0, 120.0),
        glide_early_ms=(3.0, 25.0),
        glide_late_ms=(40.0, 120.0),
        centroid_late_ms=(25.0, 100.0),
        noise_floor_ms=250.0,
        decay_floor_db=-35.0,        # hats decay fast; -30 dB clips the useful tail
    ),
    hf_band=(5000.0, 20000.0),
    fundamental_band=None,   # unpitched
    latent_dim=32,
)
