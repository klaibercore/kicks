"""The standard evaluation metric set, phrased for a given instrument.

Every drum shares the same waveform measurements — attack, decay, crest, noise
floor, high-frequency behaviour, tail character, fundamental, pitch glide,
brightness contour, onset count. What differs is the wording of a verdict and
how much each measurement matters. :func:`standard_metrics` builds the set with
the instrument's noun substituted in; profiles then drop what does not apply
(a hi-hat has no fundamental to glide) and re-word what reads wrong.
"""

from __future__ import annotations

from .profile import MetricSpec


def standard_metrics(
    noun: str,
    plural: str,
    drop: tuple[str, ...] = (),
    overrides: dict[str, dict[str, object]] | None = None,
) -> tuple[MetricSpec, ...]:
    """Build the metric set for an instrument.

    Args:
        noun: singular instrument noun used in verdicts, e.g. ``"kick"``.
        plural: plural form, e.g. ``"kicks"``.
        drop: metric keys that do not apply to this instrument.
        overrides: per-key field overrides, e.g.
            ``{"sub_hz": {"weight": 2.0, "high": "..."}}``.
    """
    specs = (
        MetricSpec(
            "attack_ms", "attack", "{:.1f} ms", 1.0, True,
            "punchy attack",
            "attack is instantaneous — may click unnaturally",
            "attack too slow — hit feels soft / swallowed",
        ),
        MetricSpec(
            "decay_ms", "decay", "{:.0f} ms", 1.0, True,
            "natural decay length",
            "dies off abruptly — sounds truncated",
            f"rings far longer than real {plural} — boomy / unresolved tail",
        ),
        MetricSpec(
            "crest_db", "punch (crest)", "{:.1f} dB", 0.5, False,
            "healthy transient-to-body dynamics",
            "over-compressed — transient is flattened",
            "thin body — all click, no weight",
        ),
        MetricSpec(
            "noise_floor_db", "tail silence", "{:.0f} dBFS", 1.5, False,
            "tail decays into silence",
            "",  # a lower noise floor than the corpus is fine
            "audible noise floor in the tail — vocoder hiss",
            multivariate=False,  # corpus tails are digital silence; saturates the stats
        ),
        MetricSpec(
            "hf_click_db", "HF click", "{:.1f} dB", 1.0, False,
            "crisp high-frequency click in the transient",
            "no high-frequency click — sounds dull / muffled",
            f"transient is all high end — sounds like a tick, not a {noun}",
        ),
        MetricSpec(
            "hf_decay_ms", "HF decay", "{:.0f} ms", 2.0, True,
            f"high end decays as fast as real {plural}",
            "high end vanishes instantly — click sounds detached",
            "high frequencies ring on — smeared / metallic top end",
        ),
        MetricSpec(
            "hf_tail_ratio_db", "HF tail energy", "{:.1f} dB", 2.0, False,
            "high-end energy correctly concentrated in the transient",
            "",  # less HF tail than the corpus is not an artefact
            "too much HF energy after 80 ms — hissy / smeared tail",
        ),
        MetricSpec(
            "tail_flatness", "tail character", "{:.2f}", 1.0, False,
            f"tail spectrum looks like a real {noun}",
            "tonal ringing in the tail — metallic vocoder artefact",
            "noise-like tail — hiss instead of pitch",
        ),
        MetricSpec(
            "sub_hz", "sub fundamental", "{:.0f} Hz", 1.0, False,
            f"fundamental in the {noun} sweet spot",
            f"fundamental below typical {plural} — may just rumble",
            f"fundamental too high — sounds like a tom, not a {noun}",
        ),
        MetricSpec(
            "pitch_glide", "pitch glide", "{:.2f}×", 1.0, False,
            f"downward pitch sweep like a real {noun}",
            f"pitch rises over time — unnatural for a {noun}",
            "extreme pitch drop — laser-like sweep",
        ),
        MetricSpec(
            "centroid_drop", "brightness contour", "{:.1f} oct", 1.0, False,
            "brightness falls from click to sub as expected",
            "spectrum stays static — sounds like a filtered blob",
            "unusually steep spectral collapse",
        ),
        # Gate: never a matter of degree. Two hits in a one-shot is an artefact
        # whatever the corpus distribution says, so it bypasses the weighted
        # score and applies a flat penalty instead.
        MetricSpec(
            "n_onsets", "onset count", "{:.0f}", 0.0, False,
            "a single clean hit",
            "",
            "separate hits detected — should be a single hit "
            "(flutter / double-trigger artefact)",
            gate=True, multivariate=False,
        ),
    )

    overrides = overrides or {}
    out = []
    for spec in specs:
        if spec.key in drop:
            continue
        if spec.key in overrides:
            spec = spec.with_text(**overrides[spec.key])
        out.append(spec)
    return tuple(out)
