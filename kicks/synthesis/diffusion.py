"""Generation through the waveform diffusion backend.

Nothing here touches the VAE, the latent GMM prior or a vocoder: the sampler
walks noise down to audio in one model. What it does share with the rest of the
project is the meaning of a slider — targets are the instrument profile's own
descriptor values, in the profile's own units, and the achieved values are
measured with the same code that calibrates the VAE's sliders.

Texture and sliders are separated on purpose. ``seed`` fixes the starting noise
(the "which hit is this" seed) and ``label_seed`` fixes any descriptor values
that were not given explicitly, so sweeping one slider across a fixed texture
seed changes only what the slider names. Fixed noise encourages continuity; it
does not guarantee that the rest of the hit holds still, which is exactly what
a control audit has to measure rather than assume.

Targets come from the corpus, not from its moments. A checkpoint carries the
training split's descriptor rows (the *label bank*), and unpinned descriptors
are drawn by resampling those rows: the values are ones a real hit had, in the
combinations real hits have. Drawing each descriptor independently from a
Gaussian — the fallback for a checkpoint without a bank — asks for targets no
hit could satisfy (a negative decay in about a quarter of snare draws, given
the corpus's heavy right tail) and ignores the correlations between sliders.
"""

from __future__ import annotations

import math
import os
import warnings

import numpy as np
import torch

from ..audio.constants import SAMPLE_RATE
from ..config import get_device, load_diffusion_from_checkpoint
from ..data.waveforms import TARGET_PEAK, label_waveform
from ..instruments import InstrumentProfile, get_profile
from ..nn.diffusion import v_sample


#: Smallest neighbourhood a pinned descriptor is matched against, and the
#: fraction of the bank used when that is larger. Enough rows that the free
#: descriptors still vary, few enough that they belong with the pinned value.
BANK_NEIGHBOURS = 16
BANK_NEIGHBOUR_FRACTION = 0.05


def draw_labels(
    model, count: int, overrides: dict[str, float] | None,
    profile: InstrumentProfile, seed: int | None = None,
) -> torch.Tensor:
    """Descriptor targets for ``count`` hits, in raw profile units.

    Unspecified descriptors are resampled from the checkpoint's label bank.
    With pinned descriptors, rows are drawn from the bank's nearest neighbours
    in the pinned dimensions (standardised distance), then the pinned values
    are set exactly — so the free descriptors are ones the corpus pairs with
    that value rather than the corpus average. A pinned value outside the
    bank's range is honoured but warned about: the model was never shown it.

    Without a bank the targets fall back to independent Gaussians around the
    training split's mean and standard deviation, with a warning; those can
    lie outside anything the model saw.
    """
    keys = [d.key for d in profile.descriptors]
    unknown = set(overrides or ()) - set(keys)
    if unknown:
        raise ValueError(
            f"{profile.name} has no descriptor(s) {sorted(unknown)}; expected {keys}",
        )
    if count < 1:
        raise ValueError("count must be positive")
    overrides = dict(overrides or {})
    generator = torch.Generator().manual_seed(42 if seed is None else seed)
    std = model.label_std.detach().cpu()
    bank = getattr(model, "label_bank", None)

    if bank is None or bank.shape[0] == 0:
        warnings.warn(
            "checkpoint has no label bank; descriptor targets are drawn from independent "
            "Gaussians and may lie outside the corpus",
            stacklevel=2,
        )
        mean = model.label_mean.detach().cpu()
        labels = mean + std * torch.randn(count, len(keys), generator=generator)
    else:
        bank = bank.to(torch.float32)
        candidates = torch.arange(bank.shape[0])
        if overrides:
            pinned = [keys.index(key) for key in overrides]
            values = torch.tensor([overrides[key] for key in overrides], dtype=torch.float32)
            low, high = bank[:, pinned].amin(dim=0), bank[:, pinned].amax(dim=0)
            outside = [key for key, value, lo, hi in zip(overrides, values, low, high)
                       if value < lo or value > hi]
            if outside:
                warnings.warn(
                    f"pinned {', '.join(outside)} outside the training range of "
                    f"{profile.name}; the model was never shown such a value",
                    stacklevel=2,
                )
            distance = (((bank[:, pinned] - values) / std[pinned]) ** 2).sum(dim=1)
            neighbours = min(
                bank.shape[0],
                max(BANK_NEIGHBOURS, math.ceil(BANK_NEIGHBOUR_FRACTION * bank.shape[0])),
            )
            candidates = distance.topk(neighbours, largest=False).indices
        rows = candidates[torch.randint(len(candidates), (count,), generator=generator)]
        labels = bank[rows].clone()

    for key, value in overrides.items():
        labels[:, keys.index(key)] = float(value)
    return labels


def sample_hits(
    model,
    labels: torch.Tensor | None,
    *,
    count: int | None = None,
    steps: int = 50,
    guidance: float = 1.0,
    seed: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample waveforms as a (count, samples) float tensor on the CPU."""
    generator = torch.Generator().manual_seed(0 if seed is None else seed)
    audio = v_sample(
        model, labels=labels, batch=count, steps=steps, guidance=guidance,
        generator=generator, device=device,
    )
    return audio.squeeze(1).detach().cpu()


def generate_diffusion(
    count: int = 10,
    instrument: str | None = None,
    checkpoint: str | None = None,
    out_dir: str | None = None,
    steps: int = 50,
    guidance: float = 1.0,
    seed: int | None = None,
    label_seed: int | None = None,
    targets: dict[str, float] | None = None,
    unconditional: bool = False,
    prefix: str = "diff",
) -> list[str]:
    """Sample one-shots from a trained denoiser and report their slider error."""
    import soundfile as sf

    if count < 1:
        raise ValueError("count must be positive")
    profile = get_profile(instrument)
    checkpoint = checkpoint or profile.paths.diffusion_checkpoint
    out_dir = out_dir or profile.paths.output_dir
    if unconditional and targets:
        raise ValueError("unconditional generation ignores descriptor targets")

    device = get_device()
    model, meta = load_diffusion_from_checkpoint(checkpoint, device, profile)
    labels = None if unconditional else draw_labels(model, count, targets, profile, label_seed)

    print(f"Sampling {count} {profile.noun if count == 1 else profile.plural}: "
          f"{steps} steps, guidance {guidance:g}, "
          f"{'unconditional' if unconditional else 'descriptor-conditioned'} "
          f"(epoch {meta.get('epoch', '?')} of run {meta.get('training', {}).get('run_id', '?')})")
    waves = sample_hits(model, labels, count=count, steps=steps,
                        guidance=guidance, seed=seed, device=device)

    os.makedirs(out_dir, exist_ok=True)
    keys = [d.key for d in profile.descriptors]
    scale = model.label_std.detach().cpu().numpy()
    paths = []
    for slot in range(count):
        audio = waves[slot]
        peak = float(audio.abs().max())
        # Only rescale what would clip on write; reporting the raw peak keeps a
        # model that drifts off the corpus's level visible instead of hidden.
        if peak > 1.0:
            audio = audio * (TARGET_PEAK / peak)
        path = os.path.join(out_dir, f"{prefix}_{slot + 1}.wav")
        sf.write(path, audio.numpy().astype(np.float32), SAMPLE_RATE, subtype="PCM_24")
        paths.append(path)

        measured = label_waveform(audio.unsqueeze(0), profile)
        if labels is None:
            print(f"  {path}: peak {peak:.3f}  "
                  + "  ".join(f"{k}={v:.3g}" for k, v in zip(keys, measured)))
        else:
            target = labels[slot].numpy()
            error = np.abs(measured - target) / np.maximum(scale, 1e-6)
            print(f"  {path}: peak {peak:.3f}  mean target error {error.mean():.2f} sd  "
                  + "  ".join(f"{k}={m:.3g}(→{t:.3g})"
                              for k, m, t in zip(keys, measured, target)))
    print(f"Wrote {len(paths)} samples to {os.path.normpath(out_dir)}/. Descriptor error is a control "
          f"proxy, not a quality rating — score the audio with `kicks eval "
          f"--pattern '{prefix}_*.wav'` and listen before drawing conclusions.")
    return paths


__all__ = ["BANK_NEIGHBOURS", "draw_labels", "generate_diffusion", "sample_hits"]
