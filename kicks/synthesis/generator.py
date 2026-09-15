"""Generation with on-manifold latent sampling and eval-guided selection.

Two corrections over naive ``z ~ N(0, 1)`` decoding:

1. Standard-normal latents land off the data manifold — that is where the blobs
   and multi-onset generations come from. A GMM is fitted to the aggregate
   posterior (the corpus's own latent mu vectors) and z is drawn from that
   instead. The fit is cached per instrument.
2. Best-of-k: each output slot decodes several candidates, scores them with the
   perceptual metrics against the corpus reference, and keeps the winner.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from ..analysis.evaluation import analyze_hit, build_reference, score_sample
from ..audio.constants import SAMPLE_RATE
from ..audio.vocoder import load_vocoder, spec_to_audio
from ..config import get_device, load_vae_from_checkpoint
from ..instruments import InstrumentProfile, get_profile


def fit_latent_prior(
    model,
    device: torch.device,
    data_dir: str,
    cache_path: str,
    n_fit: int = 512,
    n_components: int = 8,
    refresh: bool = False,
):
    """Fit (or load) a GMM over corpus latent mu vectors.

    Returns a fitted ``GaussianMixture`` whose ``sample()`` draws from the
    aggregate posterior rather than the standard-normal prior.
    """
    from sklearn.mixture import GaussianMixture

    if not refresh and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached["means"].shape[1] == model.latent_dim:
            gmm = GaussianMixture(
                n_components=len(cached["weights"]), covariance_type="full",
            )
            gmm.weights_ = cached["weights"]
            gmm.means_ = cached["means"]
            gmm.covariances_ = cached["covariances"]
            gmm.precisions_cholesky_ = cached["precisions_cholesky"]
            return gmm

    from ..audio import io
    from ..data import load_spectrogram

    files = io.list_wavs(data_dir)
    if len(files) > n_fit:
        files = sorted(random.Random(42).sample(files, n_fit))

    print(f"Encoding {len(files)} corpus samples to fit the latent prior...")
    meter = io.make_meter()
    latents: list[np.ndarray] = []
    batch: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for i, name in enumerate(files):
            spec = load_spectrogram(os.path.join(data_dir, name), meter, model.n_frames)
            if spec is not None:
                batch.append(spec)
            if len(batch) == 64 or (i == len(files) - 1 and batch):
                mu, _ = model.encode(torch.stack(batch).to(device))
                latents.append(mu.cpu().numpy())
                batch = []
    X = np.concatenate(latents).astype(np.float64)

    n_components = min(n_components, max(1, len(X) // 20))
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full",
        random_state=42, n_init=3, reg_covar=1e-4,
    )
    gmm.fit(X)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(
        cache_path,
        weights=gmm.weights_, means=gmm.means_,
        covariances=gmm.covariances_,
        precisions_cholesky=gmm.precisions_cholesky_,
    )
    print(f"Latent prior: {n_components}-component GMM on {len(X)} latents "
          f"(cached to {cache_path})")
    return gmm


def generate(
    count: int = 10,
    best_of: int = 4,
    instrument: str | None = None,
    data_dir: str | None = None,
    out_dir: str | None = None,
    checkpoint: str | None = None,
    seed: int | None = None,
    vocoder_type: str = "bigvgan",
    refresh_prior: bool = False,
) -> list[str]:
    """Generate one-shots: GMM-prior sampling plus best-of-k perceptual selection."""
    import soundfile as sf

    profile: InstrumentProfile = get_profile(instrument)
    data_dir = data_dir or profile.paths.data_dir
    out_dir = out_dir or profile.paths.output_dir
    checkpoint = checkpoint or profile.paths.checkpoint

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = get_device()
    model, _ = load_vae_from_checkpoint(checkpoint, device)
    gmm = fit_latent_prior(
        model, device, data_dir, profile.paths.latent_prior, refresh=refresh_prior,
    )
    vocoder = load_vocoder(device, vocoder_type, profile.paths.vocoder_dir)
    ref = build_reference(data_dir, profile)

    n_candidates = count * max(1, best_of)
    z, _ = gmm.sample(n_candidates)
    # gmm.sample() returns candidates grouped by component; shuffling stops every
    # output slot from drawing its whole group from one cluster.
    z = torch.from_numpy(np.random.default_rng(seed).permutation(z)).float()

    print(f"Decoding {n_candidates} candidates ({count} slots × best-of-{best_of})...")
    waves = []
    with torch.no_grad():
        for i in range(0, n_candidates, 8):
            spec = model.decode(z[i: i + 8].to(device))
            waves.append(spec_to_audio(spec, vocoder, device).cpu())
    waves = torch.cat(waves).numpy()

    scored = []
    for i in range(n_candidates):
        m = analyze_hit(waves[i].astype(np.float32), profile)
        s = score_sample(f"candidate_{i}", m, ref, profile).score if m is not None else 0.0
        scored.append((s, i))

    # Each slot picks the best of its *own* k candidates, so the output keeps the
    # diversity of `count` independent draws instead of collapsing onto the k
    # highest-scoring latents overall.
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for slot in range(count):
        group = scored[slot * best_of: (slot + 1) * best_of]
        best_score, best_i = max(group)
        path = os.path.join(out_dir, f"gen_{slot + 1}.wav")
        sf.write(path, waves[best_i], SAMPLE_RATE)
        rest = ", ".join(f"{s:.0f}" for s, _ in sorted(group, reverse=True)[1:])
        print(f"  {path}: score {best_score:.0f} (rejected: {rest})")
        paths.append(path)
    return paths
