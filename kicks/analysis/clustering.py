"""Corpus analysis report: GMM clusters over the latent space, PCA over descriptors.

Writes a single JSON blob describing the corpus — cluster membership, descriptor
statistics and correlations, PCA loadings, per-cluster averages — plus one
averaged .wav per cluster so the clusters can be listened to, not just read.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import soundfile as sf
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset

from ..audio import io
from ..audio.constants import AUDIO_LENGTH, SAMPLE_RATE
from ..config import get_device, load_vae_from_checkpoint
from ..data import DrumDataset
from ..instruments import InstrumentProfile, get_profile
from .descriptors import correlation_matrix, descriptor_matrix, descriptor_stats
from .latents import analyze_clusters, extract_latents

#: Descriptor-space components kept for the scatter plots.
N_REPORT_PCS = 3


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def _cluster_average_audio(paths: list[str]) -> np.ndarray:
    """Mean waveform of a cluster's source samples, peak-normalized.

    Averaging the *audio* rather than decoding the mean latent keeps this
    honest — it shows what the cluster contains, not what the model thinks it
    contains. Phase cancellation is the point: only what the members share
    survives.
    """
    total = torch.zeros(1, AUDIO_LENGTH)
    count = 0
    for path in paths:
        audio = io.load_waveform(path, target_lufs=None, length=AUDIO_LENGTH)
        if audio is not None:
            total += audio
            count += 1
    if not count:
        raise ValueError("No readable audio in cluster")
    avg = total / count
    return (avg / (avg.abs().max() + 1e-8)).squeeze(0).numpy()


def run_cluster(
    data: str | None = None,
    instrument: str | None = None,
    n_samples: int | None = None,
    checkpoint: str | None = None,
    out_path: str | None = None,
) -> dict:
    """Cluster the latent space and write the corpus analysis report."""
    profile: InstrumentProfile = get_profile(instrument)
    data = data or profile.paths.data_dir
    checkpoint = checkpoint or profile.paths.checkpoint
    out_path = out_path or profile.paths.cluster_analysis
    keys = profile.descriptor_keys

    dataset = DrumDataset(data, profile)
    total = len(dataset)
    if n_samples and n_samples < total:
        indices = sorted(np.random.default_rng(42).choice(total, n_samples, replace=False).tolist())
        subset: Subset | DrumDataset = Subset(dataset, indices)
    else:
        subset = dataset
    subset_indices = list(getattr(subset, "indices", range(len(subset))))

    dataloader = DataLoader(subset, batch_size=64, shuffle=False)
    print(f"Corpus: {total} {profile.plural} from {data}, clustering {len(subset)}")

    device = get_device()
    print(f"Using device: {device}")
    model, ckpt = load_vae_from_checkpoint(checkpoint, device)
    print(f"Loaded checkpoint {checkpoint} (epoch {ckpt.get('epoch', '?')})")

    print("Extracting latents...")
    latents, spectrograms = extract_latents(model, dataloader, device)
    print(f"Latents: {latents.shape}")

    # GMM on z-scored latents: without scaling, the few high-variance dimensions
    # dominate the covariance and every cluster boundary follows them alone.
    print("Running GMM clustering on normalized latents...")
    clustering = analyze_clusters(latents)
    best_k = clustering["diagnostics"]["selected_k"]
    print(f"BIC selected k={best_k}")
    labels, probs = clustering["labels"], clustering["probabilities"]
    for k in range(best_k):
        print(f"  Cluster {k}: {(labels == k).sum()} samples")

    print("Computing descriptors...")
    desc = descriptor_matrix(spectrograms, profile)

    print("Computing PCA on normalized descriptors...")
    pca = PCA(n_components=min(N_REPORT_PCS, len(keys), len(desc)))
    desc_pca = pca.fit_transform(_zscore(desc))
    print(f"Descriptor PCA variance ratio: {pca.explained_variance_ratio_}")
    print(f"  (cumulative: {pca.explained_variance_ratio_.cumsum()})")

    pc_labels = [f"pc{i + 1}" for i in range(desc_pca.shape[1])]
    pc_desc_corr = correlation_matrix(desc_pca, desc)     # (n_pcs, n_desc)
    desc_desc_corr = correlation_matrix(desc, desc)       # (n_desc, n_desc)

    print("Building output...")
    samples = []
    for i, idx in enumerate(subset_indices):
        path = dataset.paths[idx]
        info = sf.info(path)
        row = {
            "sample_idx": i,
            "filename": os.path.basename(path),
            "original_path": path,
            "cluster": int(labels[i]),
            "descriptors": {k: float(desc[i, j]) for j, k in enumerate(keys)},
            "probs": probs[i].tolist(),
            "duration_ms": float(info.frames / info.samplerate * 1000),
        }
        row.update({name: float(desc_pca[i, j]) for j, name in enumerate(pc_labels)})
        row.update({f"latent{j + 1}": float(clustering["projection"][i, j]) for j in range(3)})
        row["entropy"] = float(clustering["entropy"][i])
        samples.append(row)

    # Name each descriptor-PC after its strongest unclaimed descriptor.
    pc_names = []
    used: set[int] = set()
    for i in range(len(pc_labels)):
        candidates = [(abs(pc_desc_corr[i, j]), j) for j in range(len(keys)) if j not in used]
        best_abs, best_j = max(candidates) if candidates else (0.0, -1)
        if best_j >= 0 and best_abs >= 0.15:
            used.add(best_j)
            pc_names.append({
                "name": profile.label_for(keys[best_j]),
                "descriptor": keys[best_j],
                "correlation": float(pc_desc_corr[i, best_j]),
            })
        else:
            pc_names.append({"name": f"PC{i + 1}", "descriptor": None, "correlation": 0.0})

    loadings = {
        name: {k: float(pca.components_[i, j]) for j, k in enumerate(keys)}
        for i, name in enumerate(pc_labels)
    }
    print("PCA loadings (descriptor weights per PC):")
    for name, row in loadings.items():
        print(f"  {name}: {', '.join(f'{k}={v:+.2f}' for k, v in row.items())}")

    cluster_profiles = {}
    cluster_averages = {}
    cluster_details = {}
    for k in range(best_k):
        mask = labels == k
        if not mask.any():
            continue
        cluster_profiles[str(k)] = {
            "count": int(mask.sum()),
            **{key: float(desc[mask, j].mean()) for j, key in enumerate(keys)},
        }
        cluster_averages[str(k)] = latents[mask].mean(axis=0).tolist()
        members = np.flatnonzero(mask)
        features = clustering["features"][mask]
        representative = members[np.argmin(np.linalg.norm(features - features.mean(axis=0), axis=1))]
        cluster_details[str(k)] = {
            "mean_confidence": float(probs[mask].max(axis=1).mean()),
            "ambiguous_count": int((probs[mask].max(axis=1) < .8).sum()),
            "representative_idx": int(representative),
        }

    print("Generating cluster average audio...")
    samples_dir = profile.paths.samples_dir
    os.makedirs(samples_dir, exist_ok=True)
    for k in range(best_k):
        member_paths = [
            dataset.paths[subset_indices[i]]
            for i in range(len(subset_indices)) if labels[i] == k
        ]
        if not member_paths:
            continue
        avg = _cluster_average_audio(member_paths)
        sf.write(os.path.join(samples_dir, f"cluster_avg_{k}.wav"), avg, SAMPLE_RATE)
        print(f"  Saved cluster {k} average ({len(member_paths)} samples)")

    output = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instrument": profile.name,
        "descriptor_keys": keys,
        "descriptor_labels": profile.descriptor_labels,
        "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
        "pca_source": "descriptors_zscore",
        "n_clusters": best_k,
        "corpus": {
            "sample_rate": SAMPLE_RATE,
            "audio_length_ms": float(AUDIO_LENGTH / SAMPLE_RATE * 1000),
            "n_total": total,
            "data_dir": data,
        },
        "samples": samples,
        "cluster_averages": cluster_averages,
        "pc_names": pc_names,
        "pca_loadings": loadings,
        "pc_descriptor_correlations": {
            name: {k: float(pc_desc_corr[i, j]) for j, k in enumerate(keys)}
            for i, name in enumerate(pc_labels)
        },
        "descriptor_correlations": {
            k1: {k2: float(desc_desc_corr[i, j]) for j, k2 in enumerate(keys)}
            for i, k1 in enumerate(keys)
        },
        "cluster_profiles": cluster_profiles,
        "cluster_details": cluster_details,
        "clustering": clustering["diagnostics"],
        "latent_projection": {"method": "pca", "variance_explained": clustering["projection_variance"]},
        "descriptor_stats": descriptor_stats(desc, profile),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, cls=_NumpyEncoder)

    print(f"\nDone! Saved to {out_path}")
    print(f"  {len(samples)} samples, {best_k} clusters, {len(pc_labels)} PCs "
          "(from z-scored descriptors)")
    return output
