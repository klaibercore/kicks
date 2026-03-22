"""Cluster analysis command implementation."""

import json
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.decomposition import PCA
from torch.utils.data import Subset

from kicks import KickDataset, KickDataloader
from kicks.cluster import (
    extract_latents,
    select_n_clusters,
    fit_gmm,
    compute_descriptors,
)
from kicks.config import get_device, load_vae_from_checkpoint, BEST_CHECKPOINT
from kicks.model import SAMPLE_RATE, AUDIO_LENGTH


def run_cluster(data: str = "data/kicks", n_samples: int | None = None) -> None:
    """Run GMM clustering and PCA analysis on the latent space."""
    dataset = KickDataset(data)
    total = len(dataset)
    n = min(n_samples, total) if n_samples else total

    if n_samples and n_samples < total:
        indices = np.random.choice(total, n, replace=False).tolist()
        subset = Subset(dataset, indices)
    else:
        subset = dataset

    dataloader = KickDataloader(subset, batch_size=64, shuffle=False)
    print(f"Dataset: {total} samples from {data}, clustering {len(subset)}")

    device = get_device()
    print(f"Using device: {device}")

    model, checkpoint = load_vae_from_checkpoint(BEST_CHECKPOINT, device)
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']})")

    print("Extracting latents...")
    latents, spectrograms = extract_latents(model, dataloader, device)
    print(f"Latents: {latents.shape}")

    # Z-score normalize latents before GMM
    latent_mean = latents.mean(axis=0)
    latent_std = latents.std(axis=0) + 1e-8
    latents_normed = (latents - latent_mean) / latent_std
    print(f"Normalized latents: mean≈{latents_normed.mean():.4f}, std≈{latents_normed.std():.4f}")

    print("Running GMM clustering on normalized latents...")
    best_k, _ = select_n_clusters(latents_normed, max_k=10)
    print(f"BIC selected k={best_k}")

    gmm, cluster_labels, cluster_probs = fit_gmm(latents_normed, best_k)
    for k in range(best_k):
        count = (cluster_labels == k).sum()
        print(f"  Cluster {k}: {count} samples")

    print("Computing descriptors...")
    descriptors = [compute_descriptors(s) for s in spectrograms]
    desc_keys = ["sub", "punch", "click", "bright", "decay"]

    # PCA on z-scored descriptors (5D -> 3D)
    print("Computing PCA on normalized descriptors...")
    desc_matrix = np.array([[d[k] for k in desc_keys] for d in descriptors])
    desc_mean = desc_matrix.mean(axis=0)
    desc_std = desc_matrix.std(axis=0) + 1e-8
    desc_normed = (desc_matrix - desc_mean) / desc_std
    pca = PCA(n_components=3)
    desc_pca = pca.fit_transform(desc_normed)
    print(f"Descriptor PCA variance ratio: {pca.explained_variance_ratio_}")
    print(f"  (cumulative: {pca.explained_variance_ratio_.cumsum()})")

    subset_indices = list(subset.indices) if hasattr(subset, "indices") else list(range(len(subset)))

    print("Building output...")
    audio_data = []
    for i, idx in enumerate(subset_indices):
        sample_path = dataset.paths[idx]
        filename = os.path.basename(sample_path)
        info = sf.info(sample_path)
        duration_ms = float(info.frames / info.samplerate * 1000)
        audio_data.append({
            "sample_idx": i,
            "filename": filename,
            "original_path": sample_path,
            "cluster": int(cluster_labels[i]),
            "pc1": float(desc_pca[i, 0]),
            "pc2": float(desc_pca[i, 1]),
            "pc3": float(desc_pca[i, 2]),
            "descriptors": descriptors[i],
            "probs": cluster_probs[i].tolist(),
            "duration_ms": duration_ms,
        })

    print("Computing analytics...")
    desc_arrays = {k: np.array([d[k] for d in descriptors]) for k in desc_keys}

    # PC <-> descriptor Pearson correlations
    pc_descriptor_correlations = {}
    for pc_idx, pc_name in enumerate(["pc1", "pc2", "pc3"]):
        pc_vals = desc_pca[:, pc_idx]
        pc_mean, pc_std = pc_vals.mean(), pc_vals.std()
        corrs = {}
        for dk in desc_keys:
            dv = desc_arrays[dk]
            d_mean, d_std = dv.mean(), dv.std()
            if pc_std > 0 and d_std > 0:
                corrs[dk] = float(((pc_vals - pc_mean) * (dv - d_mean)).mean() / (pc_std * d_std))
            else:
                corrs[dk] = 0.0
        pc_descriptor_correlations[pc_name] = corrs

    # Global descriptor statistics
    descriptor_stats = {}
    for dk in desc_keys:
        dv = desc_arrays[dk]
        descriptor_stats[dk] = {
            "mean": float(dv.mean()), "std": float(dv.std()),
            "min": float(dv.min()), "max": float(dv.max()),
        }

    # Descriptor-descriptor Pearson correlations (5x5)
    descriptor_correlations = {}
    for dk1 in desc_keys:
        d1 = desc_arrays[dk1]
        d1_mean, d1_std = d1.mean(), d1.std()
        corrs = {}
        for dk2 in desc_keys:
            d2 = desc_arrays[dk2]
            d2_mean, d2_std = d2.mean(), d2.std()
            if d1_std > 0 and d2_std > 0:
                corrs[dk2] = float(((d1 - d1_mean) * (d2 - d2_mean)).mean() / (d1_std * d2_std))
            else:
                corrs[dk2] = 0.0
        descriptor_correlations[dk1] = corrs

    # PCA loadings
    pca_loadings = {}
    for pc_idx, pc_name in enumerate(["pc1", "pc2", "pc3"]):
        pca_loadings[pc_name] = {dk: float(pca.components_[pc_idx, di]) for di, dk in enumerate(desc_keys)}
    print("PCA loadings (descriptor weights per PC):")
    for pc_name, loadings in pca_loadings.items():
        parts = [f"{dk}={v:+.2f}" for dk, v in loadings.items()]
        print(f"  {pc_name}: {', '.join(parts)}")

    # Derive PC names from highest absolute correlations
    desc_name_map = {"sub": "Sub", "punch": "Punch", "click": "Click", "bright": "Bright", "decay": "Decay"}
    pc_names = []
    used: set[str] = set()
    for pc_name in ["pc1", "pc2", "pc3"]:
        corrs = pc_descriptor_correlations[pc_name]
        best_desc, best_corr = None, 0.0
        for dk, c in corrs.items():
            if dk not in used and abs(c) > abs(best_corr):
                best_desc, best_corr = dk, c
        if best_desc and abs(best_corr) >= 0.15:
            used.add(best_desc)
            pc_names.append({"name": desc_name_map.get(best_desc, best_desc.capitalize()),
                             "descriptor": best_desc, "correlation": float(best_corr)})
        else:
            pc_names.append({"name": f"PC{len(pc_names) + 1}", "descriptor": None, "correlation": 0.0})

    # Per-cluster descriptor profiles
    cluster_profiles = {}
    for k in range(best_k):
        mask = cluster_labels == k
        profile = {"count": int(mask.sum())}
        for dk in desc_keys:
            profile[dk] = float(desc_arrays[dk][mask].mean())
        cluster_profiles[str(k)] = profile

    print("Computing cluster averages...")
    cluster_averages = {}
    for k in range(best_k):
        mask = cluster_labels == k
        avg_latent = latents[mask].mean(axis=0)
        cluster_averages[k] = avg_latent.tolist()

    print("Generating cluster average audio (averaging original samples)...")
    os.makedirs("output/samples", exist_ok=True)

    for k in range(best_k):
        cluster_indices = [i for i, c in enumerate(cluster_labels) if c == k]
        if not cluster_indices:
            continue

        audios = []
        for idx in cluster_indices:
            sample_path = dataset.paths[subset_indices[idx]]
            data, sr = sf.read(sample_path, dtype="float32")
            if data.ndim == 1:
                audio = torch.from_numpy(data).unsqueeze(0)
            else:
                audio = torch.from_numpy(data.T)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                audio = resampler(audio)
            length = audio.shape[-1]
            if length > AUDIO_LENGTH:
                audio = audio[:, :AUDIO_LENGTH]
            elif length < AUDIO_LENGTH:
                audio = torch.nn.functional.pad(audio, (0, AUDIO_LENGTH - length))
            audios.append(audio)

        avg_audio = torch.stack(audios).mean(dim=0)
        avg_audio = avg_audio / (avg_audio.abs().max() + 1e-8)
        sf.write(f"output/samples/cluster_avg_{k}.wav", avg_audio.squeeze(0).numpy(), SAMPLE_RATE)
        print(f"  Saved cluster {k} average ({len(audios)} samples)")

    output = {
        "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
        "pca_source": "descriptors_zscore",
        "n_clusters": best_k,
        "corpus": {
            "sample_rate": SAMPLE_RATE,
            "audio_length_ms": float(AUDIO_LENGTH / SAMPLE_RATE * 1000),
            "n_total": total,
            "data_dir": data,
        },
        "samples": audio_data,
        "cluster_averages": {str(k): v for k, v in cluster_averages.items()},
        "pc_names": pc_names,
        "pca_loadings": pca_loadings,
        "pc_descriptor_correlations": pc_descriptor_correlations,
        "descriptor_correlations": descriptor_correlations,
        "cluster_profiles": cluster_profiles,
        "descriptor_stats": descriptor_stats,
    }

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            return super().default(o)

    with open("output/cluster_analysis.json", "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)

    print(f"\nDone! Saved to output/cluster_analysis.json")
    print(f"  {len(audio_data)} samples, {best_k} clusters, 3 PCs (from z-scored descriptors)")
    print(f"  GMM on z-scored latents, PCA on z-scored descriptors")
