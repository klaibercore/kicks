"""Shared PCA analysis: fit, name PCs, compute decorrelation.

Used by both server.py (FastAPI lifespan) and tui.py (Textual on_mount)
to avoid duplicating ~80 lines of identical PCA/descriptor logic.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA

from .cluster import compute_descriptors
from .config import N_PCS

DESC_KEYS = ["sub", "punch", "click", "bright", "decay"]
DESC_NAME_MAP = {
    "sub": "Sub", "punch": "Punch", "click": "Click",
    "bright": "Bright", "decay": "Decay",
}


@dataclass
class PCAnalysis:
    pca: PCA
    pc_projected: np.ndarray          # (n_samples, N_PCS)
    pc_names: list[str]               # e.g. ["Sub", "Punch", ...]
    pc_mins: list[float]              # 2nd percentile per PC
    pc_maxs: list[float]              # 98th percentile per PC
    decay_idx: int | None = None      # index of the "Decay" PC
    decay_compensation: np.ndarray | None = None  # per-PC ratios


def analyze_latent_space(
    latents: np.ndarray,            # (n_samples, latent_dim)
    spectrograms: np.ndarray,       # (n_samples, 1, 128, 256)
    n_pcs: int = N_PCS,
    verbose: bool = True,
) -> PCAnalysis:
    """Fit PCA, auto-name components from descriptor correlations, compute decorrelation.

    Returns a PCAnalysis dataclass with everything needed for slider mapping.
    """
    descriptors = [compute_descriptors(s) for s in spectrograms]
    desc_arrays = {k: np.array([d[k] for d in descriptors]) for k in DESC_KEYS}

    # Fit PCA
    pca = PCA(n_components=n_pcs)
    pc_projected = pca.fit_transform(latents)

    # Auto-name PCs from highest descriptor correlations and flip negative axes
    pc_names: list[str] = []
    used: set[str] = set()
    for i in range(n_pcs):
        pc_vals = pc_projected[:, i]
        pc_mean, pc_std = pc_vals.mean(), pc_vals.std()
        best_desc, best_corr = None, 0.0
        for dk in DESC_KEYS:
            if dk in used:
                continue
            dv = desc_arrays[dk]
            d_mean, d_std = dv.mean(), dv.std()
            if pc_std > 0 and d_std > 0:
                corr = float(((pc_vals - pc_mean) * (dv - d_mean)).mean() / (pc_std * d_std))
            else:
                corr = 0.0
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_desc = dk
        if best_desc and abs(best_corr) >= 0.15:
            used.add(best_desc)
            pc_names.append(DESC_NAME_MAP.get(best_desc, best_desc.capitalize()))
            if best_corr < 0:
                pca.components_[i] *= -1
                pc_projected[:, i] *= -1
                if verbose:
                    print(f"  PC{i + 1} -> {pc_names[-1]} (r={best_corr:.2f}, flipped)")
            elif verbose:
                print(f"  PC{i + 1} -> {pc_names[-1]} (r={best_corr:.2f})")
        else:
            pc_names.append(f"PC{i + 1}")
            if verbose:
                print(f"  PC{i + 1} -> PC{i + 1} (no strong correlation)")

    pc_mins = [float(np.percentile(pc_projected[:, i], 2)) for i in range(n_pcs)]
    pc_maxs = [float(np.percentile(pc_projected[:, i], 98)) for i in range(n_pcs)]

    # Decay decorrelation
    decay_idx: int | None = None
    decay_compensation: np.ndarray | None = None
    for i, name in enumerate(pc_names):
        if name == "Decay":
            decay_idx = i
            break

    if decay_idx is not None:
        decay_vals = desc_arrays["decay"]
        d_mean, d_std = decay_vals.mean(), decay_vals.std()
        betas = np.zeros(n_pcs)
        for i in range(n_pcs):
            pc_vals = pc_projected[:, i]
            pc_std = pc_vals.std()
            if pc_std > 0 and d_std > 0:
                r = float(np.corrcoef(pc_vals, decay_vals)[0, 1])
                betas[i] = r * d_std / pc_std
        decay_beta = betas[decay_idx]
        if abs(decay_beta) > 1e-8:
            ratios = betas / decay_beta
            ratios[decay_idx] = 0.0
            decay_compensation = ratios
            if verbose:
                print(f"Decay decorrelation enabled (ratios: {ratios.round(3)})")
        elif verbose:
            print("Decay decorrelation skipped (Decay PC has negligible decay correlation)")

    if verbose:
        print(f"PCA variance explained: {pca.explained_variance_ratio_}")

    return PCAnalysis(
        pca=pca,
        pc_projected=pc_projected,
        pc_names=pc_names,
        pc_mins=pc_mins,
        pc_maxs=pc_maxs,
        decay_idx=decay_idx,
        decay_compensation=decay_compensation,
    )
