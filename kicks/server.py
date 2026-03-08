"""FastAPI backend for the kick drum synthesizer."""

import io
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sklearn.decomposition import PCA

from kicks import KickDataset, KickDataloader, VAE
from kicks.cluster import extract_latents, compute_descriptors
from kicks.config import get_device, load_vae_from_checkpoint, DATA_DIR, BEST_CHECKPOINT, N_PCS
from kicks.model import SAMPLE_RATE
from kicks.vocoder import load_vocoder, spec_to_audio


class _State:
    """Module-level state populated during lifespan."""

    device: torch.device
    dataset: KickDataset
    model: VAE
    vocoder: object
    pca: PCA
    pc_projected: np.ndarray
    pc_names: list[str]
    pc_mins: list[float]
    pc_maxs: list[float]


_state = _State()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load model, vocoder, and PCA on startup."""
    data_dir = os.environ.get("KICKS_DATA_DIR", DATA_DIR)

    _state.device = get_device()
    _state.dataset = KickDataset(data_dir)
    dataloader = KickDataloader(_state.dataset, batch_size=32, shuffle=False)

    _state.model, _ = load_vae_from_checkpoint(BEST_CHECKPOINT, _state.device)

    _state.vocoder = load_vocoder(_state.device)

    latents, spectrograms = extract_latents(_state.model, dataloader, _state.device)

    # Compute perceptual descriptors for PC naming
    print("Computing descriptors for PC naming...")
    descriptors = [compute_descriptors(s) for s in spectrograms]
    desc_keys = ["sub", "punch", "click", "bright", "decay"]
    desc_name_map = {
        "sub": "Sub", "punch": "Punch", "click": "Click",
        "bright": "Bright", "decay": "Decay",
    }
    desc_arrays = {k: np.array([d[k] for d in descriptors]) for k in desc_keys}

    # Fit PCA
    _state.pca = PCA(n_components=N_PCS)
    _state.pc_projected = _state.pca.fit_transform(latents)

    # Auto-name PCs from highest descriptor correlations and flip negative axes
    _state.pc_names = []
    used: set[str] = set()
    for i in range(N_PCS):
        pc_vals = _state.pc_projected[:, i]
        pc_mean, pc_std = pc_vals.mean(), pc_vals.std()
        best_desc, best_corr = None, 0.0
        for dk in desc_keys:
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
            _state.pc_names.append(desc_name_map.get(best_desc, best_desc.capitalize()))
            if best_corr < 0:
                _state.pca.components_[i] *= -1
                _state.pc_projected[:, i] *= -1
                print(f"  PC{i + 1} -> {_state.pc_names[-1]} (r={best_corr:.2f}, flipped)")
            else:
                print(f"  PC{i + 1} -> {_state.pc_names[-1]} (r={best_corr:.2f})")
        else:
            _state.pc_names.append(f"PC{i + 1}")
            print(f"  PC{i + 1} -> PC{i + 1} (no strong correlation)")

    _state.pc_mins = [float(np.percentile(_state.pc_projected[:, i], 2)) for i in range(N_PCS)]
    _state.pc_maxs = [float(np.percentile(_state.pc_projected[:, i], 98)) for i in range(N_PCS)]

    print(f"PCA variance explained: {_state.pca.explained_variance_ratio_}")
    for i in range(N_PCS):
        print(f"{_state.pc_names[i]} range: [{_state.pc_mins[i]:.3f}, {_state.pc_maxs[i]:.3f}]")
    print("API ready")

    yield


app = FastAPI(title="Kicks API", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/config")
async def config():
    sliders = []
    for i in range(N_PCS):
        sliders.append({
            "id": i + 1,
            "name": _state.pc_names[i],
            "min": 0,
            "max": 1,
            "default": 0.5,
            "step": 0.01,
        })
    return {"sliders": sliders}


@app.get("/generate")
async def generate(
    pc1: float = Query(0.5),
    pc2: float = Query(0.5),
    pc3: float = Query(0.5),
    pc4: float = Query(0.5),
):
    pc_values = []
    for i, raw in enumerate([pc1, pc2, pc3, pc4]):
        val = _state.pc_mins[i] + raw * (_state.pc_maxs[i] - _state.pc_mins[i])
        val = max(_state.pc_mins[i], min(_state.pc_maxs[i], val))
        pc_values.append(val)

    z_np = _state.pca.inverse_transform([pc_values])
    z = torch.tensor(z_np, dtype=torch.float32).to(_state.device)

    with torch.no_grad():
        spec = _state.model.decode(z)
        waveform = spec_to_audio(spec, _state.dataset, _state.vocoder, _state.device)

    buf = io.BytesIO()
    sf.write(buf, waveform.squeeze(0).numpy(), SAMPLE_RATE, format="WAV")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")
