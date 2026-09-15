"""FastAPI synthesis backend.

The REST API is the project's only interface: the website in ``web/`` is a
client of it and nothing more, so anything the site can do, a script can do too.

Every endpoint takes an optional ``instrument`` parameter. Sliders are addressed
as ``s1..sN`` (position in [0, 1]) or by descriptor key (``sub=0.8``), since the
number and meaning of the sliders follows the instrument.

Accounts and credits are optional (see :mod:`kicks.api.auth`). When enabled,
previews stay free for signed-in users and ``POST /export`` spends one credit
per rendered file — the same audio, but licensed and recorded against the
account.
"""

from __future__ import annotations

import io
import os
import re
import uuid
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ..analysis.basis import slider_positions_to_axis_values
from ..analysis.descriptors import compute_descriptors
from ..analysis.evaluation import analyze_hit, score_sample
from ..audio import effects
from ..audio.constants import SAMPLE_RATE
from ..audio.vocoder import spec_to_audio
from ..instruments import available, get_profile, normalize
from .auth import Auth, InsufficientCredits, User
from .middleware import LRUCache, RateLimiter
from .state import InstrumentState, ServerState

state = ServerState()
auth = Auth()
rate_limiter = RateLimiter(rate=10.0)
audio_cache = LRUCache(max_size=100)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    state.startup()
    print(f"API ready — serving {state.default_instrument} "
          f"({state.vocoder_type}, {state.control_basis} control, "
          f"auth {auth.settings.mode})")
    yield
    if auth.credits is not None:
        await auth.credits.aclose()


app = FastAPI(title="Kicks API", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get("KICKS_CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
    expose_headers=["Content-Disposition", "X-Kicks-Credits-Remaining", "X-Kicks-Export-Id"],
)


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------

def _resolve(request: Request) -> InstrumentState:
    """Load (or fetch) the instrument named in the query string."""
    name = request.query_params.get("instrument")
    if name is None:
        return state.get()
    if normalize(name) not in available():
        raise HTTPException(
            status_code=404,
            detail=f"Unknown instrument {name!r}. Available: {', '.join(available())}",
        )
    try:
        return state.get(name)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Instrument {name!r} is not trained yet: {exc}",
        ) from exc


def _float_param(request: Request, key: str) -> float | None:
    raw = request.query_params.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422, detail=f"Invalid value for {key}: {raw!r}",
        ) from None


def slider_key(name: str) -> str:
    """A slider's label as a query-parameter name: "HF Click" -> "hfclick"."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _slider_positions(request: Request, inst: InstrumentState) -> list[float]:
    """Read slider positions in [0, 1], defaulting to centre.

    Three spellings per slider, checked in order: positional ``s1..sN``, the
    legacy ``pc1..pcN``, and the slider's own label (``click``, ``sizzle``).
    The label is taken from the *fitted* basis rather than the profile's
    descriptor order — under the PCA basis, slider 1 is whichever descriptor
    PC1 turned out to correlate with, which is not knowable in advance.
    """
    positions = []
    for i, label in enumerate(inst.basis.names):
        raw = None
        for name in (f"s{i + 1}", f"pc{i + 1}", slider_key(label)):
            if name in request.query_params:
                raw = _float_param(request, name)
                break
        positions.append(0.5 if raw is None else float(np.clip(raw, 0.0, 1.0)))
    return positions


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def _latent_for(request: Request, inst: InstrumentState) -> torch.Tensor:
    values = slider_positions_to_axis_values(_slider_positions(request, inst), inst.basis)
    if inst.basis.is_descriptor_basis:
        # Closed loop: decode, measure, correct. The decoder's response is
        # nonlinear, so two Newton steps roughly double slider authority versus
        # the open-loop linear map.
        z = inst.basis.basis.solve(
            values,
            measure_fn=lambda z_arr: inst.measure(z_arr, state.device),
            n_iter=2,
        )
    else:
        z = inst.basis.basis.inverse_transform([values])
    return torch.tensor(z, dtype=torch.float32).to(state.device)


def _synthesize(request: Request, inst: InstrumentState) -> tuple[torch.Tensor, torch.Tensor]:
    """Slider settings -> (decoded spectrogram, shaped waveform).

    Shared by /generate and /evaluate so the evaluation scores exactly the audio
    the client receives, shaping included.
    """
    z = _latent_for(request, inst)
    with torch.no_grad():
        spec = inst.model.decode(z)
    waveform = spec_to_audio(spec, state.vocoder, state.device)  # (B, T)

    wf = waveform.squeeze(0)
    attack_ms = _float_param(request, "attack_ms")
    decay_ms = _float_param(request, "decay_ms")
    if attack_ms is not None or decay_ms is not None:
        wf = effects.apply_envelope(wf, attack_ms, decay_ms)
    drive = _float_param(request, "drive")
    if drive is not None:
        wf = effects.apply_drive(wf, drive)
    cutoff = _float_param(request, "filter")
    if cutoff is not None:
        wf = effects.apply_lowpass(wf, cutoff)
    return spec, wf.unsqueeze(0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index() -> dict:
    return {"name": "kicks", "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "device": str(state.device),
        "vocoder": state.vocoder_type,
        "control": state.control_basis,
        "loaded": state.loaded,
        "cached_responses": len(audio_cache),
        "auth": auth.describe(),
    }


@app.get("/me")
async def me(user: User = Depends(auth.require_user)) -> dict:
    """The signed-in caller and their credit balance."""
    balance = await auth.credits.balance(user.id) if auth.credits else None
    return {"id": user.id, "email": user.email, "credits": balance}


@app.get("/instruments")
async def instruments() -> dict:
    """Every registered instrument, whether or not it has a trained model yet."""
    rows = []
    for name in available():
        profile = get_profile(name)
        rows.append({
            "name": profile.name,
            "display_name": profile.display_name,
            "description": profile.description,
            "descriptors": profile.descriptor_keys,
            "trained": os.path.exists(profile.paths.checkpoint),
            "loaded": name in state.loaded,
        })
    return {"instruments": rows, "default": state.default_instrument}


@app.get("/config")
async def config(request: Request) -> dict:
    """Slider definitions for one instrument."""
    inst = _resolve(request)
    profile = inst.profile
    sliders = [
        {
            "id": i + 1,
            "key": slider_key(name),
            "name": name,
            "min": 0,
            "max": 1,
            "default": 0.5,
            "step": 0.01,
        }
        for i, name in enumerate(inst.basis.names)
    ]
    return {
        "instrument": profile.name,
        "display_name": profile.display_name,
        "description": profile.description,
        "sliders": sliders,
        "vocoder": state.vocoder_type,
        "control": state.control_basis,
    }


@app.get("/generate")
async def generate(
    request: Request,
    _rl: None = Depends(rate_limiter),
    _user: User | None = Depends(auth.current_user),
):
    """Render a preview WAV for the given slider settings."""
    cache_key = request.url.query or ""
    if cache_key:
        cached = audio_cache.get(cache_key)
        if cached is not None:
            return StreamingResponse(io.BytesIO(cached), media_type="audio/wav")

    inst = _resolve(request)
    _, waveform = _synthesize(request, inst)

    buf = io.BytesIO()
    sf.write(buf, waveform.squeeze(0).numpy(), SAMPLE_RATE, format="WAV")
    if cache_key:
        audio_cache.put(cache_key, buf.getvalue())
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


@app.get("/evaluate")
async def evaluate(
    request: Request,
    _rl: None = Depends(rate_limiter),
    _user: User | None = Depends(auth.current_user),
) -> dict:
    """Score the hit at the given slider settings.

    Same query parameters as /generate. Returns the corpus-referenced perceptual
    verdicts plus the spectrogram descriptors.
    """
    inst = _resolve(request)
    spec, waveform = _synthesize(request, inst)
    descriptors = compute_descriptors(spec, inst.profile)

    metrics = analyze_hit(waveform.squeeze(0).numpy().astype(np.float32), inst.profile)
    if metrics is None:
        return {"instrument": inst.profile.name,
                "error": "generated audio is silent or unusable",
                "descriptors": descriptors}

    report = score_sample("api", metrics, inst.reference(), inst.profile)
    return {
        "instrument": inst.profile.name,
        "score": report.score,
        "grade": report.grade,
        "likeness_pct": report.likeness_pct,
        "verdicts": [
            {"metric": v.key, "symbol": v.symbol, "text": v.text,
             "percentile": v.percentile, "z": v.z}
            for v in report.verdicts
        ],
        "metrics": metrics,
        "descriptors": descriptors,
    }


@app.get("/spectrogram")
async def spectrogram(
    request: Request,
    _rl: None = Depends(rate_limiter),
    _user: User | None = Depends(auth.current_user),
) -> dict:
    """Raw decoded spectrogram — the model's output before the vocoder."""
    inst = _resolve(request)
    with torch.no_grad():
        spec = inst.model.decode(_latent_for(request, inst))
    return {
        "instrument": inst.profile.name,
        "shape": [inst.model.n_mels, inst.model.n_frames],
        "data": spec.squeeze().cpu().numpy().tolist(),
    }


# ---------------------------------------------------------------------------
# Paid export
# ---------------------------------------------------------------------------

_SAFE_NAME = re.compile(r"[^a-z0-9]+")


def _export_filename(inst: InstrumentState, positions: list[float]) -> str:
    """``kick_sub80_punch50_....wav`` — the settings, readable in a file browser."""
    parts = [
        f"{_SAFE_NAME.sub('', name.lower())}{round(p * 100):02d}"
        for name, p in zip(inst.basis.names, positions)
    ]
    return f"{inst.profile.name}_{'_'.join(parts)}.wav"


@app.post("/export")
async def export(
    request: Request,
    user: User = Depends(auth.require_user),
    _rl: None = Depends(rate_limiter),
):
    """Render and download a licensed WAV. Costs one credit.

    Takes the same query parameters as ``/generate``. Send an ``Idempotency-Key``
    header (any UUID) so a retried request cannot be charged twice: the ledger
    keys the charge on it and returns the same file again.
    """
    if auth.credits is None:
        raise HTTPException(status_code=503, detail="Exports are not enabled on this server")

    inst = _resolve(request)
    positions = _slider_positions(request, inst)
    export_id = request.headers.get("idempotency-key") or str(uuid.uuid4())
    try:
        uuid.UUID(export_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Idempotency-Key must be a UUID") from None

    params = {
        "instrument": inst.profile.name,
        "sliders": dict(zip(inst.basis.names, positions)),
        "control": state.control_basis,
        "vocoder": state.vocoder_type,
        **{k: v for k, v in request.query_params.items()
           if k in ("attack_ms", "decay_ms", "drive", "filter")},
    }
    try:
        remaining, generation_id = await auth.credits.charge_export(
            user.id, export_id, inst.profile.name, params,
        )
    except InsufficientCredits:
        raise HTTPException(
            status_code=402, detail="No credits left. Buy a pack to export samples.",
        ) from None

    try:
        _, waveform = _synthesize(request, inst)
        buf = io.BytesIO()
        sf.write(buf, waveform.squeeze(0).numpy(), SAMPLE_RATE, format="WAV", subtype="PCM_24")
    except Exception:
        await auth.credits.refund_export(export_id)
        raise
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{_export_filename(inst, positions)}"',
            "X-Kicks-Credits-Remaining": str(remaining),
            "X-Kicks-Export-Id": generation_id,
            "Cache-Control": "no-store",
        },
    )
