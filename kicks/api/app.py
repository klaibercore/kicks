"""FastAPI synthesis backend.

The REST API is the project's only interface: the website in ``web/`` is a
client of it and nothing more, so anything the site can do, a script can do too.

Every endpoint takes an optional ``instrument`` parameter. Sliders are addressed
as ``s1..sN`` (position in [0, 1]) or by descriptor key (``sub=0.8``), since the
number and meaning of the sliders follows the instrument.

The synthesis endpoints also take ``backend``: ``dsp``, ``vae`` or ``diffusion``.
The default follows the server's explicit selection or automatic availability.
Diffusion needs a configured waveform checkpoint and takes ``steps`` (sampler steps,
10-200, default 50) and run off the event loop, one at a time.

This local research API has no accounts or billing. ``POST /export`` writes
the same audio as the preview as a 24-bit WAV.
"""

from __future__ import annotations

import io
import math
import os
import re
from hashlib import sha256
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from ..analysis.basis import slider_positions_to_axis_values
from ..analysis.descriptors import compute_descriptors
from ..analysis.evaluation import analyze_hit, score_sample
from ..audio import effects
from ..audio.constants import N_FRAMES, SAMPLE_RATE
from ..audio.mel import spectrogram as waveform_spectrogram
from ..data.waveforms import label_waveform, peak_normalize
from ..instruments import available, get_profile, normalize
from ..synthesis.controlled import control_latent, render_controls
from ..synthesis.diffusion import render_hit, slider_range
from ..synthesis.identity import IdentityBasis
from .middleware import LRUCache, RateLimiter
from .state import DiffusionState, InstrumentState, ProceduralState, ServerState
from ..synthesis.procedural import RENDERER_VERSION, render as render_dsp

state = ServerState()
rate_limiter = RateLimiter(rate=10.0)
audio_cache = LRUCache(max_size=100)
render_cache: LRUCache[tuple[torch.Tensor, torch.Tensor]] = LRUCache(max_size=32)

BACKENDS = ("vae", "diffusion", "dsp")
DIFFUSION_STEPS = (10, 200)
DEFAULT_DIFFUSION_STEPS = 50
EFFECTS = ("attack_ms", "decay_ms", "drive", "filter")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    state.startup()
    print(f"API ready — serving {state.default_instrument} "
          f"({state.vocoder_type}, {state.control_basis} control, "
          f"{state.default_backend})")
    yield


app = FastAPI(title="Kicks API", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get("KICKS_CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    expose_headers=["Content-Disposition", "X-Kicks-Export-Id"],
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


def _backend(request: Request) -> str:
    backend = request.query_params.get("backend", state.default_backend)
    if backend not in BACKENDS:
        raise HTTPException(status_code=422, detail=f"backend must be one of {', '.join(BACKENDS)}")
    return backend


def _resolve_diffusion(request: Request) -> DiffusionState:
    """The diffusion checkpoint served for the instrument named in the query string."""
    name = request.query_params.get("instrument") or state.default_instrument
    if normalize(name) not in available():
        raise HTTPException(
            status_code=404,
            detail=f"Unknown instrument {name!r}. Available: {', '.join(available())}",
        )
    try:
        return state.get_diffusion(name)
    except KeyError:
        raise HTTPException(
            status_code=503,
            detail=f"No diffusion checkpoint is served for {name!r}. "
                   f"Start the API with `kicks serve --diffusion {normalize(name)}=PATH`.",
        ) from None


def _steps(request: Request) -> int:
    raw = request.query_params.get("steps", str(DEFAULT_DIFFUSION_STEPS))
    low, high = DIFFUSION_STEPS
    if not re.fullmatch(r"[0-9]{1,3}", raw) or not low <= int(raw) <= high:
        raise HTTPException(status_code=422, detail=f"steps must be an integer in [{low}, {high}]")
    return int(raw)


def _float_param(request: Request, key: str) -> float | None:
    raw = request.query_params.get(key)
    if raw is None:
        return None
    try:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("non-finite parameter")
        return value
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422, detail=f"Invalid value for {key}: {raw!r}",
        ) from None


def slider_key(name: str) -> str:
    """A slider's label as a query-parameter name: "HF Click" -> "hfclick"."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _texture_seed(request: Request) -> int:
    raw = request.query_params.get("seed", "0")
    if not re.fullmatch(r"[0-9]{1,10}", raw) or int(raw) > 2**32 - 1:
        raise HTTPException(status_code=422, detail="seed must be an integer in [0, 4294967295]")
    return int(raw)


def _slider_positions(request: Request, inst: InstrumentState | DiffusionState | ProceduralState) -> list[float]:
    """Read slider positions in [0, 1], defaulting to the configured values.

    Three spellings per slider, checked in order: positional ``s1..sN``, the
    legacy ``pc1..pcN``, and the slider's own label (``click``, ``sizzle``).
    For the VAE the label is taken from the *fitted* basis rather than the
    profile's descriptor order — under the PCA basis, slider 1 is whichever
    descriptor PC1 turned out to correlate with, which is not knowable in
    advance. Diffusion sliders are the checkpoint's descriptors, in order.
    """
    names = inst.names if isinstance(inst, (DiffusionState, ProceduralState)) else inst.basis.names
    positions = []
    for i, label in enumerate(names):
        raw = None
        for name in (f"s{i + 1}", f"pc{i + 1}", inst.keys[i] if isinstance(inst, ProceduralState) else slider_key(label)):
            if name in request.query_params:
                raw = _float_param(request, name)
                break
        default = 0.5
        if isinstance(inst, ProceduralState):
            parameter = inst.profile.procedural.parameters[i]
            default = parameter.position(parameter.default)
        positions.append(default if raw is None else float(np.clip(raw, 0.0, 1.0)))
    return positions


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def _latent_for(request: Request, inst: InstrumentState) -> torch.Tensor:
    z = control_latent(inst.basis, inst.response, _slider_positions(request, inst),
                       seed=_texture_seed(request))
    return torch.tensor(z, dtype=torch.float32).to(state.device)


def _synthesize(request: Request, inst: InstrumentState) -> tuple[torch.Tensor, torch.Tensor]:
    """Slider settings -> (decoded spectrogram, shaped waveform).

    Shared by /generate and /evaluate so the evaluation scores exactly the audio
    the client receives, shaping included.
    """
    positions = _slider_positions(request, inst)
    seed = _texture_seed(request)
    shaping = _shaping(request)
    vocoder = state.vocoder_for(inst.profile)
    key = repr((id(inst.model), id(inst.basis), id(vocoder), positions, seed, shaping))
    cached = render_cache.get(key)
    if cached is not None:
        return cached
    spec, waveform = render_controls(inst.model, inst.basis, inst.profile, inst.response,
                                     vocoder, state.device, positions, seed=seed)
    result = spec.cpu(), _apply_shaping(waveform, shaping)
    render_cache.put(key, result)
    return result


def _shaping(request: Request) -> tuple[float | None, ...]:
    return tuple(_float_param(request, key) for key in EFFECTS)


def _apply_shaping(waveform: torch.Tensor, shaping: tuple[float | None, ...]) -> torch.Tensor:
    """Apply the user's effects to a (1, T) waveform."""
    attack_ms, decay_ms, drive, cutoff = shaping
    wf = waveform.squeeze(0)
    if attack_ms is not None or decay_ms is not None:
        wf = effects.apply_envelope(wf, attack_ms, decay_ms)
    if drive is not None:
        wf = effects.apply_drive(wf, drive)
    if cutoff is not None:
        wf = effects.apply_lowpass(wf, cutoff)
    return wf.unsqueeze(0)


async def _synthesize_diffusion(
    request: Request, diff: DiffusionState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slider settings -> (shaped waveform, descriptor targets) from the denoiser.

    The unshaped render is cached apart from the effects, so moving an effect
    reshapes the last hit instead of walking the sampler again. Walks queue one
    at a time; a request whose client left while it waited (the studio aborts a
    superseded preview) is dropped instead of rendered for nobody.
    """
    positions = _slider_positions(request, diff)
    seed = _texture_seed(request)
    steps = _steps(request)
    shaping = _shaping(request)
    key = repr(("diffusion", id(diff.model), positions, seed, steps))
    rendered = render_cache.get(key)
    if rendered is None:
        async with diff.lock:
            rendered = render_cache.get(key)  # rendered while this request waited
            if rendered is None:
                if await request.is_disconnected():
                    raise HTTPException(status_code=499, detail="client closed the request")
                rendered = await run_in_threadpool(
                    render_hit, diff.model, positions, seed=seed, steps=steps, device=state.device,
                )
                render_cache.put(key, rendered)
    waveform, targets = rendered
    return _apply_shaping(waveform, shaping), targets


def _resolve_dsp(request: Request) -> ProceduralState:
    name = request.query_params.get("instrument") or state.default_instrument
    try:
        return state.get_dsp(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No procedural voice for {name!r}") from None


async def _synthesize_dsp(request: Request, inst: ProceduralState) -> torch.Tensor:
    positions = _slider_positions(request, inst)
    seed = _texture_seed(request)
    parameters = inst.profile.procedural.from_positions(positions)
    key = repr(("dsp", RENDERER_VERSION, inst.profile.procedural, positions, seed))
    cached = render_cache.get(key)
    if cached is None:
        audio = await run_in_threadpool(render_dsp, inst.profile, parameters, seed=seed)
        cached = (torch.from_numpy(audio)[None], torch.empty(0))
        render_cache.put(key, cached)
    return _apply_shaping(cached[0], _shaping(request))


def _dsp_config(inst: ProceduralState) -> dict:
    voice = inst.profile.procedural
    return {
        "instrument": inst.profile.name,
        "display_name": inst.profile.display_name,
        "description": "Shape a synthesized drum with physical controls.",
        "backend": "dsp", "vocoder": "none", "control": "physical",
        "calibration": None, "variation": True, "renderer_version": RENDERER_VERSION,
        "choke_group": voice.choke_group,
        "sliders": [{"id": i + 1, "key": inst.keys[i], "name": p.label,
                     "min": 0, "max": 1, "default": p.position(p.default), "step": .001,
                     "description": p.description, "unit": p.unit,
                     "physical_min": p.low, "physical_max": p.high, "logarithmic": p.logarithmic}
                    for i, p in enumerate(voice.parameters)],
        "presets": [{"name": name, "sliders": {
            inst.keys[i]: p.position(values[p.key]) for i, p in enumerate(voice.parameters)}}
            for name in voice.presets for values in [voice.resolve(preset=name)]],
    }


async def _evaluate_dsp(request: Request) -> dict:
    inst = _resolve_dsp(request)
    waveform = await _synthesize_dsp(request, inst)
    descriptors = compute_descriptors(waveform_spectrogram(waveform), inst.profile)
    metrics = analyze_hit(waveform[0].numpy(), inst.profile)
    body = {"instrument": inst.profile.name, "backend": "dsp", "descriptors": descriptors,
            "metrics": metrics, "control_fit": None, "verdicts": [],
            "parameters": inst.profile.procedural.from_positions(_slider_positions(request, inst)),
            "renderer_version": RENDERER_VERSION, "corpus_available": False}
    if metrics is None:
        return {**body, "error": "generated audio is silent or unusable"}
    if os.path.isdir(inst.profile.paths.data_dir) and any(
        name.lower().endswith('.wav') for name in os.listdir(inst.profile.paths.data_dir)
    ):
        from ..analysis.evaluation import build_reference
        reference = await run_in_threadpool(build_reference, inst.profile.paths.data_dir, inst.profile)
        report = score_sample("api", metrics, reference, inst.profile)
        body.update({"score": report.score, "grade": report.grade, "likeness_pct": report.likeness_pct,
                     "corpus_available": True,
                     "verdicts": [{"metric": v.key, "symbol": v.symbol, "text": v.text,
                                   "percentile": v.percentile, "z": v.z} for v in report.verdicts]})
    return body


async def _render(request: Request) -> tuple[InstrumentState | DiffusionState | ProceduralState, torch.Tensor]:
    """The instrument state and shaped waveform for either backend."""
    if _backend(request) == "dsp":
        inst = _resolve_dsp(request)
        return inst, await _synthesize_dsp(request, inst)
    if _backend(request) == "diffusion":
        diff = _resolve_diffusion(request)
        waveform, _ = await _synthesize_diffusion(request, diff)
        return diff, waveform
    inst = _resolve(request)
    return inst, _synthesize(request, inst)[1]


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
        "vocoders": state.vocoders,
        "control": state.control_basis,
        "loaded": state.loaded,
        "diffusion": sorted(state.diffusion_checkpoints),
        "cached_responses": len(audio_cache),
        "default_backend": state.default_backend,
        "dsp": [n for n in available() if get_profile(n).procedural is not None],
    }


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
            "choke_group": profile.procedural.choke_group if profile.procedural else None,
            "trained": os.path.exists(profile.paths.checkpoint),
            "loaded": name in state.loaded,
            "backends": sorted([
                backend for backend, served in (
                    ("vae", os.path.exists(profile.paths.checkpoint)),
                    ("diffusion", profile.name in state.diffusion_checkpoints),
                    ("dsp", profile.procedural is not None),
                ) if served
            ], key=lambda b: b != state.default_backend),
        })
    return {"instruments": rows, "default": state.default_instrument}


@app.get("/config")
async def config(request: Request) -> dict:
    """Slider definitions for one instrument and backend."""
    if _backend(request) == "dsp":
        return _dsp_config(_resolve_dsp(request))
    if _backend(request) == "diffusion":
        return _diffusion_config(_resolve_diffusion(request))
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
            "description": next((d.doc for d in profile.descriptors if d.label == name), "Latent variation"),
            "target_min": inst.basis.mins[i],
            "target_max": inst.basis.maxs[i],
        }
        for i, name in enumerate(inst.basis.names)
    ]
    return {
        "instrument": profile.name,
        "display_name": profile.display_name,
        "description": profile.description,
        "sliders": sliders,
        "backend": "vae",
        "choke_group": profile.procedural.choke_group if profile.procedural else None,
        "vocoder": state.vocoder_type_for(profile),
        "control": state.control_basis,
        "calibration": inst.basis.calibration,
        "variation": isinstance(inst.basis.basis, IdentityBasis),
    }


def _diffusion_config(diff: DiffusionState) -> dict:
    profile = diff.profile
    low, high = slider_range(diff.model)
    training = diff.meta.get("training") or {}
    return {
        "instrument": profile.name,
        "display_name": profile.display_name,
        "description": profile.description,
        "sliders": [
            {
                "id": i + 1,
                "key": slider_key(d.label),
                "name": d.label,
                "min": 0,
                "max": 1,
                "default": 0.5,
                "step": 0.01,
                "description": d.doc or d.label,
                # The corpus's 1st and 99th percentiles; the centre is its median.
                "target_min": float(low[i]),
                "target_max": float(high[i]),
            }
            for i, d in enumerate(profile.descriptors)
        ],
        "backend": "diffusion",
        "choke_group": profile.procedural.choke_group if profile.procedural else None,
        "vocoder": "none",
        "control": "descriptor",
        "calibration": None,
        "variation": True,
        "steps": {"default": DEFAULT_DIFFUSION_STEPS, "min": DIFFUSION_STEPS[0], "max": DIFFUSION_STEPS[1]},
        "checkpoint": {
            "name": os.path.basename(diff.checkpoint),
            "epoch": diff.meta.get("epoch"),
            "run_id": training.get("run_id"),
        },
    }


@app.get("/generate")
async def generate(
    request: Request,
    _rl: None = Depends(rate_limiter),
):
    """Render a preview WAV for the given slider settings."""
    cache_key = request.url.query or ""
    if cache_key:
        cached = audio_cache.get(cache_key)
        if cached is not None:
            return StreamingResponse(io.BytesIO(cached), media_type="audio/wav")

    _, waveform = await _render(request)

    buf = io.BytesIO()
    sf.write(buf, waveform.squeeze(0).numpy(), SAMPLE_RATE, format="WAV", subtype="PCM_24")
    if cache_key:
        audio_cache.put(cache_key, buf.getvalue())
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


@app.get("/evaluate")
async def evaluate(
    request: Request,
    _rl: None = Depends(rate_limiter),
) -> dict:
    """Score the hit at the given slider settings.

    Same query parameters as /generate. Returns the corpus-referenced perceptual
    verdicts plus the spectrogram descriptors.
    """
    if _backend(request) == "dsp":
        return await _evaluate_dsp(request)
    if _backend(request) == "diffusion":
        return await _evaluate_diffusion(request)
    inst = _resolve(request)
    spec, waveform = _synthesize(request, inst)
    descriptors = compute_descriptors(
        waveform_spectrogram(waveform, n_frames=inst.model.n_frames)
        if inst.profile.waveform_controls else spec,
        inst.profile,
    )

    metrics = analyze_hit(waveform.squeeze(0).numpy().astype(np.float32), inst.profile)
    if metrics is None:
        return {"instrument": inst.profile.name, "backend": "vae",
                "error": "generated audio is silent or unusable",
                "descriptors": descriptors}

    report = score_sample("api", metrics, inst.reference(), inst.profile)
    control_fit = None
    if isinstance(inst.basis.basis, IdentityBasis):
        targets = np.array(slider_positions_to_axis_values(_slider_positions(request, inst), inst.basis))
        measured = np.array([descriptors[key] for key in inst.profile.descriptor_keys])
        span = np.maximum(np.array(inst.basis.maxs) - inst.basis.mins, 1e-6)
        error = float(np.max(np.abs((measured - targets) / span)))
        control_fit = {"max_normalized_error": error, "limited": error > .05,
                       "targets": dict(zip(inst.profile.descriptor_keys, targets.tolist())),
                       "seed": _texture_seed(request),
                       "includes_effects": any(k in request.query_params for k in ("attack_ms", "decay_ms", "drive", "filter"))}
    return {
        "instrument": inst.profile.name,
        "backend": "vae",
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
        "control_fit": control_fit,
    }


async def _evaluate_diffusion(request: Request) -> dict:
    """/evaluate for a diffusion render: verdicts, and how close it came to its sliders.

    Descriptors are measured at the training labels' peak, the level the
    targets were learned at, on the audio the client receives (effects included).
    """
    diff = _resolve_diffusion(request)
    profile = diff.profile
    waveform, targets = await _synthesize_diffusion(request, diff)
    measured = label_waveform(peak_normalize(waveform), profile)
    descriptors = dict(zip(profile.descriptor_keys, (float(v) for v in measured)))

    metrics = analyze_hit(waveform.squeeze(0).numpy().astype(np.float32), profile)
    if metrics is None:
        return {"instrument": profile.name, "backend": "diffusion",
                "error": "generated audio is silent or unusable",
                "descriptors": descriptors}

    report = score_sample("api", metrics, diff.reference(), profile)
    low, high = slider_range(diff.model)
    target_values = targets.numpy()
    error = float(np.max(np.abs(measured - target_values) / np.maximum(high - low, 1e-6)))
    return {
        "instrument": profile.name,
        "backend": "diffusion",
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
        "control_fit": {
            "max_normalized_error": error,
            "limited": error > .05,
            "targets": dict(zip(profile.descriptor_keys, target_values.tolist())),
            "seed": _texture_seed(request),
            "includes_effects": any(k in request.query_params for k in EFFECTS),
        },
    }


@app.get("/spectrogram")
async def spectrogram(
    request: Request,
    _rl: None = Depends(rate_limiter),
) -> dict:
    """Raw decoded spectrogram — the model's output before the vocoder.

    A diffusion render has no spectrogram of its own; for it this is the
    normalised mel spectrogram of the rendered waveform, effects included.
    """
    if _backend(request) == "dsp":
        inst = _resolve_dsp(request)
        waveform = await _synthesize_dsp(request, inst)
        spec = waveform_spectrogram(waveform, n_frames=N_FRAMES).squeeze(0)
        return {"instrument": inst.profile.name, "backend": "dsp", "source": "render",
                "shape": list(spec.shape), "data": spec.numpy().tolist()}
    if _backend(request) == "diffusion":
        diff = _resolve_diffusion(request)
        waveform, _ = await _synthesize_diffusion(request, diff)
        spec = waveform_spectrogram(waveform, n_frames=N_FRAMES).squeeze(0)
        return {
            "instrument": diff.profile.name,
            "backend": "diffusion",
            "source": "render",
            "shape": list(spec.shape),
            "data": spec.cpu().numpy().tolist(),
        }
    inst = _resolve(request)
    with torch.no_grad():
        spec = inst.model.decode(_latent_for(request, inst))
    return {
        "instrument": inst.profile.name,
        "backend": "vae",
        "source": "decoder",
        "shape": [inst.model.n_mels, inst.model.n_frames],
        "data": spec.squeeze().cpu().numpy().tolist(),
    }


# ---------------------------------------------------------------------------
# WAV export
# ---------------------------------------------------------------------------

_SAFE_NAME = re.compile(r"[^a-z0-9]+")


def _export_filename(
    inst: InstrumentState | DiffusionState | ProceduralState, positions: list[float], seed: int = 0,
) -> str:
    """``kick_sub80_punch50_....wav`` — the settings, readable in a file browser.

    Diffusion renders are tagged ``_diffusion`` after the instrument name.
    """
    diffusion = isinstance(inst, DiffusionState)
    procedural = isinstance(inst, ProceduralState)
    names = inst.names if diffusion or procedural else inst.basis.names
    parts = [
        f"{_SAFE_NAME.sub('', name.lower())}{round(p * 100):02d}"
        for name, p in zip(names, positions)
    ]
    tag = "_diffusion" if diffusion else "_dsp" if procedural else ""
    suffix = f"_v{seed}" if seed else ""
    return f"{inst.profile.name}{tag}_{'_'.join(parts)}{suffix}.wav"


@app.post("/export")
async def export(request: Request, _rl: None = Depends(rate_limiter)):
    """Download the same rendered sound as the preview, as 24-bit WAV."""
    inst, waveform = await _render(request)
    positions = _slider_positions(request, inst)
    seed = _texture_seed(request)
    buf = io.BytesIO()
    sf.write(buf, waveform.squeeze(0).numpy(), SAMPLE_RATE, format="WAV", subtype="PCM_24")
    identity = sha256(buf.getvalue()).hexdigest()
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav", headers={
        "Content-Disposition": f'attachment; filename="{_export_filename(inst, positions, seed)}"',
        "X-Kicks-Export-Id": identity,
        "Cache-Control": "no-store",
    })
