"""FastAPI backend for the kick drum synthesizer."""

import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
import torchaudio
import soundfile as sf
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from sklearn.decomposition import PCA

from kicks import KickDataset, KickDataloader, VAE
from kicks.cluster import extract_latents
from kicks.config import get_device, load_vae_from_checkpoint, DATA_DIR, BEST_CHECKPOINT, N_PCS
from kicks.model import SAMPLE_RATE
from kicks.pca_analysis import analyze_latent_space
from kicks.vocoder import load_vocoder, spec_to_audio


class _State:
    """Module-level state populated during lifespan."""

    device: torch.device
    dataset: KickDataset
    model: VAE
    vocoder: object
    vocoder_type: str
    pca: PCA
    pc_projected: np.ndarray
    pc_names: list[str]
    pc_mins: list[float]
    pc_maxs: list[float]
    decay_compensation: np.ndarray | None  # per-PC ratios to cancel decay cross-talk
    decay_idx: int | None                  # index of the Decay PC


_state = _State()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load model, vocoder, and PCA on startup."""
    data_dir = os.environ.get("KICKS_DATA_DIR", DATA_DIR)

    _state.device = get_device()
    _state.dataset = KickDataset(data_dir)
    dataloader = KickDataloader(_state.dataset, batch_size=32, shuffle=False)

    _state.model, _ = load_vae_from_checkpoint(BEST_CHECKPOINT, _state.device)

    _state.vocoder_type = os.environ.get("KICKS_VOCODER", "bigvgan")
    _state.vocoder = load_vocoder(_state.device, vocoder_type=_state.vocoder_type)

    latents, spectrograms = extract_latents(_state.model, dataloader, _state.device)

    print("Computing descriptors for PC naming...")
    analysis = analyze_latent_space(latents, spectrograms)
    _state.pca = analysis.pca
    _state.pc_projected = analysis.pc_projected
    _state.pc_names = analysis.pc_names
    _state.pc_mins = analysis.pc_mins
    _state.pc_maxs = analysis.pc_maxs
    _state.decay_idx = analysis.decay_idx
    _state.decay_compensation = analysis.decay_compensation

    for i in range(N_PCS):
        print(f"{_state.pc_names[i]} range: [{_state.pc_mins[i]:.3f}, {_state.pc_maxs[i]:.3f}]")
    print("API ready")

    yield


# --- Rate limiter (simple token bucket) ---

class _RateLimiter:
    def __init__(self, rate: float = 10.0):
        self._rate = rate
        self._tokens = rate
        self._last = time.monotonic()

    def __call__(self) -> None:
        now = time.monotonic()
        self._tokens = min(self._rate, self._tokens + self._rate * (now - self._last))
        self._last = now
        if self._tokens < 1.0:
            from fastapi import HTTPException
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
        self._tokens -= 1.0


_rate_limiter = _RateLimiter(rate=10.0)


# --- LRU cache for generated audio (slider hash → bytes) ---

class _LRUCache:
    def __init__(self, max_size: int = 100):
        self._max_size = max_size
        self._store: dict[str, bytes] = {}
        self._order: list[str] = []

    def get(self, key: str) -> bytes | None:
        if key in self._store:
            self._order.remove(key)
            self._order.append(key)
            return self._store[key]
        return None

    def put(self, key: str, value: bytes) -> None:
        if key in self._store:
            self._order.remove(key)
        elif len(self._store) >= self._max_size:
            oldest = self._order.pop(0)
            del self._store[oldest]
        self._store[key] = value
        self._order.append(key)


_waveform_cache = _LRUCache(max_size=100)


app = FastAPI(title="Kicks API", lifespan=_lifespan)
_cors_origins = os.environ.get("KICKS_CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


_HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KICKS</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{
  background:#0d0221;color:#e0d0ff;
  font-family:'Orbitron',monospace,sans-serif;
  min-height:100vh;display:flex;justify-content:center;align-items:center;
  background-image:
    linear-gradient(rgba(42,16,80,.3) 1px,transparent 1px),
    linear-gradient(90deg,rgba(42,16,80,.3) 1px,transparent 1px);
  background-size:40px 40px;
}
.container{max-width:520px;width:100%;padding:2rem}
.logo{
  font-size:3.2rem;font-weight:900;text-align:center;
  background:linear-gradient(180deg,#ffbe0b,#ff6b35,#ff2975,#f72585,#b026ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  letter-spacing:14px;margin-bottom:.2rem;
}
.subtitle{
  text-align:center;color:#6a5a8a;letter-spacing:6px;
  font-size:.65rem;margin-bottom:2.5rem;text-transform:uppercase;
}
.slider-group{margin-bottom:1.4rem}
.slider-label{
  display:flex;justify-content:space-between;
  margin-bottom:.4rem;font-size:.7rem;letter-spacing:2px;
}
.slider-name{font-weight:700}
.slider-value{color:#e0d0ff}
input[type="range"]{
  -webkit-appearance:none;width:100%;height:4px;
  border-radius:2px;background:#3a1568;outline:none;
}
input[type="range"]::-webkit-slider-thumb{
  -webkit-appearance:none;width:18px;height:18px;
  border-radius:50%;cursor:pointer;
}
input[type="range"]::-moz-range-thumb{
  width:18px;height:18px;border-radius:50%;cursor:pointer;border:none;
}
.generate-btn{
  display:block;width:100%;margin:2rem 0 1.5rem;padding:16px;
  background:transparent;border:2px solid #ff2975;color:#ff2975;
  font-family:'Orbitron',monospace;font-size:1rem;font-weight:700;
  letter-spacing:6px;text-transform:uppercase;cursor:pointer;
  transition:all .3s;
  box-shadow:0 0 10px rgba(255,41,117,.3),inset 0 0 10px rgba(255,41,117,.05);
}
.generate-btn:hover:not(:disabled){
  background:#ff2975;color:#0d0221;
  box-shadow:0 0 20px #ff2975,0 0 40px rgba(255,41,117,.5);
}
.generate-btn:disabled{opacity:.5;cursor:wait}
.generate-btn.loading{animation:pulse 1s infinite}
@keyframes pulse{
  0%,100%{box-shadow:0 0 10px rgba(255,41,117,.3)}
  50%{box-shadow:0 0 30px rgba(255,41,117,.6)}
}
#player-section{margin-top:.5rem;text-align:center;display:none}
audio{width:100%;margin-top:.5rem;filter:hue-rotate(260deg)}
audio::-webkit-media-controls-panel{background:#150533}
.status{
  text-align:center;font-size:.6rem;letter-spacing:3px;
  color:#00fff2;margin-top:1.2rem;min-height:1.2em;
}
.sep{
  text-align:center;color:#3a1568;letter-spacing:4px;
  font-size:.5rem;margin:1.5rem 0 .5rem;
}
</style>
</head>
<body>
<div class="container">
  <h1 class="logo">KICKS</h1>
  <p class="subtitle">synthwave kick machine</p>
  <div id="sliders"></div>
  <button class="generate-btn" id="gen-btn" onclick="generate()" disabled>
    &#9654;&#8195;GENERATE
  </button>
  <div id="player-section"><audio id="player" controls></audio></div>
  <p class="status" id="status">loading...</p>
</div>
<script>
const COLORS=['#00fff2','#ff2975','#b026ff','#ff6b35'];
let sliderValues={};
async function init(){
  const res=await fetch('/config');
  const config=await res.json();
  const container=document.getElementById('sliders');
  config.sliders.forEach((s,i)=>{
    const color=COLORS[i%COLORS.length];
    sliderValues[s.id]=s.default;
    const group=document.createElement('div');
    group.className='slider-group';
    group.innerHTML=`
      <div class="slider-label">
        <span class="slider-name" style="color:${color}">${s.name}</span>
        <span class="slider-value" id="val-${s.id}">${(s.default*100).toFixed(0)}%</span>
      </div>
      <input type="range" min="${s.min}" max="${s.max}" step="${s.step}"
             value="${s.default}" data-id="${s.id}">`;
    container.appendChild(group);
    const style=document.createElement('style');
    style.textContent=`
      input[data-id="${s.id}"]::-webkit-slider-thumb{
        background:${color};box-shadow:0 0 8px ${color};
      }
      input[data-id="${s.id}"]::-moz-range-thumb{
        background:${color};box-shadow:0 0 8px ${color};
      }`;
    document.head.appendChild(style);
    group.querySelector('input').addEventListener('input',e=>{
      sliderValues[s.id]=parseFloat(e.target.value);
      document.getElementById('val-'+s.id).textContent=
        (sliderValues[s.id]*100).toFixed(0)+'%';
    });
  });
  document.getElementById('gen-btn').disabled=false;
  document.getElementById('status').textContent='ready';
}
async function generate(){
  const btn=document.getElementById('gen-btn');
  const status=document.getElementById('status');
  btn.disabled=true;btn.classList.add('loading');
  status.textContent='synthesizing...';
  try{
    const params=new URLSearchParams();
    for(const[id,val]of Object.entries(sliderValues))
      params.set('pc'+id,val);
    const res=await fetch('/generate?'+params);
    const blob=await res.blob();
    const url=URL.createObjectURL(blob);
    const player=document.getElementById('player');
    document.getElementById('player-section').style.display='block';
    player.src=url;player.play();
    status.textContent='ready';
  }catch(e){status.textContent='error: '+e.message}
  finally{btn.disabled=false;btn.classList.remove('loading')}
}
init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML_PAGE


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
    return {"sliders": sliders, "vocoder": _state.vocoder_type}


def _parse_pc_values(request: Request) -> list[float]:
    """Map slider positions [0,1] to PC-space values with decay decorrelation."""
    pc_values = []
    for i in range(N_PCS):
        raw_str = request.query_params.get(f"pc{i + 1}", "0.5")
        try:
            raw = float(raw_str)
        except (ValueError, TypeError):
            from fastapi import HTTPException
            raise HTTPException(
                status_code=422,
                detail={"error": f"Invalid value for pc{i + 1}: {raw_str!r}"},
            )
        val = _state.pc_mins[i] + raw * (_state.pc_maxs[i] - _state.pc_mins[i])
        val = max(_state.pc_mins[i], min(_state.pc_maxs[i], val))
        pc_values.append(val)

    # Compensate for decay cross-talk: adjust the Decay PC to cancel the
    # decay effect introduced by other PCs deviating from their midpoints.
    if _state.decay_compensation is not None and _state.decay_idx is not None:
        di = _state.decay_idx
        for i in range(N_PCS):
            if i == di:
                continue
            center = (_state.pc_mins[i] + _state.pc_maxs[i]) / 2.0
            pc_values[di] -= _state.decay_compensation[i] * (pc_values[i] - center)

    return pc_values


@app.get("/generate")
async def generate(request: Request, _rl: None = Depends(_rate_limiter)):
    # Check LRU cache
    cache_key = request.url.query or ""
    if cache_key:
        cached = _waveform_cache.get(cache_key)
        if cached is not None:
            return StreamingResponse(io.BytesIO(cached), media_type="audio/wav")

    pc_values = _parse_pc_values(request)

    z_np = _state.pca.inverse_transform([pc_values])
    z = torch.tensor(z_np, dtype=torch.float32).to(_state.device)

    with torch.no_grad():
        spec = _state.model.decode(z)
        waveform = spec_to_audio(spec, _state.dataset, _state.vocoder, _state.device)  # (B, T)

    # --- Optional envelope shaper ---
    attack_ms = request.query_params.get("attack_ms")
    decay_ms = request.query_params.get("decay_ms")
    if attack_ms is not None or decay_ms is not None:
        wf = waveform.squeeze(0)  # (T,)
        n = len(wf)
        env = torch.ones(n, dtype=wf.dtype)
        if attack_ms is not None:
            try:
                a_ms = max(0.0, float(attack_ms))
                a_samples = min(n, int(SAMPLE_RATE * a_ms / 1000))
                if a_samples > 0:
                    env[:a_samples] = torch.linspace(0, 1, a_samples)
            except (ValueError, TypeError):
                pass
        if decay_ms is not None:
            try:
                d_ms = max(0.0, float(decay_ms))
                d_samples = min(n, int(SAMPLE_RATE * d_ms / 1000))
                if d_samples > 0:
                    env[-d_samples:] = torch.linspace(1, 0, d_samples)
            except (ValueError, TypeError):
                pass
        waveform = (wf * env).unsqueeze(0)

    # --- Optional distortion / saturation ---
    drive_str = request.query_params.get("drive")
    filter_str = request.query_params.get("filter")
    if drive_str is not None or filter_str is not None:
        wf = waveform.squeeze(0)  # (T,)
        if drive_str is not None:
            try:
                drive = max(0.0, min(1.0, float(drive_str)))
                wf = torch.tanh(wf * (1.0 + drive * 8.0)) * (1.0 / (1.0 + drive * 0.3))
            except (ValueError, TypeError):
                pass
        if filter_str is not None:
            try:
                cutoff = max(40.0, min(18000.0, float(filter_str)))
                wf = torchaudio.functional.lowpass_biquad(
                    wf.unsqueeze(0), SAMPLE_RATE, cutoff_freq=cutoff, Q=0.707,
                ).squeeze(0)
            except (ValueError, TypeError):
                pass
        waveform = wf.unsqueeze(0)

    buf = io.BytesIO()
    sf.write(buf, waveform.squeeze(0).numpy(), SAMPLE_RATE, format="WAV")
    buf.seek(0)

    # Populate cache
    if cache_key:
        _waveform_cache.put(cache_key, buf.getvalue())
        buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")


@app.get("/spectrogram")
async def spectrogram_data(request: Request, _rl: None = Depends(_rate_limiter)):
    pc_values = _parse_pc_values(request)

    z_np = _state.pca.inverse_transform([pc_values])
    z = torch.tensor(z_np, dtype=torch.float32).to(_state.device)

    with torch.no_grad():
        spec = _state.model.decode(z)  # (1, 1, 128, 256), values in [0, 1]

    data = spec.squeeze().cpu().numpy().tolist()  # 128 rows × 256 cols
    return {"data": data}
