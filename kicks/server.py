"""FastAPI backend for the kick drum synthesizer."""

import io
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
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

    # Decay decorrelation: compute per-PC compensation ratios so that
    # non-Decay sliders don't change the perceived sample length.
    _state.decay_idx = None
    _state.decay_compensation = None
    for i, name in enumerate(_state.pc_names):
        if name == "Decay":
            _state.decay_idx = i
            break

    if _state.decay_idx is not None:
        decay_vals = desc_arrays["decay"]
        d_mean, d_std = decay_vals.mean(), decay_vals.std()
        # Regression coefficient of each PC on the decay descriptor
        betas = np.zeros(N_PCS)
        for i in range(N_PCS):
            pc_vals = _state.pc_projected[:, i]
            pc_std = pc_vals.std()
            if pc_std > 0 and d_std > 0:
                r = float(np.corrcoef(pc_vals, decay_vals)[0, 1])
                betas[i] = r * d_std / pc_std
        decay_beta = betas[_state.decay_idx]
        if abs(decay_beta) > 1e-8:
            # ratio[i] = how much Decay PC must shift per unit of PC_i to cancel its decay effect
            ratios = betas / decay_beta
            ratios[_state.decay_idx] = 0.0  # Decay slider keeps its own effect
            _state.decay_compensation = ratios
            print(f"Decay decorrelation enabled (ratios: {ratios.round(3)})")
        else:
            print("Decay decorrelation skipped (Decay PC has negligible decay correlation)")

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
        raw = float(request.query_params.get(f"pc{i + 1}", "0.5"))
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
async def generate(request: Request):
    pc_values = _parse_pc_values(request)

    z_np = _state.pca.inverse_transform([pc_values])
    z = torch.tensor(z_np, dtype=torch.float32).to(_state.device)

    with torch.no_grad():
        spec = _state.model.decode(z)
        waveform = spec_to_audio(spec, _state.dataset, _state.vocoder, _state.device)

    buf = io.BytesIO()
    sf.write(buf, waveform.squeeze(0).numpy(), SAMPLE_RATE, format="WAV")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")


@app.get("/spectrogram")
async def spectrogram_data(request: Request):
    pc_values = _parse_pc_values(request)

    z_np = _state.pca.inverse_transform([pc_values])
    z = torch.tensor(z_np, dtype=torch.float32).to(_state.device)

    with torch.no_grad():
        spec = _state.model.decode(z)  # (1, 1, 128, 256), values in [0, 1]

    data = spec.squeeze().cpu().numpy().tolist()  # 128 rows × 256 cols
    return {"data": data}
