# kicks

A VAE-powered kick drum synthesizer. Train a convolutional VAE on your kick samples, then generate new kicks in real-time by moving sliders in a web UI, TUI, or REST API.

## How it works

```
Kick samples (.wav)
  -> Strip: isolate kick hits, exclude loops
  -> LUFS loudness normalisation (-14 LUFS)
  -> Log-mel spectrograms (128x256)
  -> Fixed normalisation [-11.51, 2.5] -> [0, 1]
  -> Beta-VAE training (beta=0.3, cyclical annealing, free bits)
  -> Latent vectors (32-dim)
  -> PCA -> 5 principal components (auto-named by perceptual correlation)
  -> Slider UI (e.g. Sub, Punch, Click, Bright, Decay)
  -> PCA inverse -> z vector
  -> VAE decoder -> spectrogram
  -> BigVGAN or Griffin-LIM vocoder -> audio
```

## Features

- **VAE synthesis engine** — 2D convolutional beta-VAE trained on log-mel spectrograms
- **PCA latent space** — 5 intuitive sliders auto-named by perceptual descriptor correlation (sub, punch, click, bright, decay)
- **Decay decorrelation** — Moving non-Decay sliders no longer changes perceived sample length
- **Two vocoder backends** — BigVGAN v2 (high quality, GPU) or Griffin-LIM (CPU-only, no model download)
- **Web UI** — Next.js + shadcn/ui with 3-column layout, waveform/spectrogram visualizers, keyboard shortcuts (Space/R/S)
- **Preset management** — Save/load slider presets to `localStorage`
- **Envelope shaper** — Adjust attack/decay via query params (`attack_ms`, `decay_ms`)
- **Distortion/saturation** — Apply drive and lowpass filter via query params (`drive`, `filter`)
- **16-step sequencer** — Multi-track drum sequencing with MIDI input and BPM control
- **Terminal UI** — Synthwave-themed TUI with waveform and spectrogram displays via Textual
- **Standalone HTML UI** — Embedded in the FastAPI server at `GET /`, no Node.js needed
- **REST API** — FastAPI backend with rate limiting (10 req/s), LRU audio cache, CORS, input validation
- **Corpus analysis** — GMM clustering with BIC selection, PCA on perceptual descriptors, interactive dashboard
- **Preprocessing** — `kicks strip` isolates kick hits from drum loops, detects and excludes loops automatically
- **Vocoder fine-tuning** — GAN-based fine-tuning of BigVGAN on your kick samples
- **Docker support** — `docker compose up` for backend + frontend
- **Shared audio context** — Reuses a single `AudioContext` across all web components
- **Error states with retry** — Frontend handles connection errors, AbortController cancels stale requests

## Setup

### Requirements

- Python 3.10+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Apple Silicon (MPS), CUDA GPU, or CPU

### Install Python dependencies

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### Install frontend dependencies

```bash
cd web && npm install
```

### Add training data

Place `.wav` kick drum samples in `data/kicks/`.

## Usage

### 1. Strip (preprocessing)

```bash
uv run kicks strip --dry-run     # preview without modifying
uv run kicks strip               # run with backup (default)
```

Isolates kick hits from drum loops and full mixes. Detects kick via low-frequency envelope analysis, finds the natural decay endpoint, then fades out and zeros everything after. Automatically detects and excludes cyclical/looping samples (via energy-envelope autocorrelation) by moving them to `data/kicks_loops/`. Uses high-frequency envelope analysis to detect hi-hat/snare onsets that mark the end of the kick.

**Default behavior is non-destructive:** originals are backed up to `data/kicks_backup/` before any modification. Loops are copied (not moved) to `data/kicks_loops/`.

Options:

```
--data, -d            Path to sample directory (default: data/kicks)
--dry-run             Analyze without modifying files
--max-duration        Max kick duration in ms (default: 1200)
--min-duration        Min kick duration in ms (default: 50)
--threshold           Decay threshold, fraction of peak (default: 0.01)
--fade-ms             Fade-out length in ms (default: 10)
--backup / --no-backup  Copy originals to backup dir (default: --backup)
--exclude-loops / --keep-loops  Detect and move loops (default: exclude)
--move-loops          Move detected loops out of source dir (default: copy only)
```

### 2. Train

```bash
uv run kicks train
```

Trains for 200 epochs with cyclical beta annealing (4 cycles, beta ramping 0 → 0.3 per cycle) and a cosine annealing learning rate scheduler. Monitors reconstruction loss (multi-resolution spectral convergence + frequency-weighted L1) and KL divergence (with 0.5-nat free bits per dimension) separately. Uses a 10% validation split; saves `models/vae_best.pth` (best validation loss). Generates per-epoch loss plots and output reconstructions/samples.

Options:

```
--data, -d        Path to training data (default: data/kicks)
--epochs, -e      Number of epochs (default: 200)
--latent-dim      Latent dimension (default: 32)
--beta            KL beta weight (default: 0.3, capped in cyclical annealing)
```

### 3. Fine-tune vocoder (optional)

```bash
uv run kicks fine-tune
```

Fine-tunes the BigVGAN vocoder on your kick samples using GAN training (generator + multi-period/CQT discriminators). Freezes all but the last 2 upsampling blocks. Saves best/checkpoint/final weights to `models/vocoder/`. Supports resuming from checkpoints automatically. Verifies BigVGAN's mel configuration matches the VAE's audio constants at startup.

Options:

```
--data, -d       Path to training data (default: data/kicks)
--epochs, -e     Number of epochs (default: 200)
--batch-size, -b Batch size (default: 2)
--lr             Learning rate (default: 1e-4)
--grad-accum     Gradient accumulation steps (default: 4)
--save-every     Save checkpoint every N epochs (default: 50)
--save-dir       Directory for vocoder checkpoints (default: models/vocoder)
```

### 4. Synthesize kicks

#### Terminal UI (no browser needed)

```bash
uv run kicks tui
```

A synthwave-themed synthesizer in your terminal with waveform and spectrogram displays. PCA sliders are auto-named by perceptual correlation (e.g. Sub, Punch, Click, Bright, Decay). Auto-plays audio via `afplay` on macOS.

| Key | Action |
|-----|--------|
| Tab | Switch between sliders |
| Left/Right | Fine adjust (±2%) |
| Shift+Left/Right | Coarse adjust (±10%) |
| Space | Generate & play |
| P | Replay last |
| S | Save WAV |
| R | Randomize |
| 0 | Reset sliders to 50% |
| Q | Quit |

#### Web UI

Start the backend and frontend in two terminals:

```bash
# Terminal 1 — FastAPI backend
uv run kicks serve
```

For systems without a GPU, use the Griffin-LIM vocoder (lower quality, no neural model needed):

```bash
uv run kicks serve --griffin-lim
```

```bash
# Terminal 2 — Next.js frontend
cd web && npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The interface shows a 3-column layout: pre-vocoder spectrogram | slider controls | post-vocoder waveform. Move the sliders to generate kick drums in real-time.

**Keyboard shortcuts** (when not typing in inputs):

- **Space** — Generate new kick
- **R** — Randomize all sliders
- **S** — Download current kick as WAV

**Preset management:**

- Save: type a name and click "Save" — stored in `localStorage`
- Load: select from the dropdown and click "Load"
- Delete: select and click "Del"

**Effects (via query params on the `/generate` endpoint):**

- `attack_ms=10` — Apply attack envelope (linear ramp)
- `decay_ms=200` — Apply decay envelope (linear fade)
- `drive=0.5` — Saturation/distortion (tanh waveshaping, 0–1)
- `filter=500` — Lowpass filter cutoff in Hz (40–18000)

A standalone HTML UI is also available at [http://localhost:8080](http://localhost:8080) (no Node.js required). The standalone UI does not include sequencer, presets, or effects controls — those are web-only features.

API docs at [http://localhost:8080/docs](http://localhost:8080/docs).

#### 16-Step Sequencer

The web UI includes a built-in drum sequencer below the synthesizer controls:

- Generate a kick, then use the sequencer to build patterns
- Tracks for your generated kick, with click, clap, hat, snare, rim, and tom samples
- Adjustable BPM (60–200), play/stop transport
- MIDI input support for external controllers
- Click track labels to preview individual sounds
- Step grid with beat markers and per-track colors

### 5. Corpus analysis

```bash
uv run kicks cluster
cd web && npm run dev
```

Runs GMM clustering (BIC-selected k) on z-scored latents, PCA on z-scored perceptual descriptors (5D → 3D), and saves analysis to `output/cluster_analysis.json`. Generates per-cluster average audio samples to `output/samples/`. Open [http://localhost:3000/cluster](http://localhost:3000/cluster) to view the corpus analysis dashboard with:

- Exploratory data analysis (summary stats, descriptor correlations, duration distribution, box plots)
- PCA variance breakdown and principal component cards
- PC-descriptor correlation matrix
- 2D scatter plots and interactive 3D visualization
- Cluster profiles with audio playback
- Descriptor distributions
- Sample inspector with waveform and spectrogram views
- Per-cluster average audio playback

Options:

```
--data, -d     Path to dataset (default: data/kicks)
--samples, -n  Number of samples to cluster (default: all)
```

### 6. Docker

```bash
docker compose up --build
```

Starts both the FastAPI backend (port 8080) and Next.js frontend (port 3000). GPU support is configured in `docker-compose.yml` via `deploy.resources.reservations.devices`. CPU-only fallback works automatically.

Environment variables for Docker:

- `KICKS_DATA_DIR` — Data path inside container (default: `/app/data/kicks`)
- `KICKS_MODEL_DIR` — Model path (default: `/app/models`)
- `KICKS_OUTPUT_DIR` — Output path (default: `/app/output`)
- `KICKS_VOCODER` — Vocoder backend (`bigvgan` or `griffinlim`, default: `bigvgan`)
- `KICKS_CORS_ORIGINS` — Comma-separated allowed CORS origins (default: `http://localhost:3000`)

Data, models, and output directories are mounted as volumes from the host for persistence.

## Model

2D Convolutional VAE (latent_dim=32 by default).

| Setting | Value |
|---------|-------|
| Input | Log-mel spectrogram (1, 128, 256) |
| Sample rate | 44100 Hz |
| Audio length | ~1.49s (65536 samples) |
| N_FFT | 1024 |
| HOP_LENGTH | 256 |
| WIN_SIZE | 1024 |
| N_MELS | 128 |
| FMIN | 0 |
| FMAX | None (Nyquist) |
| Latent dim | 32 (reduced to prevent posterior collapse) |
| Beta | 0.3 (cyclical annealing, 4 cycles, 0→max ramp per cycle) |
| Free bits | 0.5 nats per latent dimension |
| Loss | Multi-resolution (scales 1, 2, 4) frequency-weighted reconstruction + β·KL |
| Vocoder | BigVGAN v2 (MIT, pretrained at 44kHz) or Griffin-LIM (no GPU needed) |
| PCA components | 5 (auto-named by perceptual correlation) |

- **Encoder**: 4 conv layers (1 → 32 → 64 → 128 → 256, stride 2, BatchNorm+ReLU), flatten, FC to mu/logvar (logvar clamped to [-10, 10])
- **Decoder**: FC, reshape, 4 transposed conv layers (mirror encoder), Sigmoid output
- **Loss**: Multi-resolution spectral convergence + frequency-weighted L1 across 3 scales + β·KL with 0.5-nat free bits per dimension
- **Optimizer**: Adam (lr=1e-3) with CosineAnnealingLR scheduler
- **Pre-processing**: LUFS loudness normalisation to -14 LUFS, BigVGAN ln-clamp normalisation [-11.51, 2.5] → [0, 1]
- **Training**: 10% validation split, best checkpoint saved by val loss
- **PCA slider naming**: Auto-correlates PCs with perceptual descriptors (sub, punch, click, bright, decay) and flips negative axes; requires |r| ≥ 0.15
- **Decay decorrelation**: Non-Decay sliders are compensated to prevent cross-talk with the Decay PC
- **Perceptual descriptors**: Sub-bass energy (bands 0–11), punch (log transient/body ratio in bass region), click (HF transient in bands 40–99), brightness (high/low energy ratio), decay (broadband early-to-late energy ratio)
- **Vocoder backends**: BigVGAN neural vocoder (high quality, GPU recommended) or Griffin-LIM phase reconstruction (CPU-only, lower quality, no model download)

## Project structure

```
kicks/
├── pyproject.toml               # Python project config (uv/pip)
├── Dockerfile                   # Backend Docker image (FastAPI + BigVGAN)
├── docker-compose.yml           # Docker Compose (backend + frontend + GPU config)
├── .python-version              # Python 3.12 (for uv)
├── .mcp.json                    # MCP server config (shadcn)
├── main.py                      # Legacy entry point → `kicks train`
├── app.py                       # Legacy entry point → `kicks serve`
├── kicks/                       # Core Python package
│   ├── __init__.py              # Exports: KickDataset, KickDataloader, VAE
│   ├── cli.py                   # Typer CLI (train, serve, tui, cluster, strip, fine-tune)
│   ├── tui.py                   # Textual TUI synthesizer (synthwave theme)
│   ├── server.py                # FastAPI backend + standalone HTML + rate limiter + cache
│   ├── config.py                # Centralized config & device detection
│   ├── model.py                 # 2D Conv VAE + audio constants
│   ├── dataset.py               # Load audio → LUFS norm → log-mel → fixed norm
│   ├── dataloader.py            # DataLoader wrapper
│   ├── train.py                 # Training loop with cyclical beta annealing + cosine LR
│   ├── loss.py                  # Multi-resolution freq-weighted loss + free-bits KL
│   ├── vocoder.py               # BigVGAN + Griffin-LIM vocoder backends
│   ├── finetune.py              # BigVGAN vocoder fine-tuning (GAN training)
│   ├── cluster.py               # GMM, PCA, perceptual descriptors
│   ├── pca_analysis.py          # Shared PCA analysis (server + TUI)
│   ├── _cluster_cmd.py          # Cluster command implementation
│   └── _strip_cmd.py            # Strip command implementation
├── web/                         # Next.js + shadcn/ui frontend
│   ├── app/
│   │   ├── page.tsx             # Synthesizer page (3-column layout + sequencer)
│   │   ├── cluster/page.tsx     # Corpus analysis dashboard
│   │   ├── math/page.tsx        # Mathematical foundations (LaTeX docs)
│   │   └── api/
│   │       ├── config/          # Slider configuration endpoint proxy
│   │       ├── generate/        # Audio generation endpoint proxy
│   │       ├── cluster-data/    # Cluster analysis data
│   │       ├── cluster-avg/     # Cluster average audio
│   │       └── play/            # Audio playback endpoint
│   ├── components/
│   │   ├── synth/               # Synthesizer widgets (waveform, spectrogram, sequencer)
│   │   ├── cluster/             # Corpus analysis components (EDA, scatter, 3D, etc.)
│   │   └── ui/                  # shadcn/ui components (slider, card, badge, etc.)
│   ├── hooks/                   # Custom React hooks
│   │   ├── use-synth.ts         # Core synth hook: API calls, presets, keyboard shortcuts
│   │   ├── use-audio-context.ts # Shared AudioContext singleton hook
│   │   ├── use-audio.ts         # Audio playback hook
│   │   ├── use-sequencer.ts     # 16-step drum sequencer hook
│   │   └── use-cluster-data.ts  # Cluster analysis data fetching hook
│   ├── types/                   # TypeScript type definitions
│   └── lib/                     # Shared utilities
├── data/kicks/                  # Input .wav samples (not tracked)
├── models/                      # Saved checkpoints (not tracked)
└── output/                      # Generated audio + analysis (not tracked)
```

## Key design decisions

- **Latent_dim=32**: Reduced from 128 to prevent posterior collapse while retaining sufficient capacity for kick drum synthesis.
- **Multi-resolution loss**: Average-pooling at scales 1, 2, 4 captures both fine transient detail and global spectral envelope.
- **Frequency-weighted L1**: Linearly decaying weights emphasize lower mel bands where kick drum energy concentrates.
- **Free bits KL**: Per-dimension KL clamped to 0.5 nats prevents individual latent dimensions from collapsing to zero.
- **5 PCA components**: A 5th component (Decay) captures the time-domain envelope, providing independent control over sample duration.
- **Decay decorrelation**: Moving non-Decay sliders no longer changes perceived sample length — Decay PC is automatically compensated via ratios computed from descriptor correlations.
- **Griffin-LIM fallback**: Uses pseudo-inverse of the mel filterbank instead of `InverseMelScale` (unsupported on MPS, prone to rank errors on CPU).
- **Fixed spectrogram bounds**: Normalization uses fixed bounds `[-11.5129, 2.5]` (BigVGAN's ln clamp), not dataset-dependent min/max — ensures consistent behavior across datasets.
- **Z-scored GMM clustering**: Latents are z-score normalized before GMM fitting, preventing magnitude differences from dominating cluster assignments.
- **CORS restricted to single origin by default**: Configurable via `KICKS_CORS_ORIGINS` environment variable.
- **Non-destructive strip defaults**: `--backup` is enabled by default; loops are copied, not moved, unless `--move-loops` is specified.

## API

The FastAPI backend exposes the following endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /` | Standalone HTML UI (embedded, no Node.js needed) |
| `GET /config` | Slider definitions (names, ranges, defaults) |
| `GET /generate` | Generate kick audio (WAV) with slider values as query params |
| `GET /spectrogram` | Raw spectrogram data (2D array) for visualization |
| `GET /docs` | OpenAPI/Swagger documentation |

**`/generate` query parameters:**

- `pc1`...`pc5` — Slider positions (0–1, default: 0.5)
- `attack_ms` — Attack envelope duration in ms
- `decay_ms` — Decay envelope duration in ms
- `drive` — Distortion/saturation amount (0–1)
- `filter` — Lowpass filter cutoff frequency (40–18000 Hz)

## License

MIT
