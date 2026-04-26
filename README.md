# kicks

A VAE-powered kick drum synthesizer. Train a convolutional VAE on your kick samples, then generate new kicks in real-time by moving sliders in a web UI or TUI.

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
uv run kicks strip --backup      # run with backup to data/kicks_backup/
```

Isolates kick hits from drum loops and full mixes. Detects kick via low-frequency envelope analysis, finds the natural decay endpoint, then fades out and zeros everything after. Automatically detects and excludes cyclical/looping samples (via energy-envelope autocorrelation) by moving them to `data/kicks_loops/`. Uses high-frequency envelope analysis to detect hi-hat/snare onsets that mark the end of the kick.

Options:

```
--data, -d         Path to sample directory (default: data/kicks)
--dry-run          Analyze without modifying files
--max-duration     Max kick duration in ms (default: 1200)
--min-duration     Min kick duration in ms (default: 50)
--threshold        Decay threshold, fraction of peak (default: 0.01)
--fade-ms          Fade-out length in ms (default: 10)
--backup           Copy originals to backup dir first
--exclude-loops / --keep-loops  Detect and move loops (default: exclude)
```

### 2. Train

```bash
uv run kicks train
```

Trains for 200 epochs with cyclical beta annealing (4 cycles, beta ramping 0 -> 0.3 per cycle) and a cosine annealing learning rate scheduler. Monitors reconstruction loss (multi-resolution spectral convergence + frequency-weighted L1) and KL divergence (with 0.5-nat free bits per dimension) separately. Uses a 10% validation split; saves `models/vae_best.pth` (best validation loss). Generates per-epoch loss plots and output reconstructions/samples.

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

A synthwave-themed synthesizer in your terminal with waveform and spectrogram displays. PCA sliders are auto-named by perceptual correlation (e.g. Sub, Punch, Click, Bright, Decay).

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
# Terminal 1 -- FastAPI backend
uv run kicks serve
```

For systems without a GPU, use the Griffin-LIM vocoder (lower quality, no neural model needed):

```bash
uv run kicks serve --griffin-lim
```

```bash
# Terminal 2 -- Next.js frontend
cd web && npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The interface shows a 3-column layout: pre-vocoder spectrogram | slider controls | post-vocoder waveform. Move the sliders to generate kick drums in real-time. Use **Randomise** to explore the latent space and **Download WAV** to save kicks.

A standalone HTML UI is also available at [http://localhost:8080](http://localhost:8080) (no Node.js required). API docs at [http://localhost:8080/docs](http://localhost:8080/docs).

#### 16-Step Sequencer

The web UI includes a built-in drum sequencer below the synthesizer controls:

- Generate a kick, then use the sequencer to build patterns
- Tracks for your generated kick, with click, clap, hat, snare, rim, and tom samples
- Adjustable BPM (60-200), play/stop transport
- MIDI input support for external controllers
- Click track labels to preview individual sounds
- Step grid with beat markers and per-track colors

### 5. Corpus analysis

```bash
uv run kicks cluster
cd web && npm run dev
```

Runs GMM clustering (BIC-selected k) on z-scored latents, PCA on z-scored perceptual descriptors (5D -> 3D), and saves analysis to `output/cluster_analysis.json`. Generates per-cluster average audio samples to `output/samples/`. Open [http://localhost:3000/cluster](http://localhost:3000/cluster) to view the corpus analysis dashboard with:

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

- **Encoder**: 4 conv layers (1->32->64->128->256, stride 2, BatchNorm+ReLU), flatten, FC to mu/logvar (logvar clamped to [-10, 10])
- **Decoder**: FC, reshape, 4 transposed conv layers (mirror), Sigmoid output
- **Loss**: Multi-resolution spectral convergence + frequency-weighted L1 across 3 scales + β·KL with 0.5-nat free bits per dimension
- **Optimizer**: Adam (lr=1e-3) with CosineAnnealingLR scheduler
- **Pre-processing**: LUFS loudness normalisation to -14 LUFS, BigVGAN ln-clamp normalisation [-11.51, 2.5] -> [0, 1]
- **Training**: 10% validation split, best checkpoint saved by val loss
- **PCA slider naming**: Auto-correlates PCs with perceptual descriptors (sub, punch, click, bright, decay) and flips negative axes; requires |r| ≥ 0.15
- **Decay decorrelation**: Non-Decay sliders are compensated to prevent cross-talk with the Decay PC
- **Perceptual descriptors**: Sub-bass energy (bands 0-11), punch (log transient/body ratio in bass region), click (HF transient in bands 40-99), brightness (high/low energy ratio), decay (broadband early-to-late energy ratio)
- **Vocoder backends**: BigVGAN neural vocoder (high quality, GPU recommended) or Griffin-LIM phase reconstruction (CPU-only, lower quality, no model download)

## Project structure

```
kicks/
├── pyproject.toml               # Python project config (uv/pip)
├── kicks/                       # Core Python package
│   ├── cli.py                   # Typer CLI (train, serve, tui, cluster, strip, fine-tune)
│   ├── tui.py                   # Textual TUI synthesizer (synthwave theme)
│   ├── server.py                # FastAPI backend + standalone HTML UI
│   ├── config.py                # Centralized config & device detection
│   ├── model.py                 # 2D Conv VAE + audio constants
│   ├── dataset.py               # Load audio -> LUFS norm -> log-mel -> fixed norm
│   ├── dataloader.py            # DataLoader wrapper
│   ├── train.py                 # Training loop with cyclical beta annealing + cosine LR
│   ├── loss.py                  # Multi-resolution freq-weighted loss + free-bits KL
│   ├── vocoder.py               # BigVGAN + Griffin-LIM vocoder backends
│   ├── finetune.py              # BigVGAN vocoder fine-tuning (GAN training)
│   ├── cluster.py               # GMM, PCA, perceptual descriptors
│   ├── _cluster_cmd.py          # Cluster command implementation
│   └── _strip_cmd.py            # Strip command implementation
├── web/                         # Next.js + shadcn/ui frontend
│   ├── app/
│   │   ├── page.tsx             # Synthesizer page (3-column layout + sequencer)
│   │   ├── cluster/page.tsx     # Corpus analysis dashboard
│   │   ├── math/page.tsx        # Mathematical foundations (LaTeX docs)
│   │   └── api/
│   │       ├── config/          # Slider configuration endpoint
│   │       ├── generate/        # Audio generation endpoint
│   │       ├── spectrogram/     # Spectrogram data endpoint
│   │       ├── cluster-data/    # Cluster analysis data
│   │       ├── cluster-avg/     # Cluster average audio
│   │       └── play/            # Audio playback endpoint
│   ├── components/
│   │   ├── synth/               # Synthesizer components (waveform, spectrogram, sequencer)
│   │   ├── cluster/             # Corpus analysis components (EDA, scatter, 3D)
│   │   └── ui/                  # shadcn components
│   ├── hooks/                   # Custom React hooks (useSynth, useSequencer, etc.)
│   ├── types/                   # TypeScript type definitions
│   └── lib/                     # Shared utilities
├── data/kicks/                  # Input .wav samples (not tracked)
├── models/                      # Saved checkpoints (not tracked)
├── output/                      # Generated audio + analysis (not tracked)
└── web/                         # Next.js frontend
```

## Key design decisions

- **Latent_dim=32**: Reduced from 128 to prevent posterior collapse while retaining sufficient capacity for kick drum synthesis.
- **Multi-resolution loss**: Average-pooling at scales 1, 2, 4 captures both fine transient detail and global spectral envelope.
- **Frequency-weighted L1**: Linearly decaying weights emphasize lower mel bands where kick drum energy concentrates.
- **Free bits KL**: Per-dimension KL clamped to 0.5 nats prevents individual latent dimensions from collapsing to zero.
- **5 PCA components**: A 5th component (Decay) captures the time-domain envelope, providing independent control over sample duration.
- **Decay decorrelation**: Moving non-Decay sliders no longer changes perceived sample length — the Decay PC is automatically compensated.
- **Griffin-LIM fallback**: The `--griffin-lim` flag enables CPU-only synthesis without downloading a neural vocoder model.
- **Z-scored GMM clustering**: Latents are z-score normalized before GMM fitting, preventing magnitude differences from dominating cluster assignments.

## License

MIT
