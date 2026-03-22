# kicks

A VAE-powered kick drum synthesizer. Train a convolutional VAE on your kick samples, then generate new kicks in real-time by moving sliders in a web UI.

## How it works

```
Kick samples (.wav)
  -> Strip: isolate kick hits, exclude loops
  -> LUFS loudness normalisation (-14 LUFS)
  -> Log-mel spectrograms (128x256)
  -> Fixed normalisation [-11.51, 2.5] -> [0, 1]
  -> Beta-VAE training (beta=1.0, cyclical annealing)
  -> Latent vectors (128-dim)
  -> PCA -> 4 principal components (auto-named by perceptual correlation)
  -> Slider UI (e.g. Sub, Punch, Click, Bright)
  -> PCA inverse -> z vector
  -> VAE decoder -> spectrogram
  -> BigVGAN vocoder (optionally fine-tuned) -> audio
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

Isolates kick hits from drum loops and full mixes. Detects kick via low-frequency envelope analysis, finds the natural decay endpoint, then fades out and zeros everything after. Automatically detects and excludes cyclical/looping samples by moving them to `data/kicks_loops/`.

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

Trains for 200 epochs with cyclical beta annealing (4 cycles, beta ramping 0 -> 1.0 per cycle). Monitors reconstruction loss (spectral convergence + L1) and KL divergence separately. Saves `models/vae_best.pth` (best validation loss). Generates 20 reconstructions and 10 latent samples in `output/`.

Options:

```
--data, -d    Path to training data (default: data/kicks)
--epochs, -e  Number of epochs (default: 200)
--latent-dim  Latent dimension (default: 128)
--beta        KL beta weight (default: 1.0)
```

### 3. Fine-tune vocoder (optional)

```bash
uv run kicks fine-tune
```

Fine-tunes the BigVGAN vocoder on your kick samples using GAN training (generator + multi-period/CQT discriminators). Freezes all but the last 2 upsampling blocks. Saves best/checkpoint/final weights to `models/vocoder/`. Supports resuming from checkpoints automatically.

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

A synthwave-themed synthesizer in your terminal with waveform and spectrogram displays. PCA sliders are auto-named by perceptual correlation (e.g. Sub, Punch, Click, Bright).

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

```bash
# Terminal 2 -- Next.js frontend
cd web && npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Move the sliders to generate kick drums in real-time. The interface shows a pre-vocoder spectrogram and post-vocoder waveform alongside the controls. Use **Randomise** to explore the latent space and **Download WAV** to save kicks.

A standalone HTML UI is also available at [http://localhost:8080](http://localhost:8080) (no Node.js required). API docs at [http://localhost:8080/docs](http://localhost:8080/docs).

### 5. Corpus analysis

```bash
uv run kicks cluster
cd web && npm run dev
```

Runs GMM clustering (BIC-selected k) on z-scored latents, PCA on perceptual descriptors, and saves analysis to `output/cluster_analysis.json`. Open [http://localhost:3000/cluster](http://localhost:3000/cluster) to view the corpus analysis dashboard with:

- Exploratory data analysis (summary stats, descriptor correlations, duration distribution, box plots)
- PCA variance breakdown and principal component cards
- PC-descriptor correlation matrix
- 2D scatter plots and interactive 3D visualization
- Cluster profiles with audio playback
- Descriptor distributions

Options:

```
--data, -d     Path to dataset (default: data/kicks)
--samples, -n  Number of samples to cluster (default: all)
```

## Model

2D Convolutional VAE (latent_dim=128).

| Setting | Value |
|---------|-------|
| Input | Log-mel spectrogram (1, 128, 256) |
| Sample rate | 44100 Hz |
| Audio length | ~1.49s (65536 samples) |
| N_FFT | 1024 |
| HOP_LENGTH | 256 |
| WIN_SIZE | 1024 |
| N_MELS | 128 |
| Latent dim | 128 |
| Beta | 1.0 (cyclical annealing, 4 cycles) |
| Vocoder | BigVGAN v2 (MIT, pretrained at 44kHz) |

- **Encoder**: 4 conv layers (1->32->64->128->256, stride 2, BatchNorm+ReLU), flatten, FC to mu/logvar
- **Decoder**: FC, reshape, 4 transposed conv layers (mirror), Sigmoid output
- **Loss**: Spectral convergence + L1 + beta * KL
- **Pre-processing**: LUFS loudness normalisation to -14 LUFS, BigVGAN ln-clamp normalisation [-11.51, 2.5] -> [0, 1]
- **Training**: 10% validation split, best checkpoint saved by val loss
- **PCA slider naming**: Auto-correlates PCs with perceptual descriptors (sub, punch, click, bright, decay) and flips negative axes
- **Perceptual descriptors**: Sub-bass energy, punch (log transient/body ratio), click (HF transient), brightness (high/low ratio), decay (exponential fit)

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
│   ├── train.py                 # Training loop with cyclical beta annealing
│   ├── loss.py                  # Spectral convergence + L1 + beta * KL
│   ├── vocoder.py               # BigVGAN vocoder (spec -> audio)
│   ├── finetune.py              # BigVGAN vocoder fine-tuning (GAN training)
│   ├── cluster.py               # GMM, PCA, descriptors
│   ├── _cluster_cmd.py          # Cluster command implementation
│   └── _strip_cmd.py            # Strip command implementation
├── web/                         # Next.js + shadcn/ui frontend
│   ├── app/
│   │   ├── page.tsx             # Synthesizer page
│   │   └── cluster/page.tsx     # Corpus analysis page
│   ├── components/
│   │   ├── synth/               # Synthesizer components (waveform, spectrogram)
│   │   ├── cluster/             # Corpus analysis components (EDA, scatter, 3D)
│   │   └── ui/                  # shadcn components
│   ├── hooks/                   # Custom React hooks
│   ├── types/                   # TypeScript type definitions
│   └── lib/                     # Shared utilities
├── data/kicks/                  # Input .wav samples (not tracked)
├── models/                      # Saved checkpoints (not tracked)
└── output/                      # Generated audio + analysis (not tracked)
```

## License

MIT
