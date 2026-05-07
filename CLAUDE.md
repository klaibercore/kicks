# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Python — install & run
uv sync                                # Install Python deps
uv run kicks train                     # Train VAE
uv run kicks serve                     # Start FastAPI backend (port 8080)
uv run kicks serve --griffin-lim       # Backend with CPU vocoder
uv run kicks tui                       # Terminal UI synthesizer
uv run kicks strip --dry-run           # Preprocess samples (preview)
uv run kicks strip --backup            # Preprocess samples (with backup)
uv run kicks cluster                   # Run corpus analysis (GMM + PCA)
uv run kicks fine-tune                 # Fine-tune BigVGAN vocoder

# Frontend
cd web && npm install                  # Install JS deps
cd web && npm run dev                  # Next.js dev server (port 3000)
cd web && npm run build                # Production build
cd web && npm run lint                 # Lint frontend

# Docker
docker compose up --build              # Build & run backend + frontend
docker compose up -d backend           # Backend-only (GPU recommended)
```

## Architecture

### Data flow

```
.wav kicks → KickDataset (LUFS norm → BigVGAN log-mel → fixed-norm [0,1])
  → VAE train → latents (32-dim μ) → PCA (5 components) → sliders
  → slider values → PCA inverse → VAE decode → vocoder → .wav
```

### Python package (`kicks/`)

- **`cli.py`** — Typer CLI entry point. All subcommands (`train`, `serve`, `tui`, `cluster`, `strip`, `fine-tune`) are defined here with their `typer.Option` signatures. Each command imports implementation lazily.
- **`model.py`** — `VAE` class (2D Conv, latent_dim=32 default) + audio constants (`SAMPLE_RATE=44100`, `AUDIO_LENGTH=65536`, `N_FFT=1024`, `HOP_LENGTH=256`, `N_MELS=128`). These constants must stay in sync with BigVGAN's config.
- **`loss.py`** — Multi-resolution reconstruction (spectral convergence + frequency-weighted L1 at scales 1,2,4) + β·KL with 0.5-nat free bits per dimension. Frequency weights are cached per (n_mels, device).
- **`dataset.py`** — `KickDataset`: loads .wav → mono → resample → pad/truncate → LUFS norm (-14 dB) → BigVGAN mel spectrogram → fixed-bounds norm [-11.51, 2.5] → [0, 1]. Uses `bigvgan.mel_spectrogram()` so representation matches vocoder input.
- **`dataloader.py`** — Thin `DataLoader` subclass (`KickDataloader`), an extension point for custom batching.
- **`train.py`** — `train()` function handles the training loop with cyclical beta annealing, CosineAnnealingLR, 10% validation split, best-checkpoint saving, and loss plots. The CLI passes its defaults to this function.
- **`config.py`** — Device detection (CUDA > MPS > CPU), checkpoint loading with auto-detected latent_dim, path constants. Paths overridable via env vars (`KICKS_DATA_DIR`, `KICKS_MODEL_DIR`, `KICKS_OUTPUT_DIR`).
- **`vocoder.py`** — Two backends: `load_bigvgan()` (neural, high quality, GPU recommended) and `load_griffin_lim()` (CPU-only, no model download). `spec_to_audio()` dispatches based on vocoder type. Both apply 25 Hz highpass + 20 kHz lowpass + peak normalization. Fine-tuned weights auto-loaded from `models/vocoder/best.pth`.
- **`server.py`** — FastAPI app with lifespan that loads model + PCA + descriptors on startup. Rate-limited (10 req/s token bucket) with LRU audio cache (100 entries), CORS restricted to `KICKS_CORS_ORIGINS`. Endpoints: `GET /config` (slider definitions), `GET /generate` (audio WAV with optional `attack_ms`, `decay_ms`, `drive`, `filter` params), `GET /spectrogram` (raw spectrogram data). Includes embedded standalone HTML UI at `GET /`.
- **`tui.py`** — Textual app (`KicksApp`) with synthwave theme. Custom widgets: `SliderBar` (focusable neon sliders), `WaveformDisplay`, `SpectrogramDisplay`, `LogoWidget`, `SunWidget` (retrowave sun art). Loads model async, supports keyboard-driven workflow. Plays audio via `afplay`.
- **`cluster.py`** — `extract_latents()`, `select_n_clusters()` (BIC), `fit_gmm()`, `compute_descriptors()` (5 perceptual features from spectrogram). Descriptors: sub, punch, click, bright, decay.
- **`_cluster_cmd.py`** — CLI implementation for clustering. Z-scores latents for GMM, runs PCA on z-scored descriptors (5D→3D), computes correlations, generates per-cluster average audio. Saves to `output/cluster_analysis.json`.
- **`_strip_cmd.py`** — CLI implementation for preprocessing. Low-frequency envelope + HF onset detection + energy-envelope autocorrelation for loop detection. Default backup is enabled (`--backup`), defaults to non-destructive copy behavior.
- **`pca_analysis.py`** — Shared PCA analysis used by both `server.py` and `tui.py`. Fits PCA (5 components), auto-names PCs by perceptual descriptor correlation (flips negative axes, requires |r| ≥ 0.15), computes decay decorrelation ratios. Returns `PCAnalysis` dataclass.
- **`finetune.py`** — BigVGAN GAN fine-tuning. Freezes all but last 2 upsampling blocks. MPD + CQT discriminators. Supports resuming from checkpoints.

### Frontend (`web/`)

- **Next.js 16** app router with shadcn/ui components, Tailwind v4, React 19
- **3-column layout**: pre-vocoder `SpectrogramVis` | controls | post-vocoder `WaveformViewerVis`
- **`useSynth` hook** — fetches config from FastAPI backend via Next.js API route proxy (`/api/config`), manages slider state, calls `/api/generate` + `/api/spectrogram`, extracts waveform envelope from generated audio. Uses `AbortController` for in-flight requests.
- **`useAudioContext` hook** — Shared `AudioContext` singleton across components, lazy creation, release on unmount, handles `suspended` state (user-gesture resume).
- **`useSequencer` hook** — 16-step drum sequencer with multi-track pattern, BPM control, MIDI input, Web Audio scheduling.
- **`useClusterData` hook** — Fetches cluster analysis, handles error states, pagination support.
- **API routes** under `web/app/api/` proxy to the FastAPI backend (port 8080), with path traversal validation.
- **Preset management** — Save/load/delete slider presets via `localStorage` (`kicks_presets` key).
- **Keyboard shortcuts** — Space=generate, R=randomize, S=download (ignored when typing in inputs).
- **Corpus analysis** at `/cluster` page: EDA, PCA variance, scatter plots (2D + 3D), cluster profiles, sample inspector.
- **Math page** at `/math`: full pipeline documentation with KaTeX LaTeX equations.

### Key data contracts

- **Spectrogram shape**: `(B, 1, 128, 256)` — normalized to [0, 1]
- **VAE latent**: 32-dim vector (μ), logvar clamped to [-10, 10]
- **PCA**: 5 components, fit on corpus latents, slider range = [2nd, 98th] percentile
- **Checkpoint format**: `{"model": state_dict, "epoch": int, "val_loss": float, "latent_dim": int}`
- **Vocoder fine-tune weights**: saved as `models/vocoder/best.pth`, loaded automatically if present
- **Vocoder selection**: `KICKS_VOCODER=griffinlim` env var or `--griffin-lim` flag
- **Rate limiter**: 10 req/s token bucket (shared across all endpoints)
- **LRU cache**: 100 entries, keyed by query string, invalidated on server restart
- **CORS**: single origin by default (`http://localhost:3000`), configurable via `KICKS_CORS_ORIGINS` (comma-separated)

### Important gotchas

- BigVGAN's `n_fft` was changed from 2048 to **1024** — must match between model constants, dataset, and fine-tuning config. Assertions in `finetune.py` verify this at startup.
- BigVGAN `from_pretrained` is patched (`_patch_bigvgan_from_pretrained`) for compatibility with huggingface_hub ≥ 1.0. Patch is idempotent.
- The Griffin-LIM vocoder uses pinverse of the mel filterbank instead of `InverseMelScale` (unsupported on MPS, prone to rank errors on CPU).
- All spectrogram normalization uses **fixed** bounds `[-11.5129, 2.5]` (derived from BigVGAN's ln clamp), not dataset-dependent min/max.
- Latent_dim is saved in checkpoints and auto-detected on load — changing latent_dim requires retraining.
- All `torch.load` calls use `weights_only=True` for security.
- The standalone HTML UI at `GET /` is embedded in `server.py` as a string constant — update both places on UI changes.
- `_strip_cmd.py` defaults to `--backup` (non-destructive) — the backup directory is `data/kicks_backup/`.
- TUI audio playback uses `afplay` (macOS only) — will fail silently on Linux.
