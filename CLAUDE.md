# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                   # Install dependencies

kicks instruments                         # List profiles + which have a trained model
kicks strip --dry-run                     # Preprocess corpus (preview)
kicks strip                               # Isolate hits (backs up by default)
kicks clean                               # Quarantine loops/outliers (dry run; --apply to move)
kicks train                               # Train the VAE
kicks serve                               # REST API on :8080 (the website in web/ is its client)
kicks serve --griffin-lim                 # CPU vocoder, no model download
kicks serve --vocoder bigvgan             # Force one backend (default: each profile's own)
kicks serve --control descriptor          # Sliders target descriptors directly
kicks generate -n 20 -k 8                 # GMM latent prior + best-of-k eval selection
kicks eval                                # Score generated output against the corpus
kicks sweep -n 40                         # Sweep the REST API slider space and score it
kicks cluster                             # GMM + descriptor PCA report
kicks publish-analysis                    # Cluster reports -> web/public/analysis/ (no filenames)

# Every command takes --instrument / -i (kick | snare | hihat).
kicks train -i snare -e 300

docker compose up --build                 # API on :8080

cd web && pnpm install && pnpm dev        # Website on :3000 (Next.js 16, shadcn/ui, static export)
cd web && pnpm lint && pnpm build         # Lint + static export to web/out
```

## Architecture

### The central idea

The pipeline is instrument-agnostic. Kicks, snares and hi-hats share one corpus
loader, VAE, vocoder and evaluator. Everything that depends on *which drum* is
being synthesised lives in an `InstrumentProfile` (`kicks/instruments/`).

**When adding behaviour, ask whether it is instrument-dependent. If it is, it
belongs in the profile, not in an `if instrument == ...` branch.** No module
outside `kicks/instruments/` should name a drum type.

Dependency direction is one-way: `instruments/` imports only
`audio/constants`, and everything else imports `instruments/`.

### Data flow

```
.wav → strip/clean → DrumDataset (LUFS → BigVGAN log-mel → fixed-norm [0,1])
  → VAE train → latents (32-dim µ) → slider basis (PCA or descriptor)
  → slider values → closed-loop solve → VAE decode → vocoder (per profile)
  → waveform correction → .wav → eval
```

### `kicks/instruments/` — the profile system

- **`profile.py`** — the data types. `Region` (a rectangle of the spectrogram),
  `DescriptorSpec` (a slider axis: `mean`, `log_ratio`, `fraction`,
  `inverse_ratio`, or the gain-invariant `power_db_ratio` / `centroid_ms` the
  calibrated profiles use — over one or two regions), `MetricSpec` (an eval
  metric plus weight and verdict phrasing; `gate=True` means a hard penalty
  rather than a weighted contribution, `multivariate=False` excludes it from the
  Mahalanobis/Fréchet stats), `OnsetSpec`, `StripSpec`, `EvalWindows`,
  `TransientLossSpec`, `PathSpec`, and `InstrumentProfile` tying them together.
  `waveform_controls` opts a profile into the closed-loop waveform correction;
  `envelope_gate` (default on) additionally vetoes slider targets whose decoded
  low-end envelope re-peaks — turn it off for instruments whose envelope
  legitimately does (hi-hat shimmer). `vocoder` names the mel-to-audio backend
  the instrument renders best with (`discoder` default; the hi-hat says
  `bigvgan`) — a measured choice, with the audit numbers in the profile comment.
- **`metrics.py`** — `standard_metrics(noun, plural, drop=(), overrides={})`
  builds the 12-metric set with the instrument's noun substituted in. Profiles
  drop what does not apply and re-word what reads wrong.
- **`kick.py` / `snare.py` / `hihat.py`** — one `PROFILE` constant each.
- **`__init__.py`** — the registry. `get_profile(name)` resolves the argument,
  then `KICKS_INSTRUMENT`, then the default; it also overlays `KICKS_DATA_DIR` /
  `KICKS_MODEL_DIR` / `KICKS_OUTPUT_DIR` onto the profile's paths.

The **kick profile is the reference**: its descriptor windows, strip heuristics
and metric weights are the ones the trained checkpoint was tuned against, and are
byte-for-byte the pre-refactor behaviour. Its `PathSpec` uses `subdir=""` so it
keeps the original flat layout (`models/vae_best.pth`, `output/eval_reference.json`).
Do not "tidy" those numbers.

### `kicks/audio/`

- **`constants.py`** — `SAMPLE_RATE=44100`, `AUDIO_LENGTH=65536`, `N_FFT=1024`,
  `HOP_LENGTH=256`, `N_MELS=128`, `N_FRAMES=256`, `LOG_MEL_MIN/MAX`,
  `TARGET_LUFS`. Fixed across instruments so one vocoder and one architecture
  serve all of them. Must stay in sync with BigVGAN's config.
- **`waveform.py`** — **numpy/scipy only, no torch import.** `load_audio`,
  `rms_envelope`, `bandpass`, `band_envelope`, `detect_onsets(x, OnsetSpec)`,
  `is_loop`, `stft_power`, `band_power`, `apply_fade_out`. Keeping torch out of
  here is why `kicks eval` and `kicks clean` start instantly — don't import torch
  into this module or anything it pulls in.
- **`io.py`** — torch-side loading. `load_waveform()` is the single definition of
  "how audio enters the model" (mono → resample → fit length → LUFS); the
  dataset and the latent prior both go through it.
- **`mel.py`** — BigVGAN log-mel plus `normalize`/`denormalize`.
- **`effects.py`** — envelope, drive, lowpass for the API's query params.
- **`controls.py`** — `correct_waveform(waveform, targets, spans, profile)`:
  bounded STFT gains (`profile.waveform_controls` masks) solved closed-loop
  against the real vocoded waveform, applied before user effects to cancel
  vocoder drift. Profiles opt in via `waveform_controls`.
- **`vocoder.py`** — three backends: DisCoder (`load_discoder`, the pinned
  official 44.1 kHz Z checkpoint, 1.72 GB, downloaded to `models/discoder/` on
  first use, mmap-loaded), BigVGAN (`load_bigvgan`, plus fine-tuned weights from
  the profile's `vocoder_dir`) and Griffin-Lim. `resolve_vocoder_type(profile,
  requested)` is the one place precedence lives: explicit > `KICKS_VOCODER` >
  `profile.vocoder`. `spec_to_audio(spec, vocoder, device)` honours a backend's
  `inference_batch_size` (DisCoder renders one hit at a time — 430M params on an
  8 GB Mac). All three share one post-chain: 25 Hz highpass, 20 kHz lowpass,
  peak normalise, `gate_tail`.

### `kicks/nn/`

- **`discoder.py`** — inference-only port of ETH DISCO's DisCoder (upstream
  commit `8aee1ee`, MIT — `DISCODER_LICENSE` ships in the wheel). Encoder is
  verified bit-exact against upstream on a reduced fixture; the DAC decoder comes
  from the `descript-audio-codec` package. Pads odd frame counts to a training
  segment and crops back.
- **`vae.py`** — below.

### `kicks/nn/vae.py`

2D conv VAE. `VAE(latent_dim=32, n_mels=128, n_frames=256)` — the spectrogram
size is a constructor argument so a short-tail instrument can use fewer frames.
`checkpoint_meta()` returns the shape metadata to save alongside the weights;
`config.load_vae_from_checkpoint()` recovers it from older checkpoints by reading
`fc_mu` and `fc_decode` shapes.

### `kicks/analysis/`

- **`descriptors.py`** — `compute_descriptors(spec, profile)` and friends. Pure
  profile evaluation; accepts torch tensors or ndarrays of any leading shape.
- **`latents.py`** — `extract_latents`, `select_n_clusters` (BIC), `fit_gmm`.
- **`basis.py`** — `analyze_latent_space(latents, specs, profile, basis=...)`
  returns a `SliderBasis`. `basis="pca"` names components by descriptor
  correlation and computes cross-talk compensation for
  `profile.decorrelated_descriptor`; `basis="descriptor"` fits a `DescriptorBasis`
  with a closed-loop Newton `solve()`. `slider_positions_to_axis_values()` maps
  [0,1] positions into basis space and applies the decorrelation.
- **`calibration.py`** — `fit_or_load_basis(...)` persists the fitted descriptor
  basis plus its calibrated slider ranges to `<checkpoint>.controls.npz`
  (arrays + JSON only, loaded with `allow_pickle=False`), fingerprinted on the
  checkpoint bytes, `CALIBRATION_VERSION`, profile and corpus. A restart reuses
  it; a mismatch refits. Bump the version whenever the range search in
  `basis._calibrate_ranges` changes. That search shrinks the slider box until
  every *corpus-supported* probe (a real hit within 15% of the span) tracks its
  target within 1%; probes the corpus never produces are recorded as
  `max_unsupported_error` but are not binding.
- **`evaluation.py`** — `analyze_hit(x, profile)`, `build_reference`,
  `reference_from_rows`, `score_sample`, `frechet_distance`, `run_eval`. numpy/
  scipy only. The reference cache is fingerprinted on the instrument *and* the
  corpus contents.
- **`clustering.py`** — `run_cluster()`, the corpus analysis report.

### `kicks/training/`

- **`loss.py`** — `vae_loss(...)`, multi-resolution reconstruction + beta·KL with
  free bits, plus `transient_loss(recon, target, TransientLossSpec)`.
- **`trainer.py`** — the loop. Saves `vae_best.pth` (val loss) *and*
  `vae_best_eval.pth` (generative eval proxy: decode latents from the val
  posterior, measure descriptor realism). Both matter; they disagree.

### `kicks/api/`

- **`app.py`** — the FastAPI app. `/health`, `/instruments`, `/config`,
  `/generate`, `/evaluate`, `/spectrogram`, `/me`, `POST /export`. Every
  synthesis endpoint takes `instrument`. There is no HTML here — the website is
  the only UI and lives in `web/`.
- **`auth.py`** — optional Supabase auth. `Auth.current_user` / `require_user`
  are FastAPI dependencies; `CreditsClient` calls the Postgres ledger functions
  (`charge_export`, `refund_export`, `credit_balance`) through PostgREST as the
  service role. Off unless `KICKS_SUPABASE_URL` is set, so local dev is open.
- **`state.py`** — `ServerState` loads instruments lazily and keeps them, so one
  server can serve several drums. Vocoders are cached per `(backend,
  weights_dir)` and resolved per instrument (`vocoder_for(profile)`); `/health`
  reports `vocoder: "profile"` plus a per-instrument `vocoders` map unless a
  backend is forced, and `/config` reports the instrument's own.
- **`middleware.py`** — token-bucket `RateLimiter`, `LRUCache`.

### `web/` — the website

Next.js 16 App Router, `output: "export"` (GitHub Pages), Tailwind v4, shadcn/ui
on Base UI (not Radix: composition is `render={<Link/>}`, not `asChild`).

- **`lib/api/`** — `KicksApi` client + the `Sound` type (instrument, slider
  positions, effects). `soundQuery()` sorts keys so equal sounds hit the API cache.
- **`hooks/use-studio.tsx`** — all studio state; `hooks/use-kit.tsx` — the pad
  bank; `lib/midi.ts` — Web MIDI; `lib/audio/engine.ts` — one AudioContext,
  decoded-buffer cache.
- **`components/studio/`** discovers instruments and sliders from the API at
  load time, so **adding an instrument needs no UI change.** The studio is
  loaded client-only (`studio-loader.tsx`) — it needs AudioContext, MIDI and
  localStorage.
- **`components/analysis/`** renders `public/analysis/*.json` written by
  `kicks publish-analysis`. Those files carry **no filenames or paths**; keep it
  that way — the corpus is not public.
- **`hooks/use-auth.tsx`**, **`lib/billing.ts`** — Supabase session, credits,
  Stripe Checkout via Edge Functions. Everything degrades gracefully when
  `NEXT_PUBLIC_SUPABASE_URL` is unset.
- **`app/legal/*`** — German legal texts; operator details come from
  `NEXT_PUBLIC_LEGAL_*` (see `lib/legal.ts`) and the pages warn until set.
- The lint config is React-Compiler-strict (`react-hooks/set-state-in-effect`,
  refs-in-render): keep effect bodies asynchronous and derive state instead of
  resetting it in effects.

### `supabase/`

- **`migrations/*_init.sql`** — schema, RLS on every table, the append-only
  `credit_ledger`, Stripe catalogue mirror, orders, kits, consents, and the
  `SECURITY DEFINER` functions the API and webhooks call. Orders are detached
  (not deleted) on account erasure — § 147 AO retention.
- **`functions/`** — Deno Edge Functions: `create-checkout-session`,
  `stripe-webhook` (idempotent on event id via `stripe_events`), `billing-portal`,
  `delete-account`.

### Other

- **`cli.py`** — Typer. Every command takes `--instrument`; implementations are
  imported lazily so `kicks eval` doesn't wait on torch.
- **`analysis/publish.py`** — `kicks publish-analysis`: slims the cluster
  reports for the browser and copies cluster audio into `web/public/analysis/`.
- **`config.py`** — `get_device()` (CUDA > MPS > CPU) and
  `load_vae_from_checkpoint()`. Paths live on profiles, not here.
- **`corpus/strip.py`, `corpus/clean.py`** — corpus preparation.
- **`synthesis/generator.py`** — GMM latent prior + best-of-k selection.
- **`sweep.py`** — drives the running REST API and scores every result.

## Key data contracts

- **Spectrogram**: `(B, 1, 128, 256)`, values in [0, 1]
- **VAE latent**: 32-dim µ by default (per-profile), logvar clamped to [-10, 10]
- **Checkpoint**: `{"model": state_dict, "instrument": str, "latent_dim": int,
  "n_mels": int, "n_frames": int, "epoch": int, "val_loss": float}`. A descriptor
  control basis adds a `<checkpoint>.controls.npz` sidecar (fitted basis +
  calibrated slider ranges, fingerprinted on checkpoint + corpus); keep the two
  together when promoting or copying a model.
- **Slider count** follows `profile.n_sliders` (one per descriptor) — never
  assume 5
- **Slider query params**: `s1..sN`, legacy `pc1..pcN`, or the slider's own
  lowercased label. Under the PCA basis a slider's label is *discovered at fit
  time*, so map names through `SliderBasis.names`, not `profile.descriptor_keys`
- **Rate limiter**: 10 req/s token bucket, shared
- **LRU cache**: 100 entries keyed on the query string
- **Export idempotency**: `POST /export` takes `Idempotency-Key: <uuid>`; the
  ledger has a unique index on it, so a retry returns the same charge
- **Static-export routes** end in `/` (`trailingSlash: true`); link to `/studio/`,
  not `/studio`
- **Vocoder selection**: per profile (`profile.vocoder`: kick and snare
  `discoder`, hi-hat `bigvgan`). `KICKS_VOCODER=…` / `--vocoder …` /
  `--griffin-lim` force one backend for every instrument. The calibration
  sidecars are vocoder-independent (decoded-spectrogram scope); the waveform
  correction closes the loop on whichever backend rendered
- **Control basis**: `KICKS_CONTROL=descriptor|pca` or `--control ...`, default
  `descriptor` (closed-loop descriptor solve; PCA remains available as fallback)

## Important gotchas

- BigVGAN's `n_fft` is **1024**, not 2048 — must match across constants, dataset
  and vocoder. `load_bigvgan` still picks up any `*.pth` in the profile's
  `vocoder_dir` (`models/vocoder/checkpoint_100.pth` is the tuned kick vocoder).
  DisCoder shares the exact same mel (`validate_discoder_config` refuses a
  checkpoint that does not), which is why no VAE was retrained for it.
- DisCoder is a generic music vocoder: it inverts *real* drum mels worse than
  BigVGAN (kick 3.7 vs 2.3 dB, hi-hat 4.2 vs 1.9 dB active-mel error) yet
  tracks the VAE's blurred, out-of-distribution mels better on pitched drums.
  Judge a vocoder on `scripts/validate_controls.py` (the generation path), not
  on `scripts/compare_vocoders.py` alone. Don't switch the hi-hat to DisCoder
  without re-running that audit.
- `BigVGAN.from_pretrained` is patched (`patch_bigvgan_from_pretrained`) for
  huggingface_hub >= 1.0. Idempotent.
- Griffin-Lim uses the mel filterbank pseudo-inverse, not `InverseMelScale`
  (unsupported on MPS, rank-unstable on CPU).
- Normalisation bounds are **fixed** (`[-11.5129, 3.0]`), not dataset-dependent.
- `kicks/audio/waveform.py` and `kicks/analysis/evaluation.py` must stay
  torch-free.
- The eval reference cache invalidates on instrument *and* corpus fingerprint;
  `--refresh-ref` forces a rebuild.
- `corpus/strip.py` backs up by default (`data/<corpus>_backup/`); `clean` moves
  rather than deletes, with a manifest.
- Every `torch.load` uses `weights_only=True`.
- `web/public/analysis/*.json` must never contain corpus filenames or paths.
- Never put a Supabase service key or Stripe secret anywhere under `web/` —
  everything there is compiled into the public page.
