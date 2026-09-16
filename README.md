# kicks

A VAE-powered drum synthesizer. Train a convolutional VAE on your samples, then
generate new one-shots by moving a handful of perceptually meaningful sliders —
through a REST API, and a website (`web/`) with a studio, a MIDI pad bank,
corpus analytics and credit-based sample exports.

The pipeline is instrument-agnostic. Kicks, snares and hi-hats all run through
the same corpus loader, VAE, vocoder and evaluator; what each drum *is* lives in
an **instrument profile** ([`kicks/instruments/`](kicks/instruments/)). Adding a
drum type means adding a profile, not editing the pipeline.

## How it works

```
Samples (.wav)
  -> strip:  isolate single hits, exclude loops        [profile: onset band, durations]
  -> clean:  quarantine double-hits and outliers       [profile: onset spacing]
  -> peak-safe loudness normalisation (up to -14 LUFS)
  -> shared DisCoder / BigVGAN log-mel spectrograms (128 x 256)
  -> fixed normalisation [-11.51, 3.0] -> [0, 1]
  -> beta-VAE training, cyclical annealing + free bits [profile: transient loss windows]
  -> latent vectors (32-dim)
  -> slider basis: PCA or supervised descriptor axes   [profile: descriptors]
  -> slider values -> inverse transform -> z
  -> VAE decoder -> spectrogram
  -> vocoder (per profile: DisCoder or BigVGAN) -> waveform descriptor correction -> audio
  -> eval:   corpus-referenced perceptual verdicts     [profile: metrics, phrasing]
```

Everything in brackets is what the profile supplies. The stages themselves never
mention a drum type.

## Instruments

| Instrument | Sliders | Corpus | Artefacts |
|---|---|---|---|
| `kick` (default) | Sub, Punch, Click, Bright, Decay | `data/kicks` | `models/`, `output/` |
| `snare` | Body, Crack, Snap, Bright, Decay | `data/snares` | `models/snare/`, `output/snare/` |
| `hihat` | Attack, Body, Sizzle, Bright, Decay | `data/hihats` | `models/hihat/`, `output/hihat/` |

Every command takes `--instrument` / `-i` and derives its paths from the profile,
so `kicks train -i snare` reads `data/snares` and writes `models/snare/` with no
further arguments. `KICKS_INSTRUMENT` sets the default. The kick keeps the
project's original flat paths, so existing checkpoints and caches keep working.

```bash
kicks instruments      # what's registered, and which have a trained model
```

Profiles differ in more than labels. A profile also fixes:

- **descriptor windows** — which mel bands and frames each slider measures,
- **evaluation metrics** — which are scored, how heavily, and the wording of a
  failure. A hi-hat drops `sub_hz` and `pitch_glide` outright (an unpitched
  source has no fundamental to track) and is not penalised for a sustained high
  end, which for a kick is the worst artefact there is,
- **eval time windows** — a hi-hat compresses a kick's windows roughly 4x,
- **onset spacing** — 150 ms for a kick, 45 ms for hi-hat rolls,
- **strip heuristics** — the band the hit lives in, and the band a *different*
  instrument would show up in,
- **transient loss windows** — the band and frames the HF fidelity term guards.

## Setup

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Apple Silicon (MPS), a CUDA GPU, or CPU

```bash
uv sync
```

The website needs Node 22+ and pnpm; see [Website](#website).

### Add training data

Drop `.wav` one-shots into the corpus directory for the instrument
(`data/kicks`, `data/snares`, `data/hihats`). Any sample rate and channel count
works — everything is resampled to 44.1 kHz mono on load.

## Usage

### 1. Prepare the corpus

```bash
kicks strip --dry-run              # preview: what would be trimmed
kicks strip                        # isolate hits (backs up to data/kicks_backup/)
kicks clean                        # preview quarantine decisions
kicks clean --apply                # move loops, double-hits and outliers out
```

`strip` finds the hit in the instrument's own band and truncates at whichever
comes first: its natural decay, or another instrument entering. `clean` then
scans with the same analyzer the evaluator uses and quarantines what is not a
clean one-shot — moved with a manifest, never deleted.

```bash
kicks strip -i hihat               # same commands, hi-hat heuristics
```

### 2. Train

```bash
kicks train                        # kick, 200 epochs
kicks train -i snare -e 300        # snare
```

Two checkpoints are written, because the two things worth optimizing disagree:

- `vae_best.pth` — lowest validation loss. Best reconstruction.
- `vae_best_eval.pth` — best generative eval proxy: latents sampled from the
  validation posterior are decoded and their descriptors compared against the
  corpus distribution. Measures descriptor statistics, not listening quality.

A model can reconstruct beautifully and still generate mush, so which one to
serve is a real choice. Copy the one you want over `vae_best.pth`.

### Training dashboard and experiment notebook

Every new training invocation records a baseline, live progress and epoch KPIs.
Start the local HTML viewer **before training**:

```bash
kicks dashboard                   # http://127.0.0.1:6060
```

In another terminal, describe the experiment as you start it:

```bash
kicks train --model-dir models/experiments/hf-baseline \
  --run-name "HF detail baseline" \
  --intent "Preserve the attack and natural high-frequency decay" \
  --hypothesis "The baseline will identify where detail is lost" \
  --success-criteria "Compare HF error, level-matched listening and slider response"
```

Use the [combined fidelity and tracking plan](docs/high-fidelity-generation.md)
to sequence loss changes, residual/latent skip experiments, audio evaluation
and checkpoint promotion. For fine-tuning, keep candidate weights in a separate
`--model-dir` and record the source checkpoint with `--resume`.

**Before training:** review a comparable run and fill in the objective,
hypothesis and success criteria. These appear in the dashboard's notebook.
**During training:** inspect validation and HF trends alongside latent activity,
beta and learning rate. Update observations with epoch numbers and listening
findings. **After training:** record the decision, supporting waveform/control
reports and the next action. KPI improvement alone does not establish sellable
audio quality.

| Notebook field | Purpose |
|---|---|
| Objective (`--intent`) | The audible problem the run should solve |
| Hypothesis (`--hypothesis`) | The change and expected reason for improvement |
| Success criteria (`--success-criteria`) | Metrics, listening and control checks required for success |
| Observations | Findings during training, including epochs and report paths |
| Decision & next action | Continue, compare, reject or promote, with evidence |

The viewer has interactive charts, optional smoothing, epoch ranges, two-run
comparison, JSON export and editable notes. It records:

- Train loss/reconstruction/KL; fixed-beta, posterior-mean validation loss and
  its reconstruction/KL components.
- Active-reference **log-mel** error in 2–16 kHz and 8–16 kHz, plus the
  profile's HF attack window. These are **before the vocoder**. Active bins lie
  within 60 dB of each reference's peak and above the mel silence floor.
- Active latent dimensions, raw KL, actual learning rate, applied beta,
  periodic descriptor-distribution proxy and epoch/elapsed time.
- Configuration, data/split fingerprint, checkpoint path and lifecycle status.

Training loss uses posterior samples and scheduled beta, so its absolute value
is not directly comparable to the validation curve. Comparisons across runs
also require matching data, split and loss settings; the viewer flags mismatches.
Unavailable KPIs remain empty, including HF error for a silent reference.

Records live in `output/training/<run-id>/`: `run.json`, `notes.json` and a
standalone `index.html` with sibling data scripts. Open the HTML directly for
a live read-only view, or use the server to edit notes and compare runs. The
viewer needs no external services, CDN assets or model loading.
`--runs-dir` / `KICKS_RUNS_DIR` override the shared records root;
`kicks dashboard --port 6061` changes the viewer port. Records survive completed,
failed and interrupted training. A hard-killed process may leave a running
record; the viewer shows when updates have become stale.

Tracking starts when the training loop begins, after corpus loading. Processes
already running before this integration retain their previously imported code;
their missing history is not reconstructed or invented. The tracking hooks
apply when the next training process starts.

### Loss and architecture experiments

The shipped objective and network are the defaults. Each experiment is one flag
away and is recorded in the run so the dashboard can tell matched runs apart:

```bash
# Stage 2 — loss only. Symmetric, reference-weighted HF detail over the whole
# hit, and frame-to-frame change matching over the attack. Unweighted terms are
# logged per epoch (Training dynamics ▸ "Val · HF detail term", …).
kicks train --hf-detail-weight 0.5 --attack-change-weight 0.25 --model-dir models/experiments/loss   --run-name "Loss · HF detail 0.5 / attack 0.25" --intent ... --hypothesis ... --success-criteria ...

# Stage 3 — architecture only. Residual blocks after every stage and the latent
# injected (FiLM) at every decoder scale; both start as the identity. Recorded
# in the checkpoint's `architecture` block and loaded back automatically.
kicks train --residual --latent-skips --latent-dim 64 --model-dir models/experiments/arch ...
```

### Waveform evidence and promotion

```bash
kicks fidelity --run <run-id> --checkpoint models/experiments/loss/vae_best.pth
```

renders the run's own held-out validation hits three ways — the reference, the
reference's real mel through the vocoder (the backend's ceiling), and the VAE's
reconstruction through the vocoder — level-matches them, and reports 2–8 kHz and
8–16 kHz attack/body error, onset timing, envelope error, spectral flatness and
unwanted late energy, plus realism scores for fresh generations. It writes
randomised blind A/B pairs (whole hit and 2 kHz+ band) with a small
`listening/listening.html` tally page, and attaches the summary to the run's
**Evidence** card. `scripts/validate_controls.py --run <run-id>` attaches the
slider audit the same way.

```bash
kicks promote -i kick --run <run-id> --checkpoint models/experiments/loss/vae_best.pth   --decision "8–16 kHz body error −1.4 dB vs baseline, 2/16 blind rejections, controls pass 100%"
```

copies the candidate over the served checkpoint (previous kept as
`vae_best_prev.pth`, calibration sidecar carried along), and refuses without a
written decision and both a fidelity and a controls report on the run —
`--allow-missing-evidence` overrides that and says so in the record. The
checkpoint's sha256 lands in the notebook, so a served model always traces back
to its experiment.

### 3. Synthesize

**Server** (REST API at `http://localhost:8080`):

```bash
kicks serve                        # each instrument renders with its profile's vocoder
kicks serve -i snare               # snare
kicks serve --vocoder bigvgan      # force one neural backend for every instrument
kicks serve --griffin-lim          # CPU-only, no model download
kicks serve --control pca          # alternative latent PCA controls
```

Instruments load lazily, so one server can serve all three once trained — the
website's instrument tabs just call `/config?instrument=...`.

### Vocoders

Each profile names the mel-to-audio backend it renders best with
(`InstrumentProfile.vocoder`). The choice is measured, not aesthetic: on the
85-render control audit, [ETH DISCO's DisCoder](https://github.com/ETH-DISCO/discoder)
scores 98.3 mean / 93.6 min on kicks (BigVGAN: 98.0 / 89.1) and 97.2 / 84.1 on
snares (96.8 / 86.4), at roughly half BigVGAN's render time on Apple Silicon;
on hi-hats it drops to 94.5 / 48.5 with three failing renders (BigVGAN: 95.4 /
70.7, none), so the hi-hat profile keeps BigVGAN. `KICKS_VOCODER` or
`--vocoder` forces one backend for every instrument.

DisCoder runs from the [official 44.1 kHz Z checkpoint](https://huggingface.co/disco-eth/discoder)
with its fine-tuned DAC decoder. Its 128-band mel representation matches this
project's existing VAE checkpoints, so switching vocoders does not require
retraining the VAE.

The first DisCoder load downloads `config.json` and the 1.72 GB `model.pt` to
`models/discoder/` (revision `6505384d8fd5f18338f171dd81dc10c9a0d34fe9`).
Set `KICKS_DISCODER_DIR` to use a different local checkpoint directory. Loading
checks the mel parameters and every model weight; incompatible checkpoints fail
explicitly. The inference adapter's upstream MIT notice is included in
[`kicks/nn/DISCODER_LICENSE`](kicks/nn/DISCODER_LICENSE).

Weights load through a memory-mapped checkpoint, and batch generation renders
one DisCoder hit at a time to limit activation memory on smaller machines.

BigVGAN picks up instrument-specific fine-tuned weights from the profile's
`vocoder_dir` (`models/vocoder/checkpoint_100.pth` for the kick). `--vocoder
griffinlim` uses classical reconstruction without a model download;
`--griffin-lim` remains an alias.

**Batch:**

```bash
kicks generate -n 20 -k 8          # 20 outputs, best of 8 candidates each
kicks eval                         # score them against the corpus
kicks sweep -n 40                  # sweep the API's slider space and score every result
```

`generate` samples from a GMM fitted to the corpus's own latents rather than
`N(0, 1)` — standard-normal draws land off the data manifold, which is where the
blobs and double-onsets come from — then keeps the best-scoring of *k* decoded
candidates per output slot.

### 4. Corpus analysis

```bash
kicks cluster                      # GMM clustering + descriptor PCA -> JSON report
kicks cluster -i snare && kicks cluster -i hihat
kicks publish-analysis             # -> web/public/analysis/ for the website
```

`cluster` writes `output/<instrument>/cluster_analysis.json` plus one averaged
`.wav` per cluster, so the clusters can be listened to rather than only read.
`publish-analysis` rewrites those reports for the browser: filenames and paths
are dropped (the site shows the shape of the corpus, never which recordings are
in it), GMM memberships collapse to one confidence, floats are rounded.

### 5. Docker

```bash
docker compose up --build          # API on :8080
```

## Website

`web/` is a Next.js 16 + shadcn/ui site, statically exported and deployed to
GitHub Pages by `.github/workflows/pages.yml`. It is a pure client of the API.

```bash
cd web
cp .env.example .env.local         # point NEXT_PUBLIC_KICKS_API_URL at a running `kicks serve`
pnpm install
pnpm dev                           # http://localhost:3000
pnpm build                         # static export -> web/out
```

| Route | What it is |
|---|---|
| `/studio/` | Sliders for every trained instrument, post-vocoder shaping, waveform + spectrogram, corpus-referenced realism check, and a 4×4 pad bank playable from the keyboard, the pointer or a MIDI controller (Web MIDI, MIDI learn, velocity). Kits persist locally and, signed in, in the account. |
| `/analysis/` | Corpus analytics per instrument: PC scatter with cluster isolation, cluster cards with averaged audio, PCA loadings, descriptor correlations, distributions, and a table view. |
| `/pricing/` | Credit packs from the Stripe catalogue mirror. |
| `/login/`, `/account/` | Google / GitHub OAuth and e-mail magic links via Supabase; credits, orders, exports, data export and account deletion. |
| `/legal/*` | Impressum, Datenschutzerklärung, AGB, Widerrufsbelehrung (German). |

### Accounts, credits and payments

Previews are free for signed-in users. `POST /export` renders a 24-bit WAV and
spends one credit; new accounts start with three. The pieces:

- **Supabase** (`supabase/`): Postgres schema with row-level security on every
  table, an append-only credit ledger, a Stripe catalogue mirror, orders, saved
  kits and consent records. Money and credits are only written through
  `SECURITY DEFINER` functions callable by the service role (`charge_export`,
  `refund_export`, `fulfill_checkout`, `refund_order`, `erase_user`); users
  read their own rows and call `credit_balance` / `export_own_data`.
- **Edge Functions**: `create-checkout-session` (Stripe Checkout with automatic
  tax, invoice creation, tax-ID collection and the § 356(5) BGB notice),
  `stripe-webhook` (signature-verified, idempotent on the event id), `billing-portal`,
  `delete-account` (removes the Stripe customer, then erases the user).
- **The API** verifies Supabase access tokens locally (JWKS, or a legacy HS256
  secret) and charges credits through PostgREST as the service role.

Setup, in order:

1. `supabase link --project-ref <ref>` (pick an EU region), `supabase db push`.
2. Enable Google and GitHub in Authentication → Providers; add
   `https://<user>.github.io/<repo>/auth/callback/` and
   `http://localhost:3000/auth/callback/` to the redirect allow-list. For magic
   links that survive being opened in another browser, use `{{ .TokenHash }}` in
   the e-mail template: `.../auth/callback/?token_hash={{ .TokenHash }}&type=email`.
3. In Stripe: complete the business profile (address, VAT ID — the invoices
   carry them), enable Stripe Tax, create one product per credit pack with a
   `credits` metadata field and a one-time EUR price (gross, tax-inclusive).
   Add a webhook endpoint for the events listed in `stripe-webhook/index.ts`.
   Re-save each product once so the webhook mirrors it.
4. `cp supabase/functions/.env.example supabase/functions/.env`, fill it in,
   `supabase secrets set --env-file supabase/functions/.env`,
   `supabase functions deploy`.
5. Run the API with `KICKS_SUPABASE_URL` and `KICKS_SUPABASE_SERVICE_KEY` set,
   and `KICKS_CORS_ORIGINS` including the site's origin.
6. Set the repository variables the Pages workflow reads (`KICKS_API_URL`,
   `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `CF_ANALYTICS_TOKEN`, `LEGAL_*`).

### German law checklist

The site is built for an operator in Germany. What is in place, and what still
needs a human:

- **Impressum** (§ 5 DDG) and **Datenschutzerklärung** (Art. 13 DSGVO) are
  generated from the `LEGAL_*` variables; the pages show a warning until they
  are set. Have both reviewed by a lawyer before going live — they are a
  faithful starting point, not legal advice.
- **Consent** (§ 25 TDDDG): Cloudflare Web Analytics loads only after an opt-in
  with equal-prominence "only necessary" / "allow analytics" buttons; sign-in
  tokens are strictly necessary and need none. Fonts are self-hosted.
- **Commerce**: prices are shown gross "inkl. MwSt." (PAngV), Stripe Tax
  charges the buyer's rate, invoices are created for every purchase (§ 14
  UStG), the AGB and Widerrufsbelehrung are accepted before checkout, the
  § 356(5) BGB waiver is a separate un-ticked checkbox and is logged with
  the legal-text version, and Stripe's "Zahlen" button is the § 312j order
  button.
- **Retention and erasure**: deleting an account removes the Stripe customer
  and every user row, but detaches rather than deletes orders (§ 147 AO).
  `export_own_data` covers Art. 15 and 20.
- The EU ODR platform was discontinued in July 2025; the pages reference § 36
  VSBG only.

## Slider bases

`--control descriptor` is the default. Each slider targets its own gain-invariant
measurement: spectral power ratios in dB, or energy-weighted duration in ms.
A bounded nonlinear solve stays within the corpus's latent region, followed by
feedback from the actual vocoded waveform to cancel reconstruction drift.
The ranges are calibrated against the checkpoint and corpus and cached in
`<checkpoint>.controls.npz`; changed weights, descriptors or data trigger a refit.

Independence is measured against these descriptors within the calibrated ranges.
It does not imply that every listener perceives every sound attribute as
independent. Explicit envelope, drive and filter effects intentionally change
the resulting measurements.

`--control pca` fits principal components of the corpus latents and
names each after the descriptor it correlates with most (|r| >= 0.15), flipping
sign so every slider reads left-to-right as "less" to "more". Honest about the
data's own structure, but each component moves several descriptors at once, so
one axis — the profile's `decorrelated_descriptor`, decay by default — gets
explicit cross-talk compensation.

## Evaluation

`kicks eval` defines "good" statistically rather than by fixed thresholds: a
generated hit passes when each perceptual metric falls inside the distribution
of the same metric over the real corpus. Per-metric robust z-scores become ✓/⚠/✗
verdicts in plain English, a Mahalanobis distance gives an overall likeness
percentile, and a Fréchet distance between the generated set and the corpus gives
one set-level number.

Metrics are computed on the final waveform — post-vocoder — because that is what
the listener hears and where vocoder artefacts actually live. The module imports
only numpy and scipy, so it starts in well under a second.

Compare actual corpus reconstructions, including a local audio comparison page:

```bash
python scripts/compare_vocoders.py --count 16 --checkpoint models/vae_best.pth \
  --out output/reconstruction
python scripts/validate_controls.py --checkpoint models/vae_best.pth \
  --corners --random 32 --out output/kick-controls      # --vocoder to override the profile
```

The second audit measures all slider axes, corners and independent random
combinations on final audio. Its report includes target error, cross-talk,
corpus realism scores and render time. Reconstruction errors and corpus realism
scores answer different questions; neither replaces listening.

To fine-tune a VAE in a separate candidate directory:

```bash
kicks train --resume models/vae_best.pth --model-dir output/candidate/models \
  --epochs 12 --learning-rate 0.00003 --beta 0.001 --preview 0
```

Training evaluates the original checkpoint before updating the best candidate,
using a fixed validation split and posterior means for repeatable comparisons.

## Project structure

```
kicks/
  instruments/       Instrument profiles — the only place a drum type is named
    profile.py         Region, DescriptorSpec, MetricSpec, OnsetSpec, StripSpec,
                       EvalWindows, TransientLossSpec, PathSpec, InstrumentProfile
    metrics.py         The standard metric set, phrased for a given instrument
    kick.py  snare.py  hihat.py
    __init__.py        Registry: get_profile(), available(), register()
  audio/             Signal processing
    constants.py       Sample rate, FFT/mel config, normalisation bounds
    waveform.py        numpy-only: loading, envelopes, onsets, loop detection, STFT
    io.py              torch-side loading: mono, resample, fit length, LUFS
    mel.py             BigVGAN log-mel + [0,1] normalisation
    effects.py         Envelope shaper, drive, lowpass (API query params)
    vocoder.py         DisCoder, BigVGAN and Griffin-Lim backends
  data/              DrumDataset, load_spectrogram
  nn/                VAE and the DisCoder inference adapter
  training/          loss.py, trainer.py
  analysis/          descriptors, latents, basis (sliders), evaluation, clustering
  synthesis/         generator.py — latent prior + best-of-k selection
  corpus/            strip.py, clean.py
  api/               app.py (endpoints), auth.py (Supabase tokens + credits), state.py, middleware.py
  cli.py  config.py  sweep.py
web/                 Next.js + shadcn/ui site (static export, GitHub Pages)
supabase/            Schema migration and Stripe Edge Functions
```

The dependency direction is one-way: `instruments/` depends on nothing but
`audio/constants`, and everything else depends on `instruments/`.

## API

Base URL `http://localhost:8080`. Every endpoint takes an optional `instrument`.

| Endpoint | Description |
|---|---|
| `GET /health` | Device, vocoder, control basis, loaded instruments, auth mode |
| `GET /instruments` | Registered instruments and whether each is trained |
| `GET /config` | Slider definitions for one instrument |
| `GET /generate` | Preview WAV for the given slider settings (free, cached) |
| `GET /evaluate` | Perceptual verdicts + descriptors for the same settings |
| `GET /spectrogram` | The decoded spectrogram, pre-vocoder |
| `GET /me` | The signed-in caller and their credit balance |
| `POST /export` | 24-bit WAV download; spends one credit. `Idempotency-Key: <uuid>` makes retries safe |

When `KICKS_SUPABASE_URL` is set, every synthesis endpoint expects
`Authorization: Bearer <supabase access token>` (`KICKS_AUTH_MODE=optional`
relaxes that for previews). Without it the server is open and `/export` is
unavailable.

Sliders accept three spellings: positional `s1..sN`, legacy `pc1..pcN`, or the
slider's own label (`?click=0.8&sub=0.3`). Anything omitted defaults to centre.
`/generate` also accepts `attack_ms`, `decay_ms`, `drive` (0-1) and `filter`
(lowpass cutoff in Hz). `/evaluate` scores exactly the audio `/generate` returns,
shaping included.

```bash
curl "localhost:8080/generate?click=0.8&decay=0.2" -o kick.wav
curl "localhost:8080/evaluate?instrument=snare&crack=0.9" | jq .grade
```

Rate limited to 10 req/s (token bucket, shared) with a 100-entry LRU cache keyed
on the query string. CORS is a single origin by default, set via
`KICKS_CORS_ORIGINS`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `KICKS_INSTRUMENT` | `kick` | Default instrument |
| `KICKS_DATA_DIR` | `data` | Corpus root |
| `KICKS_MODEL_DIR` | `models` | Checkpoint root |
| `KICKS_OUTPUT_DIR` | `output` | Output root |
| `KICKS_RUNS_DIR` | `<KICKS_OUTPUT_DIR>/training` | Shared training records and notebooks |
| `KICKS_VOCODER` | unset (each profile's own) | Force `discoder`, `bigvgan` or `griffinlim` for every instrument |
| `KICKS_DISCODER_DIR` | `<KICKS_MODEL_DIR>/discoder` | Local DisCoder config and weights |
| `KICKS_CONTROL` | `descriptor` | `descriptor` or `pca` |
| `KICKS_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated |
| `KICKS_SUPABASE_URL` | unset | Enables accounts; tokens verified via JWKS |
| `KICKS_SUPABASE_SERVICE_KEY` | unset | Enables `/export` credit charging |
| `KICKS_SUPABASE_JWT_SECRET` | unset | Legacy HS256 projects only |
| `KICKS_AUTH_MODE` | `required` if URL set, else `off` | `required`, `optional`, `off` |

## Adding an instrument

1. Copy `kicks/instruments/snare.py` and adjust the descriptor regions, metric
   overrides, eval windows, onset spacing and strip bands.
2. Register it in `kicks/instruments/__init__.py`.
3. `kicks strip -i yours && kicks clean -i yours --apply && kicks train -i yours`.

No other module needs to change.

## Model

2D convolutional beta-VAE on `(B, 1, 128, 256)` log-mel spectrograms, four
stride-2 stages (32/64/128/256 channels) down to an 8x16 bottleneck, mirrored
transposed-conv decoder with a sigmoid output. Latent dim is recorded in the
checkpoint and auto-detected on load. Spectrogram size is a constructor argument,
so a short-tail instrument can train on fewer frames without a second model class.

Loss: multi-resolution frequency-weighted reconstruction (spectral convergence +
energy-weighted L1 at scales 1, 2, 4) + beta * KL with per-dimension free bits,
plus the profile's transient fidelity term.

## Key design decisions

- **BigVGAN's `mel_spectrogram()` for the dataset** — the VAE learns exactly the
  representation the vocoder was trained to invert. `n_fft` is 1024, not 2048,
  and must match across the constants, the dataset and the vocoder.
- **Fixed normalisation bounds**, not dataset min/max — the model's output scale
  is tied to them, so a corpus change must not silently rescale the target.
- **Tail gating** — neural vocoders leave a ~-90 dBFS noise floor where real
  samples decay into true silence; the exposed hiss is the most audible artefact,
  so the tail is faded to digital zero.
- **`weights_only=True`** on every `torch.load`.
- **Griffin-Lim via mel-filterbank pseudo-inverse**, not `InverseMelScale`, which
  is unsupported on MPS and rank-unstable on CPU.

## License

MIT — see [LICENSE](LICENSE).
