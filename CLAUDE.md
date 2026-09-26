# Repository instructions

## Scope and source of truth

- Project: profile-driven drum synthesis, VAE training, audio evaluation, a FastAPI backend, a static Next.js studio, and optional Supabase/Stripe accounts.
- Human setup and product guide: `README.md`. Experiment sequence and research rationale: `docs/high-fidelity-generation.md`. Snare/hi-hat identity controls and their measured audit: `docs/audio-identity-plan.md`. Waveform diffusion backend, its 8 GB Apple Silicon measurements and known issues: `docs/waveform-diffusion.md`.
- Read current code and `git diff` before changing behavior. Local corpora, checkpoints and active jobs can change independently of Git; inspect them before making claims about the current model.
- Preserve unrelated work. Do not restart an existing training process to attach telemetry.
- Corpus identity and licensing after the numeric rename are anchored by `data/corpus_ids.json` and the three local `data/{kicks,snares,hihats}/SOURCES.md` files. Preserve them. The Modal `kicks-corpus` volume intentionally contains WAVs only; remote verification must require exact names, sizes and SHA-256 hashes but must not require or upload `SOURCES.md`.
- For files under `web/`, read `web/AGENTS.md` and `web/CLAUDE.md`; consult the installed Next.js docs they specify before editing application code.
- Implemented: HF/attack loss experiments, residual/latent options, soft log-variance bound, beta schedule controls, tracking, fidelity/listening reports, control audits and promotion records.
- Implemented and first-trained: the descriptor-conditioned waveform diffusion backend (issue #3), merged 2026-09-19, trained on hi-hat subsets of 256 and 1,000 hits on 2026-09-20 (checkpoints under `models/experiments/diffusion-hihat-*`, runs in the dashboard, numbers in `docs/waveform-diffusion.md` → *First runs*). Blind listening is prepared (`scripts/diffusion_listening.py`, report `hihat-diffusion-1000-ab`) but unjudged; nothing is served, and `kicks promote` does not know this backend. Do not describe it as a working alternative to the VAE, and do not compare the two beyond the measurements recorded. On this 8 GB Apple Silicon machine train on stratified subsets (`scripts/make_subset.py`) with `--batch-size 2 --grad-accum 4`, launched detached (`nohup … &`, verify the parent is `launchd`) so a session ending cannot kill it. Training phases wait for the user's go, each preceded by a summary of the previous phase.
- Not implemented: the proposed codec-latent sequence prior and waveform-loss training through the vocoder. Do not describe them as shipped features.

## Commands

Run Python commands from the repository root. Use `uv run` to select the project environment.

```bash
uv sync
uv run kicks --help
uv run kicks instruments
uv run kicks dashboard                       # Loopback viewer :6060
uv run kicks serve                           # API :8080
uv run kicks serve --vocoder bigvgan          # Override all instrument defaults
uv run kicks serve --griffin-lim              # No neural vocoder download; VAE still required
uv run kicks strip -i kick --dry-run
uv run kicks clean -i kick                    # Preview; --apply moves rejected files
uv run kicks generate -i kick -n 20 -k 8
uv run python scripts/make_subset.py --instrument hihat --size 2000 --seed 42   # stratified, nested subsets
uv run kicks diffusion-train -i hihat --data data/_subsets/hihat-2000 \
  --model-dir models/experiments/diffusion \
  --batch-size 2 --grad-accum 4            # 8 GB Apple Silicon; the default batch of 8 thrashes MPS
uv run kicks diffusion-generate -i hihat -n 8 --steps 50 --target decay=40
uv run --script scripts/modal_train.py check-local                # local-only corpus/license verification
uv run --script scripts/modal_train.py verify --confirm-cloud     # read-only Modal WAV verification
uv run --script scripts/modal_train.py image-test --confirm-cloud # reviewed container/protobuf gate
uv run python scripts/diffusion_listening.py --instrument hihat --checkpoint models/experiments/diffusion-hihat-1000-b/hihat/diffusion_best.pth \
  --run 20260920T124135 --out output/fidelity/hihat-diffusion-1000-ab   # blind generation A/B pairs for the Listening lab
uv run kicks eval -i kick
uv run kicks sweep -i kick -n 40              # Requires the API
uv run kicks cluster -i kick
uv run kicks publish-analysis                # Writes public browser assets
uv run pytest -q
```

Instrument-aware commands use `-i` / `--instrument`; commands such as `dashboard`, `instruments` and `publish-analysis` do not take that option. Script arguments differ: `scripts/validate_controls.py` uses `--instrument`, not `-i`.

Website, from `web/`: `pnpm install`, `pnpm dev` (:3000), `pnpm lint`, `pnpm build` (static export to `out/`). Use a static server for the production export; do not rely on `next start` for it.

`docker compose up --build` serves the API with mounted data/models/output. The supplied Compose file reserves an NVIDIA GPU; adapt that reservation for other hosts.

## Required training workflow

For remote waveform-diffusion work, follow `docs/modal-training.md`. Each cloud
operation is a separate review gate. `plan` is local-only; `launch` is detached
and must have a current `output/modal/corpus-verification.json` receipt. Corpus
data is mounted read-only and results are written to `kicks-training`; `sync`
preserves existing local `notes.json` and `notes.js`. `live` mirrors one job's
run record into the local dashboard while it trains (read-only, no checkpoints);
its lag is the launch's `--commit-interval-seconds` plus one poll.

### Before training

1. Check the active processes, available device, source checkpoint metadata and corpus. Use the existing dashboard or start `uv run kicks dashboard`.
2. Review a comparable baseline. Fix the corpus, split seed, preprocessing, vocoder and evaluation settings; isolate the change under test.
3. Supply `--run-name`, `--intent`, `--hypothesis`, `--success-criteria`. State the audible issue and measurable fidelity, listening and control criteria. Choose actual tolerances before the experiment.
4. Set a separate `--model-dir` for candidate VAE weights. It is a root: a snare candidate adds `snare/`, a hi-hat candidate adds `hihat/`, a kick uses the root directly.
5. For fine-tuning, pass `--resume` and record the source. It loads model weights with a fresh optimizer, not the old optimizer/scheduler state.
6. Use `--preview 0` when collecting rendered evidence separately. Otherwise previews run after the tracked training loop and can load/download a vocoder.

Example (settings are an experiment, not established winning values):

```bash
uv run kicks train -i kick \
  --resume models/vae_best.pth \
  --model-dir models/experiments/hf-detail \
  --epochs 12 --learning-rate 0.00003 --beta 0.001 --preview 0 \
  --hf-detail-weight 0.5 \
  --run-name "Kick · HF detail ablation" \
  --intent "Preserve audible high-frequency detail" \
  --hypothesis "Symmetric reference-weighted HF loss reduces missing detail" \
  --success-criteria "Compare HF errors, blind listening and control errors with a matched baseline"
```

### CLI training defaults and experimental options

| Option | Default | Contract |
|---|---|---|
| `--epochs` | `200` | Python `train()` has its own defaults; do not confuse them with CLI defaults |
| `--latent-dim` | Profile value, currently `32` | Resumed checkpoints retain their stored shape |
| `--batch-size` | `32` | Training and validation batch size |
| `--learning-rate` | `1e-3` new / `1e-4` resume | Adam; cosine LR schedule spans `--epochs` |
| `--beta` / `--free-bits` | `0.02` / `0.2` | Validation uses fixed target beta; free bits are per-dimension nats |
| `--beta-cycles` | `4` | Cyclical beta annealing for a new model |
| `--beta-anneal-epochs` | `--epochs` | Override beta's horizon for short screening runs; does not extend the LR horizon |
| `--beta-floor` | `0.0` | Fraction of target beta, in `[0,1]`; e.g. `.1 × .02 = .002` minimum during annealing |
| `--hf-detail-weight` | `0.0` | Symmetric reference-weighted HF loss over the hit |
| `--attack-change-weight` | `0.0` | Frame-to-frame change matching over the profile's click window |
| `--residual` | Off | Residual blocks in encoder and intermediate decoder stages |
| `--latent-skips` | Off | Latent FiLM scale/shift at decoder feature scales |
| `--soft-logvar` | Off | `10 * tanh(raw / 10)` instead of hard `[-10,10]` clipping; no added parameters |
| `--seed` | `42` | Model initialization and deterministic split/shuffle setup |
| `--preview` | `10` | Post-training sample previews; `0` skips |

- `--resume` rejects `--residual`, `--latent-skips` and `--soft-logvar`; these options are loaded from checkpoint architecture metadata. Fine-tuning uses fixed beta, ignoring the annealing horizon/floor.
- Residual and FiLM additions initialize as identity operations. Equal full-network initialization requires matching shared weights; the same seed alone does not guarantee that across different module layouts.
- A short screen with `--beta-anneal-epochs 200 --epochs 20` preserves the first 20 epochs of the beta schedule only. Record the different LR schedule when comparing runs.
- `train --model-dir` pins `KICKS_DISCODER_DIR` and `KICKS_VOCODER_DIR` to the pre-override model root when unset, then changes `KICKS_MODEL_DIR` for candidate weights. Explicit vocoder-directory overrides win; avoid duplicate downloads per experiment.

### During training

- Inspect epoch zero and revisit meaningful intervals, e.g. every five epochs or before a decision. Watch validation, HF/attack errors, raw KL, active dimensions, beta, LR and loss components together.
- Update `observations` with epoch references, measured changes, listening findings and report paths. Record uncertainty and failed comparisons.
- Never edit metrics to improve a curve. Never fabricate missing history or infer listening quality from a proxy.
- Do not promote solely because a scalar improved. Keep the decision field current with evidence and outstanding checks.

### After training: audio evidence

Evaluate the exact candidate with the intended instrument, corpus and vocoder. `--run` selects the split and report destination; it does **not** infer the candidate checkpoint, instrument or custom corpus path. Pass those explicitly when needed.

```bash
uv run kicks fidelity -i kick --run YOUR_RUN_ID \
  --checkpoint models/experiments/hf-detail/vae_best.pth \
  --out output/audits/hf-detail
uv run python scripts/validate_controls.py --instrument kick --run YOUR_RUN_ID \
  --checkpoint models/experiments/hf-detail/vae_best.pth \
  --corners --random 32 --out output/audits/hf-detail-controls
```

- Fidelity renders reference / real-mel vocoder / VAE-vocoder triples; RMS-matches them with one shared anti-clip gain; measures HF, onset, envelope, flatness and late energy; also evaluates fresh prior generations.
- Open the resulting `listening/listening.html`: randomized reference-vs-VAE pairs, whole hit and 2 kHz+ band. Its rejection tally is not persisted automatically. Copy findings and counts into the run notebook.
- With `--run`, the saved seed/ratio and corpus fingerprint must reproduce the validation split. On mismatch or missing identity, the command warns and samples the whole corpus. Check `held_out`, `hit_origin`, `hits`; do not call a fallback held out.
- Without `--run`, `--split-seed ORIGINAL_SEED --val-split ORIGINAL_RATIO` recreates a legacy split. This assumes the original corpus/order is unchanged; it has no historical fingerprint verification. `--seed` separately controls subset selection and A/B ordering. When a run is supplied, its split wins.
- The split fingerprint includes seed, ratio, dataset length and ordered resolved paths/sizes/mtimes. It is not a byte-content hash and does not detect related samples or pack leakage. Preserve this limitation in reports.
- Controls audit: `--checkpoint` and `--out` are required; `--corners` adds corner probes, `--random` controls random probes, `--run` attaches evidence. Dataset selection follows the instrument profile/environment.

### Promotion

```bash
uv run kicks promote -i kick --run YOUR_RUN_ID \
  --checkpoint models/experiments/hf-detail/vae_best.pth
```

- Write the decision in the notebook first, or pass `--decision`. Review listening and both reports against the predeclared criteria.
- The current gate requires a decision, matching run/instrument metadata and attached `fidelity` + `controls` report kinds. It does **not** judge metric thresholds or bind all report/checkpoint/run identities. Verify the candidate path/hash and report provenance yourself.
- `--allow-missing-evidence` bypasses only missing evidence and records the missing kinds. Use only within the user's requested scope and explain the absent evidence in the decision.
- Promotion copies to `profile.paths.checkpoint`, saves the prior weights as `vae_best_prev.pth`, carries an available `<stem>.controls.npz`, and writes the new SHA-256 and paths to the run.
- Check the destination root/environment and use a separate candidate path. Restart the serving process only when that operation is within the task's authorization; model instances remain cached until reloaded. Calibration detects stale sidecars and refits on first use.

## Tracking records and metric semantics

Default root: `<KICKS_OUTPUT_DIR>/training` (normally `output/training`). Override with `KICKS_RUNS_DIR` or `--runs-dir`; the writer and viewer must use the same root.

| Artifact / API | Contract |
|---|---|
| `run.json`, `data.js` | Training writer owns metrics/config/history/lifecycle; atomic snapshots |
| `notes.json`, `notes.js` | Prose is separate so telemetry cannot overwrite browser edits |
| `reports.json`, `reports.js` | Attached evidence summaries, identities and local report paths |
| `index.html` | Standalone live read-only view using sibling scripts |
| `GET /api/runs` | List run summaries, each with a `trend`: validation loss thinned to at most 48 points for the run-list sparkline |
| `GET /api/runs/<id>` | Full run, notebook and evidence |
| `POST /api/runs/<id>/notes` | Partial JSON update of string-valued notebook fields |
| `GET /api/capabilities` | Viewer version and feature flags; the template degrades to charts-only against an older server |
| `GET /api/fidelity`, `GET /api/fidelity/<slug>` | Listening-lab reports: `<output>/fidelity/*/report.json` plus every `fidelity` report attached to a run; the detail strips the blind key's `a_is`/`b_is` |
| `GET /api/fidelity/<slug>/key` | The blind assignment, only when the listener reveals it |
| `GET /api/fidelity/<slug>/audio/<name>` | One WAV from that report's `listening/`, `reconstruction/` or `generation/` directory only; name allow-listed, path-traversal refused, 256 MB cap |
| `POST /api/fidelity/<slug>/verdicts` | `{pair, band: full\|hf, choice: a\|b\|tie\|reject\|null}` merged into `listening/verdicts.json`, 64 KB cap |

Notebook keys: `objective`, `hypothesis`, `success_criteria`, `observations`, `decision`. Initial CLI prose maps `--intent` to `objective`. `attach_report()` is the Python evidence API; `find_run()` accepts a full ID or unique prefix. The HTTP API uses full IDs.

- Tracking wraps `training.train()` after corpus loading. It records baseline epoch `0`, live progress, completed epochs, and completed/failed/interrupted status. Hard kills may leave `running`; the viewer marks stale updates. Post-training CLI previews are outside this lifecycle.
- Train loss: sampled posterior and scheduled beta. Validation: posterior mean and fixed target beta. Do not equate their absolute values.
- `hf_mae_db`, `air_mae_db`, `attack_mae_db`: pre-vocoder log-mel errors; activity is within 60 dB of each reference peak and above the silence floor. Bands are 2–16 kHz, 8–16 kHz, and the profile's HF click window. Unmeasurable values are null.
- `train_term_*` / `val_term_*`: unweighted loss components. Compare weighted totals only with matching weights/objective.
- `active_dims`: per-dimension mean raw KL > 0.01 nats; `raw_kl` is unclamped to free bits. Activity does not establish fidelity.
- `eval_proxy`: descriptor-distribution mismatch, lower is better; not a perceptual quality rating.
- Waveform diffusion runs use the same writer and viewer, keyed by `config.backend`. `train_loss` draws noise levels at random with conditioning dropout; `val_loss` holds each hit at a fixed level with fixed noise and full conditioning, so the two are not comparable. `val_loss_{low,mid,high}_sigma` split that error by noise level. `control_mae` is the descriptor-target error of freshly sampled hits in training-set standard deviations — a controllability proxy, measured on a few samples at reduced step count, not a perceptual rating and not a substitute for `scripts/validate_controls.py`. The viewer refuses to treat a VAE run and a diffusion run as a like-for-like comparison.
- Fidelity waveform errors use a 512-point Hann STFT, hop 128, reference bins above −70 dBFS and profile attack/body windows. Keep them distinct from the live mel KPIs and corpus realism scores.

## Architecture map

| Location | Responsibility / entry points |
|---|---|
| `instruments/` | `InstrumentProfile`, descriptors, metric weights, windows, onsets, strip rules, paths, waveform-control masks, vocoder; `get_profile`, registry |
| `audio/constants.py` | Shared signal contract |
| `audio/waveform.py` | NumPy/SciPy loading, envelopes, filters, onsets, STFT; torch-free |
| `audio/io.py`, `data/dataset.py` | Shared waveform ingestion and `DrumDataset`; same preprocessing for dataset and prior |
| `data/waveforms.py` | `WaveformDataset`: the same ingestion plus a fixed peak, with descriptor labels measured on the returned waveform |
| `audio/mel.py`, `audio/vocoder.py` | Fixed mel normalization; DisCoder / BigVGAN / Griffin-Lim adapters |
| `audio/controls.py`, `audio/effects.py` | Bounded STFT waveform correction, then user effects |
| `nn/vae.py`, `config.py` | VAE options, device selection (CUDA > MPS > CPU), metadata-aware checkpoint loading |
| `nn/diffusion.py` | `WaveformUNet`, the angular v-prediction schedule, FiLM/null conditioning and the deterministic `v_sample` |
| `nn/discoder.py` | Inference port plus upstream `DISCODER_LICENSE`; fine-tuned DAC decoder from `descript-audio-codec` |
| `analysis/descriptors.py`, `analysis/basis.py` | Descriptor measurements; PCA or closed-loop descriptor solve |
| `analysis/calibration.py` | Fingerprinted descriptor basis and calibrated slider ranges |
| `analysis/evaluation.py` | Corpus reference, waveform metrics, robust scores, set-level distance; torch-free |
| `analysis/fidelity.py` | Matched waveform reports and blind pairs; metric helpers torch-free, rendering imports torch lazily |
| `analysis/clustering.py`, `analysis/publish.py` | Corpus report, cluster audio, sanitized browser publication |
| `training/loss.py`, `training/trainer.py` | Base/experimental objectives, baseline, schedules, checkpoint selection |
| `training/diffusion.py` | v-prediction objective, conditioning dropout, EMA, fixed-level validation and the descriptor-target proxy |
| `training/tracking.py`, `training/dashboard.html` | Standard-library loopback server, lifecycle, notes, evidence and local charts |
| `training/promotion.py` | Candidate copy and recorded decision; `PromotionRefused` |
| `synthesis/generator.py` | Corpus-fitted GMM prior and best-of-k generation |
| `synthesis/identity.py`, `synthesis/controlled.py` | Corpus-anchored `IdentityBasis` (radius 4, ±6 dB correction cap, `.identity.npz` cache) and the single rendering path shared by studio previews, exports and the control audit |
| `synthesis/diffusion.py` | Descriptor targets resampled from the checkpoint's label bank (nearest rows when pinned); sampling with separate texture and slider seeds |
| `corpus/` | Backed-up stripping; quarantine with a manifest |
| `api/` | FastAPI routes, lazy instrument/backend caches, auth, credits, rate limiting |
| `cli.py`, `sweep.py` | Lazy command implementations; API control-space sweep |

Paths in the table are relative to `kicks/`.

## Data contracts and invariants

1. Put instrument-specific DSP behavior in profiles, not scattered name branches. Profiles depend on `audio/constants`; the pipeline depends on profiles. The UI discovers instruments/sliders from the API. Do not hardcode five sliders.
2. Preserve the kick's flat paths (`models/`, `output/`). Snare/hi-hat use subdirectories. Change descriptor windows, calibration ranges or metric weights only with measured justification and corresponding audits; do not preserve stale constants merely because they are old.
3. Audio: 44,100 Hz, default 65,536 samples. Mel: FFT 1,024, hop 256, 128 bands, default 256 frames. Tensor `(B,1,128,256)` in `[0,1]`; fixed log bounds `[-11.5129,3.0]`. Keep dataset and vocoder configurations aligned; DisCoder validates compatibility.
4. Default VAE: four stride-2 stages, channels `[32,64,128,256]`; latent dimension defaults to profile value. Shape dimensions must be divisible by 16. Preserve legacy `state_dict` keys with options off and the decoder's sequential indexing used by latent injections.
5. Checkpoint metadata: `model`, `instrument`, `latent_dim`, `n_mels`, `n_frames`, `architecture` (`residual`, `latent_skips`, `soft_logvar`, `channels`), epoch/loss and `training`. Older files can omit metadata; `load_vae_from_checkpoint()` infers shapes and defaults missing architecture options to off. It returns `(model, metadata)`.
6. Use `weights_only=True` when loading checkpoints. Calibration NPZs contain arrays/JSON and use `allow_pickle=False`.
7. Outputs: `vae_best.pth` = lowest validation; `vae_best_eval.pth` = lowest periodic descriptor proxy; `vae_checkpoint.pth` = final state; `loss_curves.png` = static curves. Fine-tunes evaluate/save the source baseline before optimizer steps.
8. `multi_resolution_loss` pools a single mel at scales 1/2/4; it is not a multi-window waveform STFT objective. The dedicated transient tail term penalizes excess HF only; the general reconstruction still penalizes missing energy. Optional symmetric HF loss addresses that gap without replacing the tail penalty.
9. Calibration sidecar: replace the checkpoint suffix with `.controls.npz` (`vae_best.pth` → `vae_best.controls.npz`). Fingerprint covers checkpoint bytes, profile, corpus and `CALIBRATION_VERSION`. Bump the version when range-search semantics change. It calibrates the decoded spectrogram; waveform correction handles backend drift.
10. Independence is descriptor tracking within calibrated ranges, not guaranteed perceptual independence. Corpus-supported probes govern range calibration; unsupported errors remain reported. Test final audio with axes, corners and random combinations. Profiles with `identity_controls=True` (snare, hi-hat) anchor the texture to a corpus encoding, centre sliders on corpus medians and cap waveform correction; extreme snare combinations stay coupled by design, and the studio reports limited reach instead of forcing the numbers (`docs/audio-identity-plan.md`).
11. Keep `audio/waveform.py`, `analysis/evaluation.py`, fidelity metric imports and the standalone tracking viewer free of eager torch/model imports. Root and training package exports are lazy.
12. Eval references invalidate on instrument/corpus fingerprint (`--refresh-ref` forces rebuild). Corpus preparation must keep backups/quarantine manifests.
13. Waveform diffusion keeps its own contract, separate from the VAE's. Preprocessing identity is `peak_safe_lufs_then_peak_0.9_v1` (the shared chain plus a fixed 0.9 peak); labels are measured on the scaled waveform, so conditioning target and generation target agree whatever the descriptor kinds are. Checkpoints carry `model_kind="waveform_diffusion"`, the full `architecture` block and the descriptor keys; `load_diffusion_from_checkpoint()` refuses another kind or another instrument's sliders. Descriptor mean/std come from the training split alone and live in model buffers, so raw profile units go in and out; the split's descriptor rows travel beside the weights as `label_bank` and are rebuilt on `--resume`, and `draw_labels()` resamples them rather than drawing independent Gaussians (that path survives only as a warned fallback). Resampling factors must be 1 or even and divide the length exactly. Texture seed and slider values stay separately seeded.

## Vocoder and path configuration

- Resolution: explicit backend request > `KICKS_VOCODER` > `profile.vocoder`. Current profile defaults: kick/snare `discoder`, hi-hat `bigvgan`.
- DisCoder: pinned revision `6505384d8fd5f18338f171dd81dc10c9a0d34fe9`, config + ~1.72 GB model, mmap load; serial per-hit inference limits activation memory. `KICKS_DISCODER_DIR` overrides its directory.
- BigVGAN: local fine-tuned weights from `PathSpec.vocoder_dir`. `KICKS_VOCODER_DIR` overrides it; normally the profile's model directory plus `vocoder/`. `train --model-dir` pins the shared directory before changing the candidate root.
- Preserve `patch_bigvgan_from_pretrained` compatibility with newer `huggingface_hub`. Griffin-Lim uses a mel-filterbank pseudo-inverse, not `InverseMelScale`.
- Backends share highpass/lowpass, peak normalization and tail gating. Benchmark the actual generation/control path; reconstruction-only vocoder rankings do not establish generation quality.
- Root overrides: `KICKS_DATA_DIR`, `KICKS_MODEL_DIR`, `KICKS_OUTPUT_DIR`. Default instrument: `KICKS_INSTRUMENT` (fallback kick). Controls: `KICKS_CONTROL=descriptor|pca` (default descriptor); with `descriptor`, profiles that set `identity_controls` use the corpus-anchored basis. `scripts/validate_controls.py --control identity|legacy` selects the algorithm explicitly for audits only.

## API, web and hosted services

- `ServerState` loads instruments lazily. Vocoders cache by `(backend, weights_dir)`; `/health` reports profile mode and a backend map unless forced. Serving instances do not hot-reload promoted checkpoints.
- Routes: `/health`, `/instruments`, `/config`, `/generate`, `/evaluate`, `/spectrogram`, `/me`, `POST /export`. Synthesis requests select `instrument`; `/spectrogram` is pre-vocoder, `/evaluate` scores rendered/effected audio.
- Sliders accept `s1..sN`, legacy `pc1..pcN`, or exposed names. With PCA, map names through `SliderBasis.names`; labels are discovered during fitting. Effects: `attack_ms`, `decay_ms`, `drive`, `filter`.
- Shared token bucket: 10 requests/second; LRU: 100 entries, keyed by query. `web/lib/api/` sorts sound query keys for consistent caching.
- Auth defaults off without `KICKS_SUPABASE_URL`, required with it; `KICKS_AUTH_MODE` supports required/optional/off. `/me` and `/export` require a user. Export charges one credit; `Idempotency-Key` deduplicates ledger charging.
- Supabase tokens verify locally via JWKS or optional legacy `KICKS_SUPABASE_JWT_SECRET`. `KICKS_SUPABASE_SERVICE_KEY` is server-only. CORS comes from comma-separated `KICKS_CORS_ORIGINS` (default localhost:3000).
- `web/`: Next.js 16 / React 19, static export, Tailwind 4, shadcn on Base UI. Use Base UI `render={<Link/>}` composition, not Radix `asChild`. Routes use trailing `/`. Studio is client-only for AudioContext/MIDI/localStorage. Respect the React Compiler lint rules.
- `.github/workflows/pages.yml` is currently manual (`workflow_dispatch`). Public build variables configure API, Supabase anon access, analytics and legal-page operator details. API hosting is separate.
- Never put service-role or Stripe secrets under `web/`; `NEXT_PUBLIC_*` ships to browsers. Preserve row-level security, service-only credit writes, append-only ledger and idempotent Stripe webhook handling in `supabase/`.
- Public `web/public/analysis/*.json` must not expose corpus filenames/paths. Full local fidelity reports include them; do not publish those as public analytics.
- `scripts/fetch_drum_abuse.py [kick|snare|hihat|all]` downloads/deduplicates into `data/_staging/<plural>/abuse/`. Staging is not automatic corpus integration.

## Validation by change type

- Documentation: verify options against current `--help`, local links/anchors, artifact names and Markdown/SVG rendering. No training or full app build is needed just to edit prose.
- Loss/model/schedules/fidelity/promotion: `uv run pytest -q tests/test_high_fidelity.py`; add/run checks covering the changed behavior.
- Tracking/dashboard: `uv run pytest -q tests/test_training_tracking.py`; inspect desktop/mobile charts, live updates, note preservation and standalone HTML when UI behavior changes. The viewer serves both backends, so check a VAE run and a diffusion run, plus the Listening lab against a real `output/fidelity/*` report. The template is read per request, so HTML edits are live at once; server-side changes in `tracking.py` need the viewer restarted (the training process is separate and keeps its own copy). Fidelity detail responses include corpus sample paths — the viewer is loopback-only and never a publication path.
- Waveform diffusion: `uv run pytest -q tests/test_diffusion.py tests/test_make_subset.py tests/test_diffusion_listening.py tests/test_modal_train.py tests/test_rename_corpus.py`. The Modal wrapper tests cover its local contracts only (corpus verification, spec validation, the restart guard, periodic commits, note-preserving sync); cloud behaviour is established by the reviewed gates in `docs/modal-training.md`. The tests cover the schedule, conditioning, target drawing, determinism, checkpoints, the tracked loop, subset building and the listening-pair writer only; they establish nothing about how the backend sounds. Evidence for this backend is `kicks eval --pattern`, the direct tail-floor / onset measurements and blind pairs from `scripts/diffusion_listening.py` (a generation A/B with `held_out: false`, never a reconstruction fidelity report); `kicks fidelity` and `scripts/validate_controls.py` are VAE-only.
- Controls/calibration/vocoders: relevant `tests/test_audio_controls.py` / `tests/test_discoder.py`; use matched waveform/control audits when claiming audio improvement. For snare/hi-hat identity changes, rerun `scripts/validate_controls.py --control identity --texture-seeds ...` and `scripts/compare_identity.py`, and report multi-onset counts and target errors separately for centre, axis and random probes.
- API/auth/publication: `tests/test_api_auth.py`, `tests/test_publish.py`; full Python regression command is `uv run pytest -q`.
- Website: from `web/`, `pnpm lint` then `pnpm build`; inspect affected views. Publishing generated analysis is a separate write step, not part of ordinary lint/build checks.
- Report what was actually measured. Automated checks, proxy scores and listening evidence are separate claims.
