# Descriptor-conditioned waveform diffusion

## Status

The backend exists as code: a denoiser, a corpus loader, a tracked training
loop, a deterministic sampler and two commands. **No model has been trained,
sampled for evaluation, benchmarked against the VAE, or promoted.** Nothing in
this document reports a measurement. Every number below is a configuration
value or an arithmetic consequence of one, and the comparisons the experiment
exists to make have not been run.

This implements the plan in issue #3. The issue's open questions were settled
as recorded under [Decisions](#decisions-taken); the rest of the plan — training,
the fidelity and control audits, blind listening, latency benchmarking — is
still ahead.

Reviewed and merged into `main` on 2026-09-19. The review ran the test module
and a step-time / memory probe of the untrained network on an Apple M1 with
8 GB of unified memory; those figures are under
[Apple Silicon and other 8 GB machines](#apple-silicon-and-other-8-gb-machines).
It also found two defects, recorded under [Known issues](#known-issues); both
were fixed on 2026-09-20, before any run.

## What it is

A 1-D U-Net denoises a raw waveform directly. Neither the VAE nor a vocoder is
in the generation path, so the two failure modes the current backend has to
manage separately — what the latent space can represent, and what the vocoder
can invert — collapse into one model that is trained end to end on audio.

The trade is compute. The VAE decodes a 128×256 spectrogram once; the sampler
runs the denoiser once per step over 65,536 samples, 50 times by default.
Whether that buys anything audible is the open question.

| | VAE backend | Waveform diffusion |
|---|---|---|
| Representation | normalized log-mel, `(1, 128, 256)` | waveform, `(1, 65536)` |
| Generation path | latent GMM prior → decoder → vocoder | noise → denoiser × steps |
| Control | calibrated descriptor basis over the latent space | descriptor vector conditioning the denoiser |
| Cost per hit | one decode plus one vocoder pass | `--steps` denoiser passes, doubled under guidance |
| Trained | yes | **no** |

## The schedule

v-prediction on an angular schedule. For a noise level `sigma` in `[0, 1]`:

```
alpha = cos(sigma * pi / 2)        beta = sin(sigma * pi / 2)
x_noisy = alpha * x + beta * noise
v       = alpha * noise - beta * x
```

`sigma = 0` is the corpus, `sigma = 1` is noise, and `alpha² + beta² = 1` at
every level. Training draws a level per sample uniformly and takes the mean
squared error against the true velocity.

Sampling walks `sigma` from 1 to 0 in `--steps` equal hops. Each step reads the
predicted clean signal and the predicted noise out of one velocity, then
re-mixes them at the next level. No fresh noise is drawn along the way, so the
output is a deterministic function of the starting noise and the descriptor
targets — which is what makes two hits that differ only in one slider worth
comparing.

## Conditioning

The noise level and the instrument profile's descriptor vector are embedded,
summed, and turned into a per-channel scale and shift (FiLM) inside every
residual block. Descriptor values enter in the profile's own raw units; the
training split's mean and standard deviation are stored as buffers in the
checkpoint and applied inside the network, so a caller cannot pair a checkpoint
with the wrong normalization. The checkpoint also carries the training split's
descriptor rows (`label_bank`), which is where generation draws its targets.

A fraction of each batch (`--cond-dropout`, 0.1 by default) is trained against a
learned null embedding instead of its descriptors. That is what makes
classifier-free guidance available at sampling time: `--guidance g` returns
`v_uncond + g * (v_cond - v_uncond)`, at the cost of a second forward pass per
step for any `g` other than 1. Guidance cannot be retrofitted without
retraining, which is why the dropout is on by default rather than held back.

Labels are measured on exactly the waveform the model is trained on, so the
conditioning target and the generation target agree by construction whatever
the descriptor kinds are.

## Architecture

Eight stages, feature widths `32, 32, 64, 64, 128, 128, 256, 256`, each halving
the length; self-attention at the deepest three scales; two residual blocks per
stage. 9.56 M parameters at the default configuration.

The reference implementation uses the same widths with `factors = 1, 2, 2, 2,
2, 2, 2, 2` over a 32,768-sample window, reaching 256 frames at the bottleneck.
This project's window is twice as long, so the leading factor-1 stage becomes a
factor-2 stage: the same bottleneck length and the same attention lengths (256,
512, 1024) from a window of 65,536. `--channels` and `--factors` override both.

Every residual block's FiLM projection and output convolution start at zero, and
so does the output head, so a freshly built network is the zero map and depth
costs nothing at initialization.

## The corpus

Loading goes through `kicks.audio.io`, so the mono → 44.1 kHz → 65,536 samples →
LUFS chain is literally the one the VAE was trained on. One step is added: each
hit is scaled to a fixed peak of 0.9.

Diffusion learns an absolute signal distribution, and a corpus whose hits sit
anywhere between −30 and −6 dBFS would spend capacity modelling gain. The kick's
descriptors are power ratios and a time centroid, so they are gain-invariant and
a fixed peak changes none of them — but that is a property of the kick profile,
not a guarantee for every instrument, so labels are measured after the scaling
rather than before it. The preprocessing identity recorded in checkpoints and
run records is `peak_safe_lufs_then_peak_0.9_v1`, distinct from the VAE's
`peak_safe_lufs_v1`.

A LUFS meter needs at least 400 ms of audio, so this loader cannot be pointed at
a window shorter than about 17,640 samples.

## Training

```bash
uv run kicks diffusion-train -i kick \
  --model-dir models/experiments/diffusion \
  --epochs 200 --batch-size 8 --grad-accum 4 \
  --run-name "Kick · waveform diffusion baseline" \
  --intent "Test whether direct waveform generation beats VAE + vocoder" \
  --hypothesis "Removing the mel bottleneck and the vocoder preserves transient detail the current path loses" \
  --success-criteria "Matched fidelity report, control audit and blind listening against the promoted VAE checkpoint"
```

| Option | Default | Contract |
|---|---|---|
| `--epochs` | `200` | Cosine LR schedule spans this horizon |
| `--batch-size` / `--grad-accum` | `8` / `4` | Effective batch 32; waveform activations are far larger than spectrogram ones |
| `--learning-rate` | `1e-4` new / `3e-5` resume | Adam |
| `--cond-dropout` | `0.1` | Fraction trained unconditioned; `0` disables guidance for that checkpoint |
| `--ema-decay` | `0.999` | Validation and every saved checkpoint use the average, with a 10-step warm-up |
| `--channels` / `--factors` | see above | New models only; a resumed checkpoint keeps its own |
| `--cond-dim`, `--blocks`, `--attention-scales`, `--attention-heads` | `128`, `2`, `3`, `4` | New models only |
| `--val-split` | `0.1` | Deterministic split from `--seed`, fingerprinted like the VAE's |
| `--eval-every` | `0` (off) | Descriptor-target proxy; each evaluation costs a full sampling pass |
| `--preview` | `0` (off) | Post-training samples; each one costs a full sampling pass |

`--model-dir` moves only `KICKS_MODEL_DIR`. Unlike `kicks train`, there is no
vocoder in this path, so there is nothing to pin back to the shared root.
`--resume` keeps the checkpoint's descriptor mean/std but rebuilds the label
bank from the current training split, so targets follow the corpus being
trained on. The training loader drops a short final micro-batch and every
optimizer step divides by the number of micro-batches its accumulation group
really holds, so each step is a mean over equally sized micro-batches.

### What the run records

The same tracking writer as the VAE, so the same dashboard reads it:
`output/training/<run-id>/run.json`, the notebook, and any attached evidence.

* `train_loss` — mean squared velocity error at uniformly drawn noise levels,
  with conditioning dropout applied. `val_loss` holds each validation hit at a
  fixed noise level with a fixed noise vector and full conditioning, so the two
  are **not** comparable in absolute terms.
* `val_loss_low_sigma` / `val_loss_mid_sigma` / `val_loss_high_sigma` — the same
  loss split by noise level. High-sigma error is nearly free to reduce and low
  sigma is where fine detail lives; a flat total can hide movement in both.
  The levels sweep the whole schedule across the split, so the bands stay
  populated however small the validation set is.
* `control_mae` — mean absolute descriptor error of freshly sampled hits against
  their validation targets, in training-set standard deviations. A
  controllability proxy only. It is not a perceptual rating, it is measured on a
  handful of samples at reduced step count, and it does not replace
  `scripts/validate_controls.py`.

The dashboard reads `config.backend` and retargets itself: the detail chart
becomes loss by noise level, the latent-activity chart becomes descriptor target
error, the epoch table changes columns, and a VAE run and a diffusion run are
not offered as a like-for-like comparison.

Checkpoints: `diffusion_best.pth` (lowest validation loss), `diffusion_best_control.pth`
(lowest `control_mae`, only written when `--eval-every` is on),
`diffusion_checkpoint.pth` (final), `diffusion_loss_curves.png`.

### Apple Silicon and other 8 GB machines

Measured on 2026-09-19: Apple M1, 8 GB unified memory, torch 2.10, MPS backend,
the default 9.56 M-parameter network, random data, one forward/backward/Adam
step. `torch.mps.recommended_max_memory()` reports 5.33 GiB. These are
throughput and memory figures for an *untrained* network; they say nothing
about how the backend sounds.

| Setting | Step time | MPS memory (driver-allocated) |
|---|---|---|
| batch 1 | 0.48 s | 1.5 GiB |
| **batch 2** | **0.65 s** | **1.7 GiB** |
| batch 4 | 1.29 s | 3.4 GiB |
| batch 4, `torch.autocast` fp16 | 3.07 s — slower, no memory saved | 3.5 GiB |
| batch 8 (the CLI default) | 20.7 s — over the working set, 8× slower per sample | 6.4 GiB |
| CPU, batch 2 | 1.16 s — MPS is 1.8× faster | — |

Consequences:

* Use `--batch-size 2` on an 8 GB machine. The CLI default of `8` is right for
  a discrete GPU and wrong here; do not use autocast on MPS. For budgeted runs
  pair it with `--grad-accum 4` (effective batch 8): the cost per hit is fixed,
  so a smaller effective batch buys four times the optimizer steps of the
  documented effective 32 in the same time. Go back to `--grad-accum 16` only
  if the train curve is too noisy to read.
* Cost saturates at about 0.32 s per hit per epoch, so with the current corpora
  one epoch is roughly 27 min for hi-hats (5,824 hits), 38 min for snares
  (7,994) and 48 min for kicks (10,109). The 200-epoch default is four to seven
  days per instrument; plan short screens and resumable chunks instead.
* Sampling costs 6.4 s per hit at 50 steps and 1.95 s at 10 steps (batch 1),
  so `--eval-every`, `--preview` and audits are expensive; budget for them.
* `WaveformDataset` keeps the whole corpus in RAM as float32: 2.47 GiB for
  kicks, 1.95 GiB for snares, 1.42 GiB for hi-hats, in the same 8 GB the GPU
  uses. Stop `kicks serve` and `next dev` before training. Subsets of up to
  about 3,000 hits need no change; a compact in-memory format (int16) is
  planned before any full-corpus run, not shipped.
* Run long jobs under `caffeinate -i` inside tmux or `nohup` so sleep or a
  closed terminal does not kill them. A hard kill leaves the run `running`;
  the dashboard marks it stale. Watch `epoch_seconds` — a fanless M1 throttles.

#### Corpus size against wall-clock budget

`epoch ≈ 0.9 × N × 0.32 s`, so a budget buys a fixed number of sample-passes
(about 11k per hour) however it is split between corpus size and epochs. The
table assumes `--batch-size 2 --grad-accum 4` and the hi-hat corpus; the
snare (7,994 hits, 38 min/epoch) and kick (10,109, 48 min/epoch) equivalents
scale with the corpus.

| Budget | Corpus (stratified subset) | Epochs | s/epoch | Optimizer steps | What it can tell you |
|---|---|---|---|---|---|
| 10 min | 64 hits | 2 | 18 | 14 | The pipeline works |
| 1 h | 256 hits | ~45 | 74 | ~1,300 | Loss falls in all three sigma bands; samples stop being white noise. No listening verdicts. |
| 4 h | 1,000 hits | 50 | 288 | ~5,600 | First listen: is there a stick and a sizzle at all? |
| 10 h | 2,000 hits | ~60 | 576 | ~13,500 | First real result: blind listening against studio renders, descriptor sweeps |
| 24 h | 3,000 hits | ~100 | 864 | ~34,000 | Diversity across all families; guidance scale sweep |
| 48 h | full hi-hat, 5,824 | ~100 | 1,680 | ~66,000 | Only after a 10 h run sounded like a hi-hat; `--resume` from it |
| 4–7 days | full corpus, 200 epochs | 200 | — | — | Not on this machine |

A 2,000-hit subset gives about 120k sample-passes in ten hours — the same
compute as twenty full-corpus epochs, with three times the passes per hit —
and can be resumed onto the full corpus later.

Subsets come from `scripts/make_subset.py`:

```bash
uv run python scripts/make_subset.py --instrument hihat --size 2000 --seed 42
# -> data/_subsets/hihat-2000/, symlinks plus manifest.json
```

It draws in proportion to the families in the instrument's cluster report
(`kicks cluster` writes it), refuses a report whose file list no longer
matches the corpus, and nests: with the same seed the 256-file subset is
inside the 2,000-file one. Name the subset directory and seed in the run
notebook. Two other levers change the contract and are not used here:
halving the window to 32,768 samples (halves cost and memory but truncates
the quarter of hi-hat files longer than 743 ms) and halving `--channels`.

#### Suggested sequence

Hi-hat first (the identity problem, the smallest corpus, the fastest epochs),
then snare. Each step waits for the previous one's evidence.

1. **Smoke test, about ten minutes.** `data/_subsets/hihat-64` passed with
   `--data`, `--epochs 2 --batch-size 2 --grad-accum 4 --eval-every 1
   --eval-samples 4 --eval-steps 10`, its own `--model-dir`. Confirms the loop,
   the dashboard retargeting, the EMA checkpoint and `diffusion-generate`.
2. **One-hour signal check.** 256 hits, ~45 epochs, `--eval-every 15`. Pass or
   fail on the curves and on "not white noise", nothing else; a low-sigma band
   that never moves means look at the code, not at a longer run.
3. **Overnight, 2,000 hits, ~60 epochs.** `--eval-every 10 --preview 0`,
   notebook filled in, cosine horizon and subset recorded. Sample eight hits
   at 50 steps and listen, level-matched against studio renders.
4. **Weekend, full corpus, only if the overnight run sounded like a hi-hat**:
   `--resume` from it for ~100 epochs. `--resume` starts a fresh optimizer,
   restarts the cosine schedule and defaults the learning rate to 3e-5, so pass
   `--learning-rate 1e-4` and note the warm restart in the notebook.

Evidence stays the same as for the VAE: `kicks eval --pattern 'diff_*.wav'`
for set-level corpus distance, level-matched blind listening against studio
renders, and the achieved-versus-target print-out as a control proxy only.
`kicks fidelity` and `scripts/validate_controls.py` are VAE-specific.

## Generation

```bash
uv run kicks diffusion-generate -i kick -n 8 --steps 50 --guidance 1.5 \
  --seed 7 --target punch=9.0
```

`--seed` is the texture seed — the starting noise. `--label-seed` seeds only the
descriptor targets that were not pinned with `--target`. Holding the texture
seed while sweeping one target is how a slider is examined; holding the target
while changing the texture seed is how diversity is examined. Fixed noise
encourages continuity between neighbouring slider values but does not guarantee
that the rest of the hit holds still — that is what a control audit measures.

Each sample prints its achieved descriptor values against its targets and its
raw peak. A sample is scaled down only if it would clip on write, so a model
whose output level drifts off the corpus stays visible.

Targets come from the checkpoint's label bank — the training split's
descriptor rows. Unpinned descriptors are resampled from those rows, so every
target is a combination a real hit had. A pinned `--target` selects the bank's
nearest rows in the pinned descriptor (standardised distance; at least 16 rows
or 5 % of the bank) before setting the value exactly, so the free descriptors
are the ones the corpus pairs with that value rather than the corpus average.
A pinned value outside the bank's range is honoured and warned about. A
checkpoint without a bank falls back to independent Gaussians, with a warning;
see [Known issues](#known-issues) for why that is a poor default.

## Decisions taken

Issue #3 lists decisions to settle before implementation. These were settled as
follows.

| Question | Decision | Why |
|---|---|---|
| Mono vs stereo, 65,536 vs the reference's 32,768 | Mono, 65,536 samples | Reuses the project's preprocessing exactly, so corpus, descriptor windows, splits and every audit tool carry over. The reference's 0.74 s window truncates long kicks, and no pretrained weights are reusable at either setting. |
| Adopt the reference package or write the network | Written here, no new dependency | The reference pins a Lightning/CUDA-era stack against this project's current torch, and its conditioning hooks would need wiring regardless. |
| Conditioning mechanism | FiLM at every residual block | The issue's preferred option. Cross-attention on descriptor embeddings remains available and untried. |
| Conditioning dropout and guidance | Shipped, on by default | Cannot be retrofitted without retraining. |
| Descriptor loss on the predicted clean audio | Not implemented | The issue lists it as a "consider". It adds a weight to tune and a noise-level gate to choose, and it should be judged against a plain conditional baseline that does not yet exist. |
| Sampler | Deterministic, no added noise | Two hits that share a texture seed stay comparable, which the control audits depend on. |

## What has not been done

Everything the issue asks for after implementation:

- No training run of any length. No checkpoint exists.
- No sound-quality, HF, attack or late-energy comparison against the VAE backend.
- No generation-diversity measurement, no descriptor target error on a trained
  model, no cross-talk measurement.
- No axis, corner or random-combination sweep; no level-matched blind listening.
- No rendering-latency benchmark, so no claim about interactive use. At 50 steps
  the denoiser runs 50 times per hit, doubled under guidance; whether that is
  fast enough for a slider release is unmeasured.
- No CUDA performance figures, and no throughput figure from a real epoch. The
  MPS step-time and memory probe above used random data on an untrained
  network; training itself has not run on any device.
- Splits are fingerprinted the same way the VAE's are, with the same limitation:
  the fingerprint covers paths, sizes and mtimes, not content, so it does not
  detect near-duplicates or pack leakage. The issue asks for splits by
  pack/family and near-duplicate checks; neither is implemented.
- Nothing is wired into the API or the studio. `kicks serve` is unchanged.

A promotion path does not exist for this backend, and `kicks promote` still
means the VAE checkpoint. Deciding whether the backend is worth keeping needs
the evidence above, not the code alone.

## Known issues

Found in the 2026-09-19 review; both fixed on 2026-09-20 as described at the
end of each item. They stay here because they explain two contracts —
where targets come from, and what an optimizer step averages — that a reader
of an older checkpoint or run record needs.

1. **Descriptor targets are drawn off the corpus manifold.** `draw_labels()` in
   `kicks/synthesis/diffusion.py` samples each descriptor independently from
   `N(mean, std)` of the training split. From the published corpus statistics
   (`web/public/analysis/*.json`), `decay` alone comes out negative — a
   physically impossible target — in about 29 % of snare draws (mean 38 ms,
   std 70 ms, median ≈ 19 ms), 16 % of hi-hat draws and 17 % of kick draws,
   and the strongest descriptor correlations (snare body↔snap −0.68, hi-hat
   body↔bright −0.84) are ignored. This is the same "centred between extremes"
   failure `docs/audio-identity-plan.md` fixed for the VAE studio path.
   Training is unaffected, because training labels are measured, and the
   in-loop `control_mae` proxy resamples real validation labels, so the proxy
   can look healthy while default generation produces blips. **Fixed:** the
   training split's label matrix is saved in the checkpoint as `label_bank`
   and `draw_labels()` resamples its rows, nearest rows first when a
   descriptor is pinned; the Gaussian remains only as a warned fallback for
   checkpoints written before the bank existed.
2. **The trailing gradient-accumulation group is under-weighted.** The loop
   scales every micro-batch loss by `1 / grad_accum` and steps at
   `index % grad_accum == 0 or index == len(train_loader)`, so when the number
   of micro-batches is not a multiple of `--grad-accum` the last optimizer
   step of each epoch sees a gradient scaled by `remainder / grad_accum`, and
   with `drop_last=False` a short final micro-batch's mean counts as a full
   one. One step in a few hundred per epoch — harmless in practice, and the
   recorded `train_loss` is already size-weighted. **Fixed:** the training
   loader drops a short final micro-batch, and each loss is divided by the
   size of its own accumulation group (`_accumulation_group`), so every step
   is a mean.
