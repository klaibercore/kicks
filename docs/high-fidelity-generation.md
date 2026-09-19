# High-fidelity generation and training observability

## Recommendation

Use one workflow: establish the fidelity baseline, track every experiment,
improve the loss, test residual and latent skip connections, then consider a
codec-latent sequence with a conditional generative prior. The dashboard is the
record of the evidence at every stage. Architecture changes remain experiments
to validate, not measured improvements of a new model.

## One implementation plan

| Stage | Work | Evidence recorded in the dashboard |
|---|---|---|
| 1. Baseline and tracking | Start the local viewer, record the problem and success criteria, and measure the original checkpoint before optimization. | Configuration, data/split fingerprint, epoch-zero validation and HF error; experiment brief. |
| 2. Loss experiment | Test symmetric HF detail and attack-change losses, then tune compression strength. | Fixed-objective validation, 2–16 kHz / 8–16 kHz / attack errors, KL activity, learning rate and timing. |
| 3. Architecture experiment | Compare residual blocks and latent injections at decoder scales; assess capacity changes separately. | Matched-run comparisons, fidelity versus capacity, observations explaining regressions or gains. |
| 4. Audio validation | Render matched references and generated hits with the selected vocoder; test slider axes, corners and independent combinations. | Links/paths and summaries of waveform reports, level-matched listening observations, control errors. |
| 5. Promotion decision | Adopt only a candidate that meets the declared fidelity and control criteria. Otherwise record the next experiment. | Written decision, checkpoint identity, evidence and remaining limitations. |
| 6. Larger upgrade | If the measured ceiling remains insufficient, validate a matched audio codec and train a conditional sequence prior. | Codec reconstruction baseline, generation fidelity, compute cost and the same acceptance tests. |

### How each stage runs

| Stage | Command | Where the evidence lands |
|---|---|---|
| 1 | `kicks dashboard`, then `kicks train --run-name … --intent … --hypothesis … --success-criteria …` | `output/training/<run-id>/run.json`, notebook fields |
| 2 | `kicks train --hf-detail-weight W --attack-change-weight W [--beta B] --model-dir models/experiments/<name>` | per-epoch `val_term_*` / `train_term_*` components; weights in the run config and checkpoint |
| 3 | `kicks train --residual --latent-skips [--latent-dim 64\|128] --model-dir …` | `architecture` block in run config and checkpoint; the dashboard names a same-data, different-architecture comparison as such |
| 4 | `kicks fidelity --run <run-id> --checkpoint <candidate>`; `scripts/validate_controls.py --run <run-id> --checkpoint <candidate> --out …` | `report.json`, `listening/listening.html` (blind pairs), the run's **Evidence** card |
| 5 | `kicks promote --run <run-id> --checkpoint <candidate> --decision "…"` | promotion entry in Evidence, sha256 and paths appended to **Decision** |
| 6 | not implemented — contingent on the stage-5 outcome | — |

The stage-2 terms keep the existing transient penalty for unwanted tail noise
and add what it lacks: `hf_detail_loss` charges missing and excess HF energy
equally, weighted by how audible each bin is in the *reference* (a bin the
reference does not excite has weight 0, so the term cannot be satisfied by a
raised noise floor); `attack_change_loss` matches the frame-to-frame slope over
the profile's click window. The stage-3 options start as the identity of the
plain network, so an option run and its baseline begin from the same function.
`kicks fidelity` reproduces the run's validation split from its fingerprint and
measures on those held-out hits; when the corpus has changed since, it says so
and samples the whole corpus instead.

### Tracking workflow

Start `kicks dashboard` and open `http://127.0.0.1:6060` before training. Review
comparable runs, then pass `--run-name`, `--intent`, `--hypothesis` and
`--success-criteria` to `kicks train`. Each invocation creates a unique record
under `output/training/` (override with `--runs-dir` or `KICKS_RUNS_DIR`).

The viewer updates every three seconds, compares two runs and supports hover
inspection, epoch ranges, optional trend smoothing, JSON export and a notebook.
Metrics remain raw in the record and hover readout when smoothing is enabled.
Standalone `index.html` files also update from sibling data scripts; editing
notes and comparing runs use the local HTTP viewer. No CDN or account is needed.

Before training, complete **Objective**, **Hypothesis** and **Success criteria**.
During training, update **Observations** with epoch references, trends and
listening findings. Use **Decision & next action** for the eventual recommendation.
Changing prose never changes optimizer settings or promotes a checkpoint.

The automatically recorded high-frequency KPIs are **pre-vocoder log-mel
errors**, not the waveform STFT errors in the diagnosis below. Keep these metric
definitions distinct. Waveform and listening evidence goes in the notebook
alongside the corresponding report paths. Compare absolute validation losses
only when data, split and objective match; the viewer flags known mismatches.

## What the current pipeline shows

- The VAE flattens its convolutional features into one global vector: 64
  dimensions in the checkpoint used for the saved kick audit, 32 in the default new model.
  Four encoder stages downsample both time and frequency. Local transient
  and noise detail must survive this compression.
- The extra transient loss is symmetric over the initial click but penalizes
  only *excess* high-frequency energy in the tail. A controlled probe returned
  zero for removing an existing HF tail and 0.01653 for adding an equal error.
  The general reconstruction loss still penalizes missing energy; this is a
  limitation of the dedicated term, not a claim that missing tails are free.
  The stage-2 `hf_detail_loss` exists for exactly this probe: it returns the
  same value for both edits (`tests/test_high_fidelity.py`).
- The current multi-resolution loss pools one mel spectrogram. It does not
  analyze the waveform with several different FFT windows.
- In the saved 16-kick DisCoder comparison, mean active-mel error rises from
  3.70 dB for direct vocoding to 6.81 dB with the VAE. A separate 512-point
  STFT diagnostic finds 8–16 kHz body error rising from 5.28 to 7.28 dB.
  The latter measures reference bins above -70 dBFS during 30–150 ms, with
  equal weighting per sample. These are in-corpus diagnostics, not a held-out
  benchmark or listening-quality score. Details are in
  `output/calibration/high-end-diagnosis.json`.
- Existing corpus realism scores and accurate descriptor controls do not
  establish that fine texture, phase or transient fidelity is commercially
  adequate. Keep their role separate from fidelity and listening tests.

## Which skip connections help generation?

| Connection | Use |
|---|---|
| Residual connection around convolution blocks | Candidate for easier optimization and retaining intermediate detail. |
| Latent code injected at each decoder scale | Gives fine-resolution layers direct access to the generated sound's identity and controls. |
| Raw encoder features passed to the decoder | Useful for input-conditioned reconstruction; unavailable when sampling a fresh sound from a latent code unless a prior also generates those features. |

Generative latent skips have evidence for reducing latent collapse, but that
paper does not establish drum fidelity gains. The audited kick checkpoint had
all 64 dimensions active, so collapse prevention alone is insufficient.
[Generative skip models](https://arxiv.org/abs/1807.04863).

## First experiment: preserve the existing representation

Run controlled comparisons, keeping data split, vocoder, random seeds and
evaluation fixed:

1. **Loss only.** Add symmetric, reference-weighted HF detail reconstruction
   across the audible hit, plus temporal-change matching for the attack. Retain
   a separate penalty for unwanted tail noise. Weight audible reference detail
   rather than raising the noise floor. Sweep compression strength: the CLI's
   beta default is 0.02, versus 0.001 in the kick fine-tune. More active latent
   dimensions alone do not prove enough detail is retained.
2. **Architecture only, then combined.** Use residual convolution blocks and
   inject the latent code at multiple decoder scales. Compare 64 and 128 global
   dimensions before committing to a larger model. Version the architecture in
   checkpoints and refit the latent prior and control calibration for each.
3. **Waveform objective.** Evaluate, and on suitable hardware fine-tune with,
   real multi-window STFT losses and perceptual feature matching. Short windows
   catch attack smearing; longer windows expose tonal or spectral errors.
   Backpropagation through a frozen vocoder still requires activation memory.
   Keep the first local experiments on the smaller VAE.

The DAC work provides evidence for multi-window reconstruction and multi-band
spectral discriminators. Its recipe is a research starting point, not a set of
loss weights to copy without calibration.
[DAC paper](https://arxiv.org/html/2306.06546v2).

## Higher ceiling: generate an audio-codec latent sequence

```text
Training audio -> matched codec encoder -> latent sequence
                                           ^
Descriptors + seeded noise -> conditional diffusion / flow model
                                           |
                                    matched codec decoder
                                           |
                                  audio + control evaluation
```

The sequence retains local capacity across time instead of asking one small
vector to describe the complete hit. The conditional prior must learn plausible
sequences; increasing latent capacity without that prior can improve
reconstruction while making random generation worse. A denoising U-Net can use
encoder-to-decoder skips because its noisy input exists during generation.

First establish the codec's reconstruction ceiling on our drums. Use a matched
encoder/decoder pair: DisCoder's fine-tuned DAC decoder is not automatically
interchangeable with a stock DAC encoder. DisCoder changes that latent mapping
during fine-tuning. [DisCoder paper](https://arxiv.org/html/2502.12759v1).

Latent audio diffusion is an established approach, but the quality and training
cost of a small drum-specific implementation remain to be measured here.
[Latent audio diffusion](https://arxiv.org/abs/2402.04825).

A simpler variant — diffusion directly on the 65,536-sample waveform, with no
codec and no vocoder — is implemented and untrained in `kicks/nn/diffusion.py`;
[`docs/waveform-diffusion.md`](waveform-diffusion.md) records its design, the
8 GB Apple Silicon cost measurements, the staged run sequence and two known
issues to fix before collecting evidence. It is the first candidate for this
stage; the codec-latent prior above remains a proposal.

## Acceptance criteria

- Split by source pack or sample family where possible, and check for near
  duplicates across splits. Compare reconstruction and fresh generation.
- Report separate 2–8 kHz and 8–16 kHz errors, attack timing and envelope,
  spectral flatness/modulation, and unwanted late energy. Retain onset checks.
- Include level-matched, randomized listening comparisons of the complete hit
  and the HF band, with expert rejection counts. Higher brightness alone is
  not an improvement.
- Condition the generator on the descriptors and keep its texture seed fixed
  while a slider moves. Test the final waveform at axes, corners and independent
  combinations. Keep corrections small enough to preserve the learned texture;
  exact descriptor matching must not conceal audible damage.
- Promote a candidate only after both fidelity and control checks improve or
  meet predeclared tolerances. Preserve the current per-instrument vocoder
  choices as comparison baselines.
