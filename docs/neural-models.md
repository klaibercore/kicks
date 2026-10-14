# Neural and hybrid models

Implementation date: 14 October 2026. These are isolated research candidates
for the local Studio. Existing VAE/vocoder and waveform diffusion paths remain
available. No candidate is promoted or declared better on listening quality.

## Implemented local families

| CLI / API backend | Implementation | Training / fresh generation |
|---|---|---|
| `parameters` | Descriptor-conditioned CVAE over joint bounded physical parameters; NumPy DSP renders its output | Verified training-only fitted bank; parameter MSE + KL; seeded latent sampling |
| `ddsp` | Conditional energy-encoding VAE driving differentiable integrated-pitch/modal bodies and learned noise envelopes | Full-waveform multi-window magnitude/envelope loss + KL; seeded latent and noise |
| `noiseband` | Conditional VAE predicting positive, time-varying gains over logarithmic noise bands | Same waveform objectives, without a modal body; a compact NoiseBandNet-inspired experiment |
| `hybrid` | Learned physical/modal body plus learned band-energy texture, jointly optimized | Full-waveform objective; no arbitrary recording-minus-synth subtraction |
| `drumgan` | Descriptor-conditioned transposed-convolution generator and projection critic, trained with WGAN-GP and spectral regularization | Fresh noise-to-waveform generation at native 44.1 kHz |
| `wavevae` | Conditional waveform convolutional VAE with a sequence-valued normal prior | Waveform reconstruction + KL; fresh latent sampling; a compact RAVE-inspired baseline |
| `codec` | Descriptor-conditioned temporal transformer v-diffusion over matched DAC quantized continuous latent sequences | Frozen native 44.1 kHz DAC encoder/decoder, training-only latent normalization, seeded prior sampling |

The DDSP/noise, GAN and waveform-VAE modules are original compact implementations
of the methods. They are **not reproductions of the published DDSP, NoiseBandNet,
DrumGAN or RAVE architectures**, and their checkpoints are not interchangeable.
The historical DrumGAN work used 16 kHz; this project's GAN uses the common
44.1 kHz contract. Noise bands model stochastic energy, not phase-exact noise.
The differentiable body is a separate renderer from the fourfold-oversampled
NumPy voice. Its sinusoidal modes are bounded below Nyquist, but it has not
received the NumPy renderer's oversampling/aliasing audit.
[DDSP](https://arxiv.org/abs/2001.04643),
[NoiseBandNet](https://arxiv.org/abs/2307.08007),
[DrumGAN](https://arxiv.org/abs/2008.12073),
[RAVE](https://github.com/acids-ircam/RAVE).

## Train and generate

```bash
uv run kicks neural-models
uv run kicks benchmark-freeze --out output/benchmarks/my-corpus.json
uv run kicks neural-train --kind hybrid -i hihat \
  --manifest output/benchmarks/my-corpus.json \
  --out models/experiments/my-hihat-hybrid \
  --epochs 2 --max-steps 32 --limit 32 --validation-limit 8 --batch-size 2 \
  --intent "Improve evolving hi-hat texture" \
  --hypothesis "Time-varying band energy improves the tail" \
  --success-criteria "Finite renders, target audit, blind preference and rejection checks"

uv run kicks neural-generate \
  --checkpoint models/experiments/my-hihat-hybrid/best.pth \
  --out output/my-hihat-hybrid --count 8 --seed 42
```

Select a different `--kind` to train the other families. `parameters` additionally
requires `--bank` pointing to a content-verified `dsp-fit` bank from this same
manifest and instrument. That bank supplies training-only physical values and
the descriptor values of their resynthesized audio. This path has no held-out
parameter validation and reports that absence. `codec` additionally requires
`--codec-checkpoint` pointing to one matched local 44.1 kHz DAC artifact.

Each candidate output directory must be new. Training stops at the earlier of
the epoch or optimizer-step budget. CPU is the conservative default; `--device
cuda` is supported. MPS is excluded from this research trainer because of its
FFT requirements. Source files are hashed before loading, training and validation
folds never fall back to one another, and label/latent statistics use training
data only. The test fold is not used for training or checkpoint selection.

The trainer writes `best.pth`, `last.pth` and a normal dashboard record with the
experiment brief, exact source identities and objective. These checkpoints are
inference/candidate artifacts; restarting a run currently starts a fresh model
and optimizer in a new directory. The tracker labels these families separately
and shows their own objectives, without borrowing VAE mel-error semantics.

Generation writes native-contract WAVs, descriptor requests, checkpoint/codec
identities and `audit/report.json`. Fresh generation is explicitly unpaired;
its report does not invent reconstruction error against unrelated recordings.
`--target KEY=VALUE` overrides raw descriptor values. For matched comparisons,
pass the same `--conditions conditions.json` (a list of complete raw descriptor
rows), count and seed to each compatible candidate. Resulting case identities
can be passed to `listening-compare` using their `audit/` directories.

## Studio and API

```bash
uv run kicks serve --backend dsp \
  --neural hihat:hybrid=models/experiments/my-hihat-hybrid/best.pth \
  --neural hihat:codec=models/experiments/my-hihat-codec/best.pth
```

`--neural INSTRUMENT:KIND=PATH` is repeatable; `KICKS_RESEARCH` accepts the same
comma-separated entries. Only configured candidates are advertised. They load
lazily and must match the instrument, descriptor order, model kind, training
status and common mono 44.1 kHz / 65,536-sample contract. A codec prior refuses a
decoder with a different byte identity. `--backend KIND` can explicitly select
a configured candidate at startup; automatic startup does not promote one.

All seven use the Studio engine selector, descriptor sliders, seeded variations,
pad caching/choke groups, effects, rendered spectrograms, target measurements and
24-bit export. Codec sampling also exposes step count. Neural audition-on-release
starts disabled. Preview and export use the same checkpoint/seed/controls.
Descriptor conditioning is a request, not a calibrated accuracy guarantee.

To serve every candidate created by this implementation pilot:

```bash
uv run kicks serve --backend dsp --neural-root models/experiments/neural-20261014
```

`--neural-root` reads existing `INSTRUMENT-KIND/best.pth` directories explicitly;
it does not train or promote weights. These pilot directories are local and
ignored; a fresh checkout must train candidates or supply compatible checkpoints.
The API defaults to `127.0.0.1:8080` and the web app to port 3000, as in the
[README setup](../README.md#start-the-studio). If using `--port 8091` for a second
API, set `NEXT_PUBLIC_KICKS_API_URL=http://127.0.0.1:8091` before starting the web
app. API candidates run on CPU unless `KICKS_RESEARCH_DEVICE=cuda` is set.

The new candidates report waveform measurements and requested-versus-measured
descriptors. They do not provide a fitted corpus realism score. Existing
`kicks fidelity`, `scripts/validate_controls.py` and `kicks promote` remain
VAE-specific. Use the candidate's `audit/` directory for common waveform and
matched listening comparisons; choosing a Studio engine is not promotion.

## Optional upstream integrations

The isolated worker supports the original upstream models with **local assets**:

| Worker kind | Required local assets | Behavior |
|---|---|---|
| `stable-audio-3` | Small SFX weights, matching config and cached text/codec components | Official Stable Audio Tools conditional inpainting sampler, pingpong schedule, text and duration conditioning |
| `stable-audio-open-small` | Open Small weights/config and cached components | Official conditional diffusion sampler; retains the model's native window |
| `rave` | Upstream TorchScript export with a trained prior, or explicit `source` recording | Calls exported `prior` then `decode`; refuses an absent prior. A supplied source selects reconstruction/style transfer |
| `audiogen` | Local AudioCraft model directory plus cached conditioning/compression components | Official `AudioGen` inference API |
| `musicgen` | Local AudioCraft model directory plus cached conditioning/compression components | Official `MusicGen` inference API; a general music reference, not a specialized drum model |

Copy the corresponding JSON in `examples/models/`, replace its asset paths, and
run:

```bash
uv run kicks external-generate --spec my-model-job.json \
  --python /absolute/path/to/isolated-environment/bin/python \
  --out output/external-model-comparison
```

The worker sets Hugging Face offline mode before importing providers. It never
downloads weights, accepts access terms or shares contact information. Dependencies
of a local checkpoint, such as a text encoder, must already be cached too. The
supervisor records weight/config hashes, software versions and failure diagnostics.
These hashes cover the supplied checkpoint/configuration; nested conditioning
cache files are not individually fingerprinted. Pin those upstream components
separately when preparing a reproducible large-model comparison.
Native sample rate, channels and complete output are retained as FLOAT WAVs.
Mono conversion, polyphase resampling, first-onset cropping, bounded gain and
fades are recorded separately for the common audit; none restores missing native
bandwidth. Extra events are not automatically judged. Listen to the native output
as well as the conformed one-shot.

Text prompts are not numerical descriptor conditioning. These adapters do not
claim calibrated sliders, and remain CLI comparisons rather than misleading
Studio controls. They are tested against provider-interface doubles, not against
real upstream weights. No Stable Audio, RAVE or AudioCraft weights or optional
environments were installed by this implementation work. DAC is covered below.

`environments/stable-audio.in` pins the reviewed upstream source revision because
the PyPI release was older than the current SFX interface. Resolve/install it in
a separate environment. `environments/audiocraft.in` pins AudioCraft 1.3.0 and its
older Torch runtime; use a separate Python 3.9 environment as documented upstream.
The worker does not import the main project, so it can run there. These files are
dependency inputs, not tested platform-specific lockfiles; optional environment
resolution and actual model inference remain to be verified on the chosen host.
[Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools),
[AudioCraft installation](https://github.com/facebookresearch/audiocraft#installation).

Stable Audio 3's model access requires the owner's acceptance of its model and
component terms and contact-sharing conditions. AudioCraft's published weights
have separate noncommercial terms. This local research implementation does not
accept either on the owner's behalf.
[Stable Audio 3 Small SFX model card](https://huggingface.co/stabilityai/stable-audio-3-small-sfx),
[AudioCraft licenses](https://github.com/facebookresearch/audiocraft#license).

## What ran locally

The source manifest was
`output/benchmarks/research-20261014-corpus-v2.json`, SHA-256
`3055e152d5f684b5f11e76352408cabb7b06a34e5b540d4e7bafbee558303b94`:
23,372 training, 510 validation and 45 test hits. Unresolved-source components
remain training-only (22,690 hits). These uneven splits protect known groups
and exact duplicates; they do not establish complete provenance or rule out
near-duplicate leakage. Review source and articulation labels before scaling.

- Twenty-one candidate checkpoints: all seven families for all three instruments.
- Eighteen non-codec implementation pilots: up to four optimizer updates on at
  most eight training hits, with two validation hits where supported. The first
  hi-hat hybrid pilot used two updates/four hits. Parameter models used the
  four-hit fitted training banks and no parameter-validation fold.
- Three DAC prior pilots: sixteen updates each, eight training and two validation
  hits. The official DAC `44khz / 8kbps / 0.0.1` artifact was downloaded from its
  release, hashed and used as one matched encoder/decoder pair.
- Eighty-four fresh pilot WAVs (four per checkpoint), all finite, nonsilent and
  below the clipping threshold. Reports and target errors are retained.
- Three actual held-out hi-hat DAC round trips. They establish that the matched
  adapter runs, not that its audible ceiling is adequate; codec pretraining overlap
  with the corpus is unknown.

Artifacts are under `models/experiments/neural-20261014/`,
`output/neural-20261014/`, and `output/fidelity/dac-ceiling-20261014/`. All are
ignored local artifacts. The native DAC source/identity record lives beside
the weight file in `models/experiments/codecs/`.
`output/neural-20261014/audition.html` offers all 84 pilot WAVs in one page.
To inspect that local catalog:

```bash
python3 -m http.server 8094 --bind 127.0.0.1 --directory output/neural-20261014
```

Open `http://127.0.0.1:8094/audition.html`. This serves the existing audio and
does not generate or judge new hits.

These budgets exercise implementation and checkpoint replay; they are far too
small to establish useful learned drum quality. No listener has judged them,
and their losses are not comparable across families. Larger source-balanced
training, aliasing/control/novelty audits and blind listening remain necessary.

## Validation

After the 14 October implementation, **185 Python tests passed**, including
gradients and seeded replay for every local family, bounded training, source separation, codec
identity checks, provider interfaces and API preview/export agreement.
All **8 frontend tests**, ESLint, TypeScript and the static build passed.
Two existing upstream deprecation warnings remain in the Python suite.

Browser checks exercised the new selector, learned-hybrid preview, rendered
spectrogram and measured-vs-requested descriptors, plus DAC-prior playback and
24-bit export. The audition page contains 84 native-contract players. A separate
eight-pair matched-condition DDSP/hybrid comparison is prepared at
`output/fidelity/hihat-neural-ddsp-hybrid-20261014/listening/listening.html`.
No browser warning/error logs were observed during those checks.

The three parameter-bank pilot checkpoints' selection metadata was corrected
to identify a training objective, with `validation_loss` set to null. Weight
tensors, audio and metric histories were unchanged; original/new checkpoint
identities are retained in `output/neural-20261014/metadata-corrections.json`.
