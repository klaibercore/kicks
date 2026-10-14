# Local Studio implementation record

14 October 2026. Companion to the [framework research and generation plan](drum-generation-research-plan.md).

This record covers the original local-DSP phase. The subsequent neural/hybrid
implementation and DAC model download are recorded in [the neural guide](neural-models.md).

## Completed

- Added a deterministic NumPy/SciPy synthesizer for pitched kicks, modal/noise
  snares, and inharmonic FM/noise hi-hats. Fourfold oversampling, physical
  parameter ranges, twelve presets, velocity gain, seeded texture, smooth
  boundaries and choke releases are implemented.
- Integrated Synth beside the existing VAE and waveform diffusion engines.
  Physical sliders show Hz, milliseconds and mix values. Preview, spectrogram,
  evaluation, pads and 24-bit WAV export use the same render parameters.
  Synth starts without a corpus or checkpoint. Automatic startup prefers an
  available VAE, then configured diffusion, then Synth.
- Added reproducible JSON presets, optional hash-bound recorded texture,
  bounded multi-window inverse fitting, and joint empirical parameter-bank
  sampling. Recorded texture is sample-assisted synthesis; fitted banks are
  physical parameter priors. Neither is a trained neural texture model.
- Added immutable corpus manifests, source-group and exact-duplicate split
  protection, waveform audits shared by DSP and external engines, and blind
  matched-condition A/B preparation. Unknown source groups remain training-only,
  including components joined to known sources through exact duplicates.
- Added a local matched-DAC round-trip audit. It requires native 44.1 kHz
  weights and actual validation/test rows; it neither downloads weights nor
  falls back to training data. Adapter behavior is tested with a fake codec.
- Removed Supabase, Stripe, account/login/checkout/pricing/legal sales pages,
  credit charging, cloud kit storage, analytics, consent UI and deployment
  scaffolding. Kits autosave locally, support named saves, and import/export
  validated JSON. Studio is the home page; Analysis remains the corpus atlas.
- Removed build-time font downloads. Lint, Node regression tests, TypeScript
  and static export pass using Webpack. Existing corpora, checkpoints, research
  training tools, source records and upstream software licenses are retained.

## Local pilots and audition artifacts

Generated audio, manifests and fitted banks remain in ignored local directories,
so the repository does not acquire sample packs or model artifacts.

| Artifact | Local path |
|---|---|
| Twelve preset WAV/JSON pairs | `output/audition-20261014/presets/` |
| Starter kit with all twelve voices | `output/audition-20261014/drum-starter-kit.json` |
| Eight fitted-bank samples per instrument | `output/audition-20261014/fitted-{kick,snare,hihat}/` |
| Current frozen source manifest | `output/benchmarks/research-20261014-corpus-v2.json` |
| Three DSP baseline reports | `output/fidelity/dsp-{kick,snare,hihat}-20261014/report.json` |
| Recorded-texture hi-hat report | `output/fidelity/hihat-hybrid-20261014/report.json` |
| 72 blind full-band/HF pairs | `output/fidelity/hihat-ab-20261014/listening/listening.html` |
| Common waveform adapter pilot | `output/fidelity/kick-fitted-20261014/report.json` |
| Four-hit training parameter banks | `models/experiments/physical-{kick,snare,hihat}-20261014.json` |

The current manifest contains **23,927 hits**: 23,372 training, 510 validation
and 45 test. Its identity is
`3055e152d5f684b5f11e76352408cabb7b06a34e5b540d4e7bafbee558303b94`.
Whole-pack coverage repair produces uneven fractions. Most source labels remain
unresolved (22,690 hits); filename-inferred licenses/families need review.
These splits are a conservative starting point, not a balanced release benchmark.
The earlier manifest is retained as an immutable historical artifact.

Each DSP baseline used 24 physical settings × 3 seeds, including four presets:

| Voice | Hits | Silent / clipped | Warm render median / p95 |
|---|---:|---:|---:|
| Kick | 72 | 0 / 0 | 28.2 / 33.6 ms |
| Snare | 72 | 0 / 0 | 47.7 / 56.6 ms |
| Hi-hat | 72 | 0 / 0 | 46.9 / 56.4 ms |

These are CPU voice-render timings on this computer, recorded with concurrent
pilot work; they exclude HTTP, effects, browser decoding and playback. They are
not native real-time plug-in guarantees. The sample-assisted hi-hat run produced
72 additional hits without silence/clipping at a 40.8 ms warm median. Its texture
is the training-fold recording `hihats/#22542.wav` (original
`hat_20_chik_vl1_rr1.wav`), with mix 0.2 and source hash in every preset.

Four randomly selected training hits per voice were fitted with a strict budget
of 32 objective evaluations each. Mean active-bin multi-window magnitude loss:

| Voice | Best starting preset | Fitted settings |
|---|---:|---:|
| Kick | 9.54 | 8.70 |
| Snare | 10.17 | 8.46 |
| Hi-hat | 12.34 | 10.62 |

This tiny training pilot confirms that fitting executes and preserves or improves
its objective. It does not establish acoustic realism or held-out generalization.
No listener scores have been collected, and no engine has been declared superior.

## Validation and branch audit

Browser checks confirmed Synth presets, audition, a rendered spectrogram, WAV
export, importing and saving the starter kit, and loading the Analysis corpus
atlas. The browser reported no warning/error logs during these checks. MIDI
hardware behavior and human listening quality remain untested.

Final checks: **158 Python tests passed**, **8 frontend tests passed**, ESLint
reported no issues, and the static build completed with TypeScript validation.
The Python suite emitted two upstream deprecation warnings. The code graph was
regenerated after the implementation changes.

Main already contained the earlier training and synchronization changes. The
only remaining feature work was `studio-diffusion`, descended directly from
main. Its Studio diffusion integration is retained alongside Synth; there are
no divergent branch changes requiring conflict resolution. A fresh remote-ref
audit also pruned a remotely deleted research branch.

## Next decisions

1. Listen to the presets and the blind hi-hat pairs. Record attack, body, tail,
   metallic character, variation and preference separately. Choose the intended
   electronic/acoustic families before setting quality thresholds.
2. Resolve corpus source/articulation labels and near duplicates. Define a
   balanced family-specific evaluation set and control targets.
3. Compare existing VAE/diffusion final WAVs with the same audits and listening
   conditions. Add larger fitting pilots only after the target families are clear.
4. Run a matched codec ceiling with local DAC weights. Stable Audio 3 Small SFX
   remains an external candidate requiring its own isolated setup and model
   access. Neural texture learning, a codec prior, and Faust/WASM or JUCE ports
   remain conditional on measured quality and the intended runtime.

No pretrained model was downloaded, no candidate weights were promoted, and no
new large training job was started by this work.
