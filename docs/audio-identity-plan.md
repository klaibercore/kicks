# Preserve snare and hi-hat identity

The reported defect is loss of instrument identity: generic noise or blips.
The first intervention is the actual studio synthesis path, evaluated separately
from reconstruction of held-out examples.

## Diagnosis and implementation

1. **Center controls on typical sounds.** The current snare range puts the
   center decay target at 210 ms, while the corpus median is 19 ms. Shrinking an
   extreme-to-extreme range about its midpoint excludes the short sounds that
   dominate the corpus. Use a monotonic percentile mapping, including the median,
   for the identity-preserving control path.
2. **Preserve an actual encoded texture.** Select a deterministic corpus latent
   as the texture anchor. Optimize only the local descriptor-sensitive subspace;
   hold the remaining latent directions fixed. Bound movement around that anchor.
   A variation seed chooses texture independently of subsequent slider movement.
3. **Limit waveform reshaping.** Cap aggregate STFT correction at ±6 dB before
   peak normalization and penalize unnecessary correction. Preserve requested
   user effects after this step. Record target errors rather than erasing them
   with up to ±40 dB of spectral gain.
4. **Evaluate the newer hi-hat model.** Its existing reconstruction audit is much
   better, but prior control audits found extra onsets. Compare current and newer
   weights with the same old and new control algorithms. Keep the checkpoint
   promotion decision separate from the implementation decision.
5. **Measure and listen.** Save source / anchored reconstruction / old studio /
   new studio examples, use matched levels, and report onset count, descriptor
   errors, spectral and envelope distance, and finite/peak checks. A corpus
   realism score alone does not establish instrument identity.

## Predeclared engineering checks

- Same checkpoint, positions and seed reproduce the same latent and waveform;
  varying sliders keeps the seed's latent null-space component fixed locally.
- The center target equals the observed median, with monotonic slider mappings.
- New finite audio is peak-safe; no increase in multi-onset incidence on matched
  studio probes, and no forced 40 dB corrections to hit unsupported targets.
- Report descriptor error explicitly, separately for central/axis/random probes.
  Target p95 normalized error below 5% on central and single-axis probes; retain
  limitations if identity-preserving bounds make requested combinations unreachable.
- Quantitative gains are engineering evidence. Listening comparisons remain the
  acceptance check for the reported identity defect; do not infer rejection counts.

No model training is required to test these changes. If anchors preserve identity
but the decoder's own reconstruction still sounds inadequate, use the existing
tracked HF/attack-loss and capacity experiments before considering a new codec
sequence model. The untrained waveform diffusion backend
([`waveform-diffusion.md`](waveform-diffusion.md)) is the tracked alternative
once that ceiling is confirmed, hi-hat first. Never train or promote against a
descriptor score alone.

## Implementation and measured outcome

The studio now uses corpus-anchored controls for snare and hi-hat. Kick behavior
is unchanged. The final latent radius is 4 corpus-standardized units: radius 2
kept the snare safe but restricted decay unnecessarily. The anchor is fixed by
the seed, and only its local descriptor-sensitive subspace moves. The waveform
correction ceiling remains ±6 dB before peak normalization.

The completed audit uses 53 matching UI settings (centre, 20 axis probes and 32
random combinations), plus eight seeded centre variations. Percentile mapping
changes the physical targets at those UI settings; this is an end-to-end
comparison, not an isolated solver ablation.

| Model | Previous multi-onset renders / 53 | New / 53 | New / 61 including variations | Centre + axes p95 target error | Random p95 target error |
|---|---:|---:|---:|---:|---:|
| Served snare, epoch 198 | 3 | 0 | 0 | 14.16% | 21.41% |
| Served hi-hat, epoch 191 | 2 | 0 | 0 | 0.42% | 10.08% |
| Candidate hi-hat, epoch 187 | 20 | 1 | 1 | 5.96% | 15.43% |

Errors are measured on the vocoded waveform and normalized by each new target
range. The snare does **not** meet the proposed 5% axis-tracking target. This
tradeoff is retained explicitly: the priority is preserving the hit's structure,
and the studio reports limited reach when any target error exceeds 5%. Independent
control of all extreme combinations is not claimed. Mean realism scores alone
are not acceptance evidence for instrument identity.

The centre duration target is now the decoded corpus median: snare 17.85 ms
(previously 210.34 ms), hi-hat 38.15 ms (previously 116.44 ms). These are energy
centroids, not full audible sample lengths.

An additional four-texture comparison per instrument isolates the algorithm at
identical physical descriptor targets. It verifies each anchor against its source
encoding and RMS-matches source, reconstruction, previous solver and new solver.
Mean envelope error against the selected source falls from 13.37 to 4.84 dB for
snare and 20.37 to 13.98 dB for hi-hat. High-frequency body error is slightly worse
(7.75 to 8.05 dB and 7.10 to 7.24 dB respectively). These selected anchors are
**not held-out** examples and do not establish overall reconstruction fidelity.
Blind listening verdicts remain pending.

The newer hi-hat checkpoint remains a candidate: one multi-onset render and
weaker control tracking remain, despite its better earlier reconstruction audit.
No model weights were promoted or trained as part of this change.

Reproduce the final audit and listening comparisons:

```bash
uv run python scripts/validate_controls.py --instrument snare \
  --checkpoint models/snare/vae_best.pth --control identity --random 32 \
  --texture-seeds 1 2 3 4 5 6 7 8 --out output/identity/snare-r4
uv run python scripts/compare_identity.py --instrument snare \
  --audit output/identity/snare-r4
```

Use `--control legacy` for the previous algorithm. Substitute the instrument,
checkpoint and output path to compare hi-hats. Reports include checkpoint hashes;
private listening pages are under each audit's `listening/index.html`. The API
was also checked for reproducible WAV bytes, distinct seed 0/1 outputs, valid
spectrograms, and agreement between waveform measurements and `/evaluate`.
