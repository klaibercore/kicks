# Modal training runbook

`scripts/modal_train.py` is a standalone uv script pinned to Modal 1.5.5. It
does not add Modal to the project environment. The training image is built with
uv 0.11.7 from the project's frozen `uv.lock`; Kicks runs in `/.uv/.venv` with protobuf 3.19.6.
Modal's container runtime must not share that venv: `uv_sync` prepends
`/.uv/.venv/bin` to `PATH`, Modal starts its runtime with the first `python` on
`PATH`, and the runtime then crashes importing protobuf 3.19.6 (the first
image test on 2026-09-26 crash-looped on exactly this). The image therefore
resets `PATH` to the base image's value after `uv_sync`, and the trainer is
started through the venv's interpreter by absolute path.

Every command that contacts Modal requires `--confirm-cloud`. Run and review
one step at a time. `plan` and `check-local` are local-only.

## Local preparation

```bash
uv run --script scripts/modal_train.py check-local
```

This checks all 23,927 WAV names, sizes, and SHA-256 hashes against
`data/corpus_ids.json`; checks each `SOURCES.md`; validates the translated
subset manifests; and writes `output/modal/local-corpus.json`.

## Reviewed cloud gates

0. If the existing detached upload reports `FAILED <corpus>`, review and retry
   only that directory. For example:

   ```bash
   uv run --script scripts/modal_train.py upload --corpus kicks --confirm-cloud
   ```

   The command rechecks every local corpus byte first, then overwrites/adds
   only the selected directory's WAV files on `kicks-corpus`. It never uploads
   `SOURCES.md` and does not delete remote files. The next verification gate
   rejects missing or extra WAV names.

1. After the existing upload log says `== done` and any failed directory has
   been repaired, verify every uploaded byte:

   ```bash
   uv run --script scripts/modal_train.py verify --confirm-cloud
   ```

   Expected counts are 10,109 kicks, 7,994 snares, and 5,824 hi-hats. Every WAV
   name, size and SHA-256 digest must match `data/corpus_ids.json`. The remote
   volume intentionally contains only training WAVs; `SOURCES.md` is neither
   required nor expected there. The permanent source/license records remain
   local in the three `SOURCES.md` files and `data/corpus_ids.json`; do not
   delete or regenerate those local records. Remote SHA-256 reads use a bounded
   32-worker pool and a four-hour safety ceiling; the receipt is
   `output/modal/corpus-verification.json`. No launch is allowed without it.

2. Build and test the image on CPU:

   ```bash
   uv run --script scripts/modal_train.py image-test --confirm-cloud
   ```

   This starts Modal's runtime, checks that it is not running inside the
   project venv, then starts the locked Kicks subprocess, asserts protobuf
   3.19.6, imports Torch, and constructs a small waveform denoiser. It writes
   `output/modal/image-test.json`.

   Watch the first seconds: a container that crashes at start makes Modal retry
   it repeatedly, and the local command waits. If the log shows
   `Runner failed with exit code`, press Ctrl-C; the ephemeral app stops and
   the command reports that nothing was recorded.

3. Print the exact smoke plan locally and review it:

   ```bash
   uv run --script scripts/modal_train.py plan \
     --instrument hihat --gpu L4 --job-id hihat-smoke-1 \
     --subset-manifest data/_subsets/hihat-64/manifest.json \
     --model-dir models/experiments/modal-hihat-smoke \
     --run-name "Hi-hat · Modal CUDA smoke" \
     --intent "Validate the locked training image and establish CUDA throughput" \
     --hypothesis "The unchanged waveform-diffusion loop completes two tracked epochs on L4 without protobuf or CUDA errors" \
     --success-criteria "Image uses protobuf 3.19.6; baseline and two epochs complete; checkpoint and run record sync; no non-finite loss; record epoch seconds and peak GPU memory" \
     -- --epochs 2 --batch-size 8 --grad-accum 1 \
        --eval-every 1 --eval-samples 4 --eval-steps 10 --preview 0
   ```

   `--job-id` is required and must be unique, so the reviewed plan and the
   launch name the same job, subset path and job record. The plan also prints
   the reservation: GPU, `--memory-gib` (default 16) and `--cpu` (default 4).
   Memory and CPU are billed on the reservation, not on use; the largest corpus
   (kicks) needs about 2.5 GiB as float32 waveforms.

4. Only after approving the printed plan, repeat it with `launch` and
   `--confirm-cloud`. The job is spawned durably and the local command returns a
   Modal FunctionCall ID. The corpus volume is mounted read-only. The subset is
   rebuilt from the manifest as ephemeral symlinks, and outputs go to the
   separate `kicks-training` volume.

   While the trainer runs, the wrapper commits `kicks-training` every five
   minutes. Modal commits a volume by itself only when the container exits, so
   without this a mid-run `sync` would see nothing and a hard kill (timeout,
   OOM, preemption) would lose the run's checkpoints. Checkpoints are written to
   a temporary file and renamed, so a commit never captures half a checkpoint;
   `sync` skips in-flight temporary files.

   Modal re-runs an input after worker preemption. The wrapper refuses that
   second attempt, and any launch whose `--job-id` already has a record or
   whose `--model-dir` already holds files on the volume. A preempted job
   therefore ends as `refused` with its partial run and last committed
   checkpoints intact; resuming it (fresh optimizer, restarted cosine) is a new,
   reviewed launch with `--resume`. The 24-hour function timeout is Modal's
   ceiling: size longer work as resumable chunks.

5. After the run finishes (or for a dashboard snapshot), review and run:

   ```bash
   uv run --script scripts/modal_train.py sync --confirm-cloud
   ```

   Sync updates `output/training` and the launched candidate model directory.
   Existing local `notes.json` and `notes.js` are never overwritten.

## Experiment sequence after the smoke

Do not schedule a long run until the smoke supplies actual L4 epoch timing and
the synchronized checkpoint can generate samples. Keep the architecture,
preprocessing, seed 42, 10% validation split, and conditioning settings fixed.

1. **CUDA smoke — 64 hi-hats, 2 epochs.** Pipeline-only gate. Record image
   versions, GPU name/memory, load time, baseline validation, seconds per epoch,
   and checkpoint/sync integrity.
2. **Signal check — 256 hi-hats.** Size the epoch count from the smoke to about
   one GPU-hour, retain periodic 10-step control proxies, then generate eight
   fixed-seed 50-step hits. Pass only if all sigma-band losses fall and output
   is recognizably non-white-noise.
3. **Sampler diagnosis before more training.** Use the existing 1,000-hit
   checkpoint and fixed target/texture seeds to compare the current 50-step
   sampler with any proposed schedule change. This isolates sampling from added
   training. Judge residual floor, thin body, double onsets, corpus score, and
   blinded audio—not denoising loss alone.
4. **2,000-hit training run.** Proceed only if the sampler diagnosis does not
   remove the known floor/body defect. Budget from measured CUDA throughput,
   use a new model directory, and predeclare audible and numeric thresholds.
5. **Full-corpus continuation.** Resume only after the 2,000-hit result passes
   blind listening and control checks. A resume uses a fresh optimizer and
   cosine schedule, so its learning rate and warm restart must be explicit in
   the hypothesis and notes.

The first three cloud steps—uploaded-corpus verification, image test, and CUDA
smoke—are separate approval points. Every later launch and sync is another
approval point.
