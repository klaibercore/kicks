# Cloud diffusion training on Modal — plan

**Status: proposal, 2026-09-25. Nothing in this document is implemented.** No
Modal account, Volume, launcher or cloud run exists yet. Prices are the
published Modal rates as reported in September 2026. Throughput figures are
estimates derived from measured FLOPs and activation memory, not GPU
benchmarks. Phase D replaces them with measurements.

## Goal and constraints

- **Scope: waveform diffusion training only** (`kicks diffusion-train`, see
  [waveform-diffusion.md](waveform-diffusion.md)). VAE training, the API and
  the studio stay where they are.
- **The corpus lives in the cloud.** After a one-time upload and a verified
  checksum match, training, sampling and evidence generation run with the Mac
  switched off. The Mac is needed again only for listening, and even that is
  optional if the dashboard is hosted (see [Monitoring](#monitoring-while-the-mac-is-offline)).
- **The samples are CC0**, so storing them on third-party infrastructure has no
  licensing constraint.
- **The repository's training rules still apply.** Each phase waits for the
  user's go and is preceded by a summary of the previous one. Every run carries
  `--run-name`, `--intent`, `--hypothesis` and `--success-criteria`. Listening
  verdicts come from a person, never from a proxy.

## Why Modal needs code work first

Three facts about Modal decide the shape of this plan:

1. **GPU functions can be preempted, and the `nonpreemptible` option is not
   available for GPU functions.** Modal's recommendation is to checkpoint and
   resume, retrying with `modal.Retries` (at most 10 retries).
2. **A single function call runs for at most 24 hours.** Each retry starts a
   new timeout. Every run estimated below fits within one call.
3. **Volumes are eventually consistent between containers.** Writes are
   committed in the background every few seconds and on exit. A second
   container sees them only after a reload, and it cannot reload while it
   holds open files on that Volume.

Today `train_diffusion()` saves EMA weights only when validation improves
(`diffusion_best.pth`), and a final checkpoint at the end. `--resume` loads
weights with a *fresh* optimizer and a restarted cosine schedule
(`kicks/cli.py:198`). A preempted run therefore either starts over, losing the
money spent, or resumes as a warm restart, which is a different experiment.
**Exact resume is the one hard prerequisite** (Phase C1).

## Architecture

```
           one-time upload (Mac online)                     no Mac needed
 ┌──────────┐  modal volume put   ┌──────────────────────────────────────────────┐
 │  Mac     │ ──────────────────▶ │ Volume "kicks-store" mounted at /vol          │
 │  data/   │                     │   /vol/data/{kicks,snares,hihats}/  corpus    │
 │  output/ │ ◀────────────────── │   /vol/data/_subsets/                         │
 └──────────┘  modal volume get   │   /vol/data/_manifests/<instrument>.json      │
   (optional, when back online)   │   /vol/models/experiments/diffusion-*/        │
                                  │   /vol/output/training/<run-id>/              │
                                  │   /vol/output/[hihat/|snare/]cluster_analysis │
                                  │   /vol/output/fidelity/<report>/              │
                                  └──────────────▲───────────────────────────────┘
                                                 │ KICKS_DATA_DIR=/vol/data
                                                 │ KICKS_MODEL_DIR=/vol/models
                                                 │ KICKS_OUTPUT_DIR=/vol/output
                     ┌───────────────────────────┴───────────────────────────┐
                     │ Modal app "kicks-diffusion" (image built from uv.lock) │
                     │  verify_corpus  (CPU)   make_subset  (CPU)             │
                     │  benchmark      (GPU)   train        (GPU, retries)    │
                     │  evidence       (GPU)   dashboard    (optional, web)   │
                     └────────────────────────────────────────────────────────┘
```

- **One Volume, same layout as the repository.** The profile paths already
  resolve from `KICKS_DATA_DIR`, `KICKS_MODEL_DIR` and `KICKS_OUTPUT_DIR`, so
  the code runs unchanged against `/vol/...`. The kick keeps its flat layout
  (`/vol/output/cluster_analysis.json`), while snare and hi-hat use `snare/` and
  `hihat/` subdirectories, as they do locally.
- **The image is built from `pyproject.toml` + `uv.lock`**: `uv sync --frozen`
  on Debian slim with `libsndfile1`, mirroring the `Dockerfile`. The lock pins
  torch 2.10 from PyPI, which is the CUDA build on Linux. The launcher records
  the Git commit it built from in the run notebook, so every checkpoint traces
  back to exact code.
- **Container resources:** 2–4 CPU cores and 16 GiB of RAM for GPU functions.
  The float32 corpus takes 1.4–2.5 GiB in RAM, and a few Python-side cores keep
  kernel dispatch from becoming the bottleneck.
- **The `modal` package stays out of the runtime dependencies.** It goes in a
  `cloud` dependency group, and nothing under `kicks/` imports it.

## Phases

Estimated costs are for an A10G, including about $0.20/h for CPU and RAM.
Phases A–C cost nothing on Modal.

### Phase A — account and guard rails (Mac online)

1. Create a Modal account on the Starter plan (free, $30/month credit, 10
   concurrent GPUs). Run `modal token new` on the Mac.
2. Set a workspace spending limit or budget alert, if the account offers one
   (check in the Modal settings). Every function also gets an explicit
   `timeout` of about 1.5× its estimate and `max_retries` of 3 rather than 10,
   so a run stuck in a crash loop cannot silently spend 10 × 24 h.
3. Decide how runs will be **launched while the Mac is offline** (below).

### Phase B — move the corpus to the cloud (Mac online, one time)

1. **Freeze the corpora.** No `kicks strip`, `kicks clean` or new staging
   integration during the upload.
2. **Write a content manifest per instrument:** relative path, size and
   SHA-256 of every file, plus the file count. This is needed because
   `split_fingerprint()` hashes *resolved paths and mtimes*. Cloud paths differ
   from `/Users/...`, and uploads may not preserve mtimes, so a cloud run's
   fingerprint will never match an M1 run's even on identical files. The
   manifest is the cross-device identity. (Code change: a small script,
   Phase C4.)
3. **Upload**, from the repository root:
   - `data/kicks`, `data/snares`, `data/hihats` → `/vol/data/...`
   - the three `cluster_analysis.json` reports. `scripts/make_subset.py` needs
     them, and regenerating them in the cloud would need the VAE checkpoints,
     since `kicks cluster` encodes the corpus with the VAE.
   - the manifests → `/vol/data/_manifests/`
   - for comparison: `models/experiments/diffusion-hihat-*`, the 2026-09-20
     run directories under `output/training/`, and
     `output/fidelity/hihat-diffusion-1000-ab`.

   Size: the three corpora are 1.4–2.5 GiB each as float32 in memory. On disk
   they should be of the same order or smaller, well inside the 1 TiB/month of
   Volume storage that is included.
4. **Verify in the cloud** (`verify_corpus`, CPU only, cents): recompute every
   SHA-256 on the Volume and compare against the manifest. Then load each
   corpus through `WaveformDataset` to catch unreadable files, and record the
   ingestion time. Only after a clean match is the Mac optional.
5. **Decide the source of truth** (question 1). Recommendation: the Volume is
   authoritative for training corpora from now on, and the local copy is a
   backup. I found no documented backup or versioning guarantee for Volumes,
   and `modal volume delete` is permanent, so keep an independent copy. Any
   later corpus edit means re-uploading, re-verifying and re-running
   `kicks cluster` locally, then uploading the new report.

### Phase C — code prerequisites (one pull request, tested before any cloud run)

1. **Exact resume (required).**
   - Every epoch, plus every N minutes during long epochs, atomically write a
     full training state to the Volume and commit it: live weights, EMA shadow
     and step count, optimizer, scheduler, next epoch, noise-generator and
     loader-generator states, the torch RNG, best-validation and
     best-control values, and the tracking run ID.
   - On a retry, the same run ID continues, so the dashboard shows one run
     rather than a trail of fragments.
   - Keep this separate from `--resume`, which stays a documented fine-tune
     (warm restart).
   - Test: on CPU, a tiny config interrupted and resumed must reproduce the
     uninterrupted run's losses and weights exactly.
2. **The Modal launcher** (`cloud/modal_app.py`). It holds the image, the
   Volume mount, the environment roots and these functions:
   - `verify_corpus(instrument)`, CPU;
   - `make_subset(instrument, size, seed)`, CPU, wrapping
     `scripts/make_subset.py`;
   - `benchmark(gpu, micro_batch, steps)`, GPU: step time, peak
     `torch.cuda.max_memory_allocated`, ingestion time;
   - `train(args, gpu)`, GPU: runs `kicks diffusion-train` with
     `timeout`, `retries` and `single_use_containers=True`, as in Modal's
     long-training example;
   - `evidence(run, checkpoint)`, GPU: `kicks diffusion-generate`,
     `kicks eval --pattern`, `scripts/diffusion_listening.py`.

   The GPU type is chosen per call with `.with_options(gpu=...)`.
3. **Portable subsets.** `scripts/make_subset.py:101-104` writes absolute
   symlinks. Built *inside* the cloud against a fixed `/vol` mount they should
   resolve, but symlink behaviour on Volumes needs a test in Phase D. The
   fallback is a `--copy` mode, or relative links.
4. **The corpus manifest script** (Phase B2), and the manifest's digest
   recorded in each cloud run's config beside `split_fingerprint`.
5. **Documentation.** `CLAUDE.md` currently requires `nohup … &` with parent
   `launchd` for diffusion runs. Add the cloud equivalent: `modal run --detach`,
   check the app in the Modal dashboard, and follow the preemption and resume
   rules. Update `docs/waveform-diffusion.md` with the measured GPU figures once
   they exist.
6. *Optional, not assumed in any estimate:*
   - drop the per-micro-batch `loss.item()` host sync;
   - do the EMA update with `torch._foreach_*` instead of a Python loop over
     470 state-dict entries;
   - `torch.compile`;
   - a preprocessed corpus cache, so ingestion is not billed at GPU rates.

   bf16 autocast could give roughly 1.5–2× but may cost low-sigma detail, so it
   would need its own ablation and is not part of this plan.

Validation: `uv run pytest -q tests/test_diffusion.py tests/test_make_subset.py
tests/test_diffusion_listening.py` plus the new resume test. These establish
nothing about sound, as today.

### Phase D — smoke test, benchmark and preemption drill (≈ $1)

1. Run the diffusion test modules inside the Modal image on a GPU.
2. Smoke run: 64-hit hi-hat subset built on the Volume (this also tests the
   symlinks), `--epochs 2 --eval-every 1 --eval-samples 4 --eval-steps 10`.
3. Throughput probe: A10G, L40S and A100 40 GB at micro-batches 8 / 16 / 32,
   about 5 minutes each. Record s/sample, peak memory and OOMs. Pick the GPU
   with the lowest measured **$ per sample-pass**. The A100 wins only if it is
   more than about 1.8× faster than the A10G.
4. Preemption drill: stop the app mid-epoch (`modal app stop`), relaunch, and
   confirm exact resume on the same run ID.
5. Measure how quickly a second container sees `run.json` updates after a
   Volume reload. This decides whether a hosted dashboard is practical.

### Phase E — device parity (≈ $0.3)

Repeat the 1,000-hit hi-hat recipe: seed 42, effective batch 8 (micro-batch 8,
no accumulation), `--learning-rate 1e-4`, 50 epochs. Compare validation loss,
the three sigma bands and `control_mae` with `20260920T124135`. That M1
reference was itself a 7 + 43-epoch warm restart, and the noise draws differ
with the micro-batch size, so this is a sanity check that the GPU learns the
same way, not an identity test. Record `device` and TF32 in the notebook:
PyTorch lets cuDNN convolutions use TF32 on Ampere and newer by default.

### Phase F — full hi-hat baseline (≈ $4–7, ≈ 3–6 h)

The open question from [First runs](waveform-diffusion.md#first-runs): do the
residual tail floor and the thin body keep improving with more steps? Train on
the full hi-hat corpus (5,824 hits) with the baseline parameters below, with
`--eval-every 5`. Evidence is produced in the cloud (`evidence` function):

- 8+ hits at 50 steps with fixed texture and label seeds;
- `kicks eval --pattern`;
- the residual tail floor and onset measurements;
- blind pairs from `scripts/diffusion_listening.py`.

### Phase G — listening (human step)

The Listening lab needs a person with headphones. Either pull the report with
`modal volume get` and open the local dashboard, or use the hosted dashboard if
option M3 below is adopted. Copy the counts and findings into the run notebook.

### Phase H — snare and kick (≈ $11–22 together)

Only if the hi-hat evidence justifies it. Same recipe, one instrument at a
time, each gated on the previous result.

## Baseline training parameters on GPU

| Option | M1 runs | Cloud baseline | Reason |
|---|---|---|---|
| `--batch-size` | 2 | 16 on A10G, 32 on A100/L40S | Amortizes ~2,740 dispatched ops per micro-batch; ~0.7 GiB activations per sample |
| `--grad-accum` | 4 | 2 on A10G, 1 on A100/L40S | Effective batch 32, the documented default |
| `--learning-rate` | 1e-4 | 1e-4 | One change at a time; 2e-4 is a later screen |
| `--epochs` | 43–50 on subsets | 200 on the full corpus | ≈ 33k optimizer steps for hi-hat, 45k snare, 57k kick |
| `--ema-decay` | 0.999 | 0.999 | Half-life ≈ 700 steps, about 2 % of the run |
| `--eval-every` / `--eval-samples` / `--eval-steps` | 0–15 / 8 / 20 | 5 / 32 / 50 | ≈ 2 % overhead on a GPU; a much less noisy `control_mae` |
| `--preview` | 0 | 0 | Evidence rendered separately by `evidence` |
| `--val-split`, `--seed`, `--cond-dropout`, architecture | 0.1, 42, 0.1, default | unchanged | Comparability |
| `--data` | local subset | `/vol/data/hihats` or `/vol/data/_subsets/...` | Cloud corpus |
| `--model-dir` | local experiment dir | `/vol/models/experiments/diffusion-<instrument>-<tag>` | Separate candidate root, as required |

## Cost and time estimates

Measured on the real network (CPU, 2026-09-25): 9.56 M parameters,
**59.5 GFLOP per training sample**, 19.8 GFLOP per forward (sampling) pass,
**≈ 700 MiB of saved activations per sample** in fp32. The M1 reaches only
≈ 183 GFLOP/s, and the workload is memory-bandwidth bound, which is what the
GPU estimates scale by.

| GPU | $/h incl. CPU/RAM | Est. ms/sample | $ per 1 M sample-passes |
|---|---|---|---|
| A10G 24 GB | 1.30 | 9–18 | 3.3–6.5 |
| L40S 48 GB | 2.15 | 6–12 | 3.6–7.2 |
| A100 40 GB | 2.30 | 4–8 | 2.6–5.1 |
| L4 / T4 | 1.00 / 0.79 | 18–45 | 5.0–9.9 (not recommended: low memory bandwidth) |
| H100 80 GB | 4.15 | 2–5 | 2.3–5.8 (likely limited by kernel dispatch on a model this small) |

| Phase | Sample-passes | A10G | A100 40 GB | M1 for reference |
|---|---|---|---|---|
| D smoke + benchmark | — | ≈ $1 | ≈ $1 | — |
| E device parity | 0.05 M | 0.1–0.2 h · $0.2–0.3 | < 0.1 h · $0.1–0.2 | ≈ 4 h |
| F full hi-hat, 200 epochs | 1.11 M | 2.8–5.6 h · $3.6–7.2 | 1.2–2.5 h · $2.8–5.7 | ≈ 100 h |
| H full snare, 200 epochs | 1.53 M | 3.8–7.6 h · $5.0–9.9 | 1.7–3.4 h · $3.9–7.8 | ≈ 138 h |
| H full kick, 200 epochs | 1.93 M | 4.8–9.6 h · $6.3–12.5 | 2.1–4.3 h · $4.9–9.9 | ≈ 174 h |
| **All of the above** | 4.6 M | **≈ $16–31** | **≈ $13–25** | ≈ 16 days |

Sample-passes include ≈ 6 % for validation and the control proxy. Not included:
- corpus ingestion at each container start, billed at the GPU rate (measured
  in Phase D);
- work lost to preemptions (small once exact resume exists);
- ablations. A handful (LR 2e-4, 400 epochs, wider channels) would add
  roughly $10–40.

Volume storage fits in the included 1 TiB.

## Launching while the Mac is offline

| Option | How | Trade-off |
|---|---|---|
| **L1 (recommended to start)** | Launch from the Mac with `modal run --detach …`, then go offline | No extra setup. The run survives the Mac going offline, but the next phase needs the Mac again |
| L2 | GitHub Actions `workflow_dispatch` workflow with `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` as repository secrets | Launch from a phone via GitHub. The image builds from the pushed commit, so launched code is always committed code |
| L3 | A Claude Code cloud session, with `modal.com` allowed in the environment's network settings and the token stored as environment variables | Claude can launch, monitor and summarize phases. The Modal token can spend money, so scope it and keep the spending limit |

## Monitoring while the Mac is offline

- **M1 (no code): Modal's web dashboard.** App status, logs and GPU usage from
  any browser. The trainer logs `control_mae` lines and a progress bar.
- **M2 (no code): run records stay on the Volume.** Back online, run
  `modal volume get kicks-store /output/training output/training` and then
  `uv run kicks dashboard` for the full charts and notebook.
- **M3 (code change, decision needed): host the dashboard on Modal** with
  `@modal.web_server`, serving the Volume's run records. The viewer binds
  `127.0.0.1` (`kicks/training/tracking.py:526`) and `CLAUDE.md` calls it
  "loopback-only and never a publication path", because fidelity details
  include corpus sample paths. Hosting it needs a host option and
  authentication, plus an explicit change to that rule. Modal proxy auth
  expects `Modal-Key` / `Modal-Secret` headers, which a phone browser cannot
  send without help, so a simple login or a signed link would be needed.
  Notebook edits from the hosted viewer and the trainer's writes go to
  different files (`notes.json` versus `run.json`), which Volumes allow.

## Risks

| Risk | Mitigation |
|---|---|
| Preemption of a GPU function | Exact resume (C1), retries, drill in D4 |
| Crash loop burns credit | Per-function timeout ≈ 1.5× estimate, `max_retries=3`, spending limit |
| Cloud and M1 runs compared as if identical | Content manifest digest in config; parity phase E; device and TF32 in the notebook |
| Corpus drift between Mac and cloud | Freeze during the campaign; the Volume is the source of truth; re-verify after any change |
| Loss of the Volume | Keep the local copy as backup; never script `modal volume delete` |
| Stale dashboard data | Background commits every few seconds; the viewer reloads the Volume; latency measured in D5 |
| Estimates wrong | Phase D measures before any money beyond ≈ $1 is spent |

## Decisions needed before Phase A

1. Should the Volume become the source of truth for the corpora, with the Mac
   copy as backup, or should the Mac stay authoritative with the cloud as a
   mirror?
2. Launch option: L1, L2 or L3?
3. Monitoring: are Modal's logs plus a later `volume get` enough, or should
   hosting the dashboard (M3) be planned, which changes the loopback-only rule?
4. Effective batch 32 as the new cloud baseline, or effective batch 8 for the
   first full run, to stay comparable with the M1 runs?
5. After Phase E, go straight to the full hi-hat corpus, or keep the 2,000-hit
   step as a cheap gate?
6. Monthly ceiling: the $30 Starter credit, or higher?

## Sources

- [Modal pricing](https://modal.com/pricing)
- [Modal preemption](https://modal.com/docs/guide/preemption)
- [Modal timeouts](https://modal.com/docs/guide/timeouts)
- [Long, resumable training on Modal](https://modal.com/docs/examples/long-training)
  ([source](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/long-training.py))
- [Modal Volumes](https://modal.com/docs/guide/volumes)
- [Modal Volumes v2 announcement](https://modal.com/blog/modal-product-updates-oct-2025)
- [`modal.web_server`](https://modal.com/docs/reference/modal.web_server)
- [Modal proxy tokens](https://modal.com/docs/guide/webhook-proxy-auth)
- Secondary summaries of the pricing: [eesel](https://www.eesel.ai/blog/modal-ai-pricing),
  [Spheron](https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/),
  [CostBench](https://costbench.com/software/ai-gpu-cloud/modal/)
