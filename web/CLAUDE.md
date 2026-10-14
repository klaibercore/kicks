@AGENTS.md

# Local Studio and Analysis

Read the [repository contracts](../CLAUDE.md) and [setup guide](../README.md).
This is a local audio research frontend: `/` and `/studio/` open Studio;
`/analysis/` opens the corpus atlas. Accounts, billing, checkout, hosted kit
storage and deployment scaffolding were removed. Keep kits in localStorage
with validated JSON import/export.

- Discover instruments, engine availability and sliders from the API. Synth
  controls use physical units; neural engines request descriptors. Do not
  hardcode a fixed number of engines or sliders.
- Preserve checkpoint/backend/seed/steps/control identity across previews,
  cached pads and 24-bit export. Slow neural engines start with audition on
  release disabled. Keep stale async results from replacing a newer render.
- Research engines may return `corpus_available=false`. Show waveform and
  target measurements without inventing a corpus realism score.
- Preserve smooth hi-hat choke releases, bounded audio caches, local kit saves
  and named kit validation. Browsers need a user gesture for audio/MIDI access.
- `NEXT_PUBLIC_KICKS_API_URL` defaults to `http://localhost:8080`; the API is a
  separate loopback process. `pnpm dev` serves port 3000. See `.env.example`.
- Next.js uses Webpack, static export, trailing slashes and system fonts.
  AudioContext, Web MIDI and localStorage belong in client components. Use Base
  UI `render` composition and follow the React Compiler lint rules.
- After frontend changes, run `pnpm lint`, `pnpm test` and `pnpm build`; inspect
  affected views. Serve `out/` with a static server to inspect the export.
- Browser-safe analysis JSON must omit corpus filenames and paths. Regenerate
  it with `uv run kicks publish-analysis` from the repository root only when
  analysis assets need updating. Local fidelity/listening reports stay local.

The [neural guide](../docs/neural-models.md) records model setup, pilot budgets
and pending listening checks. Engine selection does not promote a checkpoint.
