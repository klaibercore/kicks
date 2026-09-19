"use client";

import {
  ArrowDown,
  ArrowUpRight,
  Box,
  CircleHelp,
  Layers3,
  Scan,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  AudioButton,
  ClusterCards,
  ClusterProfile,
  useClusterAudio,
} from "@/components/analysis/cluster-cards";
import { Distributions } from "@/components/analysis/distributions";
import { Heatmap } from "@/components/analysis/heatmap";
import { ModelEvidence } from "@/components/analysis/model-evidence";
import { SampleTable } from "@/components/analysis/sample-table";
import {
  ScatterCanvas,
  type AxisPair,
  type ProjectionSpace,
} from "@/components/analysis/scatter-canvas";
import { VarianceChart } from "@/components/analysis/variance-chart";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { loadIndex, loadReport } from "@/lib/analysis/load";
import { clusterColor } from "@/lib/analysis/palette";
import {
  clusterName,
  clusterSummary,
  confidenceThreshold,
  descriptorUnit,
  formatMs,
  rankedClusters,
} from "@/lib/analysis/stats";
import type {
  AnalysisIndex,
  AnalysisIndexEntry,
  Report,
  SampleRow,
} from "@/lib/analysis/types";
import { cn } from "@/lib/utils";

const number = (n: number) => n.toLocaleString("en-US");
const clusterId = (k: number) => `C${String(k + 1).padStart(2, "0")}`;

export function CorpusAnalysis() {
  const [index, setIndex] = useState<AnalysisIndex | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const [loaded, setLoaded] = useState<{ name: string; report: Report } | null>(
    null,
  );
  const [error, setError] = useState<{
    name: string | null;
    message: string;
  } | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let cancelled = false;
    loadIndex()
      .then((idx) => {
        if (!cancelled) {
          setIndex(idx);
          setActive((current) => current ?? idx.instruments[0]?.name ?? null);
        }
      })
      .catch((e: Error) => {
        if (!cancelled) setError({ name: null, message: e.message });
      });
    return () => {
      cancelled = true;
    };
  }, [attempt]);
  useEffect(() => {
    const entry = index?.instruments.find((i) => i.name === active);
    if (!entry) return;
    let cancelled = false;
    loadReport(entry.report)
      .then((report) => {
        if (!cancelled) setLoaded({ name: entry.name, report });
      })
      .catch((e: Error) => {
        if (!cancelled) setError({ name: entry.name, message: e.message });
      });
    return () => {
      cancelled = true;
    };
  }, [index, active, attempt]);
  const report = loaded?.name === active ? loaded.report : null;
  const entry = index?.instruments.find((i) => i.name === active);
  const currentError =
    error && (error.name === null || error.name === active) ? error : null;

  return (
    <div className="analysis-page mx-auto max-w-[1440px] px-4 pb-20 pt-10 sm:px-8 lg:px-12">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-6 border-b border-border pb-8">
        <div>
          <p className="mb-4 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[.22em] text-muted-foreground">
            <span className="size-1.5 rounded-full bg-blue-500" />
            Kicks research / Corpus atlas
          </p>
          <h1 className="text-4xl font-medium leading-[1.08] tracking-[-.045em] sm:text-5xl lg:text-[3.5rem]">
            The shape of sound.
          </h1>
          <p className="mt-4 max-w-xl text-sm leading-relaxed text-muted-foreground">
            Discover the families, contrasts and in-between sounds inside the
            training corpus. Follow the structure. Hear what connects it.
          </p>
        </div>
        <a
          href="#explorer"
          className="group inline-flex items-center gap-3 rounded-full border border-border px-4 py-2.5 text-xs font-medium transition-colors hover:bg-muted"
        >
          Explore the corpus
          <ArrowDown className="size-3.5 transition-transform group-hover:translate-y-0.5" />
        </a>
      </header>
      <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <Tabs
          value={active ?? ""}
          onValueChange={(value) => {
            if (value) setActive(value as string);
          }}
        >
          <TabsList variant="line" className="h-11 gap-2">
            {index
              ? index.instruments.map((inst) => (
                  <TabsTrigger
                    key={inst.name}
                    value={inst.name}
                    className="gap-3 px-3 text-sm sm:px-5"
                  >
                    {inst.display_name}
                    <span className="font-mono text-[10px] text-muted-foreground">
                      {number(inst.n_samples)}
                    </span>
                  </TabsTrigger>
                ))
              : [0, 1, 2].map((i) => <Skeleton key={i} className="h-8 w-24" />)}
          </TabsList>
        </Tabs>
        <div className="flex items-center gap-5 text-[11px] text-muted-foreground">
          <a href="#families" className="hover:text-foreground">
            Sound families
          </a>
          <a href="#evidence" className="hover:text-foreground">
            Evidence & method
          </a>
        </div>
      </div>
      {currentError ? (
        <div role="alert" className="rounded-xl border border-border p-8">
          <h2 className="text-lg font-medium">
            The corpus could not be loaded.
          </h2>
          <p className="my-3 text-sm text-muted-foreground">
            {currentError.message}
          </p>
          <Button
            variant="outline"
            onClick={() => {
              setError(null);
              setAttempt((n) => n + 1);
            }}
          >
            Try again
          </Button>
        </div>
      ) : index && !index.instruments.length ? (
        <p className="py-12 text-muted-foreground">
          No corpus reports are available yet.
        </p>
      ) : report && entry ? (
        <AnalysisReport key={entry.name} report={report} entry={entry} />
      ) : (
        <div
          role="status"
          aria-label="Loading corpus analysis"
          className="space-y-5"
        >
          <Skeleton className="h-28 w-full" />
          <Skeleton className="h-[560px] w-full" />
        </div>
      )}
    </div>
  );
}

function AnalysisReport({
  report,
  entry,
}: {
  report: Report;
  entry: AnalysisIndexEntry;
}) {
  const [isolated, setIsolated] = useState<number | null>(null);
  const [selected, setSelected] = useState<SampleRow | null>(null);
  const [space, setSpace] = useState<ProjectionSpace>("descriptors");
  const [mode, setMode] = useState<"3d" | "2d">("3d");
  const [pair, setPair] = useState<AxisPair>("12");
  const [uncertain, setUncertain] = useState(false);
  const [allFamilies, setAllFamilies] = useState(false);
  const audio = useClusterAudio(report.instrument);
  const threshold = confidenceThreshold(report);
  const clusters = useMemo(() => rankedClusters(report), [report]);
  const samples = useMemo(
    () => report.samples.filter((s) => !uncertain || s.confidence < threshold),
    [report, uncertain, threshold],
  );
  const visibleCount = samples.filter(
    (s) => isolated === null || s.cluster === isolated,
  ).length;
  const ambiguous = report.samples.filter(
    (s) => s.confidence < threshold,
  ).length;
  const largest = clusters[0];
  const largestShare =
    (report.cluster_profiles[largest].count / report.samples.length) * 100;
  const topThreeShare =
    (clusters
      .slice(0, 3)
      .reduce((sum, k) => sum + report.cluster_profiles[k].count, 0) /
      report.samples.length) *
    100;
  const meanLength =
    report.samples.reduce((sum, s) => sum + s.duration_ms, 0) /
    report.samples.length;
  const projectionVariance =
    space === "latent"
      ? report.latent_projection!.variance_explained
      : report.pca_variance_explained;
  const dimensions =
    mode === "3d" ? [0, 1, 2] : [Number(pair[0]) - 1, Number(pair[1]) - 1];
  const retained =
    dimensions.reduce((sum, i) => sum + (projectionVariance[i] ?? 0), 0) * 100;
  const correlations = report.descriptor_keys
    .flatMap((a, i) =>
      report.descriptor_keys
        .slice(i + 1)
        .map((b) => ({ a, b, r: report.descriptor_correlations[a][b] })),
    )
    .sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
  const strongest = correlations[0];
  const label = (key: string) =>
    report.descriptor_labels[report.descriptor_keys.indexOf(key)];
  const focusCluster = (cluster: number | null) => {
    setIsolated(cluster);
    setSelected(null);
  };
  const pick = (sample: SampleRow) => {
    setSelected(sample);
    setIsolated(sample.cluster);
  };
  const exploreCluster = (cluster: number | null) => {
    focusCluster(cluster);
    document.getElementById("explorer")?.scrollIntoView({ block: "start" });
  };
  const centralMember = isolated === null ? undefined : report.samples.find(
    (sample) => sample.sample_idx === report.cluster_details?.[isolated]?.representative_idx,
  );
  const confidence =
    isolated === null
      ? 0
      : report.samples
          .filter((s) => s.cluster === isolated)
          .reduce((sum, s) => sum + s.confidence, 0) /
        report.cluster_profiles[isolated].count;

  return (
    <div className="space-y-12">
      <section aria-label="Corpus overview">
        <div className="grid gap-6 pb-7 lg:grid-cols-[1.15fr_1fr] lg:gap-16">
          <div>
            <p className="mb-2 font-mono text-[10px] uppercase tracking-[.15em] text-muted-foreground">
              01 / Read the corpus
            </p>
            <h2 className="text-2xl font-medium leading-tight tracking-tight">
              {number(report.samples.length)} sounds.
              <br />
              {clusters.length} ways of belonging.
            </h2>
            <p className="mt-3 max-w-lg text-sm leading-relaxed text-muted-foreground">
              {entry.description} The {Math.min(3, clusters.length)} largest
              groups account for {topThreeShare.toFixed(0)}% of the analyzed
              corpus; the rest reveal its less common variations.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-x-6 gap-y-5 self-center">
            <Stat
              label="Corpus coverage"
              value={`${((report.samples.length / report.corpus.n_total) * 100).toFixed(0)}%`}
              hint={`${number(report.corpus.n_total)} source samples`}
            />
            <Stat
              label="Sound families"
              value={String(clusters.length)}
              hint={
                report.clustering
                  ? `Selected from 1–${report.clustering.max_k} components`
                  : "Latent-space mixture groups"
              }
            />
            <Stat
              label="Mean source length"
              value={formatMs(meanLength)}
              hint={`${formatMs(report.corpus.audio_length_ms)} analysis window`}
            />
            <Stat
              label="Uncertain membership"
              value={`${((ambiguous / report.samples.length) * 100).toFixed(1)}%`}
              hint={`Below ${threshold * 100}% model probability`}
            />
          </div>
        </div>
        <div
          className="flex h-3 gap-0.5 overflow-hidden rounded-sm"
          aria-label="Cluster population shares"
        >
          {clusters.map((k) => (
            <button
              key={k}
              onClick={() => exploreCluster(isolated === k ? null : k)}
              aria-label={`${clusterId(k)}: ${clusterName(report, k)}, ${report.cluster_profiles[k].count} samples`}
              title={`${clusterId(k)} · ${clusterName(report, k)}`}
              className="min-w-1 transition-opacity hover:opacity-70 focus-visible:z-10 focus-visible:outline-2 focus-visible:outline-foreground"
              style={{
                flex: report.cluster_profiles[k].count,
                background: clusterColor(k, false),
                opacity: isolated === null || isolated === k ? 1 : 0.25,
              }}
            />
          ))}
        </div>
        <div className="mt-5 grid gap-5 sm:grid-cols-3">
          <Insight number="A" title="The center of gravity">
            <button
              className="text-left underline decoration-border underline-offset-4 hover:decoration-foreground"
              onClick={() => exploreCluster(largest)}
            >
              {clusterName(report, largest)}
            </button>{" "}
            is the largest family, with {largestShare.toFixed(1)}% of the
            corpus.
          </Insight>
          <Insight number="B" title="The strongest relationship">
            {strongest ? (
              <>
                {label(strongest.a)} and {label(strongest.b)}{" "}
                {strongest.r >= 0
                  ? "tend to rise together"
                  : "tend to move in opposite directions"}{" "}
                (<span className="font-mono">r = {strongest.r.toFixed(2)}</span>
                ).
              </>
            ) : (
              "No descriptor pairs are available."
            )}
          </Insight>
          <Insight number="C" title="Where the boundary softens">
            {number(ambiguous)} sounds have uncertain membership.{" "}
            <button
              className="underline decoration-border underline-offset-4 hover:decoration-foreground"
              onClick={() => {
                setUncertain(true);
                exploreCluster(null);
              }}
            >
              Explore the in-between sounds.
            </button>
          </Insight>
        </div>
      </section>

      <section
        id="explorer"
        className="scroll-mt-6"
        aria-labelledby="explorer-title"
      >
        <SectionHeading
          number="02"
          id="explorer-title"
          title="Explore the sound landscape."
          description="Use the learned structure to see how the model groups sounds. Switch to descriptors to understand their measured character."
        />
        <div className="overflow-hidden rounded-xl border border-border bg-card">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
            <div className="flex flex-wrap items-center gap-3">
              <div
                className="flex rounded-lg bg-muted p-1"
                aria-label="Projection space"
              >
                {report.latent_projection && (
                  <Toggle
                    active={space === "latent"}
                    onClick={() => setSpace("latent")}
                  >
                    <Layers3 className="size-3.5" />
                    Latent structure
                  </Toggle>
                )}
                <Toggle
                  active={space === "descriptors"}
                  onClick={() => setSpace("descriptors")}
                >
                  <Scan className="size-3.5" />
                  Audio descriptors
                </Toggle>
              </div>
              <label className="flex items-center gap-2 text-[11px] text-muted-foreground">
                <input
                  type="checkbox"
                  checked={uncertain}
                  onChange={(e) => {
                    setUncertain(e.target.checked);
                    setSelected(null);
                  }}
                  className="accent-blue-600"
                />
                Uncertain only
              </label>
            </div>
            <div className="flex items-center gap-3">
              {mode === "2d" && (
                <select
                  aria-label="Projection axes"
                  value={pair}
                  onChange={(e) => setPair(e.target.value as AxisPair)}
                  className="rounded-md border border-border bg-background px-2 py-1.5 text-xs"
                >
                  {["12", "13", "23"].map((p) => (
                    <option key={p} value={p}>
                      {space === "latent" ? "L" : "PC"}
                      {p[0]} × {space === "latent" ? "L" : "PC"}
                      {p[1]}
                    </option>
                  ))}
                </select>
              )}
              <div
                className="flex rounded-lg bg-muted p-1"
                aria-label="Plot dimensions"
              >
                <Toggle active={mode === "3d"} onClick={() => setMode("3d")}>
                  <Box className="size-3.5" />
                  3D
                </Toggle>
                <Toggle active={mode === "2d"} onClick={() => setMode("2d")}>
                  2D
                </Toggle>
              </div>
            </div>
          </div>
          <div className="grid lg:grid-cols-[minmax(0,1fr)_300px] xl:grid-cols-[minmax(0,1fr)_320px]">
            <div className="min-w-0">
              <ScatterCanvas
                key={`${space}-${mode}-${pair}`}
                report={report}
                space={space}
                mode={mode}
                pair={pair}
                isolated={isolated}
                samples={samples}
                selected={selected}
                onPick={pick}
              />
              <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border px-5 py-3 text-[10px] text-muted-foreground">
                <span>
                  <span className="font-mono text-foreground">
                    {number(visibleCount)}
                  </span>{" "}
                  samples in focus{uncertain ? " · uncertain membership" : ""}
                </span>
                <span>
                  <span className="font-mono text-foreground">
                    {retained.toFixed(1)}%
                  </span>{" "}
                  of {space === "latent" ? "standardized latent" : "descriptor"}{" "}
                  variance shown
                </span>
              </div>
            </div>
            <aside
              className="flex flex-col border-t border-border p-5 lg:border-t-0 lg:border-l"
              aria-label="Cluster inspector"
            >
              <div className="mb-5 flex items-center justify-between">
                <p className="font-mono text-[10px] uppercase tracking-[.15em] text-muted-foreground">
                  {selected
                    ? "Sample inspection"
                    : isolated !== null
                      ? "Family in focus"
                      : "Reading the landscape"}
                </p>
                {isolated !== null && (
                  <button
                    onClick={() => focusCluster(null)}
                    aria-label="Clear selection"
                    className="rounded p-1 hover:bg-muted"
                  >
                    <X className="size-3.5" />
                  </button>
                )}
              </div>
              <label className="sr-only" htmlFor="family-selection">
                Select a sound family
              </label>
              <select
                id="family-selection"
                value={isolated ?? "all"}
                onChange={(e) =>
                  focusCluster(
                    e.target.value === "all" ? null : Number(e.target.value),
                  )
                }
                className="mb-5 w-full min-w-0 rounded-lg border border-border bg-background px-3 py-2 text-xs"
              >
                <option value="all">All sound families</option>
                {clusters.map((k) => (
                  <option key={k} value={k}>
                    {clusterId(k)} · {clusterName(report, k)}
                  </option>
                ))}
              </select>
              {isolated !== null ? (
                <>
                  <p className="mb-2 flex items-center gap-2 font-mono text-[10px] text-muted-foreground">
                    <span
                      className="size-2 rounded-full"
                      style={{ background: clusterColor(isolated, false) }}
                    />
                    {clusterId(isolated)} /{" "}
                    {number(report.cluster_profiles[isolated].count)} SAMPLES
                  </p>
                  <h3 className="text-xl font-medium leading-tight tracking-tight">
                    {clusterName(report, isolated)}
                  </h3>
                  <p className="mb-5 mt-3 text-xs leading-relaxed text-muted-foreground">
                    {clusterSummary(report, isolated)}
                  </p>
                  {selected ? (
                    <div className="mb-4 rounded-lg border border-border bg-muted/30 p-3">
                      <div className="mb-3 flex items-center justify-between text-xs">
                        <span className="font-medium">
                          Sample #{selected.sample_idx}
                        </span>
                        <button
                          onClick={() => setSelected(null)}
                          aria-label="Close sample inspection"
                        >
                          <X className="size-3" />
                        </button>
                      </div>
                      <dl className="space-y-1.5 text-[11px]">
                        {report.descriptor_keys.map((key, i) => (
                          <div key={key} className="flex justify-between gap-3">
                            <dt className="text-muted-foreground">
                              {report.descriptor_labels[i]}
                            </dt>
                            <dd className="font-mono">
                              {selected.descriptors[key].toFixed(1)}{" "}
                              {descriptorUnit(key, report)}
                            </dd>
                          </div>
                        ))}
                        <div className="flex justify-between border-t border-border pt-2">
                          <dt className="text-muted-foreground">Membership</dt>
                          <dd className="font-mono">
                            {(selected.confidence * 100).toFixed(1)}%
                          </dd>
                        </div>
                      </dl>
                    </div>
                  ) : (
                    <ClusterProfile
                      report={report}
                      cluster={isolated}
                      color={clusterColor(isolated, false)}
                    />
                  )}
                  <div className="mt-auto pt-5">
                    <AudioButton
                      cluster={isolated}
                      playing={audio.playing}
                      onPlay={audio.play}
                    />
                    {centralMember && !selected && <button type="button" className="mt-3 block text-[11px] underline decoration-border underline-offset-4 hover:decoration-foreground" onClick={() => { setUncertain(false); pick(centralMember); }}>Inspect the member nearest the family center</button>}
                    <p className="mt-3 text-[10px] leading-relaxed text-muted-foreground">
                      {(confidence * 100).toFixed(1)}% mean membership. Audio is
                      the averaged waveform; phase cancellation can change its
                      character. Profile and audio use all family members.
                    </p>
                  </div>
                </>
              ) : (
                <>
                  <h3 className="text-xl font-medium leading-tight tracking-tight">
                    Similarity, with room
                    <br />
                    for variation.
                  </h3>
                  <p className="mb-5 mt-3 text-xs leading-relaxed text-muted-foreground">
                    Colors show mixture membership. Nearby points are similar in
                    the displayed projection. Overlap is expected: this view
                    compresses a richer space.
                  </p>
                  <p className="mb-3 font-mono text-[9px] tracking-widest text-muted-foreground">
                    LARGEST FAMILIES
                  </p>
                  <div className="space-y-3">
                    {clusters.slice(0, 5).map((k) => (
                      <button
                        key={k}
                        className="group flex w-full items-center gap-2 text-left text-[11px]"
                        onClick={() => focusCluster(k)}
                      >
                        <span
                          className="size-1.5 shrink-0 rounded-full"
                          style={{ background: clusterColor(k, false) }}
                        />
                        <span className="font-mono text-[9px] text-muted-foreground">
                          {clusterId(k)}
                        </span>
                        <span className="truncate group-hover:underline">
                          {clusterName(report, k)}
                        </span>
                        <span className="ml-auto font-mono text-muted-foreground">
                          {(
                            (report.cluster_profiles[k].count /
                              report.samples.length) *
                            100
                          ).toFixed(0)}
                          %
                        </span>
                      </button>
                    ))}
                  </div>
                  <p className="mt-auto flex items-start gap-2 pt-6 text-[10px] leading-relaxed text-muted-foreground">
                    <CircleHelp className="mt-0.5 size-3 shrink-0" />
                    Select a family to compare its profile, or a point to
                    inspect an individual sample.
                  </p>
                </>
              )}
            </aside>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-[10px] text-muted-foreground">
          {dimensions.map((i) => (
            <span key={i}>
              <span className="mr-1.5 font-mono text-foreground">
                {space === "latent" ? "L" : "PC"}
                {i + 1}
              </span>
              {space === "descriptors" && report.pc_names[i]?.descriptor
                ? `${report.pc_names[i].name} association ${report.pc_names[i].correlation >= 0 ? "+" : "−"} · `
                : ""}
              {((projectionVariance[i] ?? 0) * 100).toFixed(1)}% variance
            </span>
          ))}
          <span className="sm:ml-auto">
            {space === "latent"
              ? "Linear PCA projection of standardized model encodings."
              : "PCA of standardized audio descriptors; colors retain latent-space membership."}
          </span>
        </div>
        {audio.error && (
          <p role="alert" className="mt-3 text-sm text-destructive">
            {audio.error}
          </p>
        )}
      </section>

      <section
        id="families"
        className="scroll-mt-6"
        aria-labelledby="families-title"
      >
        <SectionHeading
          number="03"
          id="families-title"
          title="Give each family a character."
          description="Names describe the largest measured differences from the corpus mean. Compare the profiles, then listen to the average."
        />
        <ClusterCards
          report={report}
          limit={allFamilies ? undefined : 6}
          isolated={isolated}
          onIsolate={exploreCluster}
          playing={audio.playing}
          onPlay={audio.play}
        />
        {clusters.length > 6 && (
          <div className="mt-5 flex justify-center">
            <Button
              variant="outline"
              size="lg"
              onClick={() => setAllFamilies(!allFamilies)}
            >
              {allFamilies
                ? "Show the six largest families"
                : `Explore all ${clusters.length} families`}
              <ArrowUpRight className="size-3.5" />
            </Button>
          </div>
        )}
        <p className="mt-4 text-center text-[10px] text-muted-foreground">
          Profiles use standard deviations (σ) from the corpus mean. Bars stop
          at ±3σ; labels show the full value. Family IDs apply to this report.
        </p>
      </section>

      <section aria-labelledby="descriptors-title">
        <SectionHeading
          number="04"
          id="descriptors-title"
          title="What gives the corpus its range?"
          description={
            isolated === null
              ? "The full distribution of every measured descriptor, in its original units. A wide range reveals variety along that dimension."
              : `${clusterId(isolated)} is highlighted against the full corpus. Selection changes the overlay; the measurement scale stays fixed.`
          }
        />
        {isolated !== null && (
          <Button
            variant="outline"
            size="sm"
            className="mb-4"
            onClick={() => focusCluster(null)}
          >
            Clear family selection
            <X className="size-3" />
          </Button>
        )}
        <Distributions report={report} isolated={isolated} />
      </section>

      <section
        id="evidence"
        className="scroll-mt-6"
        aria-labelledby="evidence-title"
      >
        <SectionHeading
          number="05"
          id="evidence-title"
          title="A good map shows its limits."
          description="Inspect the model-selection evidence and the information lost in projection. These are statistical groups, not verified musical categories."
        />
        <ModelEvidence report={report} />
        <details className="mt-4 rounded-xl border border-border bg-card">
          <summary className="cursor-pointer px-5 py-4 text-sm font-medium">
            Projection anatomy & descriptor relationships
            <span className="ml-3 text-xs font-normal text-muted-foreground">
              PCA loadings, variance and correlations
            </span>
          </summary>
          <div className="grid gap-8 border-t border-border p-5 sm:p-7 lg:grid-cols-2">
            <div>
              <h3 className="mb-2 text-sm font-medium">
                What the descriptor map preserves
              </h3>
              <p className="mb-4 text-xs leading-relaxed text-muted-foreground">
                Three components retain{" "}
                {(
                  report.pca_variance_explained.reduce((a, b) => a + b, 0) * 100
                ).toFixed(1)}
                % of descriptor variance. Each mixes several measurements; its
                name identifies a strong association.
              </p>
              <VarianceChart report={report} />
              <Heatmap
                caption="PCA loadings: descriptor weight per principal component"
                rows={Object.keys(report.pca_loadings)}
                cols={report.descriptor_keys}
                value={(r, c) => report.pca_loadings[r][c]}
                rowLabel={(r) => r.toUpperCase()}
                colLabel={label}
              />
            </div>
            <div>
              <h3 className="mb-2 text-sm font-medium">
                Which measurements move together?
              </h3>
              <p className="mb-6 text-xs leading-relaxed text-muted-foreground">
                Pearson correlation ranges from −1 (opposite direction) to +1
                (same direction). Correlation describes association, not
                causation.
              </p>
              <Heatmap
                caption="Descriptor correlation matrix"
                rows={report.descriptor_keys}
                cols={report.descriptor_keys}
                value={(r, c) => report.descriptor_correlations[r][c]}
                rowLabel={label}
                colLabel={label}
              />
              <div className="mt-4 flex justify-between text-[10px] text-muted-foreground">
                <span>Blue / negative</span>
                <span>Neutral / zero</span>
                <span>Red / positive</span>
              </div>
            </div>
          </div>
        </details>
        <details className="mt-3 rounded-xl border border-border bg-card">
          <summary className="cursor-pointer px-5 py-4 text-sm font-medium">
            Method & provenance
          </summary>
          <div className="grid gap-6 border-t border-border p-5 text-xs leading-relaxed text-muted-foreground sm:p-7 md:grid-cols-3">
            <p>
              <strong className="mb-2 block font-medium text-foreground">
                01 · Measure
              </strong>
              Source audio is analyzed at {number(report.corpus.sample_rate)} Hz
              in {formatMs(report.corpus.audio_length_ms)} windows. Descriptors
              quantify spectral ratios in dB and energy-weighted decay in ms.
              Source duration refers to the original file.
            </p>
            <p>
              <strong className="mb-2 block font-medium text-foreground">
                02 · Group
              </strong>
              {report.clustering
                ? `Constant latent dimensions are removed, then encodings are standardized and reduced by PCA while retaining at least 95% of variance. Regularized mixtures compare full and diagonal covariance, including one component, with seed ${report.clustering.seed}. Only converged candidates are eligible.`
                : "A Gaussian mixture groups standardized VAE encodings. BIC selects the component count."}
            </p>
            <p>
              <strong className="mb-2 block font-medium text-foreground">
                03 · Interpret
              </strong>
              Family names use up to two descriptor differences of at least
              0.25σ. Average audio can suppress out-of-phase content and is not
              a representative recording. Sample IDs reveal no source filenames.
              {report.generated_at && (
                <span className="mt-2 block font-mono text-[10px]">
                  Generated{" "}
                  {new Date(report.generated_at)
                    .toISOString()
                    .slice(0, 16)
                    .replace("T", " ")}{" "}
                  UTC
                </span>
              )}
            </p>
          </div>
        </details>
      </section>

      <section aria-labelledby="samples-title">
        <SectionHeading
          number="06"
          id="samples-title"
          title="Every point, accounted for."
          description="Inspect individual measurements and membership probabilities. The table follows the explorer’s family and uncertainty filters."
        />
        <SampleTable
          key={`${isolated}-${uncertain}`}
          report={report}
          isolated={isolated}
          uncertain={uncertain}
          selected={selected}
          onPick={(sample) => {
            pick(sample);
            document
              .getElementById("explorer")
              ?.scrollIntoView({ block: "start" });
          }}
        />
      </section>
      <footer className="flex flex-wrap justify-between gap-3 border-t border-border pt-5 font-mono text-[9px] uppercase tracking-[.12em] text-muted-foreground">
        <span>Kicks / Corpus atlas</span>
        <span>
          {number(report.samples.length)} observations ·{" "}
          {report.descriptor_keys.length} descriptors · {clusters.length}{" "}
          families
        </span>
      </footer>
    </div>
  );
}

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="border-l border-border pl-4">
      <div className="text-[10px] text-muted-foreground">{label}</div>
      <div className="mt-1 text-2xl font-medium tracking-tight tabular-nums">
        {value}
      </div>
      <div className="mt-1 text-[10px] text-muted-foreground">{hint}</div>
    </div>
  );
}
function Insight({
  number,
  title,
  children,
}: {
  number: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="flex gap-3">
      <span className="mt-0.5 font-mono text-[10px] text-muted-foreground">
        {number}
      </span>
      <div>
        <h3 className="mb-1.5 text-xs font-medium">{title}</h3>
        <p className="text-xs leading-relaxed text-muted-foreground">
          {children}
        </p>
      </div>
    </div>
  );
}
function SectionHeading({
  number,
  id,
  title,
  description,
}: {
  number: string;
  id: string;
  title: string;
  description: string;
}) {
  return (
    <div className="mb-5 flex gap-4">
      <span className="mt-1.5 font-mono text-[10px] text-muted-foreground">
        {number}
      </span>
      <div>
        <h2 id={id} className="text-xl font-medium tracking-tight sm:text-2xl">
          {title}
        </h2>
        <p className="mt-2 max-w-3xl text-xs leading-relaxed text-muted-foreground">
          {description}
        </p>
      </div>
    </div>
  );
}
function Toggle({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={cn(
        "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[11px] font-medium transition-colors focus-visible:outline-2 focus-visible:outline-ring",
        active
          ? "bg-background text-foreground shadow-sm"
          : "text-muted-foreground hover:text-foreground",
      )}
    >
      {children}
    </button>
  );
}
