"use client";

import { useEffect, useMemo, useState } from "react";
import { ClusterCards } from "@/components/analysis/cluster-cards";
import { Distributions } from "@/components/analysis/distributions";
import { Heatmap } from "@/components/analysis/heatmap";
import { SampleTable } from "@/components/analysis/sample-table";
import { ScatterCanvas } from "@/components/analysis/scatter-canvas";
import { VarianceChart } from "@/components/analysis/variance-chart";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { loadIndex, loadReport } from "@/lib/analysis/load";
import { formatMs, pcLabel } from "@/lib/analysis/stats";
import type { AnalysisIndex, Report } from "@/lib/analysis/types";

type PcKey = "pc1" | "pc2" | "pc3";

export function CorpusAnalysis() {
  const [index, setIndex] = useState<AnalysisIndex | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const [loaded, setLoaded] = useState<{ name: string; report: Report } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isolated, setIsolated] = useState<number | null>(null);
  const [xKey, setXKey] = useState<PcKey>("pc1");
  const [yKey, setYKey] = useState<PcKey>("pc2");

  useEffect(() => {
    loadIndex()
      .then((idx) => {
        setIndex(idx);
        setActive(idx.instruments[0]?.name ?? null);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!index || !active) return;
    const entry = index.instruments.find((i) => i.name === active);
    if (!entry) return;
    let cancelled = false;
    loadReport(entry.report)
      .then((report) => {
        if (!cancelled) setLoaded({ name: entry.name, report });
      })
      .catch((e: Error) => setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [index, active]);

  const report = loaded?.name === active ? loaded.report : null;

  const meanDuration = useMemo(
    () => (report ? report.samples.reduce((a, s) => a + s.duration_ms, 0) / report.samples.length : 0),
    [report],
  );

  if (error) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-16 text-sm text-destructive sm:px-6">
        Could not load the analysis data: {error}. Run <code className="font-mono">kicks cluster</code> and{" "}
        <code className="font-mono">kicks publish-analysis</code>.
      </div>
    );
  }

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-6 sm:px-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Corpus analysis</h1>
        <p className="text-sm text-muted-foreground">
          What the training data looks like: GMM clusters over the VAE latent space, and a PCA over the perceptual descriptors.
        </p>
      </div>

      <Tabs
        value={active ?? ""}
        onValueChange={(v) => {
          if (!v) return;
          setActive(v as string);
          setIsolated(null);
        }}
      >
        <TabsList variant="line" className="h-9">
          {index
            ? index.instruments.map((inst) => (
                <TabsTrigger key={inst.name} value={inst.name} className="px-3">
                  {inst.display_name}
                  <span className="ml-1 font-mono text-[10px] text-muted-foreground">{inst.n_samples}</span>
                </TabsTrigger>
              ))
            : [0, 1, 2].map((i) => <Skeleton key={i} className="h-7 w-20" />)}
        </TabsList>
      </Tabs>

      {!report ? (
        <div className="grid gap-4 sm:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-20" />
          ))}
        </div>
      ) : (
        <>
          <p className="text-sm text-muted-foreground">{index?.instruments.find((i) => i.name === active)?.description}</p>

          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Samples" value={report.samples.length.toLocaleString()} hint={`${report.corpus.n_total.toLocaleString()} in corpus`} />
            <Stat label="Clusters" value={String(report.n_clusters)} hint="chosen by BIC" />
            <Stat
              label="Variance in 3 PCs"
              value={`${(report.pca_variance_explained.reduce((a, b) => a + b, 0) * 100).toFixed(0)}%`}
              hint="of z-scored descriptors"
            />
            <Stat label="Mean length" value={formatMs(meanDuration)} hint={`clips fixed at ${formatMs(report.corpus.audio_length_ms)}`} />
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Descriptor space</CardTitle>
              <CardDescription>
                Each dot is one sample, placed by its principal components and coloured by its latent-space cluster. Click a cluster card to isolate it.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="flex flex-wrap items-center gap-2">
                <AxisSelect label="X" value={xKey} onChange={setXKey} report={report} />
                <AxisSelect label="Y" value={yKey} onChange={setYKey} report={report} />
                {isolated !== null ? (
                  <Button size="sm" variant="outline" onClick={() => setIsolated(null)}>
                    Show all clusters
                  </Button>
                ) : null}
                <div className="ml-auto flex flex-wrap gap-1.5">
                  {report.pc_names.map((pc, i) =>
                    pc.descriptor ? (
                      <Badge key={i} variant="outline" className="font-mono text-[10px]">
                        PC{i + 1} ~ {pc.descriptor} r={pc.correlation.toFixed(2)}
                      </Badge>
                    ) : null,
                  )}
                </div>
              </div>
              <ScatterCanvas
                report={report}
                xKey={xKey}
                yKey={yKey}
                xLabel={pcLabel(report, Number(xKey.slice(2)) - 1)}
                yLabel={pcLabel(report, Number(yKey.slice(2)) - 1)}
                isolated={isolated}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Clusters</CardTitle>
              <CardDescription>
                Gaussian mixture on z-scored latents. The audio is the mean waveform of the members — only what they share survives the averaging.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ClusterCards report={report} isolated={isolated} onIsolate={setIsolated} />
            </CardContent>
          </Card>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Principal components</CardTitle>
                <CardDescription>Variance explained, and which descriptor each PC loads on.</CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <VarianceChart report={report} />
                <Heatmap
                  caption="PCA loadings: descriptor weight per principal component"
                  rows={Object.keys(report.pca_loadings)}
                  cols={report.descriptor_keys}
                  value={(r, c) => report.pca_loadings[r][c]}
                  rowLabel={(r) => r.toUpperCase()}
                  colLabel={(c) => report.descriptor_labels[report.descriptor_keys.indexOf(c)]}
                />
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Descriptor correlations</CardTitle>
                <CardDescription>Pearson r between the perceptual axes. Strong off-diagonals are the cross-talk the slider basis compensates for.</CardDescription>
              </CardHeader>
              <CardContent>
                <Heatmap
                  caption="Descriptor-to-descriptor correlation matrix"
                  rows={report.descriptor_keys}
                  cols={report.descriptor_keys}
                  value={(r, c) => report.descriptor_correlations[r][c]}
                  rowLabel={(k) => report.descriptor_labels[report.descriptor_keys.indexOf(k)]}
                  colLabel={(k) => report.descriptor_labels[report.descriptor_keys.indexOf(k)]}
                />
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Descriptor distributions</CardTitle>
              <CardDescription>How each perceptual axis is spread across the corpus. Slider centre corresponds to the corpus mean.</CardDescription>
            </CardHeader>
            <CardContent>
              <Distributions report={report} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Samples</CardTitle>
              <CardDescription>Every sample in the analysis by index, with its cluster, descriptors and GMM confidence.</CardDescription>
            </CardHeader>
            <CardContent>
              <SampleTable report={report} isolated={isolated} />
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-border p-4">
      <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">{label}</div>
      <div className="tabular mt-1 text-2xl font-semibold">{value}</div>
      {hint ? <div className="text-xs text-muted-foreground">{hint}</div> : null}
    </div>
  );
}

function AxisSelect({ label, value, onChange, report }: { label: string; value: PcKey; onChange: (v: PcKey) => void; report: Report }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="font-mono text-xs text-muted-foreground">{label}</span>
      <Select
        value={value}
        onValueChange={(v) => v && onChange(v as PcKey)}
        items={Object.fromEntries(report.pca_variance_explained.map((_, i) => [`pc${i + 1}`, pcLabel(report, i)]))}
      >
        <SelectTrigger className="h-8 w-52" aria-label={`${label} axis`}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {report.pca_variance_explained.map((_, i) => (
            <SelectItem key={i} value={`pc${i + 1}`}>
              {pcLabel(report, i)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
