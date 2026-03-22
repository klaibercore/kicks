"use client";

import { useMemo } from "react";
import { useTheme } from "next-themes";
import type { ClusterData, DescriptorKey } from "@/types/cluster";
import { DESCRIPTOR_KEYS, DESCRIPTOR_LABELS } from "@/types/cluster";
import { correlationColor, correlationTextColor, descriptorColor } from "@/lib/colors";
import { card, sectionTitle } from "@/lib/styles";

function percentile(sorted: number[], p: number): number {
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const min = Math.floor(ms / 60_000);
  const sec = Math.round((ms % 60_000) / 1000);
  return `${min}m ${sec}s`;
}

// ── Summary Cards ──────────────────────────────────────────────

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className={`${card} px-4 py-3 space-y-0.5`}>
      <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">{label}</p>
      <p className="text-lg font-bold tabular-nums">{value}</p>
      {sub && <p className="text-[10px] text-muted-foreground font-mono">{sub}</p>}
    </div>
  );
}

function SummaryCards({ data }: { data: ClusterData }) {
  const durations = data.samples.map((s) => s.duration_ms).filter((d): d is number => d != null);
  const totalMs = durations.reduce((a, b) => a + b, 0);
  const avgMs = durations.length > 0 ? totalMs / durations.length : 0;
  const totalVar = data.pca_variance_explained.reduce((a, b) => a + b, 0);

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
      <StatCard label="Samples" value={data.samples.length.toLocaleString()} sub={data.corpus ? `of ${data.corpus.n_total.toLocaleString()} total` : undefined} />
      <StatCard label="Total Duration" value={formatDuration(totalMs)} sub={durations.length > 0 ? `avg ${formatDuration(avgMs)}` : undefined} />
      <StatCard label="Clusters" value={String(data.n_clusters)} sub={`BIC-selected`} />
      <StatCard label="PCA Variance" value={`${(totalVar * 100).toFixed(1)}%`} sub={`${data.pca_variance_explained.length} components`} />
      <StatCard label="Sample Rate" value={data.corpus ? `${(data.corpus.sample_rate / 1000).toFixed(1)}kHz` : "44.1kHz"} sub={data.corpus?.data_dir} />
    </div>
  );
}

// ── Descriptor Correlation Heatmap ─────────────────────────────

function DescriptorCorrelationHeatmap({ data }: { data: ClusterData }) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  const correlations = useMemo(() => {
    if (data.descriptor_correlations) return data.descriptor_correlations;
    // Fallback: compute client-side
    const result: Record<string, Record<string, number>> = {};
    for (const dk1 of DESCRIPTOR_KEYS) {
      result[dk1] = {};
      const v1 = data.samples.map((s) => s.descriptors[dk1]);
      const m1 = v1.reduce((a, b) => a + b, 0) / v1.length;
      const s1 = Math.sqrt(v1.reduce((a, b) => a + (b - m1) ** 2, 0) / v1.length);
      for (const dk2 of DESCRIPTOR_KEYS) {
        const v2 = data.samples.map((s) => s.descriptors[dk2]);
        const m2 = v2.reduce((a, b) => a + b, 0) / v2.length;
        const s2 = Math.sqrt(v2.reduce((a, b) => a + (b - m2) ** 2, 0) / v2.length);
        if (s1 > 0 && s2 > 0) {
          result[dk1][dk2] = v1.reduce((a, b, i) => a + (b - m1) * (v2[i] - m2), 0) / (v1.length * s1 * s2);
        } else {
          result[dk1][dk2] = 0;
        }
      }
    }
    return result;
  }, [data]);

  return (
    <div className="space-y-3">
      <h3 className={sectionTitle}>Descriptor Correlations</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="text-left text-muted-foreground pb-2 pr-2 font-normal" />
              {DESCRIPTOR_KEYS.map((dk) => (
                <th key={dk} className="pb-2 px-1 font-bold text-center text-[12px] text-foreground/80">
                  {DESCRIPTOR_LABELS[dk]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {DESCRIPTOR_KEYS.map((dk1) => (
              <tr key={dk1}>
                <td className="text-foreground/80 pr-2 py-1 font-medium text-[12px]">
                  {DESCRIPTOR_LABELS[dk1]}
                </td>
                {DESCRIPTOR_KEYS.map((dk2) => {
                  const v = correlations[dk1]?.[dk2] ?? 0;
                  const isDiag = dk1 === dk2;
                  return (
                    <td key={dk2} className="px-1 py-1 text-center">
                      <div
                        className="rounded-lg px-2 py-1.5 font-mono font-semibold text-[12px] transition-colors"
                        style={{
                          backgroundColor: isDiag ? "rgba(128,128,128,0.1)" : correlationColor(v),
                          color: isDiag
                            ? (isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)")
                            : Math.abs(v) > 0.25
                              ? (isDark ? "#fff" : "#000")
                              : correlationTextColor(v, isDark),
                        }}
                      >
                        {isDiag ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(2)}`}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Duration Distribution ──────────────────────────────────────

function DurationDistribution({ data }: { data: ClusterData }) {
  const durations = data.samples.map((s) => s.duration_ms).filter((d): d is number => d != null);
  if (durations.length === 0) return null;

  const bins = 20;
  const sorted = [...durations].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const range = max - min || 1;

  const histogram = new Array(bins).fill(0);
  for (const v of durations) {
    const idx = Math.min(Math.floor(((v - min) / range) * bins), bins - 1);
    histogram[idx]++;
  }
  const maxCount = Math.max(...histogram);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-[11px]">
        <span className="text-foreground/80 font-semibold">Duration</span>
        <span className="text-muted-foreground font-mono">
          {formatDuration(sorted[Math.floor(sorted.length / 2)])} median
        </span>
      </div>
      <div className="flex items-end gap-px h-14 rounded-lg bg-muted/30 p-1">
        {histogram.map((count, i) => (
          <div
            key={i}
            className="flex-1 rounded-t-sm transition-all"
            style={{
              height: `${(count / (maxCount || 1)) * 100}%`,
              backgroundColor: descriptorColor(i / bins),
              opacity: 0.75,
              minHeight: count > 0 ? 2 : 0,
            }}
          />
        ))}
      </div>
      <div className="flex justify-between text-[9px] text-muted-foreground font-mono">
        <span>{formatDuration(min)}</span>
        <span>{formatDuration(max)}</span>
      </div>
    </div>
  );
}

// ── Box Plots ──────────────────────────────────────────────────

function BoxPlots({ data }: { data: ClusterData }) {
  const stats = useMemo(() => {
    return DESCRIPTOR_KEYS.map((dk) => {
      const vals = data.samples.map((s) => s.descriptors[dk]).sort((a, b) => a - b);
      return {
        key: dk,
        min: vals[0],
        q1: percentile(vals, 25),
        median: percentile(vals, 50),
        q3: percentile(vals, 75),
        max: vals[vals.length - 1],
      };
    });
  }, [data]);

  return (
    <div className="space-y-1.5">
      <h4 className="text-[11px] font-semibold text-foreground/80">Quartiles</h4>
      <div className="space-y-2">
        {stats.map(({ key, min, q1, median, q3, max }) => {
          const range = max - min || 1;
          const toP = (v: number) => ((v - min) / range) * 100;
          return (
            <div key={key} className="flex items-center gap-2">
              <span className="text-[10px] font-medium text-muted-foreground w-10 text-right shrink-0">
                {DESCRIPTOR_LABELS[key]}
              </span>
              <div className="flex-1 relative h-5">
                {/* Whisker line */}
                <div
                  className="absolute top-1/2 -translate-y-1/2 h-px bg-muted-foreground/30"
                  style={{ left: `${toP(min)}%`, width: `${toP(max) - toP(min)}%` }}
                />
                {/* Box */}
                <div
                  className="absolute top-0.5 bottom-0.5 rounded-sm bg-violet-500/20 border border-violet-500/30"
                  style={{ left: `${toP(q1)}%`, width: `${toP(q3) - toP(q1)}%` }}
                />
                {/* Median line */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-violet-400"
                  style={{ left: `${toP(median)}%` }}
                />
                {/* Min/max caps */}
                <div className="absolute top-1 bottom-1 w-px bg-muted-foreground/40" style={{ left: `${toP(min)}%` }} />
                <div className="absolute top-1 bottom-1 w-px bg-muted-foreground/40" style={{ left: `${toP(max)}%` }} />
              </div>
              <span className="text-[9px] font-mono text-muted-foreground w-8 shrink-0">
                {median.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Main EDA Section ───────────────────────────────────────────

export function EdaSection({ data }: { data: ClusterData }) {
  return (
    <section className="space-y-5">
      <h2 className={sectionTitle}>Exploratory Data Analysis</h2>
      <SummaryCards data={data} />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className={`${card} p-4 space-y-4`}>
          <DescriptorCorrelationHeatmap data={data} />
        </div>
        <div className={`${card} p-4 space-y-5`}>
          <DurationDistribution data={data} />
          <BoxPlots data={data} />
        </div>
      </div>
    </section>
  );
}
