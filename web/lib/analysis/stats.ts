import type { Report, SampleRow } from "./types";

export interface HistogramBin {
  x0: number;
  x1: number;
  count: number;
}

export function histogram(values: number[], bins = 24, min = 0, max = 1): HistogramBin[] {
  const width = (max - min) / bins;
  const out: HistogramBin[] = Array.from({ length: bins }, (_, i) => ({
    x0: min + i * width,
    x1: min + (i + 1) * width,
    count: 0,
  }));
  for (const v of values) {
    const i = Math.min(bins - 1, Math.max(0, Math.floor((v - min) / width)));
    out[i].count++;
  }
  return out;
}

export function descriptorValues(report: Report, key: string): number[] {
  return report.samples.map((s) => s.descriptors[key]);
}

export function clusterMembers(report: Report, cluster: number): SampleRow[] {
  return report.samples.filter((s) => s.cluster === cluster);
}

export function pcLabel(report: Report, index: number): string {
  const pc = report.pc_names[index];
  const variance = report.pca_variance_explained[index];
  const base = pc?.descriptor ? `${pc.name}` : `PC${index + 1}`;
  return `${base} (PC${index + 1}, ${(variance * 100).toFixed(0)}%)`;
}

export function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}
