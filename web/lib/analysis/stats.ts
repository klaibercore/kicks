import type { Report, SampleRow } from "./types";

export interface HistogramBin {
  x0: number;
  x1: number;
  count: number;
}

export function histogram(
  values: number[],
  bins = 24,
  min = 0,
  max = 1,
): HistogramBin[] {
  if (max <= min) max = min + 1;
  const width = (max - min) / bins;
  const out: HistogramBin[] = Array.from({ length: bins }, (_, i) => ({
    x0: min + i * width,
    x1: min + (i + 1) * width,
    count: 0,
  }));
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    const i = Math.min(bins - 1, Math.max(0, Math.floor((v - min) / width)));
    out[i].count++;
  }
  return out;
}

export function descriptorUnit(key: string, report?: Report): string {
  return report?.descriptor_units?.[key] ?? (key === "decay" ? "ms" : "dB");
}

export function clusterTraits(report: Report, cluster: number) {
  const profile = report.cluster_profiles[String(cluster)];
  return report.descriptor_keys
    .map((key, i) => {
      const stat = report.descriptor_stats[key];
      return {
        key,
        label: report.descriptor_labels[i],
        value: profile[key],
        z: stat.std > 0 ? (profile[key] - stat.mean) / stat.std : 0,
      };
    })
    .sort((a, b) => Math.abs(b.z) - Math.abs(a.z));
}

const traitWords: Record<string, [string, string]> = {
  sub: ["Lighter low end", "Sub-heavy"],
  punch: ["Softer impact", "Punch-forward"],
  click: ["Less click", "Click-forward"],
  bright: ["Darker", "Brighter"],
  decay: ["Shorter sustain", "Longer sustain"],
  body: ["Lighter body", "Fuller body"],
  crack: ["Less crack", "Crack-forward"],
  snap: ["Less snap", "Snap-forward"],
  attack: ["Softer attack", "Sharper attack"],
  sizzle: ["Less sizzle", "Sizzle-rich"],
};

export function clusterName(report: Report, cluster: number): string {
  const traits = clusterTraits(report, cluster)
    .filter((t) => Math.abs(t.z) >= 0.25)
    .slice(0, 2);
  return traits.length
    ? traits
        .map(
          (t) =>
            traitWords[t.key]?.[t.z > 0 ? 1 : 0] ??
            `${t.z > 0 ? "Higher" : "Lower"} ${t.label.toLowerCase()}`,
        )
        .join(" · ")
    : "Near corpus average";
}

export function clusterSummary(report: Report, cluster: number): string {
  const traits = clusterTraits(report, cluster).slice(0, 2);
  return (
    traits
      .map(
        (t) =>
          `${t.label} is ${Math.abs(t.z).toFixed(1)} standard deviations ${t.z >= 0 ? "above" : "below"} the corpus mean`,
      )
      .join("; ") + "."
  );
}

export function rankedClusters(report: Report): number[] {
  return Object.keys(report.cluster_profiles)
    .map(Number)
    .sort(
      (a, b) =>
        report.cluster_profiles[b].count - report.cluster_profiles[a].count,
    );
}

export function confidenceThreshold(report: Report): number {
  return report.clustering?.confidence_threshold ?? 0.8;
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
