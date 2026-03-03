import type { Sample, ColorMode, ClusterData } from "@/types/cluster";

export const CLUSTER_COLORS = [
  "#a78bfa", "#f472b6", "#34d399", "#fbbf24",
  "#60a5fa", "#f87171", "#c084fc", "#22d3ee",
  "#fb923c", "#a3e635",
];

export const PC_COLORS = ["#f472b6", "#34d399", "#a78bfa"];
export const PC_LABELS_SHORT = ["PC1", "PC2", "PC3"];

export function pearsonCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  const sx = Math.sqrt(x.reduce((a, b) => a + (b - mx) ** 2, 0) / n);
  const sy = Math.sqrt(y.reduce((a, b) => a + (b - my) ** 2, 0) / n);
  if (sx === 0 || sy === 0) return 0;
  return x.reduce((a, b, i) => a + (b - mx) * (y[i] - my), 0) / (n * sx * sy);
}

export function descriptorColor(value: number): string {
  const t = Math.max(0, Math.min(1, value));
  if (t < 0.5) {
    const s = t / 0.5;
    return `rgb(${Math.round(60 + s * 60)},${Math.round(80 + s * 80)},${Math.round(160 + s * 40)})`;
  } else {
    const s = (t - 0.5) / 0.5;
    return `rgb(${Math.round(120 + s * 135)},${Math.round(160 + s * 40)},${Math.round(200 - s * 150)})`;
  }
}

export function correlationColor(v: number): string {
  const abs = Math.min(Math.abs(v), 1);
  const alpha = 0.12 + abs * 0.6;
  if (v >= 0) return `rgba(52, 211, 153, ${alpha})`;
  return `rgba(248, 113, 113, ${alpha})`;
}

export function correlationTextColor(v: number, isDark = true): string {
  if (isDark) return v >= 0 ? "#6ee7b7" : "#fca5a5";
  return v >= 0 ? "#047857" : "#dc2626";
}

export function sampleColor(
  sample: Sample,
  mode: ColorMode,
  stats?: ClusterData["descriptor_stats"],
): string {
  if (mode === "cluster") {
    return CLUSTER_COLORS[sample.cluster % CLUSTER_COLORS.length];
  }
  const val = sample.descriptors[mode];
  let normalized = val;
  if (stats && stats[mode]) {
    const { min, max } = stats[mode];
    normalized = max > min ? (val - min) / (max - min) : 0.5;
  }
  return descriptorColor(normalized);
}
