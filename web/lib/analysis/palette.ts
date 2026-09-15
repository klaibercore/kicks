/**
 * Cluster identity colours. Slots 1–8 are the validated categorical order from
 * the data-viz reference palette (fixed order, never cycled); 9 and 10 are two
 * further stepped hues for the rare corpus that clusters into more. Clusters
 * beyond eight rely on the secondary encodings the page always provides —
 * numbered labels, hover tooltips, single-cluster isolation and the table.
 */
export const CLUSTER_COLORS = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#0e7c86", "#8a5a2b"],
  dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767", "#3aa8b3", "#b98a5a"],
} as const;

export function clusterColor(cluster: number, dark: boolean): string {
  const set = dark ? CLUSTER_COLORS.dark : CLUSTER_COLORS.light;
  return set[cluster % set.length];
}

/** Sequential single hue (blue), light -> dark, from the reference palette. */
export const SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"] as const;

/** Diverging blue <-> red with a neutral midpoint. t in [-1, 1]. */
export function diverging(t: number, dark: boolean): string {
  const mid = dark ? [56, 56, 53] : [240, 239, 236];
  const blue = dark ? [57, 135, 229] : [37, 106, 191];
  const red = dark ? [230, 103, 103] : [208, 59, 59];
  const target = t < 0 ? blue : red;
  const a = Math.min(1, Math.abs(t));
  const c = mid.map((m, i) => Math.round(m + (target[i] - m) * a));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}
