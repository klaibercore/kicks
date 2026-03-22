export interface Sample {
  sample_idx: number;
  filename: string;
  original_path: string;
  cluster: number;
  pc1: number;
  pc2: number;
  pc3: number;
  descriptors: {
    sub: number;
    punch: number;
    click: number;
    bright: number;
    decay: number;
  };
  probs: number[];
  duration_ms?: number;
}

export interface ClusterProfile {
  count: number;
  sub: number;
  punch: number;
  click: number;
  bright: number;
  decay: number;
}

export interface PCName {
  name: string;
  descriptor: string | null;
  correlation: number;
}

export interface CorpusMeta {
  sample_rate: number;
  audio_length_ms: number;
  n_total: number;
  data_dir: string;
}

export interface ClusterData {
  pca_variance_explained: number[];
  pca_source?: string;
  n_clusters: number;
  corpus?: CorpusMeta;
  samples: Sample[];
  pc_names?: PCName[];
  pca_loadings?: Record<string, Record<string, number>>;
  pc_descriptor_correlations?: Record<string, Record<string, number>>;
  descriptor_correlations?: Record<string, Record<string, number>>;
  cluster_profiles?: Record<string, ClusterProfile>;
  descriptor_stats?: Record<
    string,
    { mean: number; std: number; min: number; max: number }
  >;
}

export type DescriptorKey = "sub" | "punch" | "click" | "bright" | "decay";
export type ColorMode = "cluster" | DescriptorKey;

export const DESCRIPTOR_KEYS: DescriptorKey[] = [
  "sub",
  "punch",
  "click",
  "bright",
  "decay",
];

export const DESCRIPTOR_LABELS: Record<DescriptorKey, string> = {
  sub: "Sub",
  punch: "Punch",
  click: "Click",
  bright: "Bright",
  decay: "Decay",
};
