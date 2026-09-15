export interface AnalysisIndexEntry {
  name: string;
  display_name: string;
  description: string;
  n_samples: number;
  n_clusters: number;
  descriptors: string[];
  report: string;
  cluster_audio: number;
}

export interface AnalysisIndex {
  instruments: AnalysisIndexEntry[];
}

export interface SampleRow {
  sample_idx: number;
  cluster: number;
  duration_ms: number;
  descriptors: Record<string, number>;
  confidence: number;
  pc1: number;
  pc2: number;
  pc3?: number;
}

export interface PcName {
  name: string;
  descriptor: string | null;
  correlation: number;
}

export interface DescriptorStat {
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface Report {
  instrument: string;
  descriptor_keys: string[];
  descriptor_labels: string[];
  descriptor_docs: Record<string, string>;
  pca_variance_explained: number[];
  pca_source: string;
  n_clusters: number;
  corpus: { sample_rate: number; audio_length_ms: number; n_total: number };
  samples: SampleRow[];
  pc_names: PcName[];
  pca_loadings: Record<string, Record<string, number>>;
  pc_descriptor_correlations: Record<string, Record<string, number>>;
  descriptor_correlations: Record<string, Record<string, number>>;
  cluster_profiles: Record<string, { count: number } & Record<string, number>>;
  descriptor_stats: Record<string, DescriptorStat>;
}
