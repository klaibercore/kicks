import { asset } from "@/lib/config";
import type { AnalysisIndex, Report } from "./types";

const cache = new Map<string, Promise<unknown>>();

async function fetchJson<T>(path: string): Promise<T> {
  const url = asset(path);
  if (!cache.has(url)) {
    cache.set(
      url,
      fetch(url)
        .then((res) => {
          if (!res.ok) throw new Error(`${res.status} loading ${path}`);
          return res.json();
        })
        .catch((error) => {
          cache.delete(url);
          throw error;
        }),
    );
  }
  return cache.get(url) as Promise<T>;
}

export const loadIndex = () => fetchJson<AnalysisIndex>("/analysis/index.json");
export const loadReport = (file: string) =>
  fetchJson<Report>(`/analysis/${file}`);
export const clusterAudioUrl = (instrument: string, cluster: number) =>
  asset(`/analysis/${instrument}/cluster_avg_${cluster}.wav`);
