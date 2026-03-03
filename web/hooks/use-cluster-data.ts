"use client";

import { useEffect, useMemo, useState } from "react";
import type { ClusterData, ClusterProfile, PCName } from "@/types/cluster";
import { DESCRIPTOR_KEYS, DESCRIPTOR_LABELS } from "@/types/cluster";
import type { DescriptorKey } from "@/types/cluster";
import { pearsonCorrelation } from "@/lib/colors";

export function useClusterData() {
  const [data, setData] = useState<ClusterData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/cluster-data")
      .then((r) => r.json())
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const correlations = useMemo(() => {
    if (!data) return {};
    if (data.pc_descriptor_correlations) return data.pc_descriptor_correlations;

    const result: Record<string, Record<string, number>> = {};
    for (const pc of ["pc1", "pc2", "pc3"] as const) {
      const pcVals = data.samples.map((s) => s[pc]);
      result[pc] = {};
      for (const dk of DESCRIPTOR_KEYS) {
        const dkVals = data.samples.map((s) => s.descriptors[dk]);
        result[pc][dk] = pearsonCorrelation(pcVals, dkVals);
      }
    }
    return result;
  }, [data]);

  const clusterProfiles = useMemo(() => {
    if (!data) return {};
    if (data.cluster_profiles) return data.cluster_profiles;

    const profiles: Record<string, ClusterProfile> = {};
    for (let k = 0; k < data.n_clusters; k++) {
      const cluster = data.samples.filter((s) => s.cluster === k);
      if (cluster.length === 0) continue;
      const profile: ClusterProfile = {
        count: cluster.length,
        sub: 0,
        punch: 0,
        click: 0,
        bright: 0,
        decay: 0,
      };
      for (const dk of DESCRIPTOR_KEYS) {
        profile[dk] =
          cluster.reduce((a, s) => a + s.descriptors[dk], 0) / cluster.length;
      }
      profiles[String(k)] = profile;
    }
    return profiles;
  }, [data]);

  const pcNames = useMemo((): PCName[] => {
    if (data?.pc_names) return data.pc_names;
    const used = new Set<string>();
    return ["pc1", "pc2", "pc3"].map((pc, i) => {
      const corrs = correlations[pc] || {};
      const sorted = Object.entries(corrs)
        .filter(([k]) => !used.has(k))
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
      const top = sorted[0];
      if (top && Math.abs(top[1]) >= 0.15) {
        used.add(top[0]);
        return {
          name: DESCRIPTOR_LABELS[top[0] as DescriptorKey] || top[0],
          descriptor: top[0],
          correlation: top[1],
        };
      }
      return { name: `PC${i + 1}`, descriptor: null, correlation: 0 };
    });
  }, [data, correlations]);

  return { data, loading, correlations, clusterProfiles, pcNames };
}
