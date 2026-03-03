"use client";

import type { Sample, DescriptorKey } from "@/types/cluster";
import { DESCRIPTOR_LABELS } from "@/types/cluster";
import { descriptorColor } from "@/lib/colors";

export function DescriptorDistribution({
  samples,
  descriptor,
  stats,
}: {
  samples: Sample[];
  descriptor: DescriptorKey;
  stats?: { mean: number; std: number; min: number; max: number };
}) {
  const bins = 20;
  const values = samples.map((s) => s.descriptors[descriptor]);
  const min = stats?.min ?? Math.min(...values);
  const max = stats?.max ?? Math.max(...values);
  const range = max - min || 1;

  const histogram = new Array(bins).fill(0);
  for (const v of values) {
    const idx = Math.min(Math.floor(((v - min) / range) * bins), bins - 1);
    histogram[idx]++;
  }
  const maxCount = Math.max(...histogram);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-[11px]">
        <span className="text-foreground/80 font-semibold">
          {DESCRIPTOR_LABELS[descriptor]}
        </span>
        {stats && (
          <span className="text-muted-foreground font-mono">
            {stats.mean.toFixed(2)} +/- {stats.std.toFixed(2)}
          </span>
        )}
      </div>
      <div className="flex items-end gap-px h-12 rounded-lg bg-muted/30 p-1">
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
        <span>{min.toFixed(2)}</span>
        <span>{max.toFixed(2)}</span>
      </div>
    </div>
  );
}
