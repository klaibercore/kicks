"use client";

import type { PCName } from "@/types/cluster";
import { PC_COLORS, PC_LABELS_SHORT } from "@/lib/colors";
import { card } from "@/lib/styles";

export function VarianceBar({
  variance,
  pcNames,
}: {
  variance: number[];
  pcNames: PCName[];
}) {
  const total = variance.reduce((a, b) => a + b, 0);

  return (
    <div className={`${card} p-3 sm:p-5 space-y-3`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-foreground/80">
          Variance Explained
        </span>
        <span className="font-mono text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded-md">
          {(total * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-4 rounded-full overflow-hidden flex bg-muted gap-px">
        {variance.map((v, i) => (
          <div
            key={i}
            className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${(v / total) * 100}%`,
              backgroundColor: PC_COLORS[i],
              opacity: 0.75,
            }}
            title={`${pcNames[i]?.name ?? PC_LABELS_SHORT[i]}: ${(v * 100).toFixed(1)}%`}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-2 sm:gap-5 text-xs">
        {variance.map((v, i) => (
          <div key={i} className="flex items-center gap-1.5 sm:gap-2">
            <div
              className="w-2.5 h-2.5 rounded-full"
              style={{
                backgroundColor: PC_COLORS[i],
                boxShadow: "0 0 0 2px var(--border)",
              }}
            />
            <span className="text-foreground/80 font-medium">
              {pcNames[i]?.name ?? PC_LABELS_SHORT[i]}
            </span>
            <span className="font-mono text-muted-foreground">
              {(v * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
