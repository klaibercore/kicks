"use client";

import { useTheme } from "next-themes";
import type { PCName, DescriptorKey } from "@/types/cluster";
import { DESCRIPTOR_LABELS } from "@/types/cluster";
import { PC_COLORS, correlationTextColor } from "@/lib/colors";
import { card } from "@/lib/styles";

export function PCCard({
  pcIndex,
  variance,
  correlations,
  loadings,
  pcName,
}: {
  pcIndex: number;
  variance: number;
  correlations: Record<string, number>;
  loadings?: Record<string, number>;
  pcName: PCName;
}) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
  const values = loadings || correlations;
  const sorted = Object.entries(values).sort(
    (a, b) => Math.abs(b[1]) - Math.abs(a[1]),
  );
  const maxAbs = Math.max(...sorted.map(([, v]) => Math.abs(v)), 0.01);

  return (
    <div className={`${card} p-3 sm:p-5 space-y-3 sm:space-y-4`}>
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full shrink-0"
              style={{
                backgroundColor: PC_COLORS[pcIndex],
                boxShadow: `0 0 0 2px ${PC_COLORS[pcIndex]}40`,
              }}
            />
            <span className="text-[15px] font-bold text-foreground truncate">
              {pcName.name}
            </span>
          </div>
          <div className="text-[11px] text-muted-foreground mt-1 ml-[22px] font-mono">
            PC{pcIndex + 1} {loadings ? "· loadings" : "· correlations"}
          </div>
        </div>
        <div className="text-right shrink-0">
          <div
            className="text-xl sm:text-2xl font-mono font-black tracking-tight"
            style={{ color: PC_COLORS[pcIndex] }}
          >
            {(variance * 100).toFixed(1)}
            <span className="text-sm font-medium opacity-60">%</span>
          </div>
        </div>
      </div>

      <div className="space-y-1.5 sm:space-y-2">
        {sorted.map(([desc, corr]) => {
          const width = (Math.abs(corr) / maxAbs) * 100;
          const isPositive = corr >= 0;
          return (
            <div
              key={desc}
              className="flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-xs"
            >
              <span className="w-10 sm:w-12 text-muted-foreground text-right font-medium shrink-0">
                {DESCRIPTOR_LABELS[desc as DescriptorKey] || desc}
              </span>
              <div className="flex-1 h-4 sm:h-5 relative rounded-md bg-muted/50 overflow-hidden">
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-border" />
                <div
                  className="absolute top-1 bottom-1 rounded-sm transition-all duration-500"
                  style={{
                    width: `${width / 2}%`,
                    left: isPositive ? "50%" : `${50 - width / 2}%`,
                    background: isPositive
                      ? "linear-gradient(to right, rgba(52,211,153,0.4), rgba(52,211,153,0.7))"
                      : "linear-gradient(to left, rgba(248,113,113,0.4), rgba(248,113,113,0.7))",
                  }}
                />
              </div>
              <span
                className="w-10 sm:w-12 font-mono text-right shrink-0 font-medium"
                style={{ color: correlationTextColor(corr, isDark) }}
              >
                {corr > 0 ? "+" : ""}
                {corr.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
