"use client";

import { useTheme } from "next-themes";
import type { PCName, DescriptorKey } from "@/types/cluster";
import { DESCRIPTOR_KEYS, DESCRIPTOR_LABELS } from "@/types/cluster";
import { PC_COLORS, correlationColor, correlationTextColor } from "@/lib/colors";
import { card, sectionTitle } from "@/lib/styles";

export function CorrelationHeatmap({
  correlations,
  pcNames,
}: {
  correlations: Record<string, Record<string, number>>;
  pcNames: PCName[];
}) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  return (
    <div className={`${card} p-3 sm:p-5 space-y-3 sm:space-y-4`}>
      <h3 className={sectionTitle}>Correlation Matrix</h3>
      <div className="overflow-x-auto -mx-3 sm:mx-0 px-3 sm:px-0">
        <table className="w-full text-xs min-w-[220px]">
          <thead>
            <tr>
              <th className="text-left text-muted-foreground pb-3 pr-2 sm:pr-4 font-normal" />
              {["pc1", "pc2", "pc3"].map((pc, i) => (
                <th
                  key={pc}
                  className="pb-3 px-1.5 sm:px-2 font-bold text-center text-[13px]"
                  style={{ color: PC_COLORS[i] }}
                >
                  {pcNames[i]?.name ?? pc.toUpperCase()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {DESCRIPTOR_KEYS.map((desc) => (
              <tr key={desc}>
                <td className="text-foreground/80 pr-2 sm:pr-4 py-1.5 font-medium text-[13px]">
                  {DESCRIPTOR_LABELS[desc]}
                </td>
                {["pc1", "pc2", "pc3"].map((pc) => {
                  const v = correlations[pc]?.[desc] ?? 0;
                  return (
                    <td key={pc} className="px-1.5 sm:px-2 py-1.5 text-center">
                      <div
                        className="rounded-lg px-2 sm:px-3 py-1.5 sm:py-2 font-mono font-semibold text-[13px] transition-colors"
                        style={{
                          backgroundColor: correlationColor(v),
                          color:
                            Math.abs(v) > 0.25
                              ? isDark
                                ? "#fff"
                                : "#000"
                              : correlationTextColor(v, isDark),
                        }}
                      >
                        {v > 0 ? "+" : ""}
                        {v.toFixed(2)}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
