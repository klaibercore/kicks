"use client";

import { useTheme } from "next-themes";
import { diverging } from "@/lib/analysis/palette";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface HeatmapProps {
  rows: string[];
  cols: string[];
  value: (row: string, col: string) => number;
  /** Absolute value that saturates the colour. */
  scale?: number;
  rowLabel?: (row: string) => string;
  colLabel?: (col: string) => string;
  caption: string;
}

/** Diverging blue <-> red grid with a neutral midpoint. Values are always printed. */
export function Heatmap({ rows, cols, value, scale = 1, rowLabel = (r) => r, colLabel = (c) => c, caption }: HeatmapProps) {
  const { resolvedTheme } = useTheme();
  const dark = resolvedTheme === "dark";
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-0.5 text-xs">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>
            <th aria-hidden />
            {cols.map((c) => (
              <th key={c} scope="col" className="pb-1 text-center font-mono font-normal text-muted-foreground">
                {colLabel(c)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r}>
              <th scope="row" className="pr-2 text-right font-mono font-normal text-muted-foreground">
                {rowLabel(r)}
              </th>
              {cols.map((c) => {
                const v = value(r, c);
                const t = Math.max(-1, Math.min(1, v / scale));
                const strong = Math.abs(t) > 0.55;
                return (
                  <td key={c} className="p-0">
                    <Tooltip>
                      <TooltipTrigger
                        render={
                          <div
                            className="tabular flex h-9 min-w-12 items-center justify-center rounded-sm font-mono"
                            style={{ background: diverging(t, dark), color: strong ? "#fff" : undefined }}
                          />
                        }
                      >
                        {v.toFixed(2)}
                      </TooltipTrigger>
                      <TooltipContent>
                        {rowLabel(r)} × {colLabel(c)}: {v.toFixed(3)}
                      </TooltipContent>
                    </Tooltip>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
