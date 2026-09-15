"use client";

import { Bar, BarChart, XAxis, YAxis } from "recharts";
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import { descriptorValues, histogram } from "@/lib/analysis/stats";
import type { Report } from "@/lib/analysis/types";

const chartConfig = { count: { label: "Samples", color: "var(--chart-1)" } } satisfies ChartConfig;

/** Small multiples: one histogram per descriptor, shared axes, single hue. */
export function Distributions({ report }: { report: Report }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {report.descriptor_keys.map((key, i) => {
        const bins = histogram(descriptorValues(report, key), 24);
        const stat = report.descriptor_stats[key];
        const data = bins.map((b) => ({ x: ((b.x0 + b.x1) / 2).toFixed(2), count: b.count }));
        return (
          <figure key={key} className="rounded-lg border border-border p-3">
            <figcaption className="mb-1 flex items-baseline justify-between">
              <span className="text-sm font-medium">{report.descriptor_labels[i]}</span>
              <span className="tabular font-mono text-xs text-muted-foreground">
                μ {stat.mean.toFixed(2)} · σ {stat.std.toFixed(2)}
              </span>
            </figcaption>
            <ChartContainer config={chartConfig} className="h-28 w-full">
              <BarChart data={data} margin={{ left: 0, right: 0, top: 4, bottom: 0 }} barCategoryGap={1}>
                <XAxis dataKey="x" tickLine={false} axisLine={false} fontSize={10} interval={5} />
                <YAxis hide />
                <ChartTooltip content={<ChartTooltipContent labelFormatter={(l) => `≈ ${l}`} />} cursor={{ fill: "var(--muted)" }} />
                <Bar dataKey="count" fill="var(--color-count)" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ChartContainer>
            <p className="mt-1 text-xs text-muted-foreground">{report.descriptor_docs?.[key]}</p>
          </figure>
        );
      })}
    </div>
  );
}
