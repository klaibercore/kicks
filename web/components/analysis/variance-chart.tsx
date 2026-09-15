"use client";

import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import type { Report } from "@/lib/analysis/types";

const chartConfig = { variance: { label: "Variance explained", color: "var(--chart-1)" } } satisfies ChartConfig;

/** One measure, one hue: how much descriptor variance each PC carries. */
export function VarianceChart({ report }: { report: Report }) {
  const data = report.pca_variance_explained.map((v, i) => ({
    pc: report.pc_names[i]?.descriptor ? `PC${i + 1} · ${report.pc_names[i].name}` : `PC${i + 1}`,
    variance: Math.round(v * 1000) / 10,
  }));
  return (
    <ChartContainer config={chartConfig} className="h-40 w-full">
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
        <CartesianGrid horizontal={false} strokeDasharray="3 3" />
        <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false} fontSize={11} />
        <YAxis type="category" dataKey="pc" width={120} tickLine={false} axisLine={false} fontSize={11} />
        <ChartTooltip content={<ChartTooltipContent formatter={(v) => `${v}%`} />} cursor={{ fill: "var(--muted)" }} />
        <Bar dataKey="variance" fill="var(--color-variance)" radius={[0, 4, 4, 0]} barSize={14} label={{ position: "right", fontSize: 11, formatter: (v: unknown) => `${v}%` }} />
      </BarChart>
    </ChartContainer>
  );
}
