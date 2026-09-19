"use client";

import { Bar, BarChart, ReferenceLine, XAxis, YAxis } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import {
  descriptorUnit,
  descriptorValues,
  histogram,
} from "@/lib/analysis/stats";
import type { Report } from "@/lib/analysis/types";

const chartConfig = {
  count: { label: "Corpus", color: "var(--chart-1)" },
  selected: { label: "Selected cluster", color: "var(--chart-1)" },
} satisfies ChartConfig;

export function Distributions({
  report,
  isolated,
}: {
  report: Report;
  isolated: number | null;
}) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {report.descriptor_keys.map((key, i) => {
        const stat = report.descriptor_stats[key];
        const min = stat.min;
        const max = stat.max > min ? stat.max : min + 1;
        const bins = histogram(descriptorValues(report, key), 28, min, max);
        const subset = histogram(
          report.samples
            .filter((s) => s.cluster === isolated)
            .map((s) => s.descriptors[key]),
          28,
          min,
          max,
        );
        const data = bins.map((b, j) => ({
          x: (b.x0 + b.x1) / 2,
          count: b.count - (isolated === null ? 0 : subset[j].count),
          selected: subset[j].count,
        }));
        const unit = descriptorUnit(key, report);
        return (
          <figure
            key={key}
            className="rounded-xl border border-border bg-card p-5"
          >
            <figcaption className="mb-4 flex items-baseline justify-between gap-2">
              <span className="text-sm font-medium">
                {report.descriptor_labels[i]}
              </span>
              <span className="font-mono text-[10px] text-muted-foreground">
                mean {stat.mean.toFixed(1)} {unit}
              </span>
            </figcaption>
            <ChartContainer
              config={{
                ...chartConfig,
                count: {
                  ...chartConfig.count,
                  label: isolated === null ? "Corpus" : "Other samples",
                },
              }}
              className="h-32 w-full"
            >
              <BarChart
                data={data}
                barCategoryGap={1}
                margin={{ left: 0, right: 8, top: 4, bottom: 0 }}
              >
                <XAxis
                  dataKey="x"
                  type="number"
                  domain={[min, max]}
                  tickCount={4}
                  tickFormatter={(v: number) => `${v.toFixed(0)}`}
                  tickLine={false}
                  axisLine={false}
                  fontSize={10}
                />
                <YAxis hide />
                <ChartTooltip
                  content={
                    <ChartTooltipContent
                      labelFormatter={(l) =>
                        `≈ ${Number(l).toFixed(1)} ${unit}`
                      }
                    />
                  }
                  cursor={{ fill: "var(--muted)" }}
                />
                {isolated !== null && (
                  <Bar
                    dataKey="selected"
                    stackId="population"
                    fill="var(--color-selected)"
                    isAnimationActive={false}
                  />
                )}
                <Bar
                  dataKey="count"
                  stackId="population"
                  fill="var(--color-count)"
                  fillOpacity={isolated === null ? 0.8 : 0.18}
                  radius={[2, 2, 0, 0]}
                  isAnimationActive={false}
                />
                <ReferenceLine
                  x={stat.mean}
                  stroke="var(--foreground)"
                  strokeOpacity={0.45}
                  strokeDasharray="3 3"
                />
              </BarChart>
            </ChartContainer>
            <div className="mt-1 flex justify-between font-mono text-[9px] text-muted-foreground">
              <span>{unit} · full observed range</span>
              <span>dashed line = corpus mean</span>
            </div>
            <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
              {report.descriptor_docs?.[key]}
            </p>
          </figure>
        );
      })}
    </div>
  );
}
