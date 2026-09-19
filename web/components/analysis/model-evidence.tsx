"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { Report } from "@/lib/analysis/types";

export function ModelEvidence({ report }: { report: Report }) {
  const model = report.clustering;
  if (!model)
    return (
      <div className="rounded-xl border border-border bg-muted/30 p-6 text-sm leading-relaxed text-muted-foreground">
        This report uses the earlier latent-space mixture. Its model-selection
        scores were not saved, so separation and candidate comparisons are
        unavailable. Membership confidence describes the fitted model; it does
        not establish that the clusters are perceptually distinct.
      </div>
    );
  const data = Array.from({ length: model.max_k }, (_, i) => {
    const candidates = model.candidates.filter(
      (c) => c.k === i + 1 && c.converged,
    );
    const full = candidates.find((c) => c.covariance === "full");
    const diag = candidates.find((c) => c.covariance === "diag");
    return {
      k: i + 1,
      full: full ? (full.bic - model.bic) / 1000 : null,
      diag: diag ? (diag.bic - model.bic) / 1000 : null,
    };
  });
  return (
    <div className="grid overflow-hidden rounded-xl border border-border bg-card lg:grid-cols-[1.3fr_1fr]">
      <div className="p-5 sm:p-7">
        <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-medium">
            How many groups does the evidence support?
          </h3>
          <span className="font-mono text-[10px] text-muted-foreground">
            LOWER IS BETTER
          </span>
        </div>
        <p className="mb-5 text-xs leading-relaxed text-muted-foreground">
          BIC balances fit against model complexity. The best converged
          candidate is at zero.
        </p>
        <ChartContainer
          config={{
            full: { label: "Full covariance", color: "var(--chart-1)" },
            diag: { label: "Diagonal covariance", color: "var(--chart-2)" },
          }}
          className="h-56 w-full"
        >
          <LineChart
            data={data}
            margin={{ left: 0, right: 12, top: 16, bottom: 6 }}
          >
            <CartesianGrid vertical={false} strokeDasharray="3 4" />
            <XAxis
              dataKey="k"
              type="number"
              domain={[1, Math.max(2, model.max_k)]}
              tickCount={Math.min(model.max_k, 8)}
              tickLine={false}
              axisLine={false}
              fontSize={10}
            />
            <YAxis
              width={40}
              tickFormatter={(n: number) => `${n.toFixed(0)}k`}
              tickLine={false}
              axisLine={false}
              fontSize={10}
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  labelFormatter={(k) => `${k} components`}
                  formatter={(v, name) =>
                    `${name === "full" ? "Full" : "Diagonal"}: +${(Number(v) * 1000).toLocaleString(undefined, { maximumFractionDigits: 0 })} BIC`
                  }
                />
              }
            />
            <ReferenceLine
              x={model.selected_k}
              stroke="var(--foreground)"
              strokeOpacity={0.4}
              strokeDasharray="3 4"
            />
            <Line
              dataKey="full"
              stroke="var(--color-full)"
              strokeWidth={2}
              dot={{ r: 2 }}
              isAnimationActive={false}
            />
            <Line
              dataKey="diag"
              stroke="var(--color-diag)"
              strokeWidth={2}
              strokeDasharray="4 3"
              dot={{ r: 2 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ChartContainer>
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-[10px] text-muted-foreground">
          <span>Number of mixture components</span>
          <div className="flex gap-4">
            <span>
              <span className="mr-1.5 inline-block h-0.5 w-4 bg-chart-1 align-middle" />
              Full covariance
            </span>
            <span>
              <span className="mr-1.5 inline-block h-0.5 w-4 bg-chart-2 align-middle" />
              Diagonal covariance
            </span>
          </div>
        </div>
      </div>
      <div className="flex flex-col justify-center gap-5 border-t border-border bg-muted/25 p-5 sm:p-7 lg:border-t-0 lg:border-l">
        <div>
          <p className="mb-2 font-mono text-[10px] tracking-widest text-muted-foreground">
            MODEL SELECTION
          </p>
          <h3 className="text-xl font-medium tracking-tight">
            {model.selected_k} components.{" "}
            {model.covariance === "full" ? "Full" : "Diagonal"} covariance.
          </h3>
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
            {model.candidates.length} candidates evaluated with three
            initializations each. {model.dimensions} latent principal components
            retain {(model.retained_variance * 100).toFixed(1)}% of standardized
            latent variance.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-5 border-y border-border py-4">
          <div>
            <p className="text-[11px] text-muted-foreground">
              Geometric separation
            </p>
            <p className="mt-1 font-mono text-xl">
              {model.silhouette?.toFixed(3) ?? "—"}
            </p>
            <p className="mt-1 text-[10px] leading-relaxed text-muted-foreground">
              Silhouette · −1 to +1
              <br />
              {model.silhouette_samples.toLocaleString()} sampled points
            </p>
          </div>
          <div>
            <p className="text-[11px] text-muted-foreground">
              Uncertain assignments
            </p>
            <p className="mt-1 font-mono text-xl">
              {((model.ambiguous_count / report.samples.length) * 100).toFixed(
                1,
              )}
              %
            </p>
            <p className="mt-1 text-[10px] leading-relaxed text-muted-foreground">
              {model.ambiguous_count.toLocaleString()} samples below{" "}
              {model.confidence_threshold * 100}% membership
            </p>
          </div>
        </div>
        <p className="text-xs leading-relaxed text-muted-foreground">
          {model.at_search_boundary
            ? `The best model reaches the ${model.max_k}-component search limit. Treat this grouping as provisional; a wider search may support more components. `
            : "The best model lies within the search range. "}
          {model.silhouette !== null && model.silhouette < 0.2
            ? "Low geometric separation suggests overlapping or elongated groups. "
            : ""}
          High membership confidence is not proof of distinct sound categories.
        </p>
      </div>
    </div>
  );
}
