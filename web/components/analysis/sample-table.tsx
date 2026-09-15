"use client";

import { useTheme } from "next-themes";
import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { clusterColor } from "@/lib/analysis/palette";
import { formatMs } from "@/lib/analysis/stats";
import type { Report, SampleRow } from "@/lib/analysis/types";

const PAGE = 25;

type SortKey = "sample_idx" | "cluster" | "duration_ms" | "confidence" | `d:${string}`;

/**
 * The table view of every sample — the non-visual encoding every chart needs.
 * Samples are identified by index only; the corpus files themselves are not
 * part of the site.
 */
export function SampleTable({ report, isolated }: { report: Report; isolated: number | null }) {
  const { resolvedTheme } = useTheme();
  const dark = resolvedTheme === "dark";
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "cluster", dir: 1 });
  const [page, setPage] = useState(0);

  const rows = useMemo(() => {
    const get = (s: SampleRow, key: SortKey): number =>
      key.startsWith("d:") ? s.descriptors[key.slice(2)] : (s[key as keyof SampleRow] as number);
    return report.samples
      .filter((s) => isolated === null || s.cluster === isolated)
      .sort((a, b) => (get(a, sort.key) - get(b, sort.key)) * sort.dir);
  }, [report, sort, isolated]);

  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  const current = Math.min(page, pages - 1);
  const slice = rows.slice(current * PAGE, current * PAGE + PAGE);

  const header = (label: string, key: SortKey, align = "left") => (
    <TableHead key={key} className={align === "right" ? "text-right" : ""}>
      <button
        type="button"
        className="font-medium hover:text-foreground"
        onClick={() => setSort((s) => ({ key, dir: s.key === key ? ((s.dir * -1) as 1 | -1) : 1 }))}
      >
        {label}
        {sort.key === key ? (sort.dir === 1 ? " ↑" : " ↓") : ""}
      </button>
    </TableHead>
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted-foreground">
          {rows.length} of {report.samples.length} samples
        </span>
        <div className="ml-auto flex items-center gap-1 text-xs">
          <Button size="xs" variant="ghost" disabled={current === 0} onClick={() => setPage(current - 1)}>
            Prev
          </Button>
          <span className="tabular font-mono text-muted-foreground">
            {current + 1}/{pages}
          </span>
          <Button size="xs" variant="ghost" disabled={current >= pages - 1} onClick={() => setPage(current + 1)}>
            Next
          </Button>
        </div>
      </div>
      <div className="overflow-x-auto rounded-lg border border-border">
        <Table>
          <TableHeader>
            <TableRow>
              {header("#", "sample_idx")}
              {header("Cluster", "cluster")}
              {report.descriptor_keys.map((k, i) => header(report.descriptor_labels[i], `d:${k}`, "right"))}
              {header("Length", "duration_ms", "right")}
              {header("Conf.", "confidence", "right")}
            </TableRow>
          </TableHeader>
          <TableBody>
            {slice.map((s) => (
              <TableRow key={s.sample_idx}>
                <TableCell className="tabular font-mono text-xs text-muted-foreground">{s.sample_idx}</TableCell>
                <TableCell>
                  <span className="inline-flex items-center gap-1.5 font-mono text-xs">
                    <span className="inline-block size-2 rounded-full" style={{ background: clusterColor(s.cluster, dark) }} />
                    {s.cluster}
                  </span>
                </TableCell>
                {report.descriptor_keys.map((k) => (
                  <TableCell key={k} className="tabular text-right font-mono text-xs">
                    {s.descriptors[k].toFixed(2)}
                  </TableCell>
                ))}
                <TableCell className="tabular text-right font-mono text-xs">{formatMs(s.duration_ms)}</TableCell>
                <TableCell className="tabular text-right font-mono text-xs">{(s.confidence * 100).toFixed(0)}%</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
