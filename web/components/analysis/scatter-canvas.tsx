"use client";

import { useTheme } from "next-themes";
import { useEffect, useMemo, useRef, useState } from "react";
import { clusterColor } from "@/lib/analysis/palette";
import type { Report, SampleRow } from "@/lib/analysis/types";

interface ScatterProps {
  report: Report;
  xKey: "pc1" | "pc2" | "pc3";
  yKey: "pc1" | "pc2" | "pc3";
  xLabel: string;
  yLabel: string;
  isolated: number | null;
  onPick?: (sample: SampleRow | null) => void;
}

/**
 * Canvas scatter: 4000 points stay fluid, which SVG would not. Hover finds the
 * nearest point within 10 px and shows a tooltip; the isolated cluster (if any)
 * is drawn on top with everything else dimmed, which is the secondary encoding
 * that lets clusters beyond the eight validated hues still be told apart.
 */
export function ScatterCanvas({ report, xKey, yKey, xLabel, yLabel, isolated, onPick }: ScatterProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { resolvedTheme } = useTheme();
  const dark = resolvedTheme === "dark";
  const [hover, setHover] = useState<{ sample: SampleRow; x: number; y: number; w: number } | null>(null);

  const domain = useMemo(() => {
    const xs = report.samples.map((s) => s[xKey] ?? 0);
    const ys = report.samples.map((s) => s[yKey] ?? 0);
    const pad = (arr: number[]) => {
      const lo = Math.min(...arr);
      const hi = Math.max(...arr);
      const p = (hi - lo) * 0.06 || 1;
      return [lo - p, hi + p] as const;
    };
    return { x: pad(xs), y: pad(ys) };
  }, [report, xKey, yKey]);

  const layout = useRef({ w: 0, h: 0, m: { l: 36, r: 12, t: 12, b: 32 } });

  const project = (s: SampleRow) => {
    const { w, h, m } = layout.current;
    const px = m.l + (((s[xKey] ?? 0) - domain.x[0]) / (domain.x[1] - domain.x[0])) * (w - m.l - m.r);
    const py = h - m.b - (((s[yKey] ?? 0) - domain.y[0]) / (domain.y[1] - domain.y[0])) * (h - m.t - m.b);
    return [px, py] as const;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const w = Math.floor(rect.width);
      const h = Math.floor(rect.height);
      layout.current.w = w;
      layout.current.h = h;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);
      const { m } = layout.current;
      const styles = getComputedStyle(canvas);
      const grid = styles.getPropertyValue("--border").trim();
      const ink = styles.getPropertyValue("--muted-foreground").trim();

      // Recessive axes + zero lines.
      ctx.strokeStyle = grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(m.l, m.t);
      ctx.lineTo(m.l, h - m.b);
      ctx.lineTo(w - m.r, h - m.b);
      ctx.stroke();
      const zero = project({ pc1: 0, pc2: 0, pc3: 0 } as SampleRow);
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(m.l, zero[1]);
      ctx.lineTo(w - m.r, zero[1]);
      ctx.moveTo(zero[0], m.t);
      ctx.lineTo(zero[0], h - m.b);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = ink;
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.fillText(xLabel, m.l + (w - m.l - m.r) / 2, h - 8);
      ctx.save();
      ctx.translate(12, m.t + (h - m.t - m.b) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();

      const r = report.samples.length > 2000 ? 2.2 : 3.2;
      const pass = (dim: boolean) => {
        for (const s of report.samples) {
          const isIso = isolated === null || s.cluster === isolated;
          if (dim === isIso) continue;
          const [x, y] = project(s);
          ctx.globalAlpha = dim ? 0.12 : 0.85;
          ctx.fillStyle = clusterColor(s.cluster, dark);
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fill();
        }
      };
      pass(true);
      pass(false);
      ctx.globalAlpha = 1;

      if (hover) {
        const [x, y] = project(hover.sample);
        ctx.strokeStyle = styles.getPropertyValue("--background").trim();
        ctx.lineWidth = 2;
        ctx.fillStyle = clusterColor(hover.sample.cluster, dark);
        ctx.beginPath();
        ctx.arc(x, y, r + 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [report, xKey, yKey, isolated, dark, hover, xLabel, yLabel, domain]);

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best: SampleRow | null = null;
    let bestD = 100; // 10 px squared
    for (const s of report.samples) {
      if (isolated !== null && s.cluster !== isolated) continue;
      const [x, y] = project(s);
      const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < bestD) {
        bestD = d;
        best = s;
      }
    }
    setHover(best ? { sample: best, x: mx, y: my, w: rect.width } : null);
  };

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        className="aspect-[4/3] w-full cursor-crosshair sm:aspect-[16/9]"
        role="img"
        aria-label={`Scatter plot of ${xLabel} against ${yLabel}, coloured by cluster`}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        onClick={() => onPick?.(hover?.sample ?? null)}
      />
      {hover ? (
        <div
          className="pointer-events-none absolute z-10 max-w-60 rounded-md border border-border bg-popover px-2.5 py-2 text-xs shadow-md"
          style={{
            left: Math.min(hover.x + 12, hover.w - 200),
            top: Math.max(0, hover.y - 8),
          }}
        >
          <div className="mb-1 flex items-center gap-1.5 font-medium">
            <span className="inline-block size-2 rounded-full" style={{ background: clusterColor(hover.sample.cluster, dark) }} />
            Cluster {hover.sample.cluster}
          </div>
          <div className="font-mono text-muted-foreground">
            sample #{hover.sample.sample_idx} · {Math.round(hover.sample.duration_ms)} ms
          </div>
          <div className="mt-1 grid grid-cols-2 gap-x-3 font-mono text-muted-foreground">
            {report.descriptor_keys.map((k) => (
              <span key={k} className="flex justify-between">
                <span>{k}</span>
                <span className="tabular text-foreground">{hover.sample.descriptors[k].toFixed(2)}</span>
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
