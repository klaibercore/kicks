"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTheme } from "next-themes";
import type { Sample, ColorMode, ClusterData } from "@/types/cluster";
import { sampleColor } from "@/lib/colors";
import { card } from "@/lib/styles";

const PADDING = 44;

export function ScatterPlot({
  samples,
  xKey,
  yKey,
  xLabel,
  yLabel,
  colorMode,
  stats,
  selectedIdx,
  onSelect,
}: {
  samples: Sample[];
  xKey: "pc1" | "pc2" | "pc3";
  yKey: "pc1" | "pc2" | "pc3";
  xLabel: string;
  yLabel: string;
  colorMode: ColorMode;
  stats?: ClusterData["descriptor_stats"];
  selectedIdx: number | null;
  onSelect: (idx: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    const xs = samples.map((s) => s[xKey]);
    const ys = samples.map((s) => s[yKey]);
    const xPad = (Math.max(...xs) - Math.min(...xs)) * 0.08;
    const yPad = (Math.max(...ys) - Math.min(...ys)) * 0.08;
    return {
      xMin: Math.min(...xs) - xPad,
      xMax: Math.max(...xs) + xPad,
      yMin: Math.min(...ys) - yPad,
      yMax: Math.max(...ys) + yPad,
    };
  }, [samples, xKey, yKey]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const plotW = w - PADDING * 2;
    const plotH = h - PADDING * 2;

    const toX = (v: number) =>
      PADDING + ((v - xMin) / (xMax - xMin)) * plotW;
    const toY = (v: number) =>
      PADDING + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    ctx.clearRect(0, 0, w, h);

    const bgFill = isDark ? "rgba(255,255,255,0.01)" : "rgba(0,0,0,0.02)";
    const gridStroke = isDark
      ? "rgba(255,255,255,0.04)"
      : "rgba(0,0,0,0.06)";
    const borderStroke = isDark
      ? "rgba(255,255,255,0.06)"
      : "rgba(0,0,0,0.08)";
    const labelFill = isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)";
    const tickFill = isDark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.3)";
    const selectionStroke = isDark ? "#fff" : "#000";

    ctx.fillStyle = bgFill;
    ctx.beginPath();
    ctx.roundRect(PADDING, PADDING, plotW, plotH, 6);
    ctx.fill();

    ctx.strokeStyle = gridStroke;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const x = PADDING + (plotW * i) / 4;
      const y = PADDING + (plotH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(x, PADDING);
      ctx.lineTo(x, PADDING + plotH);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(PADDING, y);
      ctx.lineTo(PADDING + plotW, y);
      ctx.stroke();
    }

    ctx.strokeStyle = borderStroke;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(PADDING, PADDING, plotW, plotH, 6);
    ctx.stroke();

    for (const sample of samples) {
      const x = toX(sample[xKey]);
      const y = toY(sample[yKey]);
      const isSelected = sample.sample_idx === selectedIdx;
      const isHovered = sample.sample_idx === hoveredIdx;
      const color = sampleColor(sample, colorMode, stats);

      ctx.beginPath();
      ctx.arc(
        x,
        y,
        isSelected ? 5.5 : isHovered ? 4.5 : 2.5,
        0,
        Math.PI * 2,
      );
      ctx.fillStyle = color;
      ctx.globalAlpha = isSelected ? 1 : isHovered ? 0.95 : 0.7;
      ctx.fill();

      if (isSelected) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = selectionStroke;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    ctx.fillStyle = labelFill;
    ctx.font = "500 11px var(--font-geist-mono, monospace)";
    ctx.textAlign = "center";
    ctx.fillText(xLabel, w / 2, h - 4);

    ctx.save();
    ctx.translate(10, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    ctx.fillStyle = tickFill;
    ctx.font = "9px var(--font-geist-mono, monospace)";
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const xVal = xMin + ((xMax - xMin) * i) / 4;
      const yVal = yMin + ((yMax - yMin) * i) / 4;
      ctx.fillText(
        xVal.toFixed(1),
        PADDING + (plotW * i) / 4,
        PADDING + plotH + 14,
      );
      ctx.textAlign = "right";
      ctx.fillText(
        yVal.toFixed(1),
        PADDING - 6,
        PADDING + plotH - (plotH * i) / 4 + 3,
      );
      ctx.textAlign = "center";
    }
  }, [
    samples,
    xKey,
    yKey,
    xMin,
    xMax,
    yMin,
    yMax,
    colorMode,
    stats,
    selectedIdx,
    hoveredIdx,
    xLabel,
    yLabel,
    isDark,
  ]);

  const findClosest = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const plotW = rect.width - PADDING * 2;
      const plotH = rect.height - PADDING * 2;

      let closest: Sample | null = null;
      let closestDist = Infinity;

      for (const sample of samples) {
        const x =
          PADDING + ((sample[xKey] - xMin) / (xMax - xMin)) * plotW;
        const y =
          PADDING +
          plotH -
          ((sample[yKey] - yMin) / (yMax - yMin)) * plotH;
        const dist = Math.hypot(mx - x, my - y);
        if (dist < closestDist && dist < 12) {
          closest = sample;
          closestDist = dist;
        }
      }
      return closest;
    },
    [samples, xKey, yKey, xMin, xMax, yMin, yMax],
  );

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const closest = findClosest(e);
      if (closest) onSelect(closest.sample_idx);
    },
    [findClosest, onSelect],
  );

  const handleCanvasMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const closest = findClosest(e);
      setHoveredIdx(closest?.sample_idx ?? null);
      const canvas = canvasRef.current;
      if (canvas) canvas.style.cursor = closest ? "pointer" : "default";
    },
    [findClosest],
  );

  return (
    <div ref={containerRef} className={`${card} overflow-hidden`}>
      <canvas
        ref={canvasRef}
        className="w-full"
        style={{ height: 300, minHeight: 240 }}
        onClick={handleCanvasClick}
        onMouseMove={handleCanvasMove}
        onMouseLeave={() => setHoveredIdx(null)}
      />
    </div>
  );
}
