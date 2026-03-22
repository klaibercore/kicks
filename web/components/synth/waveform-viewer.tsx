"use client";

import { useCallback, useEffect, useRef } from "react";

export function WaveformViewerVis({
  waveformData,
}: {
  waveformData: number[] | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const dataRef = useRef(waveformData);
  dataRef.current = waveformData;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const W = canvas.width;
    const H = canvas.height;
    const data = dataRef.current;
    const midY = H / 2;

    ctx.clearRect(0, 0, W, H);

    // Center x-axis line (always visible)
    ctx.strokeStyle = "rgba(52,211,153,0.12)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(W, midY);
    ctx.stroke();

    if (!data || data.length === 0) {
      // Empty state: faint guide lines at 25% and 75%
      ctx.strokeStyle = "rgba(52,211,153,0.05)";
      for (const frac of [0.25, 0.75]) {
        ctx.beginPath();
        ctx.moveTo(0, H * frac);
        ctx.lineTo(W, H * frac);
        ctx.stroke();
      }
      return;
    }

    const n = data.length;
    const padding = 0.85; // leave a bit of headroom

    // Build mirrored waveform path
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * W;
      const amp = data[i] * midY * padding;
      const y = midY - amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    for (let i = n - 1; i >= 0; i--) {
      const x = (i / (n - 1)) * W;
      const amp = data[i] * midY * padding;
      ctx.lineTo(x, midY + amp);
    }
    ctx.closePath();

    // Gradient fill: brighter at edges, subtle at center
    const fillGrad = ctx.createLinearGradient(0, 0, 0, H);
    fillGrad.addColorStop(0, "rgba(52,211,153,0.6)");
    fillGrad.addColorStop(0.35, "rgba(52,211,153,0.15)");
    fillGrad.addColorStop(0.5, "rgba(52,211,153,0.05)");
    fillGrad.addColorStop(0.65, "rgba(52,211,153,0.15)");
    fillGrad.addColorStop(1, "rgba(52,211,153,0.6)");
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Stroke the top and bottom edges
    ctx.save();
    ctx.shadowColor = "rgba(52,211,153,0.4)";
    ctx.shadowBlur = 6;

    // Top edge
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * W;
      const amp = data[i] * midY * padding;
      const y = midY - amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "rgba(52,211,153,0.8)";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Bottom edge (mirrored)
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * W;
      const amp = data[i] * midY * padding;
      const y = midY + amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }, []);

  // Resize canvas to match container
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      const dpr = window.devicePixelRatio || 1;
      const w = Math.round(width * dpr);
      const h = Math.round(height * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        draw();
      }
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, [draw]);

  // Redraw when data changes
  useEffect(() => {
    draw();
  }, [waveformData, draw]);

  return (
    <div className="flex flex-col h-full w-full">
      <p className="text-[10px] font-semibold tracking-widest uppercase text-muted-foreground/50 text-center pt-2 pb-1">
        Waveform
      </p>
      <div ref={containerRef} className="flex-1 relative min-h-0">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
        />
        {!waveformData && (
          <p className="absolute inset-0 flex items-center justify-center text-[10px] text-muted-foreground/30 tracking-wide">
            post-vocoder
          </p>
        )}
      </div>
    </div>
  );
}
