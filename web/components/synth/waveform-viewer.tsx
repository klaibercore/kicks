"use client";

import { useCallback, useEffect, useRef } from "react";

const DURATION_S = (256 * 256) / 44100; // ≈ 1.486 s
const TIME_TICKS = [0, 0.5, 1.0];
const AMP_TICKS = [
  { value: 1, label: "1" },
  { value: 0.5, label: ".5" },
  { value: 0, label: "0" },
  { value: -0.5, label: "-.5" },
  { value: -1, label: "-1" },
];

// CSS margins (multiplied by DPR at draw time)
const MARGIN_LEFT = 24;
const MARGIN_BOTTOM = 16;

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
    const dpr = window.devicePixelRatio || 1;
    const data = dataRef.current;

    const ML = Math.round(MARGIN_LEFT * dpr);
    const MB = Math.round(MARGIN_BOTTOM * dpr);
    const plotW = W - ML;
    const plotH = H - MB;
    const midY = plotH / 2;

    ctx.clearRect(0, 0, W, H);

    // ── Grid lines & labels ───────────────────────────
    const fontSize = Math.round(8 * dpr);
    ctx.font = `${fontSize}px ui-monospace, "Geist Mono", monospace`;

    // Amplitude ticks (Y-axis)
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tick of AMP_TICKS) {
      const y = midY - tick.value * midY;

      const isCenter = tick.value === 0;
      ctx.strokeStyle = isCenter
        ? "rgba(52,211,153,0.12)"
        : "rgba(52,211,153,0.06)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(ML, Math.round(y) + 0.5);
      ctx.lineTo(W, Math.round(y) + 0.5);
      ctx.stroke();

      ctx.fillStyle = "rgba(255,255,255,0.3)";
      ctx.fillText(tick.label, ML - 3 * dpr, y);
    }

    // Time ticks (X-axis)
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const t of TIME_TICKS) {
      const x = ML + (t / DURATION_S) * plotW;

      ctx.strokeStyle = "rgba(52,211,153,0.06)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, 0);
      ctx.lineTo(Math.round(x) + 0.5, plotH);
      ctx.stroke();

      ctx.fillStyle = "rgba(255,255,255,0.3)";
      ctx.fillText(`${t}s`, x, plotH + 2 * dpr);
    }

    // Plot border
    ctx.strokeStyle = "rgba(52,211,153,0.08)";
    ctx.lineWidth = 1;
    ctx.strokeRect(ML + 0.5, 0.5, plotW - 1, plotH - 1);

    if (!data || data.length === 0) return;

    // ── Waveform ──────────────────────────────────────
    const n = data.length;
    const padding = 0.85;

    // Clip to plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(ML, 0, plotW, plotH);
    ctx.clip();

    // Build mirrored waveform path
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = ML + (i / (n - 1)) * plotW;
      const amp = data[i] * midY * padding;
      const y = midY - amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    for (let i = n - 1; i >= 0; i--) {
      const x = ML + (i / (n - 1)) * plotW;
      const amp = data[i] * midY * padding;
      ctx.lineTo(x, midY + amp);
    }
    ctx.closePath();

    // Gradient fill: brighter at edges, subtle at center
    const fillGrad = ctx.createLinearGradient(0, 0, 0, plotH);
    fillGrad.addColorStop(0, "rgba(52,211,153,0.6)");
    fillGrad.addColorStop(0.35, "rgba(52,211,153,0.15)");
    fillGrad.addColorStop(0.5, "rgba(52,211,153,0.05)");
    fillGrad.addColorStop(0.65, "rgba(52,211,153,0.15)");
    fillGrad.addColorStop(1, "rgba(52,211,153,0.6)");
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Stroke the top and bottom edges
    ctx.shadowColor = "rgba(52,211,153,0.4)";
    ctx.shadowBlur = 6;
    ctx.strokeStyle = "rgba(52,211,153,0.8)";
    ctx.lineWidth = 1.5;

    // Top edge
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = ML + (i / (n - 1)) * plotW;
      const amp = data[i] * midY * padding;
      const y = midY - amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Bottom edge (mirrored)
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = ML + (i / (n - 1)) * plotW;
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
