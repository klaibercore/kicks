"use client";

import { useEffect, useRef } from "react";
import { peaks } from "@/lib/audio/engine";

export function WaveformView({ buffer, className }: { buffer: AudioBuffer | null; className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const { width, height } = canvas.getBoundingClientRect();
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);
      const styles = getComputedStyle(canvas);
      const fg = styles.getPropertyValue("--foreground").trim() || "currentColor";
      const muted = styles.getPropertyValue("--border").trim();

      ctx.strokeStyle = muted;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, height / 2 + 0.5);
      ctx.lineTo(width, height / 2 + 0.5);
      ctx.stroke();

      if (!buffer) return;
      const bins = Math.max(64, Math.floor(width));
      const env = peaks(buffer, bins);
      ctx.fillStyle = fg;
      const mid = height / 2;
      for (let i = 0; i < bins; i++) {
        const min = env[i * 2];
        const max = env[i * 2 + 1];
        const x = (i / bins) * width;
        const top = mid - max * mid * 0.95;
        const bottom = mid - min * mid * 0.95;
        ctx.fillRect(x, top, Math.max(1, width / bins), Math.max(1, bottom - top));
      }
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [buffer]);

  return <canvas ref={canvasRef} className={className} aria-label="Waveform of the current render" role="img" />;
}
