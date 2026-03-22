"use client";

import { useEffect, useRef } from "react";

export function WaveformViewerVis({
  waveformData,
}: {
  waveformData: number[] | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    if (!waveformData || waveformData.length === 0) {
      // Placeholder: faint center line
      ctx.strokeStyle = "rgba(52,211,153,0.12)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, H / 2);
      ctx.lineTo(W, H / 2);
      ctx.stroke();
      return;
    }

    const n = waveformData.length;
    const barW = W / n;

    for (let i = 0; i < n; i++) {
      const v = Math.min(1, Math.max(0, waveformData[i]));
      const barH = v * H;
      const x = i * barW;
      const y = H - barH;

      const gradient = ctx.createLinearGradient(0, H, 0, y);
      gradient.addColorStop(0, "rgba(52,211,153,0.9)");
      gradient.addColorStop(1, "rgba(52,211,153,0.2)");
      ctx.fillStyle = gradient;

      ctx.beginPath();
      ctx.roundRect(x + 0.5, y, Math.max(barW - 1, 1), barH, 1);
      ctx.fill();
    }
  }, [waveformData]);

  return (
    <div className="flex flex-col h-full w-full">
      <p className="text-[10px] font-semibold tracking-widest uppercase text-muted-foreground/50 text-center pt-2 pb-1">
        Waveform
      </p>
      <div className="flex-1 relative min-h-0">
        <canvas
          ref={canvasRef}
          width={128}
          height={256}
          className="w-full h-full"
          style={{ imageRendering: "pixelated" }}
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
