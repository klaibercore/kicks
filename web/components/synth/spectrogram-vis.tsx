"use client";

import { useEffect, useRef } from "react";

// Color map: 0 → near-black, mid → violet, high → pink, 1 → white
function intensityToRgb(v: number): [number, number, number] {
  // 4-stop gradient: black → violet → pink → white
  if (v < 0.33) {
    const t = v / 0.33;
    return [
      Math.round(167 * t),
      Math.round(139 * t),
      Math.round(250 * t),
    ];
  } else if (v < 0.66) {
    const t = (v - 0.33) / 0.33;
    return [
      Math.round(167 + (244 - 167) * t),
      Math.round(139 + (114 - 139) * t),
      Math.round(250 + (182 - 250) * t),
    ];
  } else {
    const t = (v - 0.66) / 0.34;
    return [
      Math.round(244 + (255 - 244) * t),
      Math.round(114 + (255 - 114) * t),
      Math.round(182 + (255 - 182) * t),
    ];
  }
}

export function SpectrogramVis({
  spectrogram,
}: {
  spectrogram: number[][] | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const W = canvas.width;   // 256 (time frames)
    const H = canvas.height;  // 128 (freq bins)

    if (!spectrogram) {
      // Placeholder: dark background with faint grid
      ctx.fillStyle = "rgba(0,0,0,0)";
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = "rgba(167,139,250,0.08)";
      ctx.lineWidth = 1;
      for (let y = 0; y < H; y += 16) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
      }
      return;
    }

    // spectrogram is 128 rows (freq bins) × 256 cols (time frames)
    // Row 0 = highest freq bin from model; we flip Y so low freq is at bottom
    const imageData = ctx.createImageData(W, H);
    const pixels = imageData.data;

    for (let freq = 0; freq < H; freq++) {
      // Flip: row 0 of canvas = highest freq (row 0 of spec = high freq already)
      // We want low freq at bottom → canvas row H-1-freq maps to spec row freq
      const specRow = spectrogram[freq];
      const canvasRow = H - 1 - freq;

      for (let t = 0; t < W; t++) {
        const v = specRow ? Math.min(1, Math.max(0, specRow[t] ?? 0)) : 0;
        const [r, g, b] = intensityToRgb(v);
        const idx = (canvasRow * W + t) * 4;
        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
        pixels[idx + 3] = v < 0.02 ? Math.round(v * 50 / 0.02) : 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [spectrogram]);

  return (
    <div className="flex flex-col h-full w-full">
      <p className="text-[10px] font-semibold tracking-widest uppercase text-muted-foreground/50 text-center pt-2 pb-1">
        Spectrogram
      </p>
      <div className="flex-1 relative min-h-0">
        <canvas
          ref={canvasRef}
          width={256}
          height={128}
          className="w-full h-full"
          style={{ imageRendering: "pixelated" }}
        />
        {!spectrogram && (
          <p className="absolute inset-0 flex items-center justify-center text-[10px] text-muted-foreground/30 tracking-wide">
            pre-vocoder
          </p>
        )}
      </div>
    </div>
  );
}
