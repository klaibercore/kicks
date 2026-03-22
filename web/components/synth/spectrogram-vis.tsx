"use client";

import { useCallback, useEffect, useRef } from "react";

const BANDS = 10;

// Discrete color palette matching the site's violet → pink → emerald theme.
// Each band is a flat color for cel-shaded / cartoon look.
const PALETTE: [number, number, number][] = [
  [  6,   6,  18],  // 0 — near-black, blends with card bg
  [ 22,  12,  55],  // 1 — deep indigo
  [ 55,  25, 110],  // 2 — dark violet
  [107,  60, 180],  // 3 — violet (slider accent range)
  [155,  80, 200],  // 4 — bright violet
  [200,  70, 160],  // 5 — magenta/pink
  [230, 100, 130],  // 6 — warm pink (generate button range)
  [240, 160,  90],  // 7 — amber
  [ 80, 220, 160],  // 8 — emerald (download button range)
  [245, 250, 235],  // 9 — near-white
];

export function SpectrogramVis({
  spectrogram,
}: {
  spectrogram: number[][] | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const spectrogramRef = useRef(spectrogram);
  spectrogramRef.current = spectrogram;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const W = canvas.width;
    const H = canvas.height;
    const spec = spectrogramRef.current;

    if (!spec) {
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = "rgba(167,139,250,0.06)";
      ctx.lineWidth = 1;
      const lines = 12;
      for (let i = 1; i < lines; i++) {
        const y = (i / lines) * H;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
      }
      return;
    }

    const specH = spec.length;
    const specW = spec[0]?.length ?? 0;
    const imageData = ctx.createImageData(W, H);
    const pixels = imageData.data;

    for (let cy = 0; cy < H; cy++) {
      const freqF = (cy / H) * (specH - 1);
      const freq0 = Math.floor(freqF);
      const freq1 = Math.min(freq0 + 1, specH - 1);
      const freqT = freqF - freq0;
      const canvasRow = H - 1 - cy;

      for (let cx = 0; cx < W; cx++) {
        const timeF = (cx / W) * (specW - 1);
        const time0 = Math.floor(timeF);
        const time1 = Math.min(time0 + 1, specW - 1);
        const timeT = timeF - time0;

        // Bilinear interpolation
        const v00 = spec[freq0]?.[time0] ?? 0;
        const v01 = spec[freq0]?.[time1] ?? 0;
        const v10 = spec[freq1]?.[time0] ?? 0;
        const v11 = spec[freq1]?.[time1] ?? 0;
        const v0 = v00 + (v01 - v00) * timeT;
        const v1 = v10 + (v11 - v10) * timeT;
        let v = Math.min(1, Math.max(0, v0 + (v1 - v0) * freqT));

        // Gamma curve to boost contrast before quantizing
        v = Math.pow(v, 0.7);

        // Posterize: snap to one of BANDS discrete levels
        const band = Math.min(BANDS - 1, Math.floor(v * BANDS));
        const [r, g, b] = PALETTE[band];

        const idx = (canvasRow * W + cx) * 4;
        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
        pixels[idx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
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
  }, [spectrogram, draw]);

  return (
    <div className="flex flex-col h-full w-full">
      <p className="text-[10px] font-semibold tracking-widest uppercase text-muted-foreground/50 text-center pt-2 pb-1">
        Spectrogram
      </p>
      <div ref={containerRef} className="flex-1 relative min-h-0">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
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
