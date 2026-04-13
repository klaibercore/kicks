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

// Mel-frequency axis helpers (128 bands, 0–22050 Hz)
const MEL_MAX = 2595 * Math.log10(1 + 22050 / 700); // ≈ 3923
function hzToFrac(hz: number): number {
  return (2595 * Math.log10(1 + hz / 700)) / MEL_MAX;
}

const DURATION_S = (256 * 256) / 44100; // ≈ 1.486 s

const FREQ_TICKS = [
  { hz: 100, label: "100" },
  { hz: 1000, label: "1k" },
  { hz: 5000, label: "5k" },
  { hz: 10000, label: "10k" },
  { hz: 20000, label: "20k" },
];

const TIME_TICKS = [0, 0.5, 1.0];

// CSS margins (multiplied by DPR at draw time)
const MARGIN_LEFT = 30;
const MARGIN_BOTTOM = 16;

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
    const dpr = window.devicePixelRatio || 1;
    const spec = spectrogramRef.current;

    const ML = Math.round(MARGIN_LEFT * dpr);
    const MB = Math.round(MARGIN_BOTTOM * dpr);
    const plotW = W - ML;
    const plotH = H - MB;

    ctx.clearRect(0, 0, W, H);

    // ── Spectrogram pixels ────────────────────────────
    if (spec) {
      const specH = spec.length;
      const specW = spec[0]?.length ?? 0;
      const imgW = Math.max(1, plotW);
      const imgH = Math.max(1, plotH);
      const imageData = ctx.createImageData(imgW, imgH);
      const pixels = imageData.data;

      for (let cy = 0; cy < imgH; cy++) {
        const freqF = (cy / imgH) * (specH - 1);
        const freq0 = Math.floor(freqF);
        const freq1 = Math.min(freq0 + 1, specH - 1);
        const freqT = freqF - freq0;
        const canvasRow = imgH - 1 - cy;

        for (let cx = 0; cx < imgW; cx++) {
          const timeF = (cx / imgW) * (specW - 1);
          const time0 = Math.floor(timeF);
          const time1 = Math.min(time0 + 1, specW - 1);
          const timeT = timeF - time0;

          const v00 = spec[freq0]?.[time0] ?? 0;
          const v01 = spec[freq0]?.[time1] ?? 0;
          const v10 = spec[freq1]?.[time0] ?? 0;
          const v11 = spec[freq1]?.[time1] ?? 0;
          const v0 = v00 + (v01 - v00) * timeT;
          const v1 = v10 + (v11 - v10) * timeT;
          let v = Math.min(1, Math.max(0, v0 + (v1 - v0) * freqT));
          v = Math.pow(v, 0.7);

          const band = Math.min(BANDS - 1, Math.floor(v * BANDS));
          const [r, g, b] = PALETTE[band];
          const idx = (canvasRow * imgW + cx) * 4;
          pixels[idx] = r;
          pixels[idx + 1] = g;
          pixels[idx + 2] = b;
          pixels[idx + 3] = 255;
        }
      }

      ctx.putImageData(imageData, ML, 0);
    }

    // ── Grid lines & labels ───────────────────────────
    const fontSize = Math.round(8 * dpr);
    ctx.font = `${fontSize}px ui-monospace, "Geist Mono", monospace`;

    // Frequency ticks (Y-axis — mel-scaled)
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tick of FREQ_TICKS) {
      const frac = hzToFrac(tick.hz);
      const y = plotH * (1 - frac);
      if (y < 2 || y > plotH - 2) continue;

      ctx.strokeStyle = "rgba(255,255,255,0.07)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(ML, Math.round(y) + 0.5);
      ctx.lineTo(W, Math.round(y) + 0.5);
      ctx.stroke();

      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.fillText(tick.label, ML - 3 * dpr, y);
    }

    // Time ticks (X-axis)
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const t of TIME_TICKS) {
      const x = ML + (t / DURATION_S) * plotW;

      ctx.strokeStyle = "rgba(255,255,255,0.07)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, 0);
      ctx.lineTo(Math.round(x) + 0.5, plotH);
      ctx.stroke();

      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.fillText(`${t}s`, x, plotH + 2 * dpr);
    }

    // Plot border
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    ctx.strokeRect(ML + 0.5, 0.5, plotW - 1, plotH - 1);
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
