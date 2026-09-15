"use client";

import { useEffect, useRef } from "react";
import type { Spectrogram } from "@/lib/api/types";

/** The decoded log-mel spectrogram (128 bands x 256 frames), low bands at the bottom. */
export function SpectrogramView({ spec, className }: { spec: Spectrogram | null; className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (!spec) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const [bands, frames] = spec.shape;
    canvas.width = frames;
    canvas.height = bands;
    const image = ctx.createImageData(frames, bands);
    for (let b = 0; b < bands; b++) {
      const row = spec.data[b];
      const y = bands - 1 - b;
      for (let f = 0; f < frames; f++) {
        const v = Math.max(0, Math.min(1, row[f]));
        const [r, g, bl] = heat(v);
        const i = (y * frames + f) * 4;
        image.data[i] = r;
        image.data[i + 1] = g;
        image.data[i + 2] = bl;
        image.data[i + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [spec]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ imageRendering: "pixelated" }}
      role="img"
      aria-label="Decoded mel spectrogram"
    />
  );
}

/** A perceptually ordered ramp: near-black -> blue -> magenta -> amber -> white. */
function heat(v: number): [number, number, number] {
  const stops: [number, [number, number, number]][] = [
    [0, [12, 10, 24]],
    [0.25, [40, 50, 160]],
    [0.5, [180, 50, 150]],
    [0.75, [245, 160, 60]],
    [1, [255, 250, 235]],
  ];
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const [t0, c0] = stops[i - 1];
      const [t1, c1] = stops[i];
      const t = (v - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * t),
        Math.round(c0[1] + (c1[1] - c0[1]) * t),
        Math.round(c0[2] + (c1[2] - c0[2]) * t),
      ];
    }
  }
  return stops[stops.length - 1][1];
}
