"use client";

import { useEffect, useRef } from "react";
import { useAudioContext } from "@/hooks/use-audio-context";

export function WaveformVis({
  audioRef,
}: {
  audioRef: React.RefObject<HTMLAudioElement | null>;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const animRef = useRef<number>(0);
  const getCtx = useAudioContext();

  useEffect(() => {
    const canvas = canvasRef.current;
    const audio = audioRef.current;
    if (!canvas || !audio) return;

    function initAudio() {
      if (sourceRef.current) return;
      const actx = getCtx();
      const analyser = actx.createAnalyser();
      analyser.fftSize = 256;
      const source = actx.createMediaElementSource(audio!);
      source.connect(analyser);
      analyser.connect(actx.destination);
      analyserRef.current = analyser;
      sourceRef.current = source;
    }

    function draw() {
      const ctx = canvas!.getContext("2d")!;
      const w = canvas!.width;
      const h = canvas!.height;
      ctx.clearRect(0, 0, w, h);

      if (analyserRef.current) {
        const bufLen = analyserRef.current.frequencyBinCount;
        const data = new Uint8Array(bufLen);
        analyserRef.current.getByteFrequencyData(data);

        const barW = w / bufLen;
        for (let i = 0; i < bufLen; i++) {
          const v = data[i] / 255;
          const barH = v * h * 0.9;

          const gradient = ctx.createLinearGradient(0, h, 0, h - barH);
          gradient.addColorStop(0, "rgba(167,139,250,0.8)");
          gradient.addColorStop(0.5, "rgba(244,114,182,0.6)");
          gradient.addColorStop(1, "rgba(52,211,153,0.4)");
          ctx.fillStyle = gradient;

          const x = i * barW;
          ctx.beginPath();
          ctx.roundRect(x + 0.5, h - barH, Math.max(barW - 1, 1), barH, 2);
          ctx.fill();
        }
      } else {
        const t = Date.now() * 0.002;
        ctx.beginPath();
        ctx.strokeStyle = "rgba(167,139,250,0.2)";
        ctx.lineWidth = 1.5;
        for (let x = 0; x < w; x++) {
          const y =
            h / 2 +
            Math.sin(x * 0.04 + t) * 8 +
            Math.sin(x * 0.02 + t * 0.7) * 5;
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      animRef.current = requestAnimationFrame(draw);
    }

    audio.addEventListener("play", initAudio);
    animRef.current = requestAnimationFrame(draw);

    return () => {
      audio.removeEventListener("play", initAudio);
      cancelAnimationFrame(animRef.current);
      if (sourceRef.current) {
        sourceRef.current.disconnect();
        sourceRef.current = null;
      }
      if (analyserRef.current) {
        analyserRef.current.disconnect();
        analyserRef.current = null;
      }
    };
  }, [audioRef, getCtx]);

  return (
    <canvas
      ref={canvasRef}
      width={480}
      height={80}
      className="w-full rounded-xl"
      style={{ height: 80 }}
    />
  );
}
