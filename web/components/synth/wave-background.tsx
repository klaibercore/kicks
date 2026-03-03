"use client";

import { useEffect, useRef } from "react";

export function WaveBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    let w = 0,
      h = 0;

    function resize() {
      w = canvas!.width = window.innerWidth;
      h = canvas!.height = window.innerHeight;
    }
    resize();
    window.addEventListener("resize", resize);

    function draw(t: number) {
      ctx.clearRect(0, 0, w, h);

      const waves = [
        { color: "rgba(167,139,250,0.07)", speed: 0.0003, amp: 80, freq: 0.003, yOff: 0.35 },
        { color: "rgba(244,114,182,0.05)", speed: 0.0005, amp: 60, freq: 0.004, yOff: 0.45 },
        { color: "rgba(52,211,153,0.04)", speed: 0.0004, amp: 50, freq: 0.005, yOff: 0.55 },
        { color: "rgba(251,191,36,0.03)", speed: 0.0006, amp: 40, freq: 0.006, yOff: 0.65 },
      ];

      for (const wave of waves) {
        ctx.beginPath();
        ctx.moveTo(0, h);
        for (let x = 0; x <= w; x += 3) {
          const y =
            h * wave.yOff +
            Math.sin(x * wave.freq + t * wave.speed) * wave.amp +
            Math.sin(x * wave.freq * 0.5 + t * wave.speed * 1.3) *
              wave.amp *
              0.5;
          ctx.lineTo(x, y);
        }
        ctx.lineTo(w, h);
        ctx.closePath();
        ctx.fillStyle = wave.color;
        ctx.fill();
      }

      const grd = ctx.createRadialGradient(
        w / 2, h * 0.38, 0,
        w / 2, h * 0.38, w * 0.5,
      );
      grd.addColorStop(0, "rgba(167,139,250,0.06)");
      grd.addColorStop(0.5, "rgba(244,114,182,0.02)");
      grd.addColorStop(1, "transparent");
      ctx.fillStyle = grd;
      ctx.fillRect(0, 0, w, h);

      animRef.current = requestAnimationFrame(draw);
    }

    animRef.current = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 0 }}
    />
  );
}
