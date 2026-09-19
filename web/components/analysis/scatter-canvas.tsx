"use client";

import { Minus, Plus, RotateCcw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { clusterColor } from "@/lib/analysis/palette";
import { clusterName } from "@/lib/analysis/stats";
import type { Report, SampleRow } from "@/lib/analysis/types";

export type ProjectionSpace = "descriptors" | "latent";
export type AxisPair = "12" | "13" | "23";
type Point = { sample: SampleRow; x: number; y: number; z: number };
type ScreenPoint = Point & {
  sx: number;
  sy: number;
  depth: number;
  size: number;
};

interface ScatterProps {
  report: Report;
  space: ProjectionSpace;
  mode: "3d" | "2d";
  pair: AxisPair;
  isolated: number | null;
  samples: SampleRow[];
  selected: SampleRow | null;
  onPick: (sample: SampleRow) => void;
}

const INITIAL_CAMERA = { yaw: -0.55, pitch: -0.4, zoom: 1 };
const AXIS_COLORS = ["#77a8ed", "#74d7b2", "#e7b77b"];

/** True 3D perspective, depth-sorted on canvas; no animation or WebGL dependency. */
export function ScatterCanvas({
  report,
  space,
  mode,
  pair,
  isolated,
  samples,
  selected,
  onPick,
}: ScatterProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const screenRef = useRef<ScreenPoint[]>([]);
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const [camera, setCamera] = useState(INITIAL_CAMERA);
  const [hover, setHover] = useState<ScreenPoint | null>(null);
  const geometry = useMemo(() => {
    const keys =
      space === "latent"
        ? (["latent1", "latent2", "latent3"] as const)
        : (["pc1", "pc2", "pc3"] as const);
    const axes =
      mode === "3d" ? [0, 1, 2] : [Number(pair[0]) - 1, Number(pair[1]) - 1];
    const mins = keys.map((key) =>
      report.samples.reduce((min, s) => Math.min(min, s[key] ?? 0), Infinity),
    );
    const maxs = keys.map((key) =>
      report.samples.reduce((max, s) => Math.max(max, s[key] ?? 0), -Infinity),
    );
    // A common scale preserves relative distances across all axes.
    const scale =
      Math.max(1e-8, ...axes.map((axis) => maxs[axis] - mins[axis])) / 2;
    const coordinate = (s: SampleRow, axis: number) =>
      ((s[keys[axis]] ?? 0) - (mins[axis] + maxs[axis]) / 2) / scale;
    return {
      points: samples.map((sample) => ({
        sample,
        x: coordinate(sample, axes[0]),
        y: coordinate(sample, axes[1]),
        z: mode === "3d" ? coordinate(sample, 2) : 0,
      })),
      centers: axes.map((axis) => (mins[axis] + maxs[axis]) / 2),
      scale,
    };
  }, [report, samples, space, mode, pair]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let frame = 0;
    const draw = () => {
      const { width: w, height: h } = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      const ctx = canvas.getContext("2d");
      if (!ctx || !w || !h) return;
      const { points } = geometry;
      ctx.scale(dpr, dpr);
      const scale =
        Math.min(w * 0.3, h * (mode === "3d" ? 0.27 : 0.34)) * camera.zoom;
      const project = (x: number, y: number, z: number) => {
        const rx =
          mode === "3d"
            ? x * Math.cos(camera.yaw) + z * Math.sin(camera.yaw)
            : x;
        const rz =
          mode === "3d"
            ? -x * Math.sin(camera.yaw) + z * Math.cos(camera.yaw)
            : 0;
        const ry =
          mode === "3d"
            ? y * Math.cos(camera.pitch) - rz * Math.sin(camera.pitch)
            : y;
        const depth =
          mode === "3d"
            ? y * Math.sin(camera.pitch) + rz * Math.cos(camera.pitch)
            : 0;
        const perspective = 4.5 / (4.5 + depth);
        return {
          sx: w / 2 + rx * scale * perspective,
          sy: h / 2 - ry * scale * perspective,
          depth,
          size: perspective,
        };
      };
      const line = (a: number[], b: number[], color: string, width = 1) => {
        const p = project(a[0], a[1], a[2]);
        const q = project(b[0], b[1], b[2]);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(p.sx, p.sy);
        ctx.lineTo(q.sx, q.sy);
        ctx.stroke();
      };
      for (let i = -1; i <= 1.01; i += 0.25) {
        if (mode === "3d") {
          line([i, -1, -1], [i, -1, 1], "#213045");
          line([-1, -1, i], [1, -1, i], "#213045");
        } else {
          line([i, -1, 0], [i, 1, 0], "#1c2a3c");
          line([-1, i, 0], [1, i, 0], "#1c2a3c");
        }
      }
      const origin = mode === "3d" ? [-1, -1, -1] : [-1, -1, 0];
      const ends =
        mode === "3d"
          ? [
              [1.14, -1, -1],
              [-1, 1.14, -1],
              [-1, -1, 1.14],
            ]
          : [
              [1.14, -1, 0],
              [-1, 1.14, 0],
            ];
      const axisNumbers =
        mode === "3d" ? [1, 2, 3] : [Number(pair[0]), Number(pair[1])];
      ends.forEach((end, i) => {
        line(origin, end, AXIS_COLORS[axisNumbers[i] - 1] + "90", 1.2);
        const p = project(...(end as [number, number, number]));
        ctx.fillStyle = AXIS_COLORS[axisNumbers[i] - 1];
        ctx.font = "11px ui-monospace, monospace";
        ctx.fillText(
          `${space === "latent" ? "L" : "PC"}${axisNumbers[i]}`,
          p.sx + 7,
          p.sy - 7,
        );
        for (const value of [-0.5, 0, 0.5]) {
          const tick = [...origin];
          tick[i] = value;
          const position = project(...(tick as [number, number, number]));
          ctx.fillStyle = "#7c8fa8";
          ctx.font = "9px ui-monospace, monospace";
          ctx.fillText(
            (geometry.centers[i] + value * geometry.scale).toFixed(1),
            position.sx + (i === 1 ? -24 : 3),
            position.sy + (i === 1 ? 3 : 13),
          );
        }
      });
      const projected = points
        .map((p) => ({ ...p, ...project(p.x, p.y, p.z) }))
        .sort((a, b) => b.depth - a.depth);
      screenRef.current = projected;
      const radius = points.length > 6000 ? 1.9 : 2.5;
      for (const dim of [true, false]) {
        for (const p of projected) {
          const visible = isolated === null || isolated === p.sample.cluster;
          if (visible === dim) continue;
          ctx.globalAlpha = dim ? 0.055 : Math.min(0.95, 0.65 + 0.12 * p.size);
          ctx.fillStyle = clusterColor(p.sample.cluster, true);
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, Math.max(1, radius * p.size), 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;
      for (const sample of [hover?.sample, selected]) {
        const p =
          sample &&
          projected.find(
            (point) => point.sample.sample_idx === sample.sample_idx,
          );
        if (!p || (isolated !== null && p.sample.cluster !== isolated))
          continue;
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, 7, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (isolated !== null) {
        const members = projected.filter((p) => p.sample.cluster === isolated);
        if (members.length) {
          const x = members.reduce((sum, p) => sum + p.sx, 0) / members.length;
          const y = members.reduce((sum, p) => sum + p.sy, 0) / members.length;
          ctx.font = "600 11px ui-monospace, monospace";
          ctx.fillStyle = "#fff";
          ctx.fillText(
            `C${String(isolated + 1).padStart(2, "0")} / ${members.length.toLocaleString()} samples`,
            Math.min(w - 180, Math.max(12, x + 14)),
            Math.max(24, y - 18),
          );
        }
      }
    };
    const schedule = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(draw);
    };
    schedule();
    const observer = new ResizeObserver(schedule);
    observer.observe(canvas);
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
    };
  }, [geometry, camera, mode, pair, space, isolated, hover, selected]);

  function nearest(clientX: number, clientY: number) {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return null;
    let nearest: ScreenPoint | null = null;
    let distance = 100;
    for (const p of screenRef.current) {
      if (isolated !== null && p.sample.cluster !== isolated) continue;
      const d =
        (p.sx - clientX + rect.left) ** 2 + (p.sy - clientY + rect.top) ** 2;
      if (d < distance) {
        distance = d;
        nearest = p;
      }
    }
    return nearest;
  }
  const adjustZoom = (amount: number) =>
    setCamera((c) => ({
      ...c,
      zoom: Math.max(0.6, Math.min(2.5, c.zoom + amount)),
    }));

  return (
    <div className="relative overflow-hidden bg-[#0b1220] text-slate-200">
      <div className="pointer-events-none absolute left-5 top-5 z-10">
        <p className="font-mono text-[10px] tracking-[.2em] text-slate-400">
          {space === "latent" ? "LEARNED STRUCTURE" : "PERCEPTUAL CHARACTER"} /{" "}
          {mode.toUpperCase()}
        </p>
        <p className="mt-1.5 text-xs text-slate-400">
          {samples.length.toLocaleString()} samples · one point, one sound
        </p>
      </div>
      <canvas
        ref={canvasRef}
        tabIndex={0}
        role="img"
        aria-label={`${mode === "3d" ? "Interactive 3D" : "2D"} ${space} scatter plot. ${mode === "3d" ? "Drag or use arrow keys to rotate. " : ""}Use plus and minus to zoom. Select samples in the sample table for a keyboard alternative.`}
        className={`h-[390px] w-full outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-400 sm:h-[520px] ${mode === "3d" ? "touch-none cursor-grab active:cursor-grabbing" : "cursor-crosshair"}`}
        onPointerDown={(e) => {
          if (mode === "3d") e.currentTarget.setPointerCapture(e.pointerId);
          drag.current = { x: e.clientX, y: e.clientY, moved: false };
        }}
        onPointerMove={(e) => {
          if (drag.current && mode === "3d") {
            const dx = e.clientX - drag.current.x;
            const dy = e.clientY - drag.current.y;
            if (Math.abs(dx) + Math.abs(dy) > 2 || drag.current.moved) {
              drag.current = { x: e.clientX, y: e.clientY, moved: true };
              setHover(null);
              setCamera((c) => ({
                ...c,
                yaw: c.yaw + dx * 0.007,
                pitch: Math.max(-1.25, Math.min(1.25, c.pitch + dy * 0.007)),
              }));
            }
          } else {
            const next = nearest(e.clientX, e.clientY);
            if (next?.sample.sample_idx !== hover?.sample.sample_idx)
              setHover(next);
          }
        }}
        onPointerUp={(e) => {
          if (!drag.current?.moved) {
            const p = nearest(e.clientX, e.clientY);
            if (p) onPick(p.sample);
          }
          drag.current = null;
          if (e.currentTarget.hasPointerCapture(e.pointerId))
            e.currentTarget.releasePointerCapture(e.pointerId);
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
        onPointerLeave={() => setHover(null)}
        onKeyDown={(e) => {
          if (
            ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(
              e.key,
            ) &&
            mode === "3d"
          ) {
            e.preventDefault();
            setCamera((c) => ({
              ...c,
              yaw:
                c.yaw +
                (e.key === "ArrowLeft"
                  ? -0.12
                  : e.key === "ArrowRight"
                    ? 0.12
                    : 0),
              pitch: Math.max(
                -1.25,
                Math.min(
                  1.25,
                  c.pitch +
                    (e.key === "ArrowUp"
                      ? -0.12
                      : e.key === "ArrowDown"
                        ? 0.12
                        : 0),
                ),
              ),
            }));
          }
          if (["+", "=", "-"].includes(e.key)) {
            e.preventDefault();
            adjustZoom(e.key === "-" ? -0.15 : 0.15);
          }
          if (e.key === "Home") {
            e.preventDefault();
            setCamera(INITIAL_CAMERA);
          }
        }}
      />
      {!samples.some(
        (sample) => isolated === null || sample.cluster === isolated,
      ) && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-10 text-center text-sm text-slate-300">
          No samples match this selection.
          <br />
          Choose another family or clear the uncertainty filter.
        </div>
      )}
      {hover && (
        <div className="pointer-events-none absolute bottom-14 left-4 right-4 rounded-lg border border-white/15 bg-slate-900/95 px-3 py-2 text-xs shadow-xl sm:left-auto sm:max-w-72">
          <p className="font-medium">
            C{String(hover.sample.cluster + 1).padStart(2, "0")} ·{" "}
            {clusterName(report, hover.sample.cluster)}
          </p>
          <p className="mt-1 text-slate-400">
            Sample #{hover.sample.sample_idx} ·{" "}
            {(hover.sample.confidence * 100).toFixed(1)}% membership · click to
            inspect
          </p>
        </div>
      )}
      <div className="absolute bottom-4 left-5 right-4 flex items-center justify-between gap-3">
        <span className="text-[11px] text-slate-400">
          {mode === "3d"
            ? "Drag to orbit · click to inspect"
            : "Click a point to inspect"}
        </span>
        <div className="flex gap-1 rounded-lg border border-white/10 bg-slate-900/90 p-1">
          {[
            { label: "Zoom out", icon: Minus, action: () => adjustZoom(-0.2) },
            { label: "Zoom in", icon: Plus, action: () => adjustZoom(0.2) },
            {
              label: "Reset camera",
              icon: RotateCcw,
              action: () => setCamera(INITIAL_CAMERA),
            },
          ].map(({ label, icon: Icon, action }) => (
            <button
              key={label}
              type="button"
              title={label}
              aria-label={label}
              onClick={action}
              className="rounded p-2 text-slate-300 hover:bg-white/10 focus-visible:outline-2 focus-visible:outline-blue-400"
            >
              <Icon className="size-3.5" />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
