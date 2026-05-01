"use client";

import { useRef } from "react";
import type { Sample, PCName } from "@/types/cluster";
import { DESCRIPTOR_KEYS, DESCRIPTOR_LABELS } from "@/types/cluster";
import { CLUSTER_COLORS, PC_COLORS } from "@/lib/colors";

let _gradientIdCounter = 0;

export function SampleInspector({
  sample,
  waveform,
  isPlaying,
  onPlayPause,
  onClose,
  pcNames,
}: {
  sample: Sample;
  waveform: number[] | null;
  isPlaying: boolean;
  onPlayPause: () => void;
  onClose: () => void;
  pcNames: PCName[];
}) {
  const gradientIdRef = useRef(`inspectorWaveGradient_${++_gradientIdCounter}`);
  const gradientId = gradientIdRef.current;

  return (
    <div className="rounded-2xl border border-border bg-gradient-to-b from-card/80 to-card/40 backdrop-blur-sm p-3 sm:p-5 space-y-3 sm:space-y-4 shadow-xl shadow-black/10 dark:shadow-black/20">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
          <button
            onClick={onPlayPause}
            className="w-9 sm:w-10 h-9 sm:h-10 rounded-full bg-muted border border-border flex items-center justify-center hover:bg-accent transition-all active:scale-95 shrink-0"
          >
            {isPlaying ? (
              <svg
                className="w-3.5 h-3.5 text-foreground/80"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <rect x="6" y="4" width="4" height="16" />
                <rect x="14" y="4" width="4" height="16" />
              </svg>
            ) : (
              <svg
                className="w-3.5 h-3.5 text-foreground/80 ml-0.5"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>
          <div className="min-w-0">
            <div className="text-sm text-foreground font-semibold truncate max-[360px]:max-w-[180px] sm:max-w-[300px]">
              {sample.filename}
            </div>
            <div className="flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-[11px] text-muted-foreground mt-0.5">
              <div
                className="w-2 h-2 rounded-full shrink-0"
                style={{
                  backgroundColor:
                    CLUSTER_COLORS[sample.cluster % CLUSTER_COLORS.length],
                }}
              />
              <span>Cl {sample.cluster}</span>
              <span className="text-muted-foreground/60 hidden sm:inline">
                |
              </span>
              <span className="font-mono truncate">
                {pcNames[0]?.name ?? "PC1"}:{sample.pc1.toFixed(1)}{" "}
                {pcNames[1]?.name ?? "PC2"}:{sample.pc2.toFixed(1)}
              </span>
            </div>
          </div>
        </div>
        <button
          onClick={onClose}
          className="w-8 h-8 rounded-full bg-muted flex items-center justify-center hover:bg-accent transition-colors text-muted-foreground hover:text-foreground shrink-0"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            strokeWidth={2}
          >
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        </button>
      </div>

      {waveform && (
        <div className="h-16 rounded-xl bg-muted border border-border overflow-hidden">
          <svg
            viewBox="0 0 200 64"
            className="w-full h-full"
            preserveAspectRatio="none"
          >
            <defs>
              <linearGradient
                id={gradientId}
                x1="0%"
                y1="0%"
                x2="100%"
                y2="0%"
              >
                <stop offset="0%" stopColor="#a78bfa" />
                <stop offset="50%" stopColor="#f472b6" />
                <stop offset="100%" stopColor="#34d399" />
              </linearGradient>
            </defs>
            {waveform.map((v, i) => (
              <line
                key={i}
                x1={i}
                y1={32 - v * 26}
                x2={i}
                y2={32 + v * 26}
                stroke={`url(#${gradientId})`}
                strokeWidth="1.2"
                opacity="0.8"
              />
            ))}
          </svg>
        </div>
      )}

      <div className="flex gap-1.5 sm:gap-2 overflow-x-auto -mx-1 px-1">
        {DESCRIPTOR_KEYS.map((dk) => {
          const val = sample.descriptors[dk];
          return (
            <div
              key={dk}
              className="flex-1 min-w-[55px] sm:min-w-0 bg-muted/50 rounded-lg sm:rounded-xl p-2 text-center"
            >
              <div className="text-[9px] sm:text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-1">
                {DESCRIPTOR_LABELS[dk]}
              </div>
              <div className="text-xs sm:text-sm font-mono font-bold text-foreground">
                {val.toFixed(2)}
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${Math.min(val * 100, 100)}%`,
                    background: `linear-gradient(to right, ${PC_COLORS[0]}, ${PC_COLORS[1]})`,
                    opacity: 0.7,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
