"use client";

import { ArrowUpRight, Pause, Play } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useRef, useState } from "react";
import { clusterAudioUrl } from "@/lib/analysis/load";
import { clusterColor } from "@/lib/analysis/palette";
import {
  clusterName,
  clusterSummary,
  rankedClusters,
} from "@/lib/analysis/stats";
import type { Report } from "@/lib/analysis/types";
import { cn } from "@/lib/utils";

export function useClusterAudio(instrument: string) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(
    () => () => {
      audioRef.current?.pause();
      audioRef.current = null;
    },
    [],
  );
  function play(cluster: number) {
    audioRef.current?.pause();
    if (playing === cluster) {
      audioRef.current = null;
      setPlaying(null);
      return;
    }
    const audio = new Audio(clusterAudioUrl(instrument, cluster));
    audioRef.current = audio;
    setPlaying(cluster);
    setError(null);
    audio.onended = () => {
      if (audioRef.current === audio) setPlaying(null);
    };
    void audio.play().catch(() => {
      if (audioRef.current === audio) {
        setPlaying(null);
        setError(
          `The average for C${String(cluster + 1).padStart(2, "0")} is unavailable. Please try again.`,
        );
      }
    });
  }
  return { playing, error, play };
}

export function AudioButton({
  cluster,
  playing,
  onPlay,
}: {
  cluster: number;
  playing: number | null;
  onPlay: (k: number) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onPlay(cluster)}
      aria-label={`${playing === cluster ? "Stop" : "Play"} cluster ${cluster + 1} average`}
      className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-2 text-xs font-medium transition-colors hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
    >
      {playing === cluster ? (
        <Pause className="size-3" />
      ) : (
        <Play className="size-3" />
      )}
      {playing === cluster ? "Stop average" : "Listen to average"}
    </button>
  );
}

export function ClusterProfile({
  report,
  cluster,
  color,
}: {
  report: Report;
  cluster: number;
  color: string;
}) {
  const profile = report.cluster_profiles[String(cluster)];
  return (
    <div>
      <dl className="space-y-2.5">
        {report.descriptor_keys.map((key, i) => {
          const stat = report.descriptor_stats[key];
          const z = stat.std > 0 ? (profile[key] - stat.mean) / stat.std : 0;
          const width = Math.min(50, (Math.abs(z) / 3) * 50);
          return (
            <div
              key={key}
              className="grid grid-cols-[3.6rem_1fr_3rem] items-center gap-3 text-[11px]"
            >
              <dt className="text-muted-foreground">
                {report.descriptor_labels[i]}
              </dt>
              <dd
                className="relative h-3 rounded-sm bg-muted/70"
                title={`${profile[key].toFixed(2)}; ${z.toFixed(2)} standard deviations from the corpus mean`}
              >
                <span className="absolute inset-y-[-2px] left-1/2 w-px bg-foreground/25" />
                <span
                  className="absolute inset-y-0.5 rounded-sm"
                  style={{
                    left: `${z >= 0 ? 50 : 50 - width}%`,
                    width: `${width}%`,
                    background: color,
                  }}
                />
              </dd>
              <dd className="text-right font-mono tabular-nums">
                {z > 0 ? "+" : ""}
                {z.toFixed(1)}σ
              </dd>
            </div>
          );
        })}
      </dl>
      <div className="ml-[4.35rem] mr-15 mt-2 flex justify-between font-mono text-[9px] text-muted-foreground">
        <span>−3σ</span>
        <span>mean</span>
        <span>+3σ</span>
      </div>
    </div>
  );
}

export function ClusterCards({
  report,
  isolated,
  onIsolate,
  playing,
  onPlay,
  limit,
}: {
  report: Report;
  isolated: number | null;
  onIsolate: (k: number | null) => void;
  playing: number | null;
  onPlay: (k: number) => void;
  limit?: number;
}) {
  const { resolvedTheme } = useTheme();
  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      {rankedClusters(report).slice(0, limit).map((k) => {
        const profile = report.cluster_profiles[String(k)];
        const color = clusterColor(k, resolvedTheme === "dark");
        const active = isolated === k;
        const share = profile.count / report.samples.length;
        return (
          <article
            key={k}
            className={cn(
              "flex flex-col rounded-xl border bg-card p-5 transition-colors",
              active
                ? "border-foreground/50 ring-1 ring-foreground/10"
                : "border-border",
            )}
          >
            <div className="mb-4 flex items-center justify-between">
              <span className="flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
                <span
                  className="size-2 rounded-full"
                  style={{ background: color }}
                />
                C{String(k + 1).padStart(2, "0")}
              </span>
              <span className="font-mono text-[11px] text-muted-foreground">
                {profile.count.toLocaleString()} / {(share * 100).toFixed(1)}%
              </span>
            </div>
            <button
              type="button"
              aria-pressed={active}
              onClick={() => onIsolate(active ? null : k)}
              className="group mb-1 flex min-h-12 items-start justify-between gap-2 text-left text-base font-medium leading-snug tracking-tight hover:underline focus-visible:outline-2 focus-visible:outline-ring"
            >
              {clusterName(report, k)}
              <ArrowUpRight className="mt-0.5 size-4 shrink-0 text-muted-foreground transition-transform group-hover:-translate-y-0.5" />
            </button>
            <p className="mb-5 min-h-12 text-[11px] leading-relaxed text-muted-foreground">
              {clusterSummary(report, k)}
            </p>
            <ClusterProfile report={report} cluster={k} color={color} />
            <div className="mt-5 flex flex-wrap items-center justify-between gap-2 border-t border-border pt-4">
              <AudioButton cluster={k} playing={playing} onPlay={onPlay} />
              {active && (
                <span className="text-[10px] font-medium">
                  Selected in explorer
                </span>
              )}
            </div>
          </article>
        );
      })}
    </div>
  );
}
