"use client";

import { PauseIcon, PlayIcon } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { clusterAudioUrl } from "@/lib/analysis/load";
import { clusterColor } from "@/lib/analysis/palette";
import type { Report } from "@/lib/analysis/types";
import { cn } from "@/lib/utils";

interface ClusterCardsProps {
  report: Report;
  isolated: number | null;
  onIsolate: (cluster: number | null) => void;
}

/**
 * One card per GMM cluster: member count, mean descriptor profile, and the
 * averaged audio of its members — what the cluster *contains*, not what the
 * model thinks it contains. Clicking a card isolates it in the scatter.
 */
export function ClusterCards({ report, isolated, onIsolate }: ClusterCardsProps) {
  const { resolvedTheme } = useTheme();
  const dark = resolvedTheme === "dark";
  const [playing, setPlaying] = useState<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    return () => audioRef.current?.pause();
  }, []);

  const play = (k: number) => {
    if (playing === k) {
      audioRef.current?.pause();
      setPlaying(null);
      return;
    }
    audioRef.current?.pause();
    const audio = new Audio(clusterAudioUrl(report.instrument, k));
    audio.onended = () => setPlaying(null);
    audioRef.current = audio;
    void audio.play();
    setPlaying(k);
  };

  const clusters = Object.keys(report.cluster_profiles)
    .map(Number)
    .sort((a, b) => report.cluster_profiles[b].count - report.cluster_profiles[a].count);
  const total = report.samples.length;

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {clusters.map((k) => {
        const profile = report.cluster_profiles[String(k)];
        const color = clusterColor(k, dark);
        const active = isolated === k;
        return (
          <div
            key={k}
            className={cn(
              "flex flex-col gap-3 rounded-lg border border-border p-3 transition-colors",
              active && "bg-muted/60",
              isolated !== null && !active && "opacity-60",
            )}
          >
            <div className="flex items-center justify-between">
              <button
                type="button"
                className="flex items-center gap-2 text-left text-sm font-medium"
                onClick={() => onIsolate(active ? null : k)}
                aria-pressed={active}
              >
                <span className="inline-block size-3 rounded-full ring-2 ring-background" style={{ background: color }} />
                Cluster {k}
              </button>
              <span className="tabular font-mono text-xs text-muted-foreground">
                {profile.count} · {((profile.count / total) * 100).toFixed(0)}%
              </span>
            </div>
            <dl className="flex flex-col gap-1">
              {report.descriptor_keys.map((key, i) => {
                const v = profile[key] ?? 0;
                const stat = report.descriptor_stats[key];
                const z = stat.std > 0 ? (v - stat.mean) / stat.std : 0;
                return (
                  <div key={key} className="grid grid-cols-[3.5rem_1fr_2.5rem] items-center gap-2 text-xs">
                    <dt className="text-muted-foreground">{report.descriptor_labels[i]}</dt>
                    <dd className="relative h-1.5 rounded-full bg-muted">
                      <span
                        className="absolute inset-y-0 left-0 rounded-full"
                        style={{ width: `${Math.max(2, v * 100)}%`, background: color }}
                      />
                    </dd>
                    <dd className={cn("tabular text-right font-mono", Math.abs(z) > 1 ? "text-foreground" : "text-muted-foreground")}>
                      {v.toFixed(2)}
                    </dd>
                  </div>
                );
              })}
            </dl>
            <Button size="sm" variant="outline" onClick={() => play(k)} className="self-start">
              {playing === k ? <PauseIcon data-icon="inline-start" /> : <PlayIcon data-icon="inline-start" />}
              {playing === k ? "Stop" : "Play average"}
            </Button>
          </div>
        );
      })}
    </div>
  );
}
