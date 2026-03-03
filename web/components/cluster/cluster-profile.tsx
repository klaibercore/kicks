"use client";

import type { ClusterProfile, ClusterData } from "@/types/cluster";
import { DESCRIPTOR_KEYS, DESCRIPTOR_LABELS } from "@/types/cluster";
import { CLUSTER_COLORS } from "@/lib/colors";
import { card } from "@/lib/styles";

export function ClusterProfileCard({
  clusterIdx,
  profile,
  globalStats,
  onPlay,
  isPlaying,
}: {
  clusterIdx: number;
  profile: ClusterProfile;
  globalStats?: ClusterData["descriptor_stats"];
  onPlay: () => void;
  isPlaying: boolean;
}) {
  const color = CLUSTER_COLORS[clusterIdx % CLUSTER_COLORS.length];

  return (
    <div className={`${card} p-3.5 space-y-2.5`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-full"
            style={{
              backgroundColor: color,
              boxShadow: `0 0 0 2px ${color}30`,
            }}
          />
          <span className="text-sm font-semibold text-foreground">
            Cluster {clusterIdx}
          </span>
          <span className="text-[11px] text-muted-foreground font-mono bg-muted px-1.5 py-0.5 rounded">
            {profile.count}
          </span>
        </div>
        <button
          onClick={onPlay}
          className={`w-7 h-7 rounded-full flex items-center justify-center transition-all ${
            isPlaying
              ? "bg-accent text-foreground scale-110"
              : "bg-muted text-muted-foreground hover:text-foreground hover:bg-accent"
          }`}
        >
          {isPlaying ? (
            <svg
              className="w-3 h-3"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <rect x="6" y="4" width="4" height="16" />
              <rect x="14" y="4" width="4" height="16" />
            </svg>
          ) : (
            <svg
              className="w-3 h-3 ml-0.5"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>
      </div>

      <div className="space-y-1.5">
        {DESCRIPTOR_KEYS.map((dk) => {
          const val = profile[dk];
          const maxVal = globalStats?.[dk]?.max ?? 1;
          const width = Math.min((val / (maxVal || 1)) * 100, 100);
          return (
            <div key={dk} className="flex items-center gap-2 text-[11px]">
              <span className="w-10 text-muted-foreground text-right font-medium shrink-0">
                {DESCRIPTOR_LABELS[dk]}
              </span>
              <div className="flex-1 h-2.5 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${width}%`,
                    backgroundColor: color,
                    opacity: 0.75,
                  }}
                />
              </div>
              <span className="w-8 text-muted-foreground font-mono text-right">
                {val.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
