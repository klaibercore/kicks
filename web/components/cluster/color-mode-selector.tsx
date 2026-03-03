"use client";

import type { ColorMode, DescriptorKey } from "@/types/cluster";
import { DESCRIPTOR_KEYS, DESCRIPTOR_LABELS } from "@/types/cluster";
import { pillActive, pillInactive } from "@/lib/styles";

export function ColorModeSelector({
  colorMode,
  setColorMode,
}: {
  colorMode: ColorMode;
  setColorMode: (m: ColorMode) => void;
}) {
  return (
    <div className="flex gap-0.5 sm:gap-1 bg-card/50 rounded-lg sm:rounded-xl p-0.5 sm:p-1 border border-border">
      {(["cluster", ...DESCRIPTOR_KEYS] as ColorMode[]).map((mode) => (
        <button
          key={mode}
          onClick={() => setColorMode(mode)}
          className={colorMode === mode ? pillActive : pillInactive}
        >
          {mode === "cluster"
            ? "Clust"
            : DESCRIPTOR_LABELS[mode as DescriptorKey]}
        </button>
      ))}
    </div>
  );
}
