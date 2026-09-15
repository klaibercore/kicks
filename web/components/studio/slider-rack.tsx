"use client";

import { Slider } from "@/components/ui/slider";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useStudio } from "@/hooks/use-studio";

/**
 * One slider per perceptual axis. The layout comes from the API — under the
 * PCA basis the axis names are discovered at fit time, so nothing here knows
 * what a "kick" is.
 */
export function SliderRack() {
  const { config, sound, setSlider, commit } = useStudio();
  if (!config || !sound) return null;

  return (
    <div className="flex flex-col gap-5">
      {config.sliders.map((def) => {
        const value = sound.sliders[def.key] ?? def.default;
        return (
          <div key={def.key} className="group">
            <div className="mb-2 flex items-baseline justify-between">
              <Tooltip>
                <TooltipTrigger
                  render={<span className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground" />}
                >
                  {def.name}
                </TooltipTrigger>
                <TooltipContent side="right">{def.description ?? def.name}</TooltipContent>
              </Tooltip>
              <button
                type="button"
                className="tabular font-mono text-xs text-muted-foreground hover:text-foreground"
                onClick={() => {
                  setSlider(def.key, def.default);
                  commit();
                }}
                title="Reset to centre"
              >
                {Math.round(value * 100)}
              </button>
            </div>
            <Slider
              aria-label={def.name}
              min={def.min}
              max={def.max}
              step={def.step}
              value={value}
              onValueChange={(v) => setSlider(def.key, Array.isArray(v) ? v[0] : v)}
              onValueCommitted={() => commit()}
            />
          </div>
        );
      })}
    </div>
  );
}
