"use client";

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useStudio } from "@/hooks/use-studio";
import type { Backend } from "@/lib/api/types";

const LABELS: Record<Backend, string> = { vae: "VAE", diffusion: "Diffusion" };

/** Sampler-step presets offered in the studio; the API accepts any integer in its range. */
const STEP_PRESETS = [25, 50, 100, 200];

/**
 * Which model renders the instrument. Shown only when the server offers more
 * than the VAE — a diffusion checkpoint is served explicitly
 * (`kicks serve --diffusion hihat=PATH`), never by default.
 */
export function EnginePicker() {
  const { backends, backend, setBackend, config, sound, setSteps, commit, renderMs } = useStudio();
  if (backends.length === 0 || (backends.length === 1 && backends[0] === "vae")) return null;

  const range = config?.steps;
  const presets = range ? STEP_PRESETS.filter((n) => n >= range.min && n <= range.max) : [];
  const steps = sound?.steps ?? range?.default;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="text-xs font-medium text-muted-foreground">Engine</span>
        <Tabs value={backend} onValueChange={(v) => v && setBackend(v as Backend)}>
          <TabsList className="h-8">
            {backends.map((b) => (
              <TabsTrigger key={b} value={b} className="px-3 text-xs">
                {LABELS[b]}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      {backend === "diffusion" && config?.backend === "diffusion" ? (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <span className="text-xs font-medium text-muted-foreground">Sampler steps</span>
            <Tabs
              value={String(steps)}
              onValueChange={(v) => {
                if (!v) return;
                setSteps(Number(v));
                commit();
              }}
            >
              <TabsList className="h-8">
                {presets.map((n) => (
                  <TabsTrigger key={n} value={String(n)} className="px-2.5 font-mono text-xs tabular">
                    {n}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          </div>
          <p className="text-xs leading-relaxed text-muted-foreground">
            Experimental. The diffusion model writes the waveform directly, with no vocoder, and follows the sliders
            approximately. Each slider spans the corpus from its 1st to its 99th percentile, centred on the median hit.
            More steps take proportionally longer.
            {renderMs !== null ? (
              <>
                {" "}
                Last render: <span className="font-mono tabular">{(renderMs / 1000).toFixed(1)} s</span>.
              </>
            ) : null}
            {config.checkpoint ? (
              <>
                {" "}
                Checkpoint <span className="font-mono">{config.checkpoint.name}</span>
                {config.checkpoint.epoch != null ? `, epoch ${config.checkpoint.epoch}` : ""}.
              </>
            ) : null}
          </p>
        </>
      ) : null}
    </div>
  );
}
