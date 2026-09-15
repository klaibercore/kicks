"use client";

import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import type { Effects } from "@/lib/api/types";
import { useStudio } from "@/hooks/use-studio";

interface EffectDef {
  key: keyof Effects;
  label: string;
  min: number;
  max: number;
  step: number;
  initial: number;
  unit: string;
  format?: (v: number) => string;
}

const EFFECTS: EffectDef[] = [
  { key: "attack_ms", label: "Attack", min: 0, max: 60, step: 1, initial: 5, unit: "ms" },
  { key: "decay_ms", label: "Decay", min: 20, max: 1500, step: 10, initial: 400, unit: "ms" },
  { key: "drive", label: "Drive", min: 0, max: 1, step: 0.01, initial: 0.3, unit: "", format: (v) => `${Math.round(v * 100)}%` },
  { key: "filter", label: "Lowpass", min: 200, max: 20000, step: 50, initial: 8000, unit: "Hz" },
];

/** Post-vocoder shaping. Off means the parameter is not sent at all. */
export function EffectsRack() {
  const { sound, setEffect, commit } = useStudio();
  if (!sound) return null;

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {EFFECTS.map((fx) => {
        const value = sound.effects[fx.key];
        const on = value !== undefined;
        return (
          <div key={fx.key} className="rounded-lg border border-border p-3">
            <div className="mb-2 flex items-center justify-between">
              <Label htmlFor={`fx-${fx.key}`} className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                {fx.label}
              </Label>
              <div className="flex items-center gap-2">
                <span className="tabular font-mono text-xs text-muted-foreground">
                  {on ? (fx.format ? fx.format(value) : `${value} ${fx.unit}`) : "off"}
                </span>
                <Switch
                  id={`fx-${fx.key}`}
                  checked={on}
                  onCheckedChange={(checked) => {
                    setEffect(fx.key, checked ? fx.initial : undefined);
                    commit();
                  }}
                />
              </div>
            </div>
            <Slider
              aria-label={fx.label}
              disabled={!on}
              min={fx.min}
              max={fx.max}
              step={fx.step}
              value={on ? value : fx.initial}
              onValueChange={(v) => setEffect(fx.key, Array.isArray(v) ? v[0] : v)}
              onValueCommitted={() => commit()}
            />
          </div>
        );
      })}
    </div>
  );
}
