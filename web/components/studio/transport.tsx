"use client";

import { DicesIcon, DownloadIcon, PlayIcon, RotateCcwIcon, ShuffleIcon } from "lucide-react";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useStudio } from "@/hooks/use-studio";

export function Transport() {
  const { sound, config, rendering, preview, autoAudition, setAutoAudition, reset, randomize, newVariation, commit, exporting, exportSample } = useStudio();
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat) return;
      const target = e.target as HTMLElement | null;
      if (target && (["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(target.tagName) || target.isContentEditable)) return;
      e.preventDefault();
      void preview(true);
    };
    window.addEventListener("keydown", down);
    return () => window.removeEventListener("keydown", down);
  }, [preview]);
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button onClick={() => void preview(true)} disabled={!sound || rendering} className="min-w-28">
        <PlayIcon data-icon="inline-start" /> {rendering ? "Rendering…" : "Preview"}
      </Button>
      {config?.variation ? (
        <Button variant="outline" disabled={!sound || rendering} onClick={() => { newVariation(); commit(); }}>
          <DicesIcon data-icon="inline-start" /> New variation
        </Button>
      ) : null}
      <Button variant="outline" size="icon" aria-label="Randomise controls" onClick={() => { randomize(); commit(); }}><ShuffleIcon /></Button>
      <Button variant="outline" size="icon" aria-label="Reset controls" onClick={() => { reset(); commit(); }}><RotateCcwIcon /></Button>
      <div className="ml-1 flex items-center gap-2">
        <Switch id="auto-audition" checked={autoAudition} onCheckedChange={setAutoAudition} />
        <Label htmlFor="auto-audition" className="text-xs text-muted-foreground">Audition on release</Label>
      </div>
      <div className="ml-auto">
        <Tooltip>
          <TooltipTrigger render={<Button variant="outline" onClick={() => void exportSample()} disabled={!sound || exporting} />}>
            <DownloadIcon data-icon="inline-start" /> {exporting ? "Exporting…" : "Export WAV"}
          </TooltipTrigger>
          <TooltipContent>24-bit / 44.1 kHz WAV</TooltipContent>
        </Tooltip>
      </div>
    </div>
  );
}
