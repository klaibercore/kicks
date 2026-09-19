"use client";

import { DicesIcon, DownloadIcon, PlayIcon, RotateCcwIcon, ShuffleIcon } from "lucide-react";
import Link from "next/link";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useAuth } from "@/hooks/use-auth";
import { useStudio } from "@/hooks/use-studio";

export function Transport() {
  const { sound, config, rendering, preview, autoAudition, setAutoAudition, reset, randomize, newVariation, commit, exporting, exportSample, health } =
    useStudio();
  const { enabled: authEnabled, user, credits } = useAuth();

  // Space previews, like every DAW.
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat) return;
      const target = e.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(target.tagName)) return;
      e.preventDefault();
      void preview(true);
    };
    window.addEventListener("keydown", down);
    return () => window.removeEventListener("keydown", down);
  }, [preview]);

  const exportsOn = health?.auth.billing ?? false;
  const noCredits = authEnabled && user && credits !== null && credits < 1;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button onClick={() => void preview(true)} disabled={!sound || rendering} className="min-w-28">
        <PlayIcon data-icon="inline-start" /> {rendering ? "Rendering…" : "Preview"}
      </Button>
      {config?.variation && (
        <Tooltip>
          <TooltipTrigger render={<Button variant="outline" disabled={!sound || rendering} onClick={() => { newVariation(); commit(); }} />}>
            <DicesIcon data-icon="inline-start" /> New variation
          </TooltipTrigger>
          <TooltipContent>A different drum texture with the same slider settings</TooltipContent>
        </Tooltip>
      )}
      <Tooltip>
        <TooltipTrigger render={<Button variant="outline" size="icon" aria-label="Randomise sliders" onClick={() => { randomize(); commit(); }} />}>
          <ShuffleIcon />
        </TooltipTrigger>
        <TooltipContent>Randomise sliders; keep this texture</TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger render={<Button variant="outline" size="icon" aria-label="Reset sliders" onClick={() => { reset(); commit(); }} />}>
          <RotateCcwIcon />
        </TooltipTrigger>
        <TooltipContent>Reset to centre</TooltipContent>
      </Tooltip>
      <div className="ml-1 flex items-center gap-2">
        <Switch id="auto-audition" checked={autoAudition} onCheckedChange={setAutoAudition} />
        <Label htmlFor="auto-audition" className="text-xs text-muted-foreground">
          Audition on release
        </Label>
      </div>

      <div className="ml-auto flex items-center gap-2">
        {exportsOn ? (
          noCredits ? (
            <Button variant="outline" render={<Link href="/pricing/" />}>
              <DownloadIcon data-icon="inline-start" /> Get credits to export
            </Button>
          ) : (
            <Tooltip>
              <TooltipTrigger
                render={
                  <Button variant="outline" onClick={() => void exportSample()} disabled={!sound || exporting} />
                }
              >
                <DownloadIcon data-icon="inline-start" /> {exporting ? "Exporting…" : "Export WAV · 1 credit"}
              </TooltipTrigger>
              <TooltipContent>24-bit / 44.1 kHz, licensed for your productions</TooltipContent>
            </Tooltip>
          )
        ) : (
          <Tooltip>
            <TooltipTrigger render={<Button variant="outline" onClick={() => void exportSample()} disabled={!sound || exporting} />}>
              <DownloadIcon data-icon="inline-start" /> {exporting ? "Exporting…" : "Export WAV"}
            </TooltipTrigger>
            <TooltipContent>This server has no billing configured; exports are open.</TooltipContent>
          </Tooltip>
        )}
      </div>
    </div>
  );
}
