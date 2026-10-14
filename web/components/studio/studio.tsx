"use client";

import { AlertCircleIcon } from "lucide-react";
import { useCallback, useState } from "react";
import { EffectsRack } from "@/components/studio/effects-rack";
import { EnginePicker } from "@/components/studio/engine-picker";
import { EvalPanel } from "@/components/studio/eval-panel";
import { KitManager } from "@/components/studio/kit-manager";
import { MidiPanel } from "@/components/studio/midi-panel";
import { PadGrid } from "@/components/studio/pad-grid";
import { SliderRack } from "@/components/studio/slider-rack";
import { SpectrogramView } from "@/components/studio/spectrogram-view";
import { Transport } from "@/components/studio/transport";
import { WaveformView } from "@/components/studio/waveform-view";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useKit } from "@/hooks/use-kit";
import { backendsOf, StudioProvider, useStudio } from "@/hooks/use-studio";
import { config } from "@/lib/config";
import { useMidi, type NoteEvent } from "@/lib/midi";

export function Studio() {
  return (
    <StudioProvider>
      <StudioBody />
    </StudioProvider>
  );
}

function StudioBody() {
  const studio = useStudio();
  const kit = useKit();
  const [learning, setLearning] = useState<number | null>(null);

  const onNote = useCallback(
    (event: NoteEvent) => {
      if (learning !== null) {
        kit.updatePad(learning, { note: event.note });
        setLearning(null);
        return;
      }
      const pad = kit.padForNote(event.note);
      if (pad) void kit.trigger(pad.index, event.velocity);
    },
    [kit, learning],
  );
  const midi = useMidi(onNote);

  if (studio.status === "offline") {
    return (
      <div className="mx-auto max-w-2xl px-4 py-16 sm:px-6">
        <Alert variant="destructive">
          <AlertCircleIcon />
          <AlertTitle>The synthesis API is not reachable</AlertTitle>
          <AlertDescription>
            <p>
              The studio talks to <code className="font-mono">{config.apiUrl}</code>. Start it with{" "}
              <code className="font-mono">kicks serve</code> or point <code className="font-mono">NEXT_PUBLIC_KICKS_API_URL</code> at
              a running instance, then reload.
            </p>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const diffusion = studio.backend === "diffusion";
  const candidate = !['vae', 'diffusion', 'dsp'].includes(studio.backend);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-6 sm:px-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Studio</h1>
          <p className="text-sm text-muted-foreground">
            Shape a drum’s character, audition it, and build your kit.
          </p>
        </div>
        {studio.health ? (
          <div className="flex flex-wrap gap-1.5 font-mono text-[11px]">
            {studio.backend === "dsp" ? (
              <Badge variant="outline">DSP synthesis</Badge>
            ) : candidate ? (
              <Badge variant="outline">{studio.backend} · experimental</Badge>
            ) : diffusion ? (
              <>
                <Badge variant="outline">diffusion</Badge>
                <Badge variant="outline">{studio.sound?.steps ?? studio.config?.steps?.default ?? "—"} steps</Badge>
              </>
            ) : (
              <>
                <Badge variant="outline">{studio.config?.vocoder ?? studio.health.vocoder}</Badge>
                <Badge variant="outline">{studio.config?.variation ? "Corpus texture" : `${studio.health.control} basis`}</Badge>
              </>
            )}
            <Badge variant="outline">{studio.backend === "dsp" ? "cpu" : studio.config?.device ?? studio.health.device}</Badge>
          </div>
        ) : null}
      </div>

      <Tabs value={studio.active ?? ""} onValueChange={(v) => v && studio.setActive(v as string)}>
        <TabsList variant="line" className="h-9">
          {studio.instruments.length === 0
            ? [0, 1, 2].map((i) => <Skeleton key={i} className="h-7 w-20" />)
            : studio.instruments.map((inst) => (
                <TabsTrigger key={inst.name} value={inst.name} disabled={backendsOf(inst).length === 0} className="px-3">
                  {inst.display_name}
                  {backendsOf(inst).length === 0 ? (
                    <span className="ml-1 text-[10px] text-muted-foreground">untrained</span>
                  ) : null}
                </TabsTrigger>
              ))}
        </TabsList>
      </Tabs>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Card>
          <CardHeader>
            <CardTitle>{studio.config?.display_name ?? "Instrument"}</CardTitle>
            <CardDescription>{studio.config?.description ?? "Loading the slider basis…"}</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-6">
            <EnginePicker />
            {studio.config ? <SliderRack /> : <SliderSkeleton />}
            {studio.config?.variation ? (
              <p className="text-xs leading-relaxed text-muted-foreground">
                {diffusion
                  ? "New variation starts the sampler from different noise: another hit aiming at the same slider targets. Each variation is saved with your pads."
                  : "Explore textures with New variation, then shape the tone and decay. Each variation is saved with your pads."}
              </p>
            ) : null}
            <div>
              <h3 className="mb-3 text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">Shaping</h3>
              <EffectsRack />
            </div>
            <Transport />
          </CardContent>
        </Card>

        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Render</CardTitle>
              <CardDescription>
                {studio.backend === "dsp"
                  ? "The synthesized waveform for the current controls."
                  : candidate ? "What the learned candidate produced for the current settings."
                  : diffusion
                  ? "What the diffusion model produced for the current settings. No vocoder is involved."
                  : "What the vocoder produced for the current settings."}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <WaveformView buffer={studio.buffer} className="h-28 w-full" />
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                  {studio.backend !== "vae" ? "Spectrogram of the render" : "Decoded spectrogram"}
                </h3>
                <Button size="xs" variant="ghost" onClick={() => void studio.loadSpectrogram()} disabled={!studio.sound}>
                  {studio.spectrogram ? "Refresh" : "Show"}
                </Button>
              </div>
              <div className="aspect-[2/1] w-full overflow-hidden rounded-md border border-border bg-[#0c0a18]">
                {studio.spectrogram ? (
                  <SpectrogramView spec={studio.spectrogram} className="size-full" />
                ) : (
                  <div className="flex size-full items-center justify-center text-xs text-muted-foreground">
                    128 mel bands × 256 frames
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <EvalPanel />
            </CardContent>
          </Card>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Pads</CardTitle>
          <CardDescription>
            Sixteen pads, played from the keyboard, the pointer or a MIDI controller. Assign the current sound to a pad from its menu.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="group mx-auto w-full max-w-md">
            <PadGrid kit={kit} learning={learning} onLearn={setLearning} />
          </div>
          <div className="flex flex-col gap-5">
            <div>
              <h3 className="mb-2 text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">MIDI</h3>
              <MidiPanel midi={midi} learning={learning} />
            </div>
            <div>
              <h3 className="mb-2 text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">Kit</h3>
              <KitManager kit={kit} />
            </div>
            <p className="text-xs text-muted-foreground">
              Keys 1–4, Q–R, A–F and Z–V trigger the rows. Space previews the current sound. Notes default to 36–51, the usual
              MPC layout; use MIDI learn on a pad to map a controller&apos;s own note numbers.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function SliderSkeleton() {
  return (
    <div className="flex flex-col gap-5">
      {[0, 1, 2, 3, 4].map((i) => (
        <div key={i} className="flex flex-col gap-2">
          <Skeleton className="h-3 w-16" />
          <Skeleton className="h-1 w-full" />
        </div>
      ))}
    </div>
  );
}
