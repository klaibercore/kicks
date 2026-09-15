"use client";

import { CableIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { useMidi } from "@/lib/midi";

type Midi = ReturnType<typeof useMidi>;

export function MidiPanel({ midi, learning }: { midi: Midi; learning: number | null }) {
  if (!midi.supported) {
    return (
      <p className="text-xs text-muted-foreground">
        Web MIDI is not available in this browser — Chrome, Edge and Opera support it. Keyboard keys still trigger the pads.
      </p>
    );
  }

  if (midi.permission !== "granted") {
    return (
      <div className="flex flex-wrap items-center gap-3">
        <Button size="sm" variant="outline" onClick={() => void midi.request()}>
          <CableIcon data-icon="inline-start" /> Connect MIDI
        </Button>
        <span className="text-xs text-muted-foreground">
          {midi.error ?? "Play the pads from a controller. The browser will ask for permission."}
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-3">
      <Select
        value={midi.selectedId}
        onValueChange={(v) => midi.select((v as string) ?? "all")}
        items={{ all: "All inputs", ...Object.fromEntries(midi.inputs.map((i) => [i.id, i.name])) }}
      >
        <SelectTrigger className="w-56" aria-label="MIDI input">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All inputs</SelectItem>
          {midi.inputs.map((input) => (
            <SelectItem key={input.id} value={input.id}>
              {input.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Badge variant="outline" className="font-mono">
        {midi.inputs.length} device{midi.inputs.length === 1 ? "" : "s"}
      </Badge>
      {midi.lastEvent ? (
        <span className="font-mono text-xs text-muted-foreground">
          note {midi.lastEvent.note} · vel {Math.round(midi.lastEvent.velocity * 127)} · ch {midi.lastEvent.channel + 1}
        </span>
      ) : null}
      {learning !== null ? (
        <Badge className="bg-amber-500 text-black">Hit a pad to map it to pad {learning + 1}</Badge>
      ) : null}
    </div>
  );
}
