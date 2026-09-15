"use client";

import { MoreHorizontalIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import type { KitApi } from "@/hooks/use-kit";
import { useStudio } from "@/hooks/use-studio";
import { padSummary, type Pad } from "@/lib/pads";
import { cn } from "@/lib/utils";

const instrumentTone: Record<string, string> = {
  kick: "border-chart-1/60 bg-chart-1/10 hover:bg-chart-1/20",
  snare: "border-chart-3/60 bg-chart-3/10 hover:bg-chart-3/20",
  hihat: "border-chart-2/60 bg-chart-2/10 hover:bg-chart-2/20",
};

interface PadGridProps {
  kit: KitApi;
  learning: number | null;
  onLearn: (index: number | null) => void;
}

export function PadGrid({ kit, learning, onLearn }: PadGridProps) {
  const { sound, loadSound } = useStudio();

  // Keyboard triggers, ignored while typing into a field.
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      const pad = kit.padForKey(e.key);
      if (pad) {
        e.preventDefault();
        void kit.trigger(pad.index, 1);
      }
    };
    window.addEventListener("keydown", down);
    return () => window.removeEventListener("keydown", down);
  }, [kit]);

  return (
    <div className="grid grid-cols-4 gap-2">
      {kit.kit.pads.map((pad) => (
        <PadButton
          key={pad.index}
          pad={pad}
          flashAt={kit.flash[pad.index]}
          learning={learning === pad.index}
          onTrigger={(v) => void kit.trigger(pad.index, v)}
          onAssign={() => sound && kit.assign(pad.index, sound)}
          onClear={() => kit.clear(pad.index)}
          onLoad={() => pad.sound && loadSound(pad.sound)}
          onLearn={() => onLearn(learning === pad.index ? null : pad.index)}
          onUpdate={(patch) => kit.updatePad(pad.index, patch)}
          canAssign={Boolean(sound)}
        />
      ))}
    </div>
  );
}

interface PadButtonProps {
  pad: Pad;
  flashAt?: number;
  learning: boolean;
  canAssign: boolean;
  onTrigger: (velocity: number) => void;
  onAssign: () => void;
  onClear: () => void;
  onLoad: () => void;
  onLearn: () => void;
  onUpdate: (patch: Partial<Pad>) => void;
}

function PadButton({ pad, flashAt, learning, canAssign, onTrigger, onAssign, onClear, onLoad, onLearn, onUpdate }: PadButtonProps) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const tone = pad.sound ? (instrumentTone[pad.sound.instrument] ?? "border-border bg-muted/40") : "border-dashed border-border";

  return (
    <div
      className={cn(
        "group/pad relative aspect-square select-none rounded-lg border transition-[transform,box-shadow,background-color] duration-75",
        tone,
        learning && "ring-2 ring-amber-500",
      )}
    >
      {flashAt ? <span key={flashAt} aria-hidden className="pad-flash pointer-events-none absolute inset-0 rounded-lg" /> : null}
      <button
        type="button"
        className="flex size-full flex-col items-start justify-between p-2 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring"
        onPointerDown={(e) => {
          if (e.button !== 0) return;
          // Pointer pressure is 0.5 for mice; treat that as full velocity.
          const v = e.pointerType === "mouse" ? 1 : Math.max(0.3, e.pressure || 1);
          onTrigger(v);
        }}
        aria-label={pad.name ? `Pad ${pad.index + 1}: ${pad.name}` : `Pad ${pad.index + 1}, empty`}
      >
        <span className="flex w-full items-center justify-between font-mono text-[10px] uppercase text-muted-foreground">
          <span>{pad.key}</span>
          <span>{pad.note}</span>
        </span>
        <span className="min-w-0">
          <span className="block truncate text-xs font-medium">{pad.name ?? "—"}</span>
          <span className="block truncate font-mono text-[10px] text-muted-foreground">{padSummary(pad)}</span>
        </span>
      </button>

      <DropdownMenu>
        <DropdownMenuTrigger
          render={
            <Button
              variant="ghost"
              size="icon-xs"
              className="absolute top-1 right-1 opacity-0 transition-opacity group-hover/pad:opacity-100 hover:opacity-100 focus-visible:opacity-100 data-[popup-open]:opacity-100 [@media(hover:none)]:opacity-100"
              aria-label={`Pad ${pad.index + 1} options`}
            />
          }
        >
          <MoreHorizontalIcon />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-52">
          <DropdownMenuItem disabled={!canAssign} onClick={onAssign}>
            Assign current sound
          </DropdownMenuItem>
          <DropdownMenuItem disabled={!pad.sound} onClick={onLoad}>
            Load into sliders
          </DropdownMenuItem>
          <DropdownMenuItem onClick={onLearn}>{learning ? "Cancel MIDI learn" : "MIDI learn"}</DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={() => setSettingsOpen(true)}>Name, note &amp; key…</DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem disabled={!pad.sound} onClick={onClear} variant="destructive">
            Clear pad
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <PadSettings pad={pad} onUpdate={onUpdate} open={settingsOpen} onOpenChange={setSettingsOpen} />
    </div>
  );
}

function PadSettings({
  pad,
  onUpdate,
  open,
  onOpenChange,
}: {
  pad: Pad;
  onUpdate: (patch: Partial<Pad>) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Pad {pad.index + 1}</DialogTitle>
          <DialogDescription>Trigger note and key, and a label for the pad.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-3">
          <div className="grid gap-1.5">
            <Label htmlFor={`pad-name-${pad.index}`}>Name</Label>
            <Input
              id={`pad-name-${pad.index}`}
              value={pad.name ?? ""}
              placeholder="e.g. Tight kick"
              maxLength={24}
              onChange={(e) => onUpdate({ name: e.target.value || null })}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="grid gap-1.5">
              <Label htmlFor={`pad-note-${pad.index}`}>MIDI note</Label>
              <Input
                id={`pad-note-${pad.index}`}
                type="number"
                min={0}
                max={127}
                value={pad.note}
                onChange={(e) => onUpdate({ note: Math.max(0, Math.min(127, Number(e.target.value) || 0)) })}
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor={`pad-key-${pad.index}`}>Key</Label>
              <Input
                id={`pad-key-${pad.index}`}
                value={pad.key}
                maxLength={1}
                onChange={(e) => onUpdate({ key: e.target.value.slice(-1).toLowerCase() })}
              />
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
