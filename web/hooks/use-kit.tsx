"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useStudio } from "@/hooks/use-studio";
import { engine } from "@/lib/audio/engine";
import type { Sound } from "@/lib/api/types";
import { emptyKit, loadLocalKit, saveLocalKit, loadNamedKits, saveNamedKits, parseKit, type Kit, type Pad, type SavedKit } from "@/lib/pads";

/** Active and named kits stay on this computer; JSON files make them portable. */
export function useKit() {
  const { bufferFor, instruments } = useStudio();
  const [kit, setKit] = useState<Kit>(() => loadLocalKit() ?? emptyKit());
  const [flash, setFlash] = useState<Record<number, number>>({});
  const [savedKits, setSavedKits] = useState<SavedKit[]>(loadNamedKits);
  const kitRef = useRef(kit);
  const triggerOrder = useRef<Record<string, number>>({});

  useEffect(() => { kitRef.current = kit; saveLocalKit(kit); }, [kit]);
  useEffect(() => { saveNamedKits(savedKits); }, [savedKits]);
  useEffect(() => {
    for (const pad of kitRef.current.pads) {
      if (pad.sound) void bufferFor(pad.sound).catch(() => undefined);
    }
  }, [bufferFor]);

  const trigger = useCallback(async (index: number, velocity = 1) => {
    const pad = kitRef.current.pads[index];
    if (!pad?.sound) return;
    const group = pad.sound.choke_group ?? instruments.find((i) => i.name === pad.sound?.instrument)?.choke_group ?? undefined;
    const order = group ? (triggerOrder.current[group] ?? 0) + 1 : 0;
    if (group) triggerOrder.current[group] = order;
    setFlash((f) => ({ ...f, [index]: Date.now() }));
    try {
      const buffer = await bufferFor(pad.sound);
      if (group && triggerOrder.current[group] !== order) return;
      engine.play(buffer, velocity, group);
    } catch { /* renderer errors appear in the studio */ }
  }, [bufferFor, instruments]);

  const assign = useCallback((index: number, sound: Sound, name?: string) => {
    setKit((k) => ({ ...k, pads: k.pads.map((p) => p.index === index ? { ...p, sound, name: name ?? p.name ?? sound.instrument } : p) }));
    void bufferFor(sound).catch(() => undefined);
  }, [bufferFor]);
  const clear = useCallback((index: number) => setKit((k) => ({ ...k, pads: k.pads.map((p) => p.index === index ? { ...p, sound: null, name: null } : p) })), []);
  const updatePad = useCallback((index: number, patch: Partial<Pad>) => setKit((k) => ({ ...k, pads: k.pads.map((p) => p.index === index ? { ...p, ...patch } : p) })), []);
  const rename = useCallback((name: string) => setKit((k) => ({ ...k, name })), []);
  const clearAll = useCallback(() => { engine.stopAll(); setKit((k) => emptyKit(k.name)); }, []);
  const padForNote = useCallback((note: number) => kitRef.current.pads.find((p) => p.note === note) ?? null, []);
  const padForKey = useCallback((key: string) => kitRef.current.pads.find((p) => p.key === key.toLowerCase()) ?? null, []);
  const save = useCallback(() => setSavedKits((all) => {
    const saved = { ...kit, id: crypto.randomUUID() };
    return [...all.filter((k) => k.name !== kit.name), saved];
  }), [kit]);
  const open = useCallback((saved: Kit) => { engine.stopAll(); setKit(parseKit(saved)); }, []);
  const remove = useCallback((id: string) => setSavedKits((all) => all.filter((k) => k.id !== id)), []);

  return useMemo(() => ({ kit, flash, trigger, assign, clear, updatePad, rename, clearAll, padForNote, padForKey,
    savedKits, save, open, remove }), [kit, flash, trigger, assign, clear, updatePad, rename, clearAll, padForNote, padForKey, savedKits, save, open, remove]);
}

export type KitApi = ReturnType<typeof useKit>;
