"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { useAuth } from "@/hooks/use-auth";
import { useStudio } from "@/hooks/use-studio";
import { engine } from "@/lib/audio/engine";
import type { Sound } from "@/lib/api/types";
import { emptyKit, loadLocalKit, saveLocalKit, type Kit, type Pad } from "@/lib/pads";
import { getSupabase } from "@/lib/supabase/client";
import type { Kit as SavedKit } from "@/lib/supabase/types";

/**
 * The pad bank. Sixteen pads, each holding a complete sound. Pads always
 * persist locally; a signed-in user can also keep named kits in their account.
 */
export function useKit() {
  const { bufferFor } = useStudio();
  const { user } = useAuth();
  // The studio is rendered client-only, so localStorage is safe to read here.
  const [kit, setKit] = useState<Kit>(() => loadLocalKit() ?? emptyKit());
  const [flash, setFlash] = useState<Record<number, number>>({});
  const [savedKits, setSavedKits] = useState<SavedKit[]>([]);
  const [kitId, setKitId] = useState<string | null>(null);
  const kitRef = useRef(kit);

  useEffect(() => {
    kitRef.current = kit;
    saveLocalKit(kit);
  }, [kit]);

  // Warm the audio cache for every assigned pad once the API is reachable.
  useEffect(() => {
    for (const pad of kitRef.current.pads) {
      if (pad.sound) void bufferFor(pad.sound).catch(() => undefined);
    }
  }, [bufferFor]);

  const trigger = useCallback(
    async (index: number, velocity = 1) => {
      const pad = kitRef.current.pads[index];
      if (!pad?.sound) return;
      setFlash((f) => ({ ...f, [index]: Date.now() }));
      try {
        engine.play(await bufferFor(pad.sound), velocity);
      } catch {
        /* the studio already reports render errors */
      }
    },
    [bufferFor],
  );

  const assign = useCallback((index: number, sound: Sound, name?: string) => {
    setKit((k) => ({
      ...k,
      pads: k.pads.map((p) => (p.index === index ? { ...p, sound, name: name ?? p.name ?? sound.instrument } : p)),
    }));
    void bufferFor(sound).catch(() => undefined);
  }, [bufferFor]);

  const clear = useCallback((index: number) => {
    setKit((k) => ({ ...k, pads: k.pads.map((p) => (p.index === index ? { ...p, sound: null, name: null } : p)) }));
  }, []);

  const updatePad = useCallback((index: number, patch: Partial<Pad>) => {
    setKit((k) => ({ ...k, pads: k.pads.map((p) => (p.index === index ? { ...p, ...patch } : p)) }));
  }, []);

  const rename = useCallback((name: string) => setKit((k) => ({ ...k, name })), []);

  const clearAll = useCallback(() => {
    setKit((k) => emptyKit(k.name));
    setKitId(null);
  }, []);

  const padForNote = useCallback((note: number) => kitRef.current.pads.find((p) => p.note === note) ?? null, []);
  const padForKey = useCallback(
    (key: string) => kitRef.current.pads.find((p) => p.key === key.toLowerCase()) ?? null,
    [],
  );

  // -- account persistence ------------------------------------------------

  const fetchSaved = useCallback(async (): Promise<SavedKit[]> => {
    const sb = getSupabase();
    if (!sb || !user) return [];
    const { data } = await sb.from("kits").select("*").order("updated_at", { ascending: false });
    return (data as SavedKit[]) ?? [];
  }, [user]);

  const refreshSaved = useCallback(async () => setSavedKits(await fetchSaved()), [fetchSaved]);

  useEffect(() => {
    let cancelled = false;
    fetchSaved().then((kits) => {
      if (!cancelled) setSavedKits(kits);
    });
    return () => {
      cancelled = true;
    };
  }, [fetchSaved]);

  const saveToAccount = useCallback(async () => {
    const sb = getSupabase();
    if (!sb || !user) {
      toast.error("Sign in to save kits to your account.");
      return;
    }
    const payload = { user_id: user.id, name: kit.name, pads: kit.pads };
    const query = kitId
      ? sb.from("kits").update(payload).eq("id", kitId).select("id").single()
      : sb.from("kits").insert(payload).select("id").single();
    const { data, error } = await query;
    if (error) {
      toast.error(error.message);
      return;
    }
    setKitId((data as { id: string }).id);
    toast.success(`Saved "${kit.name}".`);
    await refreshSaved();
  }, [kit, kitId, user, refreshSaved]);

  const loadFromAccount = useCallback((saved: SavedKit) => {
    setKit({ name: saved.name, pads: saved.pads as Pad[] });
    setKitId(saved.id);
    for (const pad of saved.pads as Pad[]) {
      if (pad.sound) void bufferFor(pad.sound).catch(() => undefined);
    }
  }, [bufferFor]);

  const deleteFromAccount = useCallback(
    async (id: string) => {
      const sb = getSupabase();
      if (!sb) return;
      const { error } = await sb.from("kits").delete().eq("id", id);
      if (error) toast.error(error.message);
      if (kitId === id) setKitId(null);
      await refreshSaved();
    },
    [kitId, refreshSaved],
  );

  return useMemo(
    () => ({
      kit, kitId, flash,
      trigger, assign, clear, updatePad, rename, clearAll, padForNote, padForKey,
      savedKits, saveToAccount, loadFromAccount, deleteFromAccount,
    }),
    [
      kit, kitId, flash,
      trigger, assign, clear, updatePad, rename, clearAll, padForNote, padForKey,
      savedKits, saveToAccount, loadFromAccount, deleteFromAccount,
    ],
  );
}

export type KitApi = ReturnType<typeof useKit>;
