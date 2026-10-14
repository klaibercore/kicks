import type { Sound } from "@/lib/api/types";

export interface Pad {
  index: number;
  /** MIDI note that triggers this pad. Defaults follow an MPC-style 36..51 layout. */
  note: number;
  /** Keyboard key that triggers this pad. */
  key: string;
  name: string | null;
  sound: Sound | null;
}

export interface Kit {
  name: string;
  pads: Pad[];
}

export interface SavedKit extends Kit { id: string }

/** Validate imported and persisted data before it becomes a render request. */
export function parseKit(value: unknown): Kit {
  if (!value || typeof value !== "object") throw new Error("Invalid kit");
  const kit = value as Kit;
  if (typeof kit.name !== "string" || kit.name.length > 60 || !Array.isArray(kit.pads) || kit.pads.length !== PAD_COUNT) throw new Error("Invalid kit");
  const pads = kit.pads.map((pad, index) => {
    if (!pad || pad.index !== index || !Number.isInteger(pad.note) || pad.note < 0 || pad.note > 127 || typeof pad.key !== "string" || pad.key.length !== 1 || (pad.name !== null && (typeof pad.name !== "string" || pad.name.length > 120))) throw new Error("Invalid pad");
    const s = pad.sound;
    if (s !== null) {
      if (!s || typeof s.instrument !== "string" || !/^[a-z0-9_-]{1,64}$/.test(s.instrument) || ![undefined, "vae", "diffusion", "dsp", "parameters", "ddsp", "noiseband", "hybrid", "drumgan", "codec", "wavevae"].includes(s.backend)) throw new Error("Invalid sound");
      if (!s.sliders || typeof s.sliders !== "object" || Object.entries(s.sliders).some(([k, v]) => !/^[a-z0-9_]{1,64}$/.test(k) || !Number.isFinite(v) || v < 0 || v > 1)) throw new Error("Invalid controls");
      if (!s.effects || typeof s.effects !== "object" || Object.entries(s.effects).some(([k, v]) => !["attack_ms", "decay_ms", "drive", "filter"].includes(k) || !Number.isFinite(v) || v! < 0)) throw new Error("Invalid effects");
      if (s.seed !== undefined && (!Number.isInteger(s.seed) || s.seed < 0 || s.seed >= 2 ** 32)) throw new Error("Invalid seed");
      if (s.steps !== undefined && (!Number.isInteger(s.steps) || s.steps < 10 || s.steps > 200)) throw new Error("Invalid steps");
      if (s.choke_group !== undefined && !/^[a-z0-9_-]{1,64}$/.test(s.choke_group)) throw new Error("Invalid choke group");
    }
    return { index, note: pad.note, key: pad.key, name: pad.name, sound: s };
  });
  return { name: kit.name, pads };
}

export const PAD_COUNT = 16;

/** Top row first, so the grid reads like a keyboard. */
export const PAD_KEYS = ["1", "2", "3", "4", "q", "w", "e", "r", "a", "s", "d", "f", "z", "x", "c", "v"];

/** MPC/Maschine convention: pad 1 is bottom-left, notes rise left-to-right, bottom-to-top. */
export function defaultNote(index: number): number {
  const row = Math.floor(index / 4); // 0 = top
  const col = index % 4;
  return 36 + (3 - row) * 4 + col;
}

export function emptyKit(name = "Untitled kit"): Kit {
  return {
    name,
    pads: Array.from({ length: PAD_COUNT }, (_, index) => ({
      index,
      note: defaultNote(index),
      key: PAD_KEYS[index],
      name: null,
      sound: null,
    })),
  };
}

const STORAGE_KEY = "kicks.kit";

export function loadLocalKit(): Kit | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return parseKit(JSON.parse(raw));
  } catch {
    return null;
  }
}

export function loadNamedKits(): SavedKit[] {
  try {
    const entries = JSON.parse(localStorage.getItem("kicks.saved-kits") ?? "[]") as SavedKit[];
    if (!Array.isArray(entries)) return [];
    return entries.filter((k) => typeof k.id === "string").map((k) => ({ ...parseKit(k), id: k.id }));
  } catch { return []; }
}

export function saveNamedKits(kits: SavedKit[]): void {
  try { localStorage.setItem("kicks.saved-kits", JSON.stringify(kits)); } catch { /* storage unavailable */ }
}

export function saveLocalKit(kit: Kit): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(kit));
  } catch {
    /* storage unavailable */
  }
}

/** Short label for a pad: "kick · sub 80 dec 20". */
export function padSummary(pad: Pad): string {
  if (!pad.sound) return "";
  const parts = Object.entries(pad.sound.sliders)
    .slice(0, 3)
    .map(([k, v]) => `${k.slice(0, 3)} ${Math.round(v * 100)}`);
  if (pad.sound.backend === "diffusion") parts.unshift("diff");
  if (pad.sound.backend === "dsp") parts.unshift("dsp");
  if (pad.sound.backend && !["vae", "diffusion", "dsp"].includes(pad.sound.backend)) parts.unshift(pad.sound.backend);
  return parts.join(" · ");
}
