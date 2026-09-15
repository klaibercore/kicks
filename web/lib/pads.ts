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
    const parsed = JSON.parse(raw) as Kit;
    if (!Array.isArray(parsed.pads) || parsed.pads.length !== PAD_COUNT) return null;
    return parsed;
  } catch {
    return null;
  }
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
  return parts.join(" · ");
}
