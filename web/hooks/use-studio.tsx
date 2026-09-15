"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { toast } from "sonner";
import { ApiError, soundQuery } from "@/lib/api/client";
import type {
  Effects,
  Evaluation,
  Health,
  InstrumentConfig,
  InstrumentInfo,
  Sound,
  Spectrogram,
} from "@/lib/api/types";
import { engine } from "@/lib/audio/engine";
import { useAuth } from "@/hooks/use-auth";

export type ApiStatus = "connecting" | "online" | "offline";

interface StudioContextValue {
  status: ApiStatus;
  health: Health | null;
  instruments: InstrumentInfo[];
  configs: Record<string, InstrumentConfig>;
  active: string | null;
  setActive: (name: string) => void;
  sound: Sound | null;
  config: InstrumentConfig | null;
  setSlider: (key: string, value: number) => void;
  setEffect: <K extends keyof Effects>(key: K, value: Effects[K]) => void;
  reset: () => void;
  randomize: () => void;
  loadSound: (sound: Sound) => void;

  buffer: AudioBuffer | null;
  rendering: boolean;
  preview: (play?: boolean) => Promise<AudioBuffer | null>;
  autoAudition: boolean;
  setAutoAudition: (on: boolean) => void;
  /** Called by sliders when a drag ends. */
  commit: () => void;

  evaluation: Evaluation | null;
  evaluating: boolean;
  evaluate: () => Promise<void>;
  spectrogram: Spectrogram | null;
  loadSpectrogram: () => Promise<void>;

  exporting: boolean;
  exportSample: () => Promise<void>;
  /** Fetch (or reuse) the decoded buffer for any sound, e.g. a pad. */
  bufferFor: (sound: Sound) => Promise<AudioBuffer>;
}

const StudioContext = createContext<StudioContextValue | null>(null);

function centredSound(config: InstrumentConfig): Sound {
  return {
    instrument: config.instrument,
    sliders: Object.fromEntries(config.sliders.map((s) => [s.key, s.default])),
    effects: {},
  };
}

export function StudioProvider({ children }: { children: ReactNode }) {
  const { api, user, enabled: authEnabled, refreshCredits } = useAuth();
  const [status, setStatus] = useState<ApiStatus>("connecting");
  const [health, setHealth] = useState<Health | null>(null);
  const [instruments, setInstruments] = useState<InstrumentInfo[]>([]);
  const [configs, setConfigs] = useState<Record<string, InstrumentConfig>>({});
  const [sounds, setSounds] = useState<Record<string, Sound>>({});
  const [active, setActiveState] = useState<string | null>(null);
  const [buffer, setBuffer] = useState<AudioBuffer | null>(null);
  const [rendering, setRendering] = useState(false);
  const [autoAudition, setAutoAudition] = useState(true);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [evaluating, setEvaluating] = useState(false);
  const [spectrogram, setSpectrogram] = useState<Spectrogram | null>(null);
  const [exporting, setExporting] = useState(false);
  const renderSeq = useRef(0);

  // Discover instruments once the API answers.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [h, list] = await Promise.all([api.health(), api.instruments()]);
        if (cancelled) return;
        setHealth(h);
        setInstruments(list.instruments);
        setStatus("online");
        const first = list.instruments.find((i) => i.name === list.default && i.trained) ?? list.instruments.find((i) => i.trained);
        if (first) setActiveState(first.name);
      } catch {
        if (!cancelled) setStatus("offline");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [api]);

  // Load the slider layout for the active instrument on first use.
  useEffect(() => {
    if (!active || configs[active]) return;
    let cancelled = false;
    api
      .config(active)
      .then((cfg) => {
        if (cancelled) return;
        setConfigs((c) => ({ ...c, [active]: cfg }));
        setSounds((s) => (s[active] ? s : { ...s, [active]: centredSound(cfg) }));
      })
      .catch((err: unknown) => {
        toast.error(err instanceof ApiError ? err.message : "Could not load the instrument.");
      });
    return () => {
      cancelled = true;
    };
  }, [active, api, configs]);

  const sound = active ? (sounds[active] ?? null) : null;
  const config = active ? (configs[active] ?? null) : null;

  const bufferFor = useCallback(
    async (target: Sound): Promise<AudioBuffer> => {
      const key = soundQuery(target);
      const hit = engine.cached(key);
      if (hit) return hit;
      const bytes = await api.generate(target);
      const decoded = await engine.decode(bytes);
      engine.remember(key, decoded);
      return decoded;
    },
    [api],
  );

  const preview = useCallback(
    async (play = true) => {
      if (!sound) return null;
      const seq = ++renderSeq.current;
      setRendering(true);
      try {
        const decoded = await bufferFor(sound);
        if (seq !== renderSeq.current) return decoded; // a newer render superseded this one
        setBuffer(decoded);
        if (play) engine.play(decoded);
        return decoded;
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          toast.error("Sign in to use the synthesizer.");
        } else if (err instanceof ApiError && err.status === 429) {
          toast.warning("Slow down — the renderer is rate-limited.");
        } else {
          toast.error(err instanceof Error ? err.message : "Render failed.");
        }
        return null;
      } finally {
        if (seq === renderSeq.current) setRendering(false);
      }
    },
    [sound, bufferFor],
  );

  const update = useCallback(
    (fn: (s: Sound) => Sound) => {
      if (!active) return;
      setSounds((all) => (all[active] ? { ...all, [active]: fn(all[active]) } : all));
      setEvaluation(null);
      setSpectrogram(null);
    },
    [active],
  );

  const setSlider = useCallback(
    (key: string, value: number) => update((s) => ({ ...s, sliders: { ...s.sliders, [key]: value } })),
    [update],
  );

  const setEffect = useCallback(
    <K extends keyof Effects>(key: K, value: Effects[K]) =>
      update((s) => {
        const effects = { ...s.effects };
        if (value === undefined) delete effects[key];
        else effects[key] = value;
        return { ...s, effects };
      }),
    [update],
  );

  const reset = useCallback(() => {
    if (config) update(() => centredSound(config));
  }, [config, update]);

  const randomize = useCallback(() => {
    if (!config) return;
    update((s) => ({
      ...s,
      sliders: Object.fromEntries(config.sliders.map((d) => [d.key, Math.round(Math.random() * 100) / 100])),
    }));
  }, [config, update]);

  const loadSound = useCallback(
    (next: Sound) => {
      setActiveState(next.instrument);
      setSounds((all) => ({ ...all, [next.instrument]: next }));
      setEvaluation(null);
      setSpectrogram(null);
    },
    [],
  );

  const commit = useCallback(() => {
    if (autoAudition) void preview(true);
  }, [autoAudition, preview]);

  const evaluate = useCallback(async () => {
    if (!sound) return;
    setEvaluating(true);
    try {
      setEvaluation(await api.evaluate(sound));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Evaluation failed.");
    } finally {
      setEvaluating(false);
    }
  }, [api, sound]);

  const loadSpectrogram = useCallback(async () => {
    if (!sound) return;
    try {
      setSpectrogram(await api.spectrogram(sound));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not load the spectrogram.");
    }
  }, [api, sound]);

  const exportSample = useCallback(async () => {
    if (!sound) return;
    if (authEnabled && !user) {
      toast.error("Sign in to export samples.");
      return;
    }
    setExporting(true);
    try {
      const result = await api.exportSample(sound, crypto.randomUUID());
      const url = URL.createObjectURL(result.blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = result.filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
      await refreshCredits();
      toast.success(
        Number.isFinite(result.remaining)
          ? `Exported ${result.filename} — ${result.remaining} credit${result.remaining === 1 ? "" : "s"} left.`
          : `Exported ${result.filename}.`,
      );
    } catch (err) {
      if (err instanceof ApiError && err.status === 402) {
        toast.error("No credits left.", {
          description: "Buy a pack on the pricing page to export samples.",
        });
      } else if (err instanceof ApiError && err.status === 503) {
        toast.error("Exports are not enabled on this server.");
      } else {
        toast.error(err instanceof Error ? err.message : "Export failed.");
      }
    } finally {
      setExporting(false);
    }
  }, [api, sound, authEnabled, user, refreshCredits]);

  const setActive = useCallback((name: string) => {
    setActiveState(name);
    setEvaluation(null);
    setSpectrogram(null);
    setBuffer(null);
  }, []);

  const value = useMemo<StudioContextValue>(
    () => ({
      status, health, instruments, configs, active, setActive, sound, config,
      setSlider, setEffect, reset, randomize, loadSound,
      buffer, rendering, preview, autoAudition, setAutoAudition, commit,
      evaluation, evaluating, evaluate, spectrogram, loadSpectrogram,
      exporting, exportSample, bufferFor,
    }),
    [
      status, health, instruments, configs, active, setActive, sound, config,
      setSlider, setEffect, reset, randomize, loadSound,
      buffer, rendering, preview, autoAudition, commit,
      evaluation, evaluating, evaluate, spectrogram, loadSpectrogram,
      exporting, exportSample, bufferFor,
    ],
  );

  return <StudioContext.Provider value={value}>{children}</StudioContext.Provider>;
}

export function useStudio(): StudioContextValue {
  const ctx = useContext(StudioContext);
  if (!ctx) throw new Error("useStudio must be used inside <StudioProvider>");
  return ctx;
}
