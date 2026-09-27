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
  Backend,
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
  /** Backends the server offers for the active instrument. */
  backends: Backend[];
  backend: Backend;
  setBackend: (backend: Backend) => void;
  sound: Sound | null;
  config: InstrumentConfig | null;
  setSteps: (steps: number) => void;
  setSlider: (key: string, value: number) => void;
  setEffect: <K extends keyof Effects>(key: K, value: Effects[K]) => void;
  reset: () => void;
  randomize: () => void;
  newVariation: () => void;
  loadSound: (sound: Sound) => void;

  buffer: AudioBuffer | null;
  rendering: boolean;
  /** Wall time of the last render the server actually computed (cache hits excluded). */
  renderMs: number | null;
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
  bufferFor: (sound: Sound, signal?: AbortSignal) => Promise<AudioBuffer>;
}

const StudioContext = createContext<StudioContextValue | null>(null);

function centredSound(config: InstrumentConfig): Sound {
  const sound: Sound = {
    instrument: config.instrument,
    sliders: Object.fromEntries(config.sliders.map((s) => [s.key, s.default])),
    effects: {},
  };
  if (config.backend === "diffusion") {
    sound.backend = "diffusion";
    sound.steps = config.steps?.default;
  }
  return sound;
}

/** Backends an instrument can render with; servers without the field offer the VAE when it is trained. */
export function backendsOf(info: InstrumentInfo | undefined): Backend[] {
  if (!info) return [];
  return info.backends ?? (info.trained ? ["vae"] : []);
}

/** Configs and sounds are kept per instrument and backend, so switching back restores the settings. */
function slotKey(instrument: string, backend: Backend): string {
  return backend === "vae" ? instrument : `${instrument}:${backend}`;
}

export function StudioProvider({ children }: { children: ReactNode }) {
  const { api, user, enabled: authEnabled, refreshCredits } = useAuth();
  const [status, setStatus] = useState<ApiStatus>("connecting");
  const [health, setHealth] = useState<Health | null>(null);
  const [instruments, setInstruments] = useState<InstrumentInfo[]>([]);
  const [configs, setConfigs] = useState<Record<string, InstrumentConfig>>({});
  const [sounds, setSounds] = useState<Record<string, Sound>>({});
  const [active, setActiveState] = useState<string | null>(null);
  const [chosenBackends, setChosenBackends] = useState<Record<string, Backend>>({});
  const [buffer, setBuffer] = useState<AudioBuffer | null>(null);
  const [rendering, setRendering] = useState(false);
  const [renderMs, setRenderMs] = useState<number | null>(null);
  // Diffusion renders take seconds, so auditioning every slider release is opt-in there.
  const [autoAuditionBy, setAutoAuditionBy] = useState<Record<Backend, boolean>>({ vae: true, diffusion: false });
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [evaluating, setEvaluating] = useState(false);
  const [spectrogram, setSpectrogram] = useState<Spectrogram | null>(null);
  const [exporting, setExporting] = useState(false);
  const renderSeq = useRef(0);
  // Bumped when the sound itself changes; evaluations and spectrograms of an older sound are dropped.
  const soundSeq = useRef(0);
  const inflight = useRef<{ key: string; controller: AbortController } | null>(null);
  // Event handlers can update and audition in the same tick (reset buttons,
  // effect switches). Keep that pending sound available before React renders.
  const pendingSound = useRef<Sound | null>(null);

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
        const servable = (i: InstrumentInfo) => backendsOf(i).length > 0;
        const first =
          list.instruments.find((i) => i.name === list.default && servable(i)) ?? list.instruments.find(servable);
        if (first) setActiveState(first.name);
      } catch {
        if (!cancelled) setStatus("offline");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [api]);

  const backends = useMemo(() => backendsOf(instruments.find((i) => i.name === active)), [instruments, active]);
  const backend: Backend = (active && chosenBackends[active]) || backends[0] || "vae";
  const slot = active ? slotKey(active, backend) : null;

  // Load the slider layout for the active instrument and backend on first use.
  useEffect(() => {
    if (!active || !slot || configs[slot]) return;
    let cancelled = false;
    api
      .config(active, backend)
      .then((cfg) => {
        if (cancelled) return;
        setConfigs((c) => ({ ...c, [slot]: cfg }));
        setSounds((s) => (s[slot] ? s : { ...s, [slot]: centredSound(cfg) }));
      })
      .catch((err: unknown) => {
        toast.error(err instanceof ApiError ? err.message : "Could not load the instrument.");
      });
    return () => {
      cancelled = true;
    };
  }, [active, backend, slot, api, configs]);

  const sound = slot ? (sounds[slot] ?? null) : null;
  const config = slot ? (configs[slot] ?? null) : null;
  const autoAudition = autoAuditionBy[backend];
  const setAutoAudition = useCallback(
    (on: boolean) => setAutoAuditionBy((all) => ({ ...all, [backend]: on })),
    [backend],
  );

  const bufferFor = useCallback(
    async (target: Sound, signal?: AbortSignal): Promise<AudioBuffer> => {
      const key = soundQuery(target);
      const hit = engine.cached(key);
      if (hit) return hit;
      const bytes = await api.generate(target, signal);
      const decoded = await engine.decode(bytes);
      engine.remember(key, decoded);
      return decoded;
    },
    [api],
  );

  const preview = useCallback(
    async (play = true) => {
      const target = pendingSound.current ?? sound;
      if (!target) return null;
      const seq = ++renderSeq.current;
      const key = soundQuery(target);
      // A newer sound supersedes the render in flight; the server then skips it if it is still queued.
      if (inflight.current && inflight.current.key !== key) inflight.current.controller.abort();
      const controller = new AbortController();
      inflight.current = { key, controller };
      setRendering(true);
      try {
        const fresh = !engine.cached(key);
        const started = performance.now();
        const decoded = await bufferFor(target, controller.signal);
        if (seq !== renderSeq.current) return decoded; // a newer render superseded this one
        if (fresh) setRenderMs(performance.now() - started);
        setBuffer(decoded);
        if (play) engine.play(decoded);
        return decoded;
      } catch (err) {
        if (controller.signal.aborted) return null;
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
      if (!slot) return;
      const current = pendingSound.current ?? sound;
      if (!current) return;
      const next = fn(current);
      pendingSound.current = next;
      ++renderSeq.current;
      ++soundSeq.current;
      setRendering(false);
      setSounds((all) => ({ ...all, [slot]: next }));
      setEvaluation(null);
      setSpectrogram(null);
    },
    [slot, sound],
  );

  const setSteps = useCallback(
    (steps: number) => {
      if (backend === "diffusion") update((s) => ({ ...s, steps }));
    },
    [backend, update],
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
    // Sliders and texture go back to centre; the chosen sampler steps stay.
    if (config) update((s) => ({ ...centredSound(config), ...(s.steps ? { steps: s.steps } : {}) }));
  }, [config, update]);

  const randomize = useCallback(() => {
    if (!config) return;
    update((s) => ({
      ...s,
      sliders: Object.fromEntries(config.sliders.map((d) => [d.key, Math.round(Math.random() * 100) / 100])),
    }));
  }, [config, update]);

  const newVariation = useCallback(() => {
    if (!config?.variation) return;
    update((s) => ({ ...s, seed: crypto.getRandomValues(new Uint32Array(1))[0] }));
  }, [config, update]);

  const loadSound = useCallback(
    (next: Sound) => {
      const nextBackend = next.backend ?? "vae";
      pendingSound.current = next;
      ++renderSeq.current;
      ++soundSeq.current;
      setRendering(false);
      setActiveState(next.instrument);
      setChosenBackends((all) => ({ ...all, [next.instrument]: nextBackend }));
      setSounds((all) => ({ ...all, [slotKey(next.instrument, nextBackend)]: next }));
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
    const seq = soundSeq.current;
    setEvaluating(true);
    try {
      const result = await api.evaluate(pendingSound.current ?? sound);
      if (seq === soundSeq.current) setEvaluation(result);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Evaluation failed.");
    } finally {
      setEvaluating(false);
    }
  }, [api, sound]);

  const loadSpectrogram = useCallback(async () => {
    if (!sound) return;
    const seq = soundSeq.current;
    try {
      const result = await api.spectrogram(pendingSound.current ?? sound);
      if (seq === soundSeq.current) setSpectrogram(result);
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
    pendingSound.current = null;
    ++renderSeq.current;
    ++soundSeq.current;
    setRendering(false);
    setActiveState(name);
    setEvaluation(null);
    setSpectrogram(null);
    setBuffer(null);
  }, []);

  const setBackend = useCallback(
    (next: Backend) => {
      if (!active) return;
      pendingSound.current = null;
      ++renderSeq.current;
      ++soundSeq.current;
      setRendering(false);
      setChosenBackends((all) => ({ ...all, [active]: next }));
      setEvaluation(null);
      setSpectrogram(null);
      setBuffer(null);
      setRenderMs(null);
    },
    [active],
  );

  const value = useMemo<StudioContextValue>(
    () => ({
      status, health, instruments, configs, active, setActive, backends, backend, setBackend,
      sound, config, setSteps,
      setSlider, setEffect, reset, randomize, newVariation, loadSound,
      buffer, rendering, renderMs, preview, autoAudition, setAutoAudition, commit,
      evaluation, evaluating, evaluate, spectrogram, loadSpectrogram,
      exporting, exportSample, bufferFor,
    }),
    [
      status, health, instruments, configs, active, setActive, backends, backend, setBackend,
      sound, config, setSteps,
      setSlider, setEffect, reset, randomize, newVariation, loadSound,
      buffer, rendering, renderMs, preview, autoAudition, setAutoAudition, commit,
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
