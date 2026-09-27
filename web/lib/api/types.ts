export type InstrumentName = string;

/** How a sound is synthesised: the VAE and a vocoder, or the waveform diffusion model. */
export type Backend = "vae" | "diffusion";

export interface InstrumentInfo {
  name: InstrumentName;
  display_name: string;
  description: string;
  descriptors: string[];
  trained: boolean;
  loaded: boolean;
  /** Backends this server can render the instrument with. Older servers omit it. */
  backends?: Backend[];
}

export interface InstrumentsResponse {
  instruments: InstrumentInfo[];
  default: InstrumentName;
}

export interface SliderDef {
  id: number;
  key: string;
  name: string;
  min: number;
  max: number;
  default: number;
  step: number;
  description?: string;
  target_min?: number;
  target_max?: number;
}

export interface InstrumentConfig {
  instrument: InstrumentName;
  display_name: string;
  description: string;
  sliders: SliderDef[];
  /** Older servers omit it and serve the VAE only. */
  backend?: Backend;
  /** The vocoder this instrument renders with; "none" for diffusion. */
  vocoder: string;
  control: "pca" | "descriptor";
  /** Supports a stable drum texture independently of slider settings. */
  variation?: boolean;
  /** Diffusion only: the sampler-step range the server accepts. */
  steps?: { default: number; min: number; max: number };
  /** Diffusion only: which checkpoint is served. */
  checkpoint?: { name: string; epoch: number | null; run_id: string | null };
}

export interface Verdict {
  metric: string;
  symbol: string;
  text: string;
  percentile: number | null;
  z: number | null;
}

export interface Evaluation {
  instrument: InstrumentName;
  backend?: Backend;
  score?: number;
  grade?: string;
  likeness_pct?: number;
  verdicts?: Verdict[];
  metrics?: Record<string, number>;
  descriptors: Record<string, number>;
  control_fit?: {
    max_normalized_error: number;
    limited: boolean;
    targets: Record<string, number>;
    seed: number;
    includes_effects: boolean;
  } | null;
  error?: string;
}

export interface Spectrogram {
  instrument: InstrumentName;
  /** "decoder": the VAE's output before the vocoder. "render": measured from the rendered audio. */
  source?: "decoder" | "render";
  shape: [number, number];
  data: number[][];
}

export interface Health {
  status: string;
  device: string;
  /** A forced backend, or "profile" when each instrument renders with its own. */
  vocoder: string;
  /** Loaded instrument -> backend actually in use. */
  vocoders?: Record<string, string>;
  control: string;
  loaded: string[];
  cached_responses: number;
  /** Instruments served with a diffusion checkpoint. */
  diffusion?: string[];
  auth: { mode: "required" | "optional" | "off"; billing: boolean };
}

export interface Me {
  id: string;
  email: string | null;
  credits: number | null;
}

/** Effects the API applies after the vocoder. Undefined = not sent = off. */
export interface Effects {
  attack_ms?: number;
  decay_ms?: number;
  drive?: number;
  filter?: number;
}

/** A complete, reproducible sound: which instrument, where every slider sits, which effects. */
export interface Sound {
  instrument: InstrumentName;
  /** Omitted for the VAE, which older presets always used. */
  backend?: Backend;
  /** Diffusion sampler steps; omitted means the server default. */
  steps?: number;
  /** Omitted in older presets; the server treats it as the default texture. */
  seed?: number;
  /** Slider key -> position in [0, 1]. Keys come from InstrumentConfig.sliders. */
  sliders: Record<string, number>;
  effects: Effects;
}
