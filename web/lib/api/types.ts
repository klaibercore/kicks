export type InstrumentName = string;

export interface InstrumentInfo {
  name: InstrumentName;
  display_name: string;
  description: string;
  descriptors: string[];
  trained: boolean;
  loaded: boolean;
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
  vocoder: string;
  control: "pca" | "descriptor";
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
  score?: number;
  grade?: string;
  likeness_pct?: number;
  verdicts?: Verdict[];
  metrics?: Record<string, number>;
  descriptors: Record<string, number>;
  error?: string;
}

export interface Spectrogram {
  instrument: InstrumentName;
  shape: [number, number];
  data: number[][];
}

export interface Health {
  status: string;
  device: string;
  vocoder: string;
  control: string;
  loaded: string[];
  cached_responses: number;
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
  /** Slider key -> position in [0, 1]. Keys come from InstrumentConfig.sliders. */
  sliders: Record<string, number>;
  effects: Effects;
}
