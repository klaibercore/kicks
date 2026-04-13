export interface SliderConfig {
  id: number;
  name: string;
  min: number;
  max: number;
  default: number;
  step: number;
}

export type VocoderType = "bigvgan" | "griffinlim";

export const SLIDER_COLORS = [
  { accent: "#a78bfa", glow: "rgba(167,139,250,0.35)" },
  { accent: "#f472b6", glow: "rgba(244,114,182,0.35)" },
  { accent: "#34d399", glow: "rgba(52,211,153,0.35)" },
  { accent: "#fbbf24", glow: "rgba(251,191,36,0.35)" },
  { accent: "#22d3ee", glow: "rgba(34,211,238,0.35)" },
];
