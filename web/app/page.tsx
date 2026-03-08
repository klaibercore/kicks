"use client";

import { Slider } from "@/components/ui/slider";
import { SLIDER_COLORS } from "@/types/synth";
import { WaveBackground } from "@/components/synth/wave-background";
import { WaveformVis } from "@/components/synth/waveform-vis";
import { useSynth } from "@/hooks/use-synth";

export default function Home() {
  const {
    sliders,
    values,
    status,
    playerRef,
    handleSliderChange,
    handleGenerate,
    randomize,
    download,
  } = useSynth();

  return (
    <>
      <WaveBackground />

      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center p-6 sm:p-8">
        <div className="w-full max-w-lg">
          {/* Header */}
          <div className="mb-10 text-center">
            <h1 className="text-5xl sm:text-6xl font-black tracking-tighter mb-3 bg-gradient-to-r from-violet-400 via-pink-400 to-emerald-400 bg-clip-text text-transparent">
              Kick Synth
            </h1>
            <p className="text-muted-foreground text-sm sm:text-base leading-relaxed max-w-sm mx-auto">
              Neural kick drum synthesizer powered by a variational autoencoder.
              Shape your sound with four latent dimensions.
            </p>
          </div>

          {/* Main Card */}
          <div className="rounded-2xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-xl p-6 sm:p-8 space-y-7 shadow-2xl shadow-violet-500/5">
            {/* Sliders */}
            <div className="space-y-5">
              {sliders.map((s, i) => {
                const color = SLIDER_COLORS[i % SLIDER_COLORS.length];
                return (
                  <div key={s.id} className="space-y-2.5">
                    <div className="flex items-center justify-between">
                      <label
                        className="text-sm font-semibold tracking-wide uppercase"
                        style={{ color: color.accent }}
                      >
                        {s.name}
                      </label>
                      <span className="font-mono text-xs tabular-nums text-muted-foreground bg-white/5 px-2.5 py-1 rounded-lg">
                        {values[i]?.toFixed(2)}
                      </span>
                    </div>
                    <div
                      className="slider-colored"
                      style={
                        {
                          "--slider-accent": color.accent,
                          "--slider-glow": color.glow,
                        } as React.CSSProperties
                      }
                    >
                      <Slider
                        min={s.min}
                        max={s.max}
                        step={s.step}
                        value={[values[i] ?? s.default]}
                        onValueChange={(v) => handleSliderChange(i, v)}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Generate Button */}
            <button
              onClick={handleGenerate}
              disabled={status === "Generating..." || sliders.length === 0}
              className="w-full py-3.5 rounded-xl border border-pink-500/40 bg-pink-500/15 text-base font-bold tracking-widest uppercase text-pink-300 hover:bg-pink-500/25 hover:border-pink-500/60 hover:shadow-lg hover:shadow-pink-500/20 disabled:opacity-50 disabled:cursor-wait transition-all duration-200"
            >
              {status === "Generating..." ? "Generating..." : "Generate"}
            </button>

            {/* Action Buttons */}
            <div className="flex gap-3">
              <button
                onClick={randomize}
                className="flex-1 py-2.5 rounded-xl border border-violet-500/30 bg-violet-500/10 text-sm font-semibold tracking-wide uppercase text-violet-300 hover:bg-violet-500/20 hover:border-violet-500/50 transition-all duration-200"
              >
                Randomise
              </button>
              <button
                onClick={download}
                className="flex-1 py-2.5 rounded-xl border border-emerald-500/30 bg-emerald-500/10 text-sm font-semibold tracking-wide uppercase text-emerald-300 hover:bg-emerald-500/20 hover:border-emerald-500/50 transition-all duration-200"
              >
                Download WAV
              </button>
            </div>

            {/* Divider */}
            <div className="border-t border-white/[0.06]" />

            {/* Waveform Visualiser */}
            <div className="space-y-3">
              <label className="text-sm font-semibold tracking-wide uppercase text-muted-foreground">
                Output
              </label>
              <div className="rounded-xl bg-black/30 border border-white/[0.06] p-3 space-y-3">
                <WaveformVis audioRef={playerRef} />
                <audio
                  ref={playerRef}
                  controls
                  className="w-full audio-player"
                />
              </div>
            </div>

            {/* Status */}
            <div className="h-5 flex items-center justify-center gap-2">
              {status && (
                <>
                  {status === "Generating..." && (
                    <div className="size-3 rounded-full border-2 border-violet-400/40 border-t-violet-400 animate-spin" />
                  )}
                  <p className="text-xs text-muted-foreground">{status}</p>
                </>
              )}
            </div>
          </div>

          {/* Footer */}
          <div className="mt-8 text-center">
            <p className="text-xs text-muted-foreground/50">
              &copy; {new Date().getFullYear()} Kevin Paul Klaiber
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
