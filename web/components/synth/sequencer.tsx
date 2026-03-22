"use client";

import {
  useSequencer,
  TRACK_COLORS,
  NUM_STEPS,
} from "@/hooks/use-sequencer";
import { Slider } from "@/components/ui/slider";

interface SequencerProps {
  kickBuffer: AudioBuffer | null;
}

export function Sequencer({ kickBuffer }: SequencerProps) {
  const {
    tracks,
    pattern,
    playing,
    currentStep,
    bpm,
    midiStatus,
    setBpm,
    toggleStep,
    triggerSound,
    play,
    stop,
  } = useSequencer(kickBuffer);

  return (
    <div className="space-y-4">
      {/* Transport bar */}
      <div className="flex items-center gap-4 flex-wrap">
        <button
          onClick={playing ? stop : play}
          className="px-6 py-2.5 rounded-xl border border-violet-500/40 bg-violet-500/15 text-sm font-bold tracking-widest uppercase text-violet-300 hover:bg-violet-500/25 hover:border-violet-500/60 hover:shadow-lg hover:shadow-violet-500/20 transition-all duration-200"
        >
          {playing ? "Stop" : "Play"}
        </button>

        <div className="flex items-center gap-3 w-48">
          <label className="text-[10px] font-semibold tracking-widest uppercase text-muted-foreground whitespace-nowrap">
            BPM
          </label>
          <div className="slider-colored flex-1" style={{ "--slider-accent": "#a78bfa", "--slider-glow": "rgba(167,139,250,0.35)" } as React.CSSProperties}>
            <Slider
              min={60}
              max={200}
              step={1}
              value={[bpm]}
              onValueChange={([v]) => setBpm(v)}
            />
          </div>
          <span className="font-mono text-xs tabular-nums text-muted-foreground bg-white/5 px-2.5 py-1 rounded-lg min-w-[3rem] text-center">
            {bpm}
          </span>
        </div>

        <span className="text-[10px] font-medium tracking-wide uppercase text-muted-foreground/50 ml-auto">
          {midiStatus}
        </span>
      </div>

      {/* Step grid */}
      <div className="overflow-x-auto">
        <div className="min-w-[600px] space-y-1">
          {/* Beat markers */}
          <div className="flex items-center">
            <div className="w-20 shrink-0" />
            <div
              className="flex-1 grid gap-1"
              style={{ gridTemplateColumns: `repeat(${NUM_STEPS}, minmax(0, 1fr))` }}
            >
              {Array.from({ length: NUM_STEPS }, (_, i) => (
                <div
                  key={i}
                  className="text-center text-[9px] font-mono text-muted-foreground/40"
                >
                  {i % 4 === 0 ? i / 4 + 1 : ""}
                </div>
              ))}
            </div>
          </div>

          {/* Track rows */}
          {tracks.map((track, trackIdx) => {
            const color = TRACK_COLORS[trackIdx];
            return (
              <div key={track.type} className="flex items-center gap-2">
                {/* Track label (click to preview) */}
                <button
                  onClick={() => triggerSound(trackIdx)}
                  className="w-20 shrink-0 text-right text-[10px] font-semibold tracking-wide uppercase pr-2 hover:brightness-125 transition-all cursor-pointer select-none"
                  style={{ color }}
                  title={`Preview ${track.name}`}
                >
                  {track.name}
                </button>

                {/* Steps */}
                <div
                  className="flex-1 grid gap-1"
                  style={{ gridTemplateColumns: `repeat(${NUM_STEPS}, minmax(0, 1fr))` }}
                >
                  {Array.from({ length: NUM_STEPS }, (_, stepIdx) => {
                    const active = pattern[trackIdx][stepIdx];
                    const isCurrent = stepIdx === currentStep && playing;
                    const onBeat = stepIdx % 4 === 0;

                    return (
                      <button
                        key={stepIdx}
                        onClick={() => toggleStep(trackIdx, stepIdx)}
                        className="aspect-square rounded-[3px] border transition-all duration-75"
                        style={{
                          borderColor: onBeat
                            ? "rgba(255,255,255,0.1)"
                            : "rgba(255,255,255,0.04)",
                          background: active
                            ? color
                            : isCurrent
                              ? "rgba(255,255,255,0.08)"
                              : "rgba(255,255,255,0.02)",
                          boxShadow: active
                            ? `0 0 8px ${color}50`
                            : isCurrent
                              ? "inset 0 0 6px rgba(255,255,255,0.06)"
                              : "none",
                          opacity: active ? 1 : isCurrent ? 0.8 : 0.5,
                        }}
                      />
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Hint when no kick generated */}
      {!kickBuffer && (
        <p className="text-[10px] text-muted-foreground/40 text-center">
          Generate a kick above to hear it in the sequencer
        </p>
      )}
    </div>
  );
}
