"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useAudioContext } from "./use-audio-context";

export const NUM_STEPS = 16;

export const TRACKS = [
  { name: "Kick", type: "kick" },
  { name: "Snare", type: "snare-1" },
  { name: "Snare 2", type: "snare-2" },
  { name: "Clap", type: "clap" },
  { name: "Closed HH", type: "hat-closed" },
  { name: "Closed HH 2", type: "hat-closed-2" },
  { name: "Open HH", type: "hat-open" },
] as const;

export const TRACK_COLORS = [
  "#a78bfa", // kick - violet
  "#f472b6", // snare 1 - pink
  "#ec4899", // snare 2 - hot pink
  "#f9a8d4", // clap - light pink
  "#34d399", // closed hh - emerald
  "#6ee7b7", // closed hh 2 - light emerald
  "#10b981", // open hh - green
];

// GM drum note -> track index
const MIDI_MAP: Record<number, number> = {
  36: 0, // C1 - bass drum 1
  35: 0, // B0 - bass drum 2
  38: 1, // D1 - acoustic snare
  40: 2, // E1 - electric snare
  39: 3, // D#1 - hand clap
  37: 3, // C#1 - side stick (alt clap)
  42: 4, // F#1 - closed hi-hat
  44: 5, // G#1 - pedal hi-hat
  46: 6, // A#1 - open hi-hat
};

type Pattern = boolean[][];

const LOOKAHEAD_MS = 25;
const SCHEDULE_AHEAD_S = 0.1;

// ── Shared noise buffer ──────────────────────────────────

let sharedNoiseBuf: AudioBuffer | null = null;

function getNoise(ctx: AudioContext): AudioBuffer {
  if (!sharedNoiseBuf || sharedNoiseBuf.sampleRate !== ctx.sampleRate) {
    const len = ctx.sampleRate; // 1 second
    sharedNoiseBuf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = sharedNoiseBuf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
  }
  return sharedNoiseBuf;
}

// ── Sound synthesis ──────────────────────────────────────

function playKick(
  ctx: AudioContext,
  buf: AudioBuffer | null,
  t: number,
) {
  if (!buf) return;
  const src = ctx.createBufferSource();
  src.buffer = buf;
  const g = ctx.createGain();
  g.gain.value = 0.9;
  src.connect(g).connect(ctx.destination);
  src.start(t);
}

function playSnare1(ctx: AudioContext, t: number) {
  // Tight snare: short noise + sine body
  const dur = 0.15;
  const g = ctx.createGain();
  g.gain.setValueAtTime(0.55, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + dur);
  g.connect(ctx.destination);

  const n = ctx.createBufferSource();
  n.buffer = getNoise(ctx);
  const bp = ctx.createBiquadFilter();
  bp.type = "bandpass";
  bp.frequency.value = 3000;
  bp.Q.value = 0.8;
  n.connect(bp).connect(g);
  n.start(t);
  n.stop(t + dur);

  const osc = ctx.createOscillator();
  osc.type = "sine";
  osc.frequency.setValueAtTime(180, t);
  osc.frequency.exponentialRampToValueAtTime(80, t + 0.05);
  const bg = ctx.createGain();
  bg.gain.setValueAtTime(0.45, t);
  bg.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
  osc.connect(bg).connect(ctx.destination);
  osc.start(t);
  osc.stop(t + dur);
}

function playSnare2(ctx: AudioContext, t: number) {
  // Fat / loose snare: longer noise + lower body
  const dur = 0.25;
  const g = ctx.createGain();
  g.gain.setValueAtTime(0.5, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + dur);
  g.connect(ctx.destination);

  const n = ctx.createBufferSource();
  n.buffer = getNoise(ctx);
  const bp = ctx.createBiquadFilter();
  bp.type = "bandpass";
  bp.frequency.value = 2200;
  bp.Q.value = 0.6;
  n.connect(bp).connect(g);
  n.start(t);
  n.stop(t + dur);

  const osc = ctx.createOscillator();
  osc.type = "sine";
  osc.frequency.setValueAtTime(150, t);
  osc.frequency.exponentialRampToValueAtTime(60, t + 0.07);
  const bg = ctx.createGain();
  bg.gain.setValueAtTime(0.55, t);
  bg.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
  osc.connect(bg).connect(ctx.destination);
  osc.start(t);
  osc.stop(t + dur);
}

function playClap(ctx: AudioContext, t: number) {
  // Layered noise bursts
  for (let layer = 0; layer < 3; layer++) {
    const off = layer * 0.008;
    const dur = 0.1;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.35, t + off);
    g.gain.exponentialRampToValueAtTime(0.001, t + off + dur);
    g.connect(ctx.destination);

    const n = ctx.createBufferSource();
    n.buffer = getNoise(ctx);
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass";
    bp.frequency.value = 4000;
    bp.Q.value = 1.2;
    n.connect(bp).connect(g);
    n.start(t + off);
    n.stop(t + off + dur);
  }

  const osc = ctx.createOscillator();
  osc.type = "triangle";
  osc.frequency.setValueAtTime(300, t);
  osc.frequency.exponentialRampToValueAtTime(150, t + 0.04);
  const bg = ctx.createGain();
  bg.gain.setValueAtTime(0.3, t);
  bg.gain.exponentialRampToValueAtTime(0.001, t + 0.06);
  osc.connect(bg).connect(ctx.destination);
  osc.start(t);
  osc.stop(t + 0.1);
}

function playHatClosed(ctx: AudioContext, t: number) {
  const dur = 0.04;
  const g = ctx.createGain();
  g.gain.setValueAtTime(0.35, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + dur);
  g.connect(ctx.destination);

  const n = ctx.createBufferSource();
  n.buffer = getNoise(ctx);
  const hp = ctx.createBiquadFilter();
  hp.type = "highpass";
  hp.frequency.value = 8000;
  const bp = ctx.createBiquadFilter();
  bp.type = "bandpass";
  bp.frequency.value = 10000;
  bp.Q.value = 1.5;
  n.connect(hp).connect(bp).connect(g);
  n.start(t);
  n.stop(t + dur);
}

function playHatClosed2(ctx: AudioContext, t: number) {
  const dur = 0.08;
  const g = ctx.createGain();
  g.gain.setValueAtTime(0.3, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + dur);
  g.connect(ctx.destination);

  const n = ctx.createBufferSource();
  n.buffer = getNoise(ctx);
  const hp = ctx.createBiquadFilter();
  hp.type = "highpass";
  hp.frequency.value = 6000;
  const bp = ctx.createBiquadFilter();
  bp.type = "bandpass";
  bp.frequency.value = 8500;
  bp.Q.value = 1.0;
  n.connect(hp).connect(bp).connect(g);
  n.start(t);
  n.stop(t + dur);
}

function playHatOpen(ctx: AudioContext, t: number) {
  const dur = 0.35;
  const g = ctx.createGain();
  g.gain.setValueAtTime(0.3, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + dur);
  g.connect(ctx.destination);

  const n = ctx.createBufferSource();
  n.buffer = getNoise(ctx);
  const hp = ctx.createBiquadFilter();
  hp.type = "highpass";
  hp.frequency.value = 5000;
  const bp = ctx.createBiquadFilter();
  bp.type = "bandpass";
  bp.frequency.value = 8000;
  bp.Q.value = 0.7;
  n.connect(hp).connect(bp).connect(g);
  n.start(t);
  n.stop(t + dur);
}

const PLAY_FNS: ((ctx: AudioContext, t: number) => void)[] = [
  () => {}, // kick handled separately
  playSnare1,
  playSnare2,
  playClap,
  playHatClosed,
  playHatClosed2,
  playHatOpen,
];

// ── Hook ─────────────────────────────────────────────────

function createEmptyPattern(): Pattern {
  return TRACKS.map(() => Array(NUM_STEPS).fill(false));
}

export function useSequencer(kickBuffer: AudioBuffer | null) {
  const [pattern, setPattern] = useState<Pattern>(createEmptyPattern);
  const [playing, setPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  const [bpm, setBpm] = useState(120);
  const [midiStatus, setMidiStatus] = useState("No MIDI");

  const timerRef = useRef<number | null>(null);
  const nextNoteTimeRef = useRef(0);
  const stepRef = useRef(0);
  const patternRef = useRef(pattern);
  const kickBufRef = useRef(kickBuffer);
  const playingRef = useRef(false);
  const bpmRef = useRef(bpm);
  const getCtx = useAudioContext();

  useEffect(() => {
    patternRef.current = pattern;
  }, [pattern]);
  useEffect(() => {
    kickBufRef.current = kickBuffer;
  }, [kickBuffer]);
  useEffect(() => {
    bpmRef.current = bpm;
  }, [bpm]);

  const triggerSound = useCallback(
    (trackIndex: number) => {
      const ctx = getCtx();
      if (trackIndex === 0) {
        playKick(ctx, kickBufRef.current, ctx.currentTime);
      } else {
        PLAY_FNS[trackIndex]?.(ctx, ctx.currentTime);
      }
    },
    [getCtx],
  );

  const toggleStep = useCallback((track: number, step: number) => {
    setPattern((prev) => {
      const next = prev.map((row) => [...row]);
      next[track][step] = !next[track][step];
      return next;
    });
  }, []);

  // ── Lookahead scheduler ──────────────────────────────

  const tick = useCallback(() => {
    if (!playingRef.current) return;
    const ctx = getCtx();

    while (nextNoteTimeRef.current < ctx.currentTime + SCHEDULE_AHEAD_S) {
      const step = stepRef.current;
      const t = nextNoteTimeRef.current;

      for (let track = 0; track < TRACKS.length; track++) {
        if (patternRef.current[track][step]) {
          if (track === 0) playKick(ctx, kickBufRef.current, t);
          else PLAY_FNS[track]?.(ctx, t);
        }
      }

      // Schedule visual update close to the actual audio time
      const delay = Math.max(0, (t - ctx.currentTime) * 1000);
      const s = step;
      setTimeout(() => setCurrentStep(s), delay);

      nextNoteTimeRef.current += 60 / bpmRef.current / 4; // 16th notes
      stepRef.current = (step + 1) % NUM_STEPS;
    }

    timerRef.current = window.setTimeout(tick, LOOKAHEAD_MS);
  }, [getCtx]);

  const play = useCallback(() => {
    const ctx = getCtx();
    playingRef.current = true;
    setPlaying(true);
    stepRef.current = 0;
    nextNoteTimeRef.current = ctx.currentTime;
    tick();
  }, [getCtx, tick]);

  const stop = useCallback(() => {
    playingRef.current = false;
    setPlaying(false);
    setCurrentStep(-1);
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // ── MIDI input ───────────────────────────────────────

  useEffect(() => {
    if (!navigator.requestMIDIAccess) {
      setMidiStatus("MIDI not supported");
      return;
    }

    let disposed = false;

    navigator.requestMIDIAccess().then(
      (access) => {
        if (disposed) return;

        function bindInputs() {
          const inputs = Array.from(access.inputs.values());
          if (inputs.length > 0) {
            setMidiStatus(`MIDI: ${inputs[0].name}`);
          } else {
            setMidiStatus("No MIDI devices");
          }
          for (const input of inputs) {
            input.onmidimessage = handleMessage;
          }
        }

        function handleMessage(e: MIDIMessageEvent) {
          if (!e.data || e.data.length < 3) return;
          const [status, note, velocity] = e.data;
          const cmd = status & 0xf0;
          // Note on with velocity > 0
          if (cmd === 0x90 && velocity > 0) {
            const trackIdx = MIDI_MAP[note];
            if (trackIdx !== undefined) triggerSound(trackIdx);
          }
        }

        bindInputs();
        access.onstatechange = () => {
          if (!disposed) bindInputs();
        };
      },
      () => setMidiStatus("MIDI access denied"),
    );

    return () => {
      disposed = true;
    };
  }, [triggerSound]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current);
    };
  }, []);

  return {
    tracks: TRACKS,
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
  };
}
