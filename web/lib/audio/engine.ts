"use client";

/**
 * One AudioContext for the whole page, created on the first user gesture (the
 * autoplay policy will not let it start any earlier). Decoded previews are
 * cached by their query string so a pad that was assigned once never waits on
 * the network again.
 */
class AudioEngine {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private readonly buffers = new Map<string, AudioBuffer>();
  private readonly voices = new Set<{ source: AudioBufferSourceNode; gain: GainNode }>();
  private readonly groups = new Map<string, { source: AudioBufferSourceNode; gain: GainNode }>();

  get context(): AudioContext {
    if (!this.ctx) {
      const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new Ctor({ latencyHint: "interactive" });
      this.master = this.ctx.createGain();
      this.master.gain.value = 0.9;
      this.master.connect(this.ctx.destination);
    }
    if (this.ctx.state === "suspended") void this.ctx.resume();
    return this.ctx;
  }

  async decode(bytes: ArrayBuffer): Promise<AudioBuffer> {
    // decodeAudioData detaches the buffer, so copy in case the caller reuses it.
    return this.context.decodeAudioData(bytes.slice(0));
  }

  cached(key: string): AudioBuffer | undefined {
    return this.buffers.get(key);
  }

  remember(key: string, buffer: AudioBuffer): void {
    if (!this.buffers.has(key) && this.buffers.size >= 256) {
      const oldest = this.buffers.keys().next().value;
      if (oldest !== undefined) this.buffers.delete(oldest);
    }
    this.buffers.set(key, buffer);
  }

  /** Fire-and-forget one-shot. Velocity in [0, 1] maps to a perceptual gain curve. */
  play(buffer: AudioBuffer, velocity = 1, chokeGroup?: string): AudioBufferSourceNode {
    const ctx = this.context;
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    const gain = ctx.createGain();
    gain.gain.value = Math.pow(Math.max(0, Math.min(1, velocity)), 1.6);
    source.connect(gain);
    gain.connect(this.master!);
    const voice = { source, gain };
    if (chokeGroup) {
      const previous = this.groups.get(chokeGroup);
      if (previous) this.release(previous);
      this.groups.set(chokeGroup, voice);
    }
    this.voices.add(voice);
    source.onended = () => {
      this.voices.delete(voice);
      if (chokeGroup && this.groups.get(chokeGroup) === voice) this.groups.delete(chokeGroup);
      source.disconnect(); gain.disconnect();
    };
    source.start();
    return source;
  }

  private release(voice: { source: AudioBufferSourceNode; gain: GainNode }): void {
    const now = this.context.currentTime;
    voice.gain.gain.cancelScheduledValues(now);
    voice.gain.gain.setValueAtTime(voice.gain.gain.value, now);
    voice.gain.gain.linearRampToValueAtTime(0, now + 0.005);
    voice.source.stop(now + 0.006);
  }

  stopAll(): void {
    for (const voice of this.voices) this.release(voice);
    this.groups.clear();
  }
}

export const engine = new AudioEngine();

/** Mono peak envelope for drawing: `bins` (min, max) pairs across the buffer. */
export function peaks(buffer: AudioBuffer, bins: number): Float32Array {
  const data = buffer.getChannelData(0);
  const out = new Float32Array(bins * 2);
  const step = data.length / bins;
  for (let i = 0; i < bins; i++) {
    let min = 1;
    let max = -1;
    const start = Math.floor(i * step);
    const end = Math.min(data.length, Math.floor((i + 1) * step));
    for (let j = start; j < end; j++) {
      const v = data[j];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    out[i * 2] = min;
    out[i * 2 + 1] = max;
  }
  return out;
}
