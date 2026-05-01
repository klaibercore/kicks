"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { SliderConfig, VocoderType } from "@/types/synth";
import { useAudioContext } from "./use-audio-context";

const API = "/api";
const PRESETS_KEY = "kicks_presets";

export interface Preset {
  name: string;
  values: number[];
  timestamp: number;
}

function loadPresetsFromStorage(): Preset[] {
  try {
    const raw = localStorage.getItem(PRESETS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function savePresetsToStorage(presets: Preset[]) {
  try {
    localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
  } catch {}
}

export function useSynth() {
  const [sliders, setSliders] = useState<SliderConfig[]>([]);
  const [values, setValues] = useState<number[]>([]);
  const [status, setStatus] = useState("Loading...");
  const [spectrogram, setSpectrogram] = useState<number[][] | null>(null);
  const [waveformData, setWaveformData] = useState<number[] | null>(null);
  const [kickBuffer, setKickBuffer] = useState<AudioBuffer | null>(null);
  const [vocoder, setVocoder] = useState<VocoderType>("bigvgan");
  const [presets, setPresets] = useState<Preset[]>([]);
  const playerRef = useRef<HTMLAudioElement>(null);
  const blobUrlRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const getCtx = useAudioContext();

  useEffect(() => {
    fetch(`${API}/config`)
      .then((r) => r.json())
      .then((data) => {
        setSliders(data.sliders);
        setValues(data.sliders.map((s: SliderConfig) => s.default));
        if (data.vocoder) setVocoder(data.vocoder);
        setStatus("");
      })
      .catch(() => setStatus("Cannot connect to backend"));
    setPresets(loadPresetsFromStorage());
  }, []);

  const generate = useCallback((vals: number[]) => {
    if (vals.length === 0) return;

    // Abort any in-flight request
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setStatus("Generating...");
    const params = vals.map((v, i) => `pc${i + 1}=${v}`).join("&");

    const audioPromise = fetch(`${API}/generate?${params}`, { signal: controller.signal })
      .then((r) => r.blob())
      .then(async (blob) => {
        if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
        const url = URL.createObjectURL(blob);
        blobUrlRef.current = url;
        const player = playerRef.current;
        if (player) {
          player.src = url;
          player.play().catch(() => {});
        }

        // Extract waveform envelope from generated audio
        const arrayBuf = await blob.arrayBuffer();
        const ctx = getCtx();
        const audioBuf = await ctx.decodeAudioData(arrayBuf);
        setKickBuffer(audioBuf);
        const raw = audioBuf.getChannelData(0);
        const n = 256;
        const block = Math.floor(raw.length / n);
        const wf: number[] = [];
        for (let i = 0; i < n; i++) {
          let sum = 0;
          for (let j = 0; j < block; j++) sum += Math.abs(raw[i * block + j]);
          wf.push(sum / block);
        }
        const mx = Math.max(...wf);
        setWaveformData(wf.map((v) => v / (mx || 1)));
      });

    const specPromise = fetch(`${API}/spectrogram?${params}`, { signal: controller.signal })
      .then((r) => r.json())
      .then((json) => setSpectrogram(json.data as number[][]));

    Promise.all([audioPromise, specPromise])
      .then(() => setStatus(""))
      .catch((err) => {
        if (err.name !== "AbortError") setStatus("Error generating");
      });
  }, [getCtx]);

  const handleSliderChange = useCallback(
    (index: number, newValue: number[]) => {
      setValues((prev) => {
        const updated = [...prev];
        updated[index] = newValue[0];
        return updated;
      });
    },
    [],
  );

  const handleGenerate = useCallback(() => {
    generate(values);
  }, [generate, values]);

  const randomize = useCallback(() => {
    if (!sliders.length) return;
    const randomized = sliders.map(() => Math.random());
    setValues(randomized);
    generate(randomized);
  }, [sliders, generate]);

  const download = useCallback(() => {
    if (!blobUrlRef.current) return;
    const a = document.createElement("a");
    a.href = blobUrlRef.current;
    a.download = "kick.wav";
    a.click();
  }, []);

  // ── Presets ──────────────────────────────────────────

  const savePreset = useCallback((name: string) => {
    const trimmed = name.trim();
    if (!trimmed || values.length === 0) return;
    setPresets((prev) => {
      const filtered = prev.filter((p) => p.name !== trimmed);
      const updated = [...filtered, { name: trimmed, values: [...values], timestamp: Date.now() }];
      savePresetsToStorage(updated);
      return updated;
    });
  }, [values]);

  const loadPreset = useCallback((name: string) => {
    setPresets((prev) => {
      const preset = prev.find((p) => p.name === name);
      if (preset) {
        setValues(preset.values);
        generate(preset.values);
      }
      return prev;
    });
  }, [generate]);

  const deletePreset = useCallback((name: string) => {
    setPresets((prev) => {
      const updated = prev.filter((p) => p.name !== name);
      savePresetsToStorage(updated);
      return updated;
    });
  }, []);

  // ── Keyboard shortcuts ───────────────────────────────

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      // Ignore when typing in inputs
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (e.key === " " || e.code === "Space") {
        e.preventDefault();
        handleGenerate();
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        randomize();
      } else if (e.key === "s" || e.key === "S") {
        e.preventDefault();
        download();
      }
    }

    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [handleGenerate, randomize, download]);

  return {
    sliders,
    values,
    status,
    playerRef,
    spectrogram,
    waveformData,
    kickBuffer,
    vocoder,
    presets,
    handleSliderChange,
    handleGenerate,
    randomize,
    download,
    savePreset,
    loadPreset,
    deletePreset,
  };
}
