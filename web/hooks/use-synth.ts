"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { SliderConfig } from "@/types/synth";

const API = "/api";

export function useSynth() {
  const [sliders, setSliders] = useState<SliderConfig[]>([]);
  const [values, setValues] = useState<number[]>([]);
  const [status, setStatus] = useState("Loading...");
  const [spectrogram, setSpectrogram] = useState<number[][] | null>(null);
  const [waveformData, setWaveformData] = useState<number[] | null>(null);
  const playerRef = useRef<HTMLAudioElement>(null);
  const blobUrlRef = useRef<string | null>(null);

  useEffect(() => {
    fetch(`${API}/config`)
      .then((r) => r.json())
      .then((data) => {
        setSliders(data.sliders);
        setValues(data.sliders.map((s: SliderConfig) => s.default));
        setStatus("");
      })
      .catch(() => setStatus("Cannot connect to backend"));
  }, []);

  const generate = useCallback((vals: number[]) => {
    if (vals.length === 0) return;
    setStatus("Generating...");
    const params = vals.map((v, i) => `pc${i + 1}=${v}`).join("&");

    const audioPromise = fetch(`${API}/generate?${params}`)
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
        const actx = new AudioContext();
        const audioBuf = await actx.decodeAudioData(arrayBuf);
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

    const specPromise = fetch(`${API}/spectrogram?${params}`)
      .then((r) => r.json())
      .then((json) => setSpectrogram(json.data as number[][]));

    Promise.all([audioPromise, specPromise])
      .then(() => setStatus(""))
      .catch(() => setStatus("Error generating"));
  }, []);

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

  return {
    sliders,
    values,
    status,
    playerRef,
    spectrogram,
    waveformData,
    handleSliderChange,
    handleGenerate,
    randomize,
    download,
  };
}
