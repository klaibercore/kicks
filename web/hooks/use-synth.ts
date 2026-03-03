"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { SliderConfig } from "@/types/synth";

const API = "/api";

export function useSynth() {
  const [sliders, setSliders] = useState<SliderConfig[]>([]);
  const [values, setValues] = useState<number[]>([]);
  const [status, setStatus] = useState("Loading...");
  const playerRef = useRef<HTMLAudioElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const blobUrlRef = useRef<string | null>(null);
  const initialGenDone = useRef(false);

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
    fetch(`${API}/generate?${params}`)
      .then((r) => r.blob())
      .then((blob) => {
        if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
        const url = URL.createObjectURL(blob);
        blobUrlRef.current = url;
        const player = playerRef.current;
        if (player) {
          player.src = url;
          player.play().catch(() => {});
        }
        setStatus("");
      })
      .catch(() => setStatus("Error generating"));
  }, []);

  useEffect(() => {
    if (values.length > 0 && !initialGenDone.current) {
      initialGenDone.current = true;
      generate(values);
    }
  }, [values, generate]);

  const handleSliderChange = useCallback(
    (index: number, newValue: number[]) => {
      setValues((prev) => {
        const updated = [...prev];
        updated[index] = newValue[0];

        if (debounceRef.current) clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(() => generate(updated), 150);

        return updated;
      });
    },
    [generate],
  );

  const randomize = useCallback(() => {
    if (!sliders.length) return;
    const randomized = sliders.map(() => Math.random());
    setValues(randomized);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => generate(randomized), 150);
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
    handleSliderChange,
    randomize,
    download,
  };
}
