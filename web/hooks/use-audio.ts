"use client";

import { useCallback, useRef, useState } from "react";

export function useAudioPlayer() {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [waveform, setWaveform] = useState<number[] | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playingCluster, setPlayingCluster] = useState<number | null>(null);

  const loadSample = useCallback((idx: number) => {
    const audioSrc = `/api/play?idx=${idx}`;

    if (audioRef.current) {
      audioRef.current.src = audioSrc;
      audioRef.current.play().catch(() => {});
      setIsPlaying(true);
      setPlayingCluster(null);
    }

    fetch(audioSrc)
      .then((r) => r.arrayBuffer())
      .then((buf) => {
        const ctx = new AudioContext();
        return ctx.decodeAudioData(buf);
      })
      .then((audioBuf) => {
        const raw = audioBuf.getChannelData(0);
        const n = 200;
        const block = Math.floor(raw.length / n);
        const wf: number[] = [];
        for (let i = 0; i < n; i++) {
          let sum = 0;
          for (let j = 0; j < block; j++) sum += Math.abs(raw[i * block + j]);
          wf.push(sum / block);
        }
        const mx = Math.max(...wf);
        setWaveform(wf.map((v) => v / (mx || 1)));
      })
      .catch(() => setWaveform(null));
  }, []);

  const clearSample = useCallback(() => {
    setWaveform(null);
  }, []);

  const handlePlayPause = useCallback(() => {
    if (!audioRef.current) return;
    if (isPlaying) audioRef.current.pause();
    else audioRef.current.play();
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  const handlePlayCluster = useCallback((k: number) => {
    setPlayingCluster(k);
    if (audioRef.current) {
      audioRef.current.src = `/api/cluster-avg?cluster=${k}`;
      audioRef.current.play().catch(() => {});
      setIsPlaying(true);
    }
  }, []);

  const handleAudioEnded = useCallback(() => {
    setIsPlaying(false);
    setPlayingCluster(null);
  }, []);

  return {
    audioRef,
    waveform,
    isPlaying,
    playingCluster,
    loadSample,
    clearSample,
    handlePlayPause,
    handlePlayCluster,
    handleAudioEnded,
  };
}
