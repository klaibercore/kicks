"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface MidiInputInfo {
  id: string;
  name: string;
  manufacturer: string;
}

export interface NoteEvent {
  note: number;
  velocity: number; // 0..1
  channel: number;
  inputId: string;
}

export interface MidiState {
  supported: boolean;
  permission: "unknown" | "granted" | "denied" | "prompt";
  inputs: MidiInputInfo[];
  selectedId: string | "all";
  lastEvent: NoteEvent | null;
  error: string | null;
}

/**
 * Web MIDI (Chrome, Edge, Opera; Firefox behind a flag; not Safari). Note-on
 * messages from the selected input — or every input — are delivered to
 * `onNoteOn`. The hook never asks for SysEx, which keeps the permission prompt
 * minimal.
 */
export function useMidi(onNoteOn: (event: NoteEvent) => void) {
  const [state, setState] = useState<MidiState>({
    supported: false,
    permission: "unknown",
    inputs: [],
    selectedId: "all",
    lastEvent: null,
    error: null,
  });
  const accessRef = useRef<MIDIAccess | null>(null);
  const handlerRef = useRef(onNoteOn);
  useEffect(() => {
    handlerRef.current = onNoteOn;
  }, [onNoteOn]);
  const selectedRef = useRef<string>("all");

  const refreshInputs = useCallback(() => {
    const access = accessRef.current;
    if (!access) return;
    const inputs: MidiInputInfo[] = [];
    access.inputs.forEach((input) => {
      inputs.push({ id: input.id, name: input.name ?? "MIDI input", manufacturer: input.manufacturer ?? "" });
    });
    setState((s) => ({ ...s, inputs }));
  }, []);

  const attach = useCallback(
    (access: MIDIAccess) => {
      accessRef.current = access;
      access.inputs.forEach((input) => {
        input.onmidimessage = (message: MIDIMessageEvent) => {
          const data = message.data;
          if (!data || data.length < 3) return;
          const status = data[0] & 0xf0;
          const channel = data[0] & 0x0f;
          const note = data[1];
          const velocity = data[2];
          if (status !== 0x90 || velocity === 0) return; // note-on only
          if (selectedRef.current !== "all" && selectedRef.current !== input.id) return;
          const event: NoteEvent = { note, velocity: velocity / 127, channel, inputId: input.id };
          setState((s) => ({ ...s, lastEvent: event }));
          handlerRef.current(event);
        };
      });
      access.onstatechange = () => {
        refreshInputs();
        attach(access);
      };
      refreshInputs();
    },
    [refreshInputs],
  );

  const request = useCallback(async () => {
    if (typeof navigator === "undefined" || !("requestMIDIAccess" in navigator)) {
      setState((s) => ({ ...s, supported: false, error: "Web MIDI is not available in this browser." }));
      return;
    }
    try {
      const access = await navigator.requestMIDIAccess({ sysex: false });
      setState((s) => ({ ...s, supported: true, permission: "granted", error: null }));
      attach(access);
    } catch (err) {
      setState((s) => ({
        ...s,
        supported: true,
        permission: "denied",
        error: err instanceof Error ? err.message : "MIDI access was refused.",
      }));
    }
  }, [attach]);

  const select = useCallback((id: string | "all") => {
    selectedRef.current = id;
    setState((s) => ({ ...s, selectedId: id }));
  }, []);

  useEffect(() => {
    setState((s) => ({ ...s, supported: typeof navigator !== "undefined" && "requestMIDIAccess" in navigator }));
  }, []);

  return { ...state, request, select };
}
