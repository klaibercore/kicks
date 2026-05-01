"use client";

import { useCallback, useEffect, useRef } from "react";

let _sharedCtx: AudioContext | null = null;
let _sharedRefCount = 0;

function _getSharedCtx(): AudioContext {
  if (!_sharedCtx || _sharedCtx.state === "closed") {
    _sharedCtx = new AudioContext();
    _sharedRefCount = 0;
  }
  _sharedRefCount++;
  return _sharedCtx;
}

function _releaseSharedCtx() {
  _sharedRefCount--;
  if (_sharedRefCount <= 0 && _sharedCtx) {
    _sharedCtx.close().catch(() => {});
    _sharedCtx = null;
    _sharedRefCount = 0;
  }
}

/**
 * Shared AudioContext hook — reuses a single AudioContext across all
 * components. Creates lazily, releases on unmount, and handles suspended
 * state (user-gesture resume).
 */
export function useAudioContext() {
  const ctxRef = useRef<AudioContext | null>(null);
  const ownedRef = useRef(false);

  const getCtx = useCallback((): AudioContext => {
    if (!ctxRef.current || ctxRef.current.state === "closed") {
      ctxRef.current = _getSharedCtx();
      ownedRef.current = true;
    }
    if (ctxRef.current.state === "suspended") {
      ctxRef.current.resume().catch(() => {});
    }
    return ctxRef.current;
  }, []);

  useEffect(() => {
    return () => {
      if (ownedRef.current) {
        _releaseSharedCtx();
        ownedRef.current = false;
        ctxRef.current = null;
      }
    };
  }, []);

  return getCtx;
}
