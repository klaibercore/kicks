"use client";

import { useCallback, useEffect, useState } from "react";
import { config } from "@/lib/config";

/**
 * Consent under § 25 TDDDG / Art. 6(1)(a) GDPR for anything that is not
 * strictly necessary. Right now that is one thing: Cloudflare Web Analytics.
 * Sign-in tokens are stored regardless — they are what the visitor asked for.
 *
 * The record is versioned against the legal texts so a material change to the
 * privacy policy asks again instead of relying on stale consent.
 */
export interface ConsentState {
  version: string;
  decidedAt: string;
  analytics: boolean;
}

const KEY = "kicks.consent";
const EVENT = "kicks:consent";

export function readConsent(): ConsentState | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as ConsentState;
    return parsed.version === config.legalVersion ? parsed : null;
  } catch {
    return null;
  }
}

export function writeConsent(analytics: boolean): ConsentState {
  const state: ConsentState = {
    version: config.legalVersion,
    decidedAt: new Date().toISOString(),
    analytics,
  };
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    /* storage blocked: the banner will simply ask again next visit */
  }
  window.dispatchEvent(new CustomEvent(EVENT));
  return state;
}

export function clearConsent(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
  window.dispatchEvent(new CustomEvent(EVENT));
}

export function useConsent() {
  const [consent, setConsent] = useState<ConsentState | null | undefined>(undefined);

  useEffect(() => {
    const sync = () => setConsent(readConsent());
    sync();
    window.addEventListener(EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  const decide = useCallback((analytics: boolean) => setConsent(writeConsent(analytics)), []);
  const reset = useCallback(() => {
    clearConsent();
    setConsent(null);
  }, []);

  return { consent, decided: consent !== null && consent !== undefined, loading: consent === undefined, decide, reset };
}
