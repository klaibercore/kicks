"use client";

import { useEffect } from "react";
import { config } from "@/lib/config";
import { useConsent } from "@/lib/consent";

/**
 * Cloudflare Web Analytics beacon. It sets no cookies and does not fingerprint,
 * but it is still a third party receiving request data, so it is only injected
 * once the visitor has opted in — and removed again when they opt out.
 */
export function CloudflareAnalytics() {
  const { consent } = useConsent();
  const enabled = Boolean(config.cloudflareToken) && consent?.analytics === true;

  useEffect(() => {
    const id = "cf-beacon";
    const existing = document.getElementById(id);
    if (!enabled) {
      existing?.remove();
      return;
    }
    if (existing) return;
    const script = document.createElement("script");
    script.id = id;
    script.defer = true;
    script.src = "https://static.cloudflareinsights.com/beacon.min.js";
    script.setAttribute("data-cf-beacon", JSON.stringify({ token: config.cloudflareToken }));
    document.body.appendChild(script);
  }, [enabled]);

  return null;
}
