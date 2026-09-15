"use client";

import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { config, hasSupabase } from "@/lib/config";

let client: SupabaseClient | null = null;

/**
 * The browser Supabase client, or null when the site is built without a
 * project (local development against the API alone still works).
 *
 * PKCE is the OAuth flow for public clients: the code lands on /auth/callback
 * and is exchanged from the browser with a verifier only this browser holds.
 */
export function getSupabase(): SupabaseClient | null {
  if (!hasSupabase) return null;
  if (typeof window === "undefined") return null;
  if (!client) {
    client = createClient(config.supabase.url, config.supabase.anonKey, {
      auth: {
        flowType: "pkce",
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: false,
      },
    });
  }
  return client;
}
