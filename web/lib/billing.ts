"use client";

import { config } from "@/lib/config";
import { getSupabase } from "@/lib/supabase/client";
import type { Price, Product } from "@/lib/supabase/types";

export interface Pack {
  price: Price;
  product: Product;
}

/** Active credit packs from the Stripe catalogue mirror, cheapest first. */
export async function loadPacks(): Promise<Pack[]> {
  const sb = getSupabase();
  if (!sb) return [];
  const { data, error } = await sb
    .from("prices")
    .select("*, products(*)")
    .eq("active", true)
    .eq("type", "one_time")
    .order("unit_amount", { ascending: true });
  if (error) throw new Error(error.message);
  return ((data ?? []) as Array<Price & { products: Product | null }>)
    .filter((row) => row.products?.active)
    .map((row) => ({ price: row, product: row.products! }));
}

async function invoke<T>(name: string, body: Record<string, unknown>): Promise<T> {
  const sb = getSupabase();
  if (!sb) throw new Error("Accounts are not configured on this site.");
  const { data, error } = await sb.functions.invoke<T & { error?: string }>(name, { body });
  if (error) {
    // Edge Functions return their message in the JSON body.
    let detail = error.message;
    try {
      const ctx = (error as { context?: Response }).context;
      if (ctx) detail = (await ctx.json()).error ?? detail;
    } catch {
      /* keep the generic message */
    }
    throw new Error(detail);
  }
  if (data && typeof data === "object" && "error" in data && data.error) throw new Error(String(data.error));
  return data as T;
}

/** Redirects to Stripe Checkout. Resolves only if the redirect could not happen. */
export async function startCheckout(priceId: string): Promise<void> {
  const { url } = await invoke<{ url: string }>("create-checkout-session", {
    price_id: priceId,
    withdrawal_waiver: true,
    legal_version: config.legalVersion,
  });
  window.location.assign(url);
}

export async function openBillingPortal(): Promise<void> {
  const { url } = await invoke<{ url: string }>("billing-portal", {});
  window.location.assign(url);
}

export async function deleteAccount(): Promise<void> {
  await invoke<{ ok: true }>("delete-account", { confirm: "DELETE" });
}

export function formatMoney(minor: number, currency: string, locale = "de-DE"): string {
  return new Intl.NumberFormat(locale, { style: "currency", currency: currency.toUpperCase() }).format(minor / 100);
}
