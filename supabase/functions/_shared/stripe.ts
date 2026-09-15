import Stripe from "npm:stripe@18.5.0";

export const stripe = new Stripe(Deno.env.get("STRIPE_SECRET_KEY") ?? "", {
  apiVersion: "2025-08-27.basil",
  httpClient: Stripe.createFetchHttpClient(),
  appInfo: { name: "kicks", url: "https://github.com/klaibercore/kicks" },
});

export const cryptoProvider = Stripe.createSubtleCryptoProvider();

export function siteUrl(path: string): string {
  const base = (Deno.env.get("SITE_URL") ?? "http://localhost:3000").replace(/\/$/, "");
  const prefix = Deno.env.get("SITE_BASE_PATH") ?? "";
  return `${base}${prefix}${path}`;
}
