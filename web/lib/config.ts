/**
 * Build-time configuration. Everything here is public: the site is a static
 * export, so there is no server to keep a secret on. Secrets live in Supabase
 * Edge Function environment variables and in the kicks API's environment.
 */

export const config = {
  apiUrl: (process.env.NEXT_PUBLIC_KICKS_API_URL ?? "http://localhost:8080").replace(/\/$/, ""),
  basePath: process.env.NEXT_PUBLIC_BASE_PATH ?? "",
  supabase: {
    url: process.env.NEXT_PUBLIC_SUPABASE_URL ?? "",
    anonKey: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "",
  },
  cloudflareToken: process.env.NEXT_PUBLIC_CF_ANALYTICS_TOKEN ?? "",
  siteName: "kicks",
  /** Bumped whenever the legal texts change; consents are recorded against it. */
  legalVersion: "2026-09-15",
} as const;

export const hasSupabase = Boolean(config.supabase.url && config.supabase.anonKey);

/** Prefix a public asset path with the base path for GitHub Pages sub-paths. */
export function asset(path: string): string {
  return `${config.basePath}${path.startsWith("/") ? path : `/${path}`}`;
}

/** Absolute URL of a route on this site — used for OAuth and Stripe redirects. */
export function siteUrl(path: string): string {
  if (typeof window === "undefined") return path;
  return `${window.location.origin}${config.basePath}${path}`;
}
