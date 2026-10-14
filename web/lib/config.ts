/** Public configuration for the local studio and analysis app. */
export const config = {
  apiUrl: (process.env.NEXT_PUBLIC_KICKS_API_URL ?? "http://localhost:8080").replace(/\/$/, ""),
  siteName: "kicks",
} as const;

export function asset(path: string): string {
  return path.startsWith("/") ? path : `/${path}`;
}
