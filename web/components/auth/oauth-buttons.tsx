"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { siteUrl } from "@/lib/config";
import { getSupabase } from "@/lib/supabase/client";

type Provider = "google" | "github";

const providers: Array<{ id: Provider; label: string; icon: React.ReactNode }> = [
  {
    id: "github",
    label: "Continue with GitHub",
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden className="size-4 fill-current">
        <path d="M12 .5C5.7.5.5 5.7.5 12c0 5.1 3.3 9.4 7.9 10.9.6.1.8-.3.8-.6v-2c-3.2.7-3.9-1.4-3.9-1.4-.5-1.3-1.3-1.7-1.3-1.7-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.7 1.3 3.4 1 .1-.8.4-1.3.7-1.6-2.6-.3-5.3-1.3-5.3-5.7 0-1.3.5-2.3 1.2-3.1-.1-.3-.5-1.5.1-3.1 0 0 1-.3 3.2 1.2a11 11 0 0 1 5.8 0c2.2-1.5 3.2-1.2 3.2-1.2.6 1.6.2 2.8.1 3.1.8.8 1.2 1.8 1.2 3.1 0 4.4-2.7 5.4-5.3 5.7.4.4.8 1.1.8 2.2v3.2c0 .3.2.7.8.6 4.6-1.5 7.9-5.8 7.9-10.9C23.5 5.7 18.3.5 12 .5z" />
      </svg>
    ),
  },
  {
    id: "google",
    label: "Continue with Google",
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden className="size-4">
        <path fill="#4285F4" d="M23 12.3c0-.8-.1-1.6-.2-2.3H12v4.5h6.2a5.3 5.3 0 0 1-2.3 3.5v2.9h3.7c2.2-2 3.4-5 3.4-8.6z" />
        <path fill="#34A853" d="M12 24c3.1 0 5.7-1 7.6-2.8l-3.7-2.9c-1 .7-2.3 1.1-3.9 1.1-3 0-5.5-2-6.4-4.7H1.8v3C3.7 21.4 7.5 24 12 24z" />
        <path fill="#FBBC05" d="M5.6 14.7a7.2 7.2 0 0 1 0-4.6V7.1H1.8a12 12 0 0 0 0 10.8l3.8-3.2z" />
        <path fill="#EA4335" d="M12 4.8c1.7 0 3.2.6 4.4 1.7l3.3-3.3C17.7 1.3 15.1.2 12 .2 7.5.2 3.7 2.8 1.8 7.1l3.8 3c.9-2.7 3.4-5.3 6.4-5.3z" />
      </svg>
    ),
  },
];

export function OAuthButtons({ next }: { next: string }) {
  const [busy, setBusy] = useState<Provider | null>(null);

  const start = async (provider: Provider) => {
    const sb = getSupabase();
    if (!sb) return;
    setBusy(provider);
    const { error } = await sb.auth.signInWithOAuth({
      provider,
      options: { redirectTo: siteUrl(`/auth/callback/?next=${encodeURIComponent(next)}`) },
    });
    if (error) {
      toast.error(error.message);
      setBusy(null);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      {providers.map((p) => (
        <Button key={p.id} variant="outline" size="lg" className="justify-start" disabled={busy !== null} onClick={() => void start(p.id)}>
          {p.icon}
          {busy === p.id ? "Redirecting…" : p.label}
        </Button>
      ))}
    </div>
  );
}
