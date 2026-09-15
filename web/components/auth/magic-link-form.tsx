"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { siteUrl } from "@/lib/config";
import { getSupabase } from "@/lib/supabase/client";

/** Passwordless sign-in: a one-time link by e-mail. Nothing to reuse, nothing to leak. */
export function MagicLinkForm({ next }: { next: string }) {
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const sb = getSupabase();
    if (!sb) return;
    setBusy(true);
    const { error } = await sb.auth.signInWithOtp({
      email: email.trim(),
      options: { emailRedirectTo: siteUrl(`/auth/callback/?next=${encodeURIComponent(next)}`), shouldCreateUser: true },
    });
    setBusy(false);
    if (error) {
      toast.error(error.message);
      return;
    }
    setSent(true);
  };

  if (sent) {
    return (
      <p className="rounded-md border border-border p-3 text-sm text-muted-foreground">
        Check <span className="font-medium text-foreground">{email}</span> for a sign-in link. It is valid for 15 minutes and
        works once.
      </p>
    );
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-3">
      <div className="grid gap-1.5">
        <Label htmlFor="email">E-mail</Label>
        <Input
          id="email"
          type="email"
          autoComplete="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="you@example.com"
        />
      </div>
      <Button type="submit" disabled={busy || !email}>
        {busy ? "Sending…" : "Send sign-in link"}
      </Button>
    </form>
  );
}
