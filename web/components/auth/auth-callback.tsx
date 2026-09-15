"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { getSupabase } from "@/lib/supabase/client";

function safeNext(raw: string | null): string {
  return raw && raw.startsWith("/") && !raw.startsWith("//") ? raw : "/studio/";
}

/**
 * Where OAuth and magic links land. PKCE gives us `?code=`; a magic link
 * opened in another browser (no verifier) can still be verified through its
 * `token_hash`, which the e-mail template exposes as `{{ .TokenHash }}`.
 */
export function AuthCallback() {
  const params = useSearchParams();
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const next = safeNext(params.get("next"));
    const code = params.get("code");
    const tokenHash = params.get("token_hash");
    const type = params.get("type");

    // Everything below is asynchronous by construction; the effect body itself sets no state.
    (async () => {
      const sb = getSupabase();
      if (!sb) return setError("Accounts are not configured on this site.");
      const described =
        params.get("error_description") ?? new URLSearchParams(window.location.hash.slice(1)).get("error_description");
      if (described) return setError(described);
      if (code) {
        const { error } = await sb.auth.exchangeCodeForSession(code);
        if (error) return setError(error.message);
      } else if (tokenHash && type) {
        const { error } = await sb.auth.verifyOtp({ token_hash: tokenHash, type: type as "email" | "magiclink" });
        if (error) return setError(error.message);
      } else {
        const { data } = await sb.auth.getSession();
        if (!data.session) return setError("No sign-in code in this link. Please request a new one.");
      }
      window.history.replaceState({}, "", window.location.pathname); // drop the code from history
      router.replace(next);
    })();
  }, [params, router]);

  return (
    <div className="mx-auto max-w-md px-4 py-24 text-center sm:px-6">
      {error ? (
        <>
          <p className="text-sm text-destructive">{error}</p>
          <a href="/login/" className="mt-3 inline-block text-sm underline underline-offset-4">
            Back to sign in
          </a>
        </>
      ) : (
        <p className="text-sm text-muted-foreground">Signing you in…</p>
      )}
    </div>
  );
}
