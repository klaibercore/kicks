"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect } from "react";
import { MagicLinkForm } from "@/components/auth/magic-link-form";
import { OAuthButtons } from "@/components/auth/oauth-buttons";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/hooks/use-auth";
import { legalDocuments } from "@/lib/legal";

function safeNext(raw: string | null): string {
  // Only same-site paths; never an absolute URL someone pasted into a link.
  return raw && raw.startsWith("/") && !raw.startsWith("//") ? raw : "/studio/";
}

export function LoginPanel() {
  const { enabled, ready, user } = useAuth();
  const params = useSearchParams();
  const router = useRouter();
  const next = safeNext(params.get("next"));

  useEffect(() => {
    if (ready && user) router.replace(next);
  }, [ready, user, router, next]);

  if (!enabled) {
    return (
      <Card className="mx-auto max-w-md">
        <CardHeader>
          <CardTitle>Accounts are not configured</CardTitle>
          <CardDescription>
            This build has no Supabase project. Set <code className="font-mono">NEXT_PUBLIC_SUPABASE_URL</code> and{" "}
            <code className="font-mono">NEXT_PUBLIC_SUPABASE_ANON_KEY</code> and rebuild.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card className="mx-auto w-full max-w-md">
      <CardHeader>
        <CardTitle>Sign in</CardTitle>
        <CardDescription>Previews are free once you are signed in. New accounts start with three export credits.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <OAuthButtons next={next} />
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <Separator className="flex-1" /> or <Separator className="flex-1" />
        </div>
        <MagicLinkForm next={next} />
        <p className="text-xs text-muted-foreground">
          By signing in you accept the{" "}
          <Link href={legalDocuments.agb.href} className="underline underline-offset-4">
            AGB
          </Link>
          . How your data is handled is described in the{" "}
          <Link href={legalDocuments.datenschutz.href} className="underline underline-offset-4">
            Datenschutzerklärung
          </Link>
          . Your e-mail address and provider profile name are stored to run your account; nothing else is collected at
          sign-in.
        </p>
      </CardContent>
    </Card>
  );
}
