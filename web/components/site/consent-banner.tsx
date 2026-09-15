"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { config } from "@/lib/config";
import { useConsent } from "@/lib/consent";
import { legalDocuments } from "@/lib/legal";

/**
 * Equal-prominence choices (BGH I ZR 7/16 and the DSK guidance): "only
 * necessary" is as visible as "accept all", there is no pre-ticked box and no
 * dark pattern, and the decision can be changed from the footer at any time.
 */
export function ConsentBanner() {
  const { loading, decided, decide } = useConsent();
  if (loading || decided) return null;
  if (!config.cloudflareToken) return null; // nothing optional to consent to

  return (
    <div
      role="dialog"
      aria-live="polite"
      aria-label="Privacy choices"
      className="fixed inset-x-0 bottom-0 z-50 border-t border-border bg-background/95 p-4 backdrop-blur supports-[backdrop-filter]:bg-background/80"
      style={{ paddingBottom: "calc(1rem + env(safe-area-inset-bottom, 0px))" }}
    >
      <div className="mx-auto flex max-w-5xl flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <p className="text-sm text-muted-foreground">
          We use Cloudflare Web Analytics to count visits. It sets no cookies, but it is a third party, so
          it only runs if you allow it. Signing in stores a session token — that is necessary and not
          optional. Details in the{" "}
          <Link href={legalDocuments.datenschutz.href} className="underline underline-offset-4">
            Datenschutzerklärung
          </Link>
          .
        </p>
        <div className="flex shrink-0 gap-2">
          <Button variant="outline" onClick={() => decide(false)}>
            Only necessary
          </Button>
          <Button variant="outline" onClick={() => decide(true)}>
            Allow analytics
          </Button>
        </div>
      </div>
    </div>
  );
}
