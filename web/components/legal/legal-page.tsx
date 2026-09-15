import type { ReactNode } from "react";
import { config } from "@/lib/config";
import { placeholdersPresent } from "@/lib/legal";

/**
 * Shell for the German legal texts. Rendered in German because the operator
 * is German and these documents must be intelligible to German consumers;
 * the product UI stays English. Unfilled operator details are called out
 * loudly rather than hidden — an incomplete Impressum is a § 5 DDG problem.
 */
export function LegalPage({ title, children }: { title: string; children: ReactNode }) {
  return (
    <article className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
      <p className="mb-2 text-xs text-muted-foreground">
        Provided in German, as required for a service operated from Germany. The German text is authoritative.
      </p>
      <h1 className="mb-8 text-3xl font-semibold tracking-tight">{title}</h1>
      {placeholdersPresent ? (
        <p className="mb-8 rounded-md border border-amber-500/50 bg-amber-500/10 p-3 text-sm">
          Betreiberangaben fehlen. Setze <code className="font-mono">NEXT_PUBLIC_LEGAL_*</code> in der Build-Umgebung, bevor die Seite
          veröffentlicht wird.
        </p>
      ) : null}
      <div className="legal-prose">{children}</div>
      <p className="mt-12 text-xs text-muted-foreground">Stand: {config.legalVersion}</p>
    </article>
  );
}
