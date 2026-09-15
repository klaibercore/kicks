"use client";

import Link from "next/link";
import { config } from "@/lib/config";
import { useConsent } from "@/lib/consent";
import { legalDocuments } from "@/lib/legal";

export function SiteFooter() {
  const { reset } = useConsent();
  return (
    <footer className="mt-auto border-t border-border">
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-8 text-xs text-muted-foreground sm:px-6 md:flex-row md:items-center md:justify-between">
        <p className="font-mono">
          kicks — neural drum synthesis. Sample generation runs on a VAE trained on a curated one-shot corpus.
        </p>
        <nav className="flex flex-wrap gap-x-4 gap-y-2" aria-label="Legal">
          {Object.values(legalDocuments).map((doc) => (
            <Link key={doc.href} href={doc.href} className="hover:text-foreground">
              {doc.title}
            </Link>
          ))}
          {config.cloudflareToken ? (
            <button type="button" onClick={reset} className="hover:text-foreground">
              Cookie-Einstellungen
            </button>
          ) : null}
        </nav>
      </div>
    </footer>
  );
}
