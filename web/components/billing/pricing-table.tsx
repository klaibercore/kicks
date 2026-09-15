"use client";

import { CheckIcon } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/hooks/use-auth";
import { formatMoney, loadPacks, startCheckout, type Pack } from "@/lib/billing";
import { legalDocuments } from "@/lib/legal";

/**
 * Credit packs. Prices are shown gross ("inkl. MwSt.", § 1 PAngV); Stripe
 * Checkout shows the exact tax for the buyer's country before the final
 * "Zahlen" button, which is the § 312j BGB order button. The two checkboxes
 * below are the pre-contractual consents: the AGB, and the § 356(5) BGB
 * waiver — digital content, delivered immediately, no withdrawal after that.
 */
export function PricingTable() {
  const { enabled, user, ready } = useAuth();
  const [packs, setPacks] = useState<Pack[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [terms, setTerms] = useState(false);
  const [waiver, setWaiver] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) return;
    loadPacks().then(setPacks).catch((e: Error) => setError(e.message));
  }, [enabled]);

  const buy = async (pack: Pack) => {
    setBusy(pack.price.id);
    try {
      await startCheckout(pack.price.id);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Checkout could not be started.");
      setBusy(null);
    }
  };

  if (!enabled) {
    return (
      <p className="text-sm text-muted-foreground">
        This build has no account backend, so there is nothing to buy — exports are open on the local server.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-8">
      {error ? <p className="text-sm text-destructive">{error}</p> : null}

      <div className="grid gap-4 md:grid-cols-3">
        {packs === null
          ? [0, 1, 2].map((i) => <Skeleton key={i} className="h-64" />)
          : packs.length === 0
            ? (
              <p className="text-sm text-muted-foreground md:col-span-3">
                No credit packs are configured yet. Create products in Stripe with a <code className="font-mono">credits</code>{" "}
                metadata field; the webhook mirrors them here.
              </p>
            )
            : packs.map((pack, i) => {
                const perCredit = pack.price.unit_amount / pack.product.credits;
                return (
                  <Card key={pack.price.id} className={i === 1 ? "border-foreground/40" : undefined}>
                    <CardHeader>
                      <CardTitle>{pack.product.name}</CardTitle>
                      <CardDescription>{pack.product.description ?? `${pack.product.credits} sample exports`}</CardDescription>
                    </CardHeader>
                    <CardContent className="flex flex-col gap-3">
                      <div>
                        <span className="tabular text-3xl font-semibold">{formatMoney(pack.price.unit_amount, pack.price.currency)}</span>
                        <span className="ml-1 text-xs text-muted-foreground">inkl. MwSt.</span>
                      </div>
                      <ul className="flex flex-col gap-1.5 text-sm text-muted-foreground">
                        <li className="flex gap-2">
                          <CheckIcon className="mt-0.5 size-4 shrink-0" /> {pack.product.credits} export credits
                        </li>
                        <li className="flex gap-2">
                          <CheckIcon className="mt-0.5 size-4 shrink-0" /> {formatMoney(perCredit, pack.price.currency)} per 24-bit WAV
                        </li>
                        <li className="flex gap-2">
                          <CheckIcon className="mt-0.5 size-4 shrink-0" /> Royalty-free licence for your productions
                        </li>
                        <li className="flex gap-2">
                          <CheckIcon className="mt-0.5 size-4 shrink-0" /> Credits never expire
                        </li>
                      </ul>
                    </CardContent>
                    <CardFooter>
                      {!ready ? (
                        <Skeleton className="h-8 w-full" />
                      ) : !user ? (
                        <Button className="w-full" variant="outline" render={<Link href="/login/?next=/pricing/" />}>
                          Sign in to buy
                        </Button>
                      ) : (
                        <Button className="w-full" disabled={!terms || !waiver || busy !== null} onClick={() => void buy(pack)}>
                          {busy === pack.price.id ? "Redirecting to Stripe…" : "Continue to checkout"}
                        </Button>
                      )}
                    </CardFooter>
                  </Card>
                );
              })}
      </div>

      {user && packs && packs.length > 0 ? (
        <div className="flex max-w-2xl flex-col gap-3 rounded-lg border border-border p-4">
          <div className="flex items-start gap-3">
            <Checkbox id="terms" checked={terms} onCheckedChange={(v) => setTerms(v === true)} className="mt-0.5" />
            <Label htmlFor="terms" className="text-sm leading-snug font-normal">
              Ich habe die{" "}
              <Link href={legalDocuments.agb.href} className="underline underline-offset-4" target="_blank">
                AGB
              </Link>{" "}
              und die{" "}
              <Link href={legalDocuments.widerruf.href} className="underline underline-offset-4" target="_blank">
                Widerrufsbelehrung
              </Link>{" "}
              gelesen und akzeptiere sie.
            </Label>
          </div>
          <div className="flex items-start gap-3">
            <Checkbox id="waiver" checked={waiver} onCheckedChange={(v) => setWaiver(v === true)} className="mt-0.5" />
            <Label htmlFor="waiver" className="text-sm leading-snug font-normal">
              Ich stimme ausdrücklich zu, dass die Credits sofort nach Zahlung bereitgestellt werden, und nehme zur Kenntnis,
              dass ich damit mein Widerrufsrecht verliere (§ 356 Abs. 5 BGB).
            </Label>
          </div>
          <p className="text-xs text-muted-foreground">
            Payment is handled by Stripe. The exact VAT for your country and the final gross amount are shown on the Stripe page
            before you confirm the order. You will receive an invoice by e-mail.
          </p>
        </div>
      ) : null}
    </div>
  );
}
