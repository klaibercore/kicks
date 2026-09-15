"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";

/** After Stripe: the webhook books the credits; we poll the balance until it moves. */
export function CheckoutSuccess() {
  const params = useSearchParams();
  const { credits, refreshCredits, ready, user } = useAuth();
  const [initial] = useState(() => credits);
  const [attempts, setAttempts] = useState(0);
  const sessionId = params.get("session_id");
  const landed = credits !== null && initial !== null && credits > initial;

  useEffect(() => {
    if (!ready || !user || landed || attempts >= 10) return;
    const t = setTimeout(() => {
      void refreshCredits().then(() => setAttempts((a) => a + 1));
    }, 1500);
    return () => clearTimeout(t);
  }, [ready, user, landed, attempts, refreshCredits]);

  return (
    <div className="mx-auto flex max-w-md flex-col gap-4 px-4 py-24 sm:px-6">
      <h1 className="text-2xl font-semibold tracking-tight">Thank you</h1>
      <p className="text-sm text-muted-foreground">
        {landed
          ? `Your credits are in your account — you now have ${credits}.`
          : attempts >= 10
            ? "Payment received. The credits usually appear within a minute; reload the account page if they have not."
            : "Payment received. Booking your credits…"}
      </p>
      {sessionId ? <p className="font-mono text-xs text-muted-foreground">Reference {sessionId.slice(-12)}</p> : null}
      <p className="text-xs text-muted-foreground">Your invoice is on its way by e-mail and is always available under Account → Billing.</p>
      <div className="flex gap-2">
        <Button render={<Link href="/studio/" />}>Open the studio</Button>
        <Button variant="outline" render={<Link href="/account/" />}>
          Account
        </Button>
      </div>
    </div>
  );
}
