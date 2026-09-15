import type { Metadata } from "next";
import Link from "next/link";
import { Button } from "@/components/ui/button";

export const metadata: Metadata = { title: "Checkout cancelled", robots: { index: false } };

export default function CancelPage() {
  return (
    <div className="mx-auto flex max-w-md flex-col gap-4 px-4 py-24 sm:px-6">
      <h1 className="text-2xl font-semibold tracking-tight">Nothing was charged</h1>
      <p className="text-sm text-muted-foreground">The checkout was cancelled. Your credits are unchanged.</p>
      <div className="flex gap-2">
        <Button render={<Link href="/pricing/" />}>Back to pricing</Button>
        <Button variant="outline" render={<Link href="/studio/" />}>
          Studio
        </Button>
      </div>
    </div>
  );
}
