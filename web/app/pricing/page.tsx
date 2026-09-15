import type { Metadata } from "next";
import { PricingTable } from "@/components/billing/pricing-table";

export const metadata: Metadata = { title: "Pricing" };

export default function PricingPage() {
  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-8 px-4 py-10 sm:px-6">
      <div className="max-w-2xl">
        <h1 className="text-2xl font-semibold tracking-tight">Credits</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Previewing sounds in the studio is free. Exporting a sample as a 24-bit WAV you can use in your music costs one
          credit. New accounts start with three.
        </p>
      </div>
      <PricingTable />
      <section className="grid max-w-3xl gap-4 text-sm text-muted-foreground md:grid-cols-2">
        <div>
          <h2 className="mb-1 font-medium text-foreground">What you get</h2>
          <p>
            Each export is a unique render at your slider settings, delivered as 44.1 kHz / 24-bit WAV, with a royalty-free
            licence to use it in commercial and non-commercial productions. Reselling the samples themselves is not permitted.
          </p>
        </div>
        <div>
          <h2 className="mb-1 font-medium text-foreground">Refunds</h2>
          <p>
            Credits are digital content delivered immediately, so the statutory right of withdrawal ends once they are
            provided. Unused credits from a pack can still be refunded within 14 days on request — see the AGB.
          </p>
        </div>
      </section>
    </div>
  );
}
