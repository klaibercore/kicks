import type { Metadata } from "next";
import { Suspense } from "react";
import { CheckoutSuccess } from "@/components/billing/checkout-result";

export const metadata: Metadata = { title: "Order complete", robots: { index: false } };

export default function SuccessPage() {
  return (
    <Suspense>
      <CheckoutSuccess />
    </Suspense>
  );
}
