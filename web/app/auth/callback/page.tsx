import type { Metadata } from "next";
import { Suspense } from "react";
import { AuthCallback } from "@/components/auth/auth-callback";

export const metadata: Metadata = { title: "Signing in", robots: { index: false } };

export default function CallbackPage() {
  return (
    <Suspense>
      <AuthCallback />
    </Suspense>
  );
}
