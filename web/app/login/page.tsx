import type { Metadata } from "next";
import { Suspense } from "react";
import { LoginPanel } from "@/components/auth/login-panel";

export const metadata: Metadata = { title: "Sign in" };

export default function LoginPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
      <Suspense>
        <LoginPanel />
      </Suspense>
    </div>
  );
}
