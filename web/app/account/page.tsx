import type { Metadata } from "next";
import { AccountPanel } from "@/components/account/account-panel";

export const metadata: Metadata = { title: "Account", robots: { index: false } };

export default function AccountPage() {
  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-10 sm:px-6">
      <h1 className="text-2xl font-semibold tracking-tight">Account</h1>
      <AccountPanel />
    </div>
  );
}
