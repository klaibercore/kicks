"use client";

import dynamic from "next/dynamic";
import { Skeleton } from "@/components/ui/skeleton";

// Client-only: the studio needs AudioContext, Web MIDI and localStorage, none of
// which exist at build time, and pre-rendering an empty pad bank would only
// flash before the real one hydrates.
const Studio = dynamic(() => import("@/components/studio/studio").then((m) => m.Studio), {
  ssr: false,
  loading: () => (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-6 sm:px-6">
      <Skeleton className="h-8 w-40" />
      <Skeleton className="h-9 w-64" />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Skeleton className="h-[520px]" />
        <Skeleton className="h-[520px]" />
      </div>
    </div>
  ),
});

export function StudioLoader() {
  return <Studio />;
}
