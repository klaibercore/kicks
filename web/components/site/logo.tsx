import Link from "next/link";
import { cn } from "@/lib/utils";

export function Logo({ className }: { className?: string }) {
  return (
    <Link href="/" className={cn("flex items-center gap-2 font-mono text-sm font-semibold tracking-tight", className)}>
      <span aria-hidden className="inline-block size-2.5 rounded-full bg-foreground" />
      kicks
    </Link>
  );
}
