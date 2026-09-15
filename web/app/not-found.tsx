import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="mx-auto flex max-w-2xl flex-col items-start gap-4 px-4 py-24 sm:px-6">
      <p className="font-mono text-xs uppercase tracking-[0.3em] text-muted-foreground">404</p>
      <h1 className="text-2xl font-semibold tracking-tight">Nothing here</h1>
      <Button variant="outline" render={<Link href="/" />}>
        Back to the start
      </Button>
    </div>
  );
}
