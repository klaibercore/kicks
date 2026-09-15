"use client";

import { CoinsIcon, LogOutIcon, UserIcon } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from "@/hooks/use-auth";

export function UserMenu() {
  const { enabled, ready, user, profile, credits, signOut } = useAuth();
  const router = useRouter();

  if (!enabled) return null;
  if (!ready) return <div className="size-7 rounded-full bg-muted" aria-hidden />;
  if (!user) {
    return (
      <Button size="sm" variant="outline" render={<Link href="/login/" />}>
        Sign in
      </Button>
    );
  }

  const name = profile?.display_name ?? user.user_metadata?.name ?? user.email ?? "Account";
  const avatar = user.user_metadata?.avatar_url as string | undefined;

  return (
    <div className="flex items-center gap-2">
      <Link
        href="/pricing/"
        className="hidden items-center gap-1 rounded-md border border-border px-2 py-1 font-mono text-xs text-muted-foreground hover:text-foreground sm:flex"
        title="Credits — one credit exports one sample"
      >
        <CoinsIcon className="size-3.5" />
        {credits ?? "–"}
      </Link>
      <DropdownMenu>
        <DropdownMenuTrigger
          render={
            <Button variant="ghost" size="icon-sm" className="rounded-full" aria-label="Account menu" />
          }
        >
          <Avatar className="size-7">
            {avatar ? <AvatarImage src={avatar} alt="" /> : null}
            <AvatarFallback>{String(name).slice(0, 1).toUpperCase()}</AvatarFallback>
          </Avatar>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-56">
          <DropdownMenuLabel className="truncate font-normal">
            <span className="block truncate text-sm">{name}</span>
            <span className="block truncate text-xs text-muted-foreground">{user.email}</span>
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={() => router.push("/account/")}>
            <UserIcon /> Account
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => router.push("/pricing/")}>
            <CoinsIcon /> Credits: {credits ?? "–"}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onClick={async () => {
              await signOut();
              router.push("/");
            }}
          >
            <LogOutIcon /> Sign out
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
