"use client";

import { DownloadIcon, ExternalLinkIcon } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useAuth } from "@/hooks/use-auth";
import { deleteAccount, formatMoney, openBillingPortal } from "@/lib/billing";
import { getSupabase } from "@/lib/supabase/client";
import type { Generation, LedgerEntry, Order } from "@/lib/supabase/types";

const reasonLabel: Record<LedgerEntry["reason"], string> = {
  welcome: "Welcome credits",
  purchase: "Purchase",
  refund: "Refund",
  export: "Export",
  export_refund: "Export refunded",
  adjustment: "Adjustment",
};

interface Activity {
  ledger: LedgerEntry[];
  orders: Order[];
  generations: Generation[];
}

export function AccountPanel() {
  const { enabled, ready, user, profile, credits, refreshCredits, signOut } = useAuth();
  const router = useRouter();
  const [name, setName] = useState<string | null>(null);
  const [activity, setActivity] = useState<Activity | null>(null);

  useEffect(() => {
    if (ready && enabled && !user) router.replace("/login/?next=/account/");
  }, [ready, enabled, user, router]);

  useEffect(() => {
    const sb = getSupabase();
    if (!sb || !user) return;
    let cancelled = false;
    Promise.all([
      sb.from("credit_ledger").select("*").order("created_at", { ascending: false }).limit(50),
      sb.from("orders").select("*").order("created_at", { ascending: false }),
      sb.from("generations").select("*").order("created_at", { ascending: false }).limit(50),
    ]).then(([l, o, g]) => {
      if (cancelled) return;
      setActivity({
        ledger: (l.data as LedgerEntry[]) ?? [],
        orders: (o.data as Order[]) ?? [],
        generations: (g.data as Generation[]) ?? [],
      });
    });
    return () => {
      cancelled = true;
    };
  }, [user, credits]);

  if (!enabled) {
    return <p className="text-sm text-muted-foreground">Accounts are not configured on this site.</p>;
  }
  if (!ready || !user) {
    return <Skeleton className="h-64 w-full" />;
  }

  const displayName = name ?? profile?.display_name ?? "";

  const saveName = async () => {
    const sb = getSupabase();
    if (!sb) return;
    const { error } = await sb.from("profiles").update({ display_name: displayName.trim() || null }).eq("id", user.id);
    if (error) toast.error(error.message);
    else {
      toast.success("Saved.");
      await refreshCredits();
    }
  };

  const exportData = async () => {
    const sb = getSupabase();
    if (!sb) return;
    const { data, error } = await sb.rpc("export_own_data");
    if (error) {
      toast.error(error.message);
      return;
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `kicks-account-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  };

  const providers = (user.app_metadata?.providers as string[] | undefined)?.join(", ") ?? user.app_metadata?.provider ?? "e-mail";

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
      <div className="flex flex-col gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Profile</CardTitle>
            <CardDescription>{user.email}</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div className="grid gap-1.5">
              <Label htmlFor="display-name">Display name</Label>
              <div className="flex gap-2">
                <Input id="display-name" value={displayName} maxLength={60} onChange={(e) => setName(e.target.value)} />
                <Button variant="outline" onClick={() => void saveName()}>
                  Save
                </Button>
              </div>
            </div>
            <p className="text-xs text-muted-foreground">Signed in via {providers}.</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Credits</CardTitle>
            <CardDescription>One credit exports one sample.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex items-baseline gap-3">
              <span className="tabular text-4xl font-semibold">{credits ?? "–"}</span>
              <Button size="sm" render={<Link href="/pricing/" />}>
                Buy more
              </Button>
            </div>
            {activity === null ? (
              <Skeleton className="h-24" />
            ) : activity.ledger.length === 0 ? (
              <p className="text-sm text-muted-foreground">No activity yet.</p>
            ) : (
              <ul className="flex flex-col divide-y divide-border text-sm">
                {activity.ledger.map((e) => (
                  <li key={e.id} className="flex items-center justify-between py-1.5">
                    <span>
                      {reasonLabel[e.reason]}
                      {e.note ? <span className="text-muted-foreground"> · {e.note}</span> : null}
                    </span>
                    <span className="flex items-center gap-3">
                      <span className="text-xs text-muted-foreground">{new Date(e.created_at).toLocaleDateString()}</span>
                      <span className={`tabular font-mono ${e.delta > 0 ? "text-emerald-600 dark:text-emerald-400" : ""}`}>
                        {e.delta > 0 ? `+${e.delta}` : e.delta}
                      </span>
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Your data</CardTitle>
            <CardDescription>Art. 15, 17 and 20 GDPR — take it with you or remove it.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={() => void exportData()}>
              <DownloadIcon data-icon="inline-start" /> Download my data (JSON)
            </Button>
            <DeleteAccountDialog
              onDeleted={async () => {
                await signOut();
                router.replace("/");
              }}
            />
          </CardContent>
        </Card>
      </div>

      <div className="flex flex-col gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Billing</CardTitle>
            <CardDescription>Invoices and receipts are kept by Stripe; the portal shows every one.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <Button
              variant="outline"
              className="self-start"
              disabled={!activity || activity.orders.length === 0}
              onClick={() => void openBillingPortal().catch((e: Error) => toast.error(e.message))}
            >
              <ExternalLinkIcon data-icon="inline-start" /> Open billing portal
            </Button>
            {activity === null ? (
              <Skeleton className="h-24" />
            ) : activity.orders.length === 0 ? (
              <p className="text-sm text-muted-foreground">No purchases yet.</p>
            ) : (
              <div className="overflow-x-auto rounded-lg border border-border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Date</TableHead>
                      <TableHead>Credits</TableHead>
                      <TableHead className="text-right">Total</TableHead>
                      <TableHead className="text-right">Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {activity.orders.map((o) => (
                      <TableRow key={o.id}>
                        <TableCell className="text-xs">{new Date(o.created_at).toLocaleDateString()}</TableCell>
                        <TableCell className="tabular font-mono text-xs">{o.credits}</TableCell>
                        <TableCell className="tabular text-right font-mono text-xs">
                          {formatMoney(o.amount_total, o.currency)}
                          <span className="ml-1 text-muted-foreground">(incl. {formatMoney(o.amount_tax, o.currency)} VAT)</span>
                        </TableCell>
                        <TableCell className="text-right">
                          <Badge variant={o.status === "paid" ? "secondary" : "outline"} className="text-[10px]">
                            {o.status}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Exports</CardTitle>
            <CardDescription>Every sample you have exported, with the settings that made it.</CardDescription>
          </CardHeader>
          <CardContent>
            {activity === null ? (
              <Skeleton className="h-24" />
            ) : activity.generations.length === 0 ? (
              <p className="text-sm text-muted-foreground">Nothing exported yet.</p>
            ) : (
              <ul className="flex flex-col divide-y divide-border text-sm">
                {activity.generations.map((g) => {
                  const sliders = (g.params.sliders ?? {}) as Record<string, number>;
                  return (
                    <li key={g.id} className="flex flex-wrap items-center justify-between gap-2 py-2">
                      <span className="flex items-center gap-2">
                        <Badge variant="outline" className="font-mono text-[10px]">
                          {g.instrument}
                        </Badge>
                        <span className="font-mono text-xs text-muted-foreground">
                          {Object.entries(sliders)
                            .map(([k, v]) => `${k} ${Math.round(v * 100)}`)
                            .join(" · ")}
                        </span>
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {new Date(g.created_at).toLocaleString()}
                        {g.refunded ? " · refunded" : ""}
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function DeleteAccountDialog({ onDeleted }: { onDeleted: () => Promise<void> }) {
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    try {
      await deleteAccount();
      toast.success("Your account has been deleted.");
      await onDeleted();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Deletion failed.");
      setBusy(false);
    }
  };

  return (
    <Dialog>
      <DialogTrigger render={<Button variant="destructive" />}>Delete account</DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete your account?</DialogTitle>
          <DialogDescription>
            This removes your profile, credits, kits, export history and consent records immediately and cannot be undone.
            Invoices for past purchases are retained without your profile, as tax law requires (§ 147 AO). Unused credits are
            forfeited.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-1.5">
          <Label htmlFor="confirm-delete">
            Type <span className="font-mono">DELETE</span> to confirm
          </Label>
          <Input id="confirm-delete" value={confirm} onChange={(e) => setConfirm(e.target.value)} autoComplete="off" />
        </div>
        <DialogFooter>
          <Button variant="destructive" disabled={confirm !== "DELETE" || busy} onClick={() => void run()}>
            {busy ? "Deleting…" : "Delete permanently"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
