"use client";

import { DownloadIcon, FolderOpenIcon, SaveIcon, UploadIcon } from "lucide-react";
import { useRef } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import type { KitApi } from "@/hooks/use-kit";
import { parseKit } from "@/lib/pads";

export function KitManager({ kit }: { kit: KitApi }) {
  const file = useRef<HTMLInputElement>(null);
  const download = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(kit.kit, null, 2)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url; a.download = `${kit.kit.name.replace(/[^a-z0-9_-]/gi, "_") || "kit"}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Input aria-label="Kit name" value={kit.kit.name} maxLength={60} onChange={(e) => kit.rename(e.target.value)} className="h-8 w-44" />
      <Button size="sm" variant="outline" onClick={() => { kit.save(); toast.success("Kit saved on this computer."); }}><SaveIcon data-icon="inline-start" /> Save</Button>
      <DropdownMenu>
        <DropdownMenuTrigger render={<Button size="sm" variant="outline" />}><FolderOpenIcon data-icon="inline-start" /> Open</DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-64">
          <DropdownMenuLabel>Saved on this computer</DropdownMenuLabel><DropdownMenuSeparator />
          {kit.savedKits.length ? kit.savedKits.map((saved) => (
            <DropdownMenuItem key={saved.id} onClick={() => kit.open(saved)}>{saved.name}</DropdownMenuItem>
          )) : <DropdownMenuItem disabled>No named kits yet</DropdownMenuItem>}
        </DropdownMenuContent>
      </DropdownMenu>
      <Button size="sm" variant="ghost" onClick={download}><DownloadIcon data-icon="inline-start" /> JSON</Button>
      <Button size="sm" variant="ghost" onClick={() => file.current?.click()}><UploadIcon data-icon="inline-start" /> Import</Button>
      <input ref={file} type="file" accept="application/json,.json" className="hidden" aria-label="Import kit JSON" onChange={async (e) => {
        const selected = e.target.files?.[0]; e.target.value = "";
        if (!selected) return;
        if (selected.size > 1_000_000) { toast.error("Kit file is too large."); return; }
        try { kit.open(parseKit(JSON.parse(await selected.text()))); toast.success("Kit imported."); }
        catch { toast.error("This file is not a valid kicks kit."); }
      }} />
      <Button size="sm" variant="ghost" onClick={kit.clearAll}>Clear all</Button>
    </div>
  );
}
