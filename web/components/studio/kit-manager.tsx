"use client";

import { FolderOpenIcon, SaveIcon, Trash2Icon } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/hooks/use-auth";
import type { KitApi } from "@/hooks/use-kit";

export function KitManager({ kit }: { kit: KitApi }) {
  const { user, enabled } = useAuth();
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Input
        aria-label="Kit name"
        value={kit.kit.name}
        maxLength={60}
        onChange={(e) => kit.rename(e.target.value)}
        className="h-8 w-44"
      />
      {enabled ? (
        <>
          <Button size="sm" variant="outline" onClick={() => void kit.saveToAccount()} disabled={!user}>
            <SaveIcon data-icon="inline-start" /> {kit.kitId ? "Update" : "Save"}
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger render={<Button size="sm" variant="outline" disabled={!user} />}>
              <FolderOpenIcon data-icon="inline-start" /> Open
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-64">
              <DropdownMenuLabel>Saved kits</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {kit.savedKits.length === 0 ? (
                <DropdownMenuItem disabled>No kits saved yet</DropdownMenuItem>
              ) : (
                kit.savedKits.map((saved) => (
                  <DropdownMenuItem key={saved.id} onClick={() => kit.loadFromAccount(saved)}>
                    <span className="flex-1 truncate">{saved.name}</span>
                    <button
                      type="button"
                      className="text-muted-foreground hover:text-destructive"
                      aria-label={`Delete ${saved.name}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        void kit.deleteFromAccount(saved.id);
                      }}
                    >
                      <Trash2Icon className="size-3.5" />
                    </button>
                  </DropdownMenuItem>
                ))
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        </>
      ) : null}
      <Button size="sm" variant="ghost" onClick={kit.clearAll}>
        Clear all
      </Button>
      {!user && enabled ? <span className="text-xs text-muted-foreground">Sign in to keep kits in your account.</span> : null}
    </div>
  );
}
