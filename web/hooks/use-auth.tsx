"use client";

import type { Session, User } from "@supabase/supabase-js";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { KicksApi } from "@/lib/api/client";
import { hasSupabase } from "@/lib/config";
import { getSupabase } from "@/lib/supabase/client";
import type { Profile } from "@/lib/supabase/types";

interface AuthContextValue {
  /** False until the first session check has completed. */
  ready: boolean;
  enabled: boolean;
  session: Session | null;
  user: User | null;
  profile: Profile | null;
  credits: number | null;
  api: KicksApi;
  refreshCredits: () => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(!hasSupabase);
  const [session, setSession] = useState<Session | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [credits, setCredits] = useState<number | null>(null);

  const api = useMemo(
    () =>
      new KicksApi(undefined, async () => {
        const sb = getSupabase();
        if (!sb) return null;
        const { data } = await sb.auth.getSession();
        return data.session?.access_token ?? null;
      }),
    [],
  );

  const loadAccount = useCallback(async (userId: string | undefined) => {
    const sb = getSupabase();
    if (!sb || !userId) {
      setProfile(null);
      setCredits(null);
      return;
    }
    const [{ data: prof }, { data: balance }] = await Promise.all([
      sb.from("profiles").select("*").eq("id", userId).maybeSingle(),
      sb.rpc("credit_balance"),
    ]);
    setProfile((prof as Profile | null) ?? null);
    setCredits(typeof balance === "number" ? balance : null);
  }, []);

  useEffect(() => {
    const sb = getSupabase();
    if (!sb) return;
    let cancelled = false;
    sb.auth.getSession().then(({ data }) => {
      if (cancelled) return;
      setSession(data.session);
      loadAccount(data.session?.user.id).finally(() => setReady(true));
    });
    const { data: sub } = sb.auth.onAuthStateChange((_event, next) => {
      setSession(next);
      loadAccount(next?.user.id);
    });
    return () => {
      cancelled = true;
      sub.subscription.unsubscribe();
    };
  }, [loadAccount]);

  const refreshCredits = useCallback(async () => loadAccount(session?.user.id), [loadAccount, session]);

  const signOut = useCallback(async () => {
    await getSupabase()?.auth.signOut();
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      ready,
      enabled: hasSupabase,
      session,
      user: session?.user ?? null,
      profile,
      credits,
      api,
      refreshCredits,
      signOut,
    }),
    [ready, session, profile, credits, api, refreshCredits, signOut],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
