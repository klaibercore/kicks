/** Hand-maintained mirror of supabase/migrations. Regenerate with `supabase gen types` once linked. */

export interface Profile {
  id: string;
  display_name: string | null;
  created_at: string;
  updated_at: string;
}

export interface Product {
  id: string;
  active: boolean;
  name: string;
  description: string | null;
  credits: number;
  metadata: Record<string, string>;
  updated_at: string;
}

export interface Price {
  id: string;
  product_id: string;
  active: boolean;
  currency: string;
  unit_amount: number;
  type: "one_time" | "recurring";
  updated_at: string;
}

export interface Order {
  id: string;
  user_id: string | null;
  stripe_checkout_session_id: string;
  stripe_payment_intent_id: string | null;
  stripe_invoice_id: string | null;
  price_id: string | null;
  credits: number;
  amount_subtotal: number;
  amount_tax: number;
  amount_total: number;
  currency: string;
  status: "paid" | "refunded" | "partially_refunded";
  created_at: string;
  updated_at: string;
}

export interface LedgerEntry {
  id: number;
  user_id: string;
  delta: number;
  reason: "welcome" | "purchase" | "refund" | "export" | "export_refund" | "adjustment";
  ref_type: string | null;
  ref_id: string | null;
  note: string | null;
  created_at: string;
}

export interface Generation {
  id: string;
  user_id: string;
  instrument: string;
  params: Record<string, unknown>;
  ledger_id: number | null;
  license: string;
  refunded: boolean;
  created_at: string;
}

export interface Kit {
  id: string;
  user_id: string;
  name: string;
  pads: unknown[];
  created_at: string;
  updated_at: string;
}

export type ConsentKind = "analytics" | "terms" | "privacy" | "withdrawal_waiver";
