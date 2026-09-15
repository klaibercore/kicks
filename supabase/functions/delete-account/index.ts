/**
 * POST { confirm: "DELETE" } — Art. 17 GDPR erasure.
 *
 * Order: remove the Stripe customer (Stripe keeps the invoices it is legally
 * required to keep, detached from any personal profile we control), then let
 * the database detach orders and cascade-delete everything else.
 */
import { json, preflight } from "../_shared/cors.ts";
import { stripe } from "../_shared/stripe.ts";
import { admin, userFromRequest } from "../_shared/supabase.ts";

Deno.serve(async (req) => {
  const pre = preflight(req);
  if (pre) return pre;
  if (req.method !== "POST") return json(req, { error: "method not allowed" }, 405);

  const user = await userFromRequest(req);
  if (!user) return json(req, { error: "unauthorized" }, 401);

  let body: { confirm?: string } = {};
  try {
    body = await req.json();
  } catch {
    /* empty body */
  }
  if (body.confirm !== "DELETE") return json(req, { error: "confirmation required" }, 400);

  const { data: customer } = await admin
    .from("billing_customers")
    .select("stripe_customer_id")
    .eq("user_id", user.id)
    .maybeSingle();
  if (customer) {
    try {
      await stripe.customers.del(customer.stripe_customer_id);
    } catch (err) {
      console.error("stripe customer deletion failed", err);
      return json(req, { error: "could not remove billing profile; try again later" }, 502);
    }
  }

  const { error } = await admin.rpc("erase_user", { p_user_id: user.id });
  if (error) {
    console.error("erase_user failed", error);
    return json(req, { error: "deletion failed" }, 500);
  }
  return json(req, { ok: true });
});
