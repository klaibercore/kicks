/** POST -> { url }: a Stripe customer-portal session for invoices and receipts. */
import { json, preflight } from "../_shared/cors.ts";
import { siteUrl, stripe } from "../_shared/stripe.ts";
import { admin, userFromRequest } from "../_shared/supabase.ts";

Deno.serve(async (req) => {
  const pre = preflight(req);
  if (pre) return pre;
  if (req.method !== "POST") return json(req, { error: "method not allowed" }, 405);

  const user = await userFromRequest(req);
  if (!user) return json(req, { error: "unauthorized" }, 401);

  const { data } = await admin.from("billing_customers").select("stripe_customer_id").eq("user_id", user.id).maybeSingle();
  if (!data) return json(req, { error: "no purchases yet" }, 404);

  const session = await stripe.billingPortal.sessions.create({
    customer: data.stripe_customer_id,
    return_url: siteUrl("/account/"),
  });
  return json(req, { url: session.url });
});
