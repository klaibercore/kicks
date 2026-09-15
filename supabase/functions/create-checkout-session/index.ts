/**
 * POST { price_id, withdrawal_waiver: true }
 *
 * Creates a Stripe Checkout Session for one credit pack. The customer is
 * looked up or created so every purchase lands on the same Stripe customer,
 * which is what makes the billing portal and invoice history work.
 *
 * German consumer law shapes a few of the options below:
 *   - automatic_tax: VAT is charged at the customer's rate, and the invoice
 *     carries it (§ 14 UStG).
 *   - invoice_creation: every one-time purchase gets a proper invoice.
 *   - consent_collection.terms_of_service: the AGB are accepted on the
 *     Stripe page as well, so the acceptance is logged by Stripe too.
 *   - custom_text.submit: the § 356(5) BGB notice — digital content is
 *     delivered immediately and the right of withdrawal expires with it.
 *     The site collects the explicit checkbox before redirecting; we refuse
 *     to create a session without it and record it in `consents`.
 */
import { json, preflight } from "../_shared/cors.ts";
import { siteUrl, stripe } from "../_shared/stripe.ts";
import { admin, userFromRequest } from "../_shared/supabase.ts";

Deno.serve(async (req) => {
  const pre = preflight(req);
  if (pre) return pre;
  if (req.method !== "POST") return json(req, { error: "method not allowed" }, 405);

  const user = await userFromRequest(req);
  if (!user) return json(req, { error: "unauthorized" }, 401);

  let body: { price_id?: string; withdrawal_waiver?: boolean; legal_version?: string };
  try {
    body = await req.json();
  } catch {
    return json(req, { error: "invalid json" }, 400);
  }
  if (!body.price_id) return json(req, { error: "price_id required" }, 400);
  if (body.withdrawal_waiver !== true) {
    return json(req, { error: "withdrawal_waiver must be acknowledged" }, 400);
  }

  // Only sell what the catalogue mirror says is active.
  const { data: price } = await admin
    .from("prices")
    .select("id, product_id, active, currency, unit_amount, products(credits, name, active)")
    .eq("id", body.price_id)
    .maybeSingle();
  const product = price?.products as { credits: number; name: string; active: boolean } | null;
  if (!price?.active || !product?.active) return json(req, { error: "price not available" }, 404);

  // Find or create the Stripe customer for this user.
  const { data: existing } = await admin
    .from("billing_customers")
    .select("stripe_customer_id")
    .eq("user_id", user.id)
    .maybeSingle();
  let customerId = existing?.stripe_customer_id as string | undefined;
  if (!customerId) {
    const customer = await stripe.customers.create({
      email: user.email ?? undefined,
      metadata: { supabase_user_id: user.id },
    });
    customerId = customer.id;
    await admin.from("billing_customers").insert({ user_id: user.id, stripe_customer_id: customerId });
  }

  // Record the pre-contractual consents against the user (Art. 7(1) GDPR).
  const version = body.legal_version ?? "unknown";
  await admin.from("consents").insert([
    { user_id: user.id, kind: "withdrawal_waiver", version, granted: true },
    { user_id: user.id, kind: "terms", version, granted: true },
  ]);

  const session = await stripe.checkout.sessions.create({
    mode: "payment",
    customer: customerId,
    client_reference_id: user.id,
    line_items: [{ price: price.id, quantity: 1 }],
    success_url: siteUrl("/checkout/success/?session_id={CHECKOUT_SESSION_ID}"),
    cancel_url: siteUrl("/checkout/cancel/"),
    locale: "auto",
    automatic_tax: { enabled: true },
    tax_id_collection: { enabled: true },
    billing_address_collection: "required",
    customer_update: { address: "auto", name: "auto" },
    invoice_creation: {
      enabled: true,
      invoice_data: {
        description: `${product.credits} kicks credits — ${product.name}`,
        metadata: { supabase_user_id: user.id, credits: String(product.credits) },
      },
    },
    consent_collection: { terms_of_service: "required" },
    custom_text: {
      submit: {
        message:
          "Digitale Inhalte: Mit dem Kauf stimmen Sie zu, dass die Credits sofort bereitgestellt werden, " +
          "und nehmen zur Kenntnis, dass Ihr Widerrufsrecht damit erlischt (§ 356 Abs. 5 BGB).",
      },
      terms_of_service_acceptance: {
        message: `Ich habe die [AGB](${siteUrl("/legal/agb/")}) und die [Widerrufsbelehrung](${siteUrl("/legal/widerruf/")}) gelesen und akzeptiere sie.`,
      },
    },
    metadata: {
      supabase_user_id: user.id,
      price_id: price.id,
      credits: String(product.credits),
      legal_version: version,
    },
    payment_intent_data: {
      metadata: { supabase_user_id: user.id, credits: String(product.credits) },
    },
    expires_at: Math.floor(Date.now() / 1000) + 30 * 60,
  });

  return json(req, { url: session.url, id: session.id });
});
