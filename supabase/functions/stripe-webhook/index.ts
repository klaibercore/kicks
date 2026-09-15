/**
 * Stripe -> Supabase. Signature-verified, idempotent on the event id, and
 * every state change goes through a SECURITY DEFINER function so the ledger
 * invariants hold even if this function is deployed twice.
 *
 * Subscribe this endpoint to:
 *   checkout.session.completed, checkout.session.async_payment_succeeded,
 *   charge.refunded, product.created, product.updated, product.deleted,
 *   price.created, price.updated, price.deleted
 */
import type Stripe from "npm:stripe@18.5.0";
import { cryptoProvider, stripe } from "../_shared/stripe.ts";
import { admin } from "../_shared/supabase.ts";

const webhookSecret = Deno.env.get("STRIPE_WEBHOOK_SECRET") ?? "";

async function upsertProduct(product: Stripe.Product) {
  const credits = Number.parseInt(product.metadata?.credits ?? "", 10);
  if (!Number.isFinite(credits) || credits <= 0) {
    console.warn(`product ${product.id} has no metadata.credits; skipping`);
    return;
  }
  await admin.from("products").upsert({
    id: product.id,
    active: product.active,
    name: product.name,
    description: product.description,
    credits,
    metadata: product.metadata ?? {},
  });
}

async function upsertPrice(price: Stripe.Price) {
  const productId = typeof price.product === "string" ? price.product : price.product.id;
  // Make sure the parent exists (price events can arrive before product ones).
  const { data: parent } = await admin.from("products").select("id").eq("id", productId).maybeSingle();
  if (!parent) {
    const product = await stripe.products.retrieve(productId);
    await upsertProduct(product);
  }
  await admin.from("prices").upsert({
    id: price.id,
    product_id: productId,
    active: price.active,
    currency: price.currency,
    unit_amount: price.unit_amount ?? 0,
    type: price.type,
  });
}

async function fulfill(session: Stripe.Checkout.Session) {
  if (session.payment_status !== "paid") return;
  const userId = session.metadata?.supabase_user_id ?? session.client_reference_id;
  const credits = Number.parseInt(session.metadata?.credits ?? "", 10);
  const priceId = session.metadata?.price_id ?? null;
  if (!userId || !Number.isFinite(credits)) {
    throw new Error(`session ${session.id} is missing user or credits metadata`);
  }
  const { error } = await admin.rpc("fulfill_checkout", {
    p_user_id: userId,
    p_session_id: session.id,
    p_payment_intent_id: typeof session.payment_intent === "string" ? session.payment_intent : session.payment_intent?.id ?? null,
    p_invoice_id: typeof session.invoice === "string" ? session.invoice : session.invoice?.id ?? null,
    p_price_id: priceId,
    p_credits: credits,
    p_amount_subtotal: session.amount_subtotal ?? 0,
    p_amount_tax: session.total_details?.amount_tax ?? 0,
    p_amount_total: session.amount_total ?? 0,
    p_currency: session.currency ?? "eur",
  });
  if (error) throw error;
}

async function handle(event: Stripe.Event) {
  switch (event.type) {
    case "checkout.session.completed":
    case "checkout.session.async_payment_succeeded":
      await fulfill(event.data.object as Stripe.Checkout.Session);
      break;
    case "charge.refunded": {
      const charge = event.data.object as Stripe.Charge;
      const intent = typeof charge.payment_intent === "string" ? charge.payment_intent : charge.payment_intent?.id;
      if (!intent) break;
      const { error } = await admin.rpc("refund_order", {
        p_payment_intent_id: intent,
        p_amount_refunded: charge.amount_refunded,
        p_full: charge.refunded,
      });
      if (error) throw error;
      break;
    }
    case "product.created":
    case "product.updated":
      await upsertProduct(event.data.object as Stripe.Product);
      break;
    case "product.deleted":
      await admin.from("products").update({ active: false }).eq("id", (event.data.object as Stripe.Product).id);
      break;
    case "price.created":
    case "price.updated":
      await upsertPrice(event.data.object as Stripe.Price);
      break;
    case "price.deleted":
      await admin.from("prices").update({ active: false }).eq("id", (event.data.object as Stripe.Price).id);
      break;
    default:
      // Unsubscribed event types are acknowledged and ignored.
      break;
  }
}

Deno.serve(async (req) => {
  const signature = req.headers.get("stripe-signature");
  if (!signature) return new Response("missing signature", { status: 400 });
  const body = await req.text();

  let event: Stripe.Event;
  try {
    event = await stripe.webhooks.constructEventAsync(body, signature, webhookSecret, undefined, cryptoProvider);
  } catch (err) {
    console.error("signature verification failed", err);
    return new Response("invalid signature", { status: 400 });
  }

  // Claim the event; a duplicate delivery is acknowledged without re-processing.
  const { error: claimError } = await admin.from("stripe_events").insert({ id: event.id, type: event.type });
  if (claimError) {
    if (claimError.code === "23505") return new Response("duplicate", { status: 200 });
    console.error("could not record event", claimError);
    return new Response("storage error", { status: 500 });
  }

  try {
    await handle(event);
    await admin.from("stripe_events").update({ processed_at: new Date().toISOString() }).eq("id", event.id);
    return new Response("ok", { status: 200 });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`event ${event.id} failed:`, message);
    // Release the claim so Stripe's retry can process it.
    await admin.from("stripe_events").delete().eq("id", event.id);
    return new Response("processing failed", { status: 500 });
  }
});
