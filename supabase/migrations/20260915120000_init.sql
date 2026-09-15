-- kicks: accounts, credits, orders, kits.
--
-- Design notes
--   * Credits are an append-only ledger, not a balance column. A balance is a
--     sum; every change has a reason and a reference, and refunds are just more
--     rows. That is what an accountant (and a chargeback dispute) wants to see.
--   * Every table has row-level security. Users read their own rows. Nothing
--     is written from the browser except profiles, kits and consent records;
--     money and credits are only ever written by the service role (webhooks,
--     the synthesis API) through SECURITY DEFINER functions below.
--   * Orders keep the Stripe identifiers, amounts and tax — the invoice itself
--     lives in Stripe. When an account is deleted the order rows stay behind
--     with user_id set to NULL: § 147 AO / § 14b UStG require invoice records
--     to be kept for ten years, which Art. 17(3)(b) GDPR expressly allows.
--   * The Stripe catalogue (products + prices) is mirrored by webhook so the
--     pricing page never needs a secret key.

create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- profiles
-- ---------------------------------------------------------------------------

create table public.profiles (
  id            uuid primary key references auth.users (id) on delete cascade,
  display_name  text,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now(),
  constraint display_name_length check (display_name is null or char_length(display_name) between 1 and 60)
);

comment on table public.profiles is 'Public-facing profile, one per auth user. Email stays in auth.users.';

alter table public.profiles enable row level security;

create policy "profiles: read own"   on public.profiles for select using (auth.uid() = id);
create policy "profiles: update own" on public.profiles for update using (auth.uid() = id) with check (auth.uid() = id);

-- ---------------------------------------------------------------------------
-- billing customers (user -> Stripe customer)
-- ---------------------------------------------------------------------------

create table public.billing_customers (
  user_id             uuid primary key references auth.users (id) on delete cascade,
  stripe_customer_id  text not null unique,
  created_at          timestamptz not null default now()
);

alter table public.billing_customers enable row level security;
-- No policies: service role only. Users never need the raw Stripe id.

-- ---------------------------------------------------------------------------
-- Stripe catalogue mirror
-- ---------------------------------------------------------------------------

create table public.products (
  id           text primary key,                -- Stripe product id
  active       boolean not null default true,
  name         text not null,
  description  text,
  credits      integer not null check (credits > 0),
  metadata     jsonb not null default '{}'::jsonb,
  updated_at   timestamptz not null default now()
);

create table public.prices (
  id            text primary key,               -- Stripe price id
  product_id    text not null references public.products (id) on delete cascade,
  active        boolean not null default true,
  currency      text not null check (currency = lower(currency)),
  unit_amount   bigint not null check (unit_amount >= 0),  -- minor units, gross
  type          text not null check (type in ('one_time', 'recurring')),
  updated_at    timestamptz not null default now()
);

alter table public.products enable row level security;
alter table public.prices   enable row level security;

create policy "products: public read active" on public.products for select using (active);
create policy "prices: public read active"   on public.prices   for select using (active);

-- ---------------------------------------------------------------------------
-- orders
-- ---------------------------------------------------------------------------

create table public.orders (
  id                          uuid primary key default gen_random_uuid(),
  user_id                     uuid references auth.users (id) on delete set null,
  stripe_checkout_session_id  text not null unique,
  stripe_payment_intent_id    text unique,
  stripe_invoice_id           text,
  price_id                    text references public.prices (id),
  credits                     integer not null check (credits > 0),
  amount_subtotal             bigint not null,
  amount_tax                  bigint not null default 0,
  amount_total                bigint not null,
  currency                    text not null,
  status                      text not null check (status in ('paid', 'refunded', 'partially_refunded')),
  created_at                  timestamptz not null default now(),
  updated_at                  timestamptz not null default now()
);

create index orders_user_id_idx on public.orders (user_id, created_at desc);

alter table public.orders enable row level security;
create policy "orders: read own" on public.orders for select using (auth.uid() = user_id);

-- ---------------------------------------------------------------------------
-- credit ledger
-- ---------------------------------------------------------------------------

create table public.credit_ledger (
  id          bigint generated always as identity primary key,
  user_id     uuid not null references auth.users (id) on delete cascade,
  delta       integer not null check (delta <> 0),
  reason      text not null check (reason in ('welcome', 'purchase', 'refund', 'export', 'export_refund', 'adjustment')),
  ref_type    text,
  ref_id      text,
  note        text,
  created_at  timestamptz not null default now()
);

-- One ledger row per (reason, reference): a webhook delivered twice or an
-- export retried with the same idempotency key cannot double-book.
create unique index credit_ledger_ref_idx on public.credit_ledger (reason, ref_type, ref_id) where ref_id is not null;
create index credit_ledger_user_idx on public.credit_ledger (user_id, created_at desc);

alter table public.credit_ledger enable row level security;
create policy "ledger: read own" on public.credit_ledger for select using (auth.uid() = user_id);

-- ---------------------------------------------------------------------------
-- generations (exported samples)
-- ---------------------------------------------------------------------------

create table public.generations (
  id          uuid primary key,                 -- the export's idempotency key
  user_id     uuid not null references auth.users (id) on delete cascade,
  instrument  text not null,
  params      jsonb not null,
  ledger_id   bigint references public.credit_ledger (id),
  license     text not null default 'kicks-sample-license-v1',
  refunded    boolean not null default false,
  created_at  timestamptz not null default now()
);

create index generations_user_idx on public.generations (user_id, created_at desc);

alter table public.generations enable row level security;
create policy "generations: read own" on public.generations for select using (auth.uid() = user_id);

-- ---------------------------------------------------------------------------
-- kits (saved pad layouts)
-- ---------------------------------------------------------------------------

create table public.kits (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users (id) on delete cascade,
  name        text not null check (char_length(name) between 1 and 60),
  pads        jsonb not null,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now(),
  constraint pads_is_array check (jsonb_typeof(pads) = 'array')
);

create index kits_user_idx on public.kits (user_id, updated_at desc);

alter table public.kits enable row level security;
create policy "kits: read own"   on public.kits for select using (auth.uid() = user_id);
create policy "kits: insert own" on public.kits for insert with check (auth.uid() = user_id);
create policy "kits: update own" on public.kits for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "kits: delete own" on public.kits for delete using (auth.uid() = user_id);

-- ---------------------------------------------------------------------------
-- consent + legal acceptance records (Art. 7(1) GDPR: the controller must be
-- able to demonstrate consent; § 312j BGB: the order confirmation).
-- ---------------------------------------------------------------------------

create table public.consents (
  id          bigint generated always as identity primary key,
  user_id     uuid not null references auth.users (id) on delete cascade,
  kind        text not null check (kind in ('analytics', 'terms', 'privacy', 'withdrawal_waiver')),
  version     text not null,
  granted     boolean not null,
  created_at  timestamptz not null default now()
);

create index consents_user_idx on public.consents (user_id, kind, created_at desc);

alter table public.consents enable row level security;
create policy "consents: read own"   on public.consents for select using (auth.uid() = user_id);
create policy "consents: insert own" on public.consents for insert with check (auth.uid() = user_id);

-- ---------------------------------------------------------------------------
-- stripe events (webhook idempotency)
-- ---------------------------------------------------------------------------

create table public.stripe_events (
  id            text primary key,               -- Stripe event id
  type          text not null,
  received_at   timestamptz not null default now(),
  processed_at  timestamptz,
  error         text
);

alter table public.stripe_events enable row level security;
-- Service role only.

-- ---------------------------------------------------------------------------
-- housekeeping triggers
-- ---------------------------------------------------------------------------

create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end $$;

create trigger profiles_updated_at before update on public.profiles for each row execute function public.set_updated_at();
create trigger orders_updated_at   before update on public.orders   for each row execute function public.set_updated_at();
create trigger kits_updated_at     before update on public.kits     for each row execute function public.set_updated_at();
create trigger products_updated_at before update on public.products for each row execute function public.set_updated_at();
create trigger prices_updated_at   before update on public.prices   for each row execute function public.set_updated_at();

-- New sign-up: create the profile and grant the welcome credits.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, display_name)
  values (new.id, nullif(new.raw_user_meta_data ->> 'name', ''));

  insert into public.credit_ledger (user_id, delta, reason, ref_type, ref_id, note)
  values (new.id, 3, 'welcome', 'user', new.id::text, 'welcome credits');

  return new;
end $$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ---------------------------------------------------------------------------
-- credits API (service role and the account holder)
-- ---------------------------------------------------------------------------

create or replace function public.credit_balance(p_user_id uuid default auth.uid())
returns integer
language sql
stable
security definer
set search_path = public
as $$
  select coalesce(sum(delta), 0)::integer
  from public.credit_ledger
  where user_id = p_user_id
    and (p_user_id = auth.uid() or auth.role() = 'service_role');
$$;

comment on function public.credit_balance is 'Current balance. Users may ask for their own; the service role for anyone.';

-- Spend one credit for an export. Idempotent on p_export_id: a retried request
-- returns the existing charge instead of taking another credit. Raises
-- 'insufficient_credits' when the balance is empty.
create or replace function public.charge_export(
  p_user_id    uuid,
  p_export_id  uuid,
  p_instrument text,
  p_params     jsonb
)
returns table (balance integer, generation_id uuid)
language plpgsql
security definer
set search_path = public
as $$
declare
  v_balance   integer;
  v_ledger_id bigint;
begin
  if auth.role() <> 'service_role' then
    raise exception 'forbidden' using errcode = '42501';
  end if;

  -- Serialise concurrent charges for one user.
  perform pg_advisory_xact_lock(hashtext(p_user_id::text));

  if exists (select 1 from public.generations g where g.id = p_export_id) then
    return query
      select public.credit_balance(p_user_id), p_export_id;
    return;
  end if;

  select coalesce(sum(delta), 0) into v_balance
  from public.credit_ledger where user_id = p_user_id;

  if v_balance < 1 then
    raise exception 'insufficient_credits' using errcode = 'P0001';
  end if;

  insert into public.credit_ledger (user_id, delta, reason, ref_type, ref_id)
  values (p_user_id, -1, 'export', 'generation', p_export_id::text)
  returning id into v_ledger_id;

  insert into public.generations (id, user_id, instrument, params, ledger_id)
  values (p_export_id, p_user_id, p_instrument, p_params, v_ledger_id);

  return query select v_balance - 1, p_export_id;
end $$;

-- Undo a charge when rendering failed after the credit was taken.
create or replace function public.refund_export(p_export_id uuid)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare
  v_user uuid;
begin
  if auth.role() <> 'service_role' then
    raise exception 'forbidden' using errcode = '42501';
  end if;

  select user_id into v_user from public.generations
  where id = p_export_id and not refunded;
  if v_user is null then
    return;
  end if;

  insert into public.credit_ledger (user_id, delta, reason, ref_type, ref_id, note)
  values (v_user, 1, 'export_refund', 'generation', p_export_id::text, 'render failed')
  on conflict do nothing;

  update public.generations set refunded = true where id = p_export_id;
end $$;

-- Fulfil a paid Checkout Session: record the order and book the credits in one
-- transaction. Idempotent on the session id.
create or replace function public.fulfill_checkout(
  p_user_id            uuid,
  p_session_id         text,
  p_payment_intent_id  text,
  p_invoice_id         text,
  p_price_id           text,
  p_credits            integer,
  p_amount_subtotal    bigint,
  p_amount_tax         bigint,
  p_amount_total       bigint,
  p_currency           text
)
returns uuid
language plpgsql
security definer
set search_path = public
as $$
declare
  v_order_id uuid;
begin
  if auth.role() <> 'service_role' then
    raise exception 'forbidden' using errcode = '42501';
  end if;

  insert into public.orders (
    user_id, stripe_checkout_session_id, stripe_payment_intent_id, stripe_invoice_id,
    price_id, credits, amount_subtotal, amount_tax, amount_total, currency, status
  ) values (
    p_user_id, p_session_id, p_payment_intent_id, p_invoice_id,
    p_price_id, p_credits, p_amount_subtotal, p_amount_tax, p_amount_total, lower(p_currency), 'paid'
  )
  on conflict (stripe_checkout_session_id) do nothing
  returning id into v_order_id;

  if v_order_id is null then
    select id into v_order_id from public.orders where stripe_checkout_session_id = p_session_id;
    return v_order_id;
  end if;

  insert into public.credit_ledger (user_id, delta, reason, ref_type, ref_id)
  values (p_user_id, p_credits, 'purchase', 'order', v_order_id::text);

  return v_order_id;
end $$;

-- A refund on Stripe takes the credits back (the balance may go negative if
-- they were already spent; that is the honest state of the account).
create or replace function public.refund_order(
  p_payment_intent_id text,
  p_amount_refunded   bigint,
  p_full              boolean
)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare
  v_order public.orders%rowtype;
  v_credits integer;
begin
  if auth.role() <> 'service_role' then
    raise exception 'forbidden' using errcode = '42501';
  end if;

  select * into v_order from public.orders where stripe_payment_intent_id = p_payment_intent_id;
  if v_order.id is null then
    return;
  end if;

  v_credits := case
    when p_full then v_order.credits
    else greatest(1, round(v_order.credits * p_amount_refunded::numeric / nullif(v_order.amount_total, 0))::integer)
  end;

  update public.orders
  set status = case when p_full then 'refunded' else 'partially_refunded' end
  where id = v_order.id;

  if v_order.user_id is not null then
    insert into public.credit_ledger (user_id, delta, reason, ref_type, ref_id, note)
    values (v_order.user_id, -v_credits, 'refund', 'order', v_order.id::text,
            case when p_full then 'full refund' else 'partial refund' end)
    on conflict do nothing;
  end if;
end $$;

-- ---------------------------------------------------------------------------
-- data subject rights
-- ---------------------------------------------------------------------------

-- Art. 15 / Art. 20 GDPR: everything we hold about the caller, as one JSON
-- document they can download.
create or replace function public.export_own_data()
returns jsonb
language sql
stable
security definer
set search_path = public
as $$
  select jsonb_build_object(
    'exported_at', now(),
    'user', (select jsonb_build_object('id', u.id, 'email', u.email, 'created_at', u.created_at,
                                       'last_sign_in_at', u.last_sign_in_at,
                                       'providers', u.raw_app_meta_data -> 'providers')
             from auth.users u where u.id = auth.uid()),
    'profile',     (select to_jsonb(p) from public.profiles p where p.id = auth.uid()),
    'credits',     (select jsonb_build_object('balance', public.credit_balance(auth.uid()),
                                              'ledger', coalesce(jsonb_agg(to_jsonb(l) order by l.created_at), '[]'::jsonb))
                    from public.credit_ledger l where l.user_id = auth.uid()),
    'orders',      (select coalesce(jsonb_agg(to_jsonb(o) order by o.created_at), '[]'::jsonb) from public.orders o where o.user_id = auth.uid()),
    'generations', (select coalesce(jsonb_agg(to_jsonb(g) order by g.created_at), '[]'::jsonb) from public.generations g where g.user_id = auth.uid()),
    'kits',        (select coalesce(jsonb_agg(to_jsonb(k) order by k.created_at), '[]'::jsonb) from public.kits k where k.user_id = auth.uid()),
    'consents',    (select coalesce(jsonb_agg(to_jsonb(c) order by c.created_at), '[]'::jsonb) from public.consents c where c.user_id = auth.uid())
  );
$$;

revoke all on function public.export_own_data() from public;
grant execute on function public.export_own_data() to authenticated;

-- Art. 17 GDPR. Called by the delete-account Edge Function (service role) after
-- it has removed the Stripe customer. Order rows are detached, not deleted —
-- see the header comment.
create or replace function public.erase_user(p_user_id uuid)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  if auth.role() <> 'service_role' then
    raise exception 'forbidden' using errcode = '42501';
  end if;
  update public.orders set user_id = null where user_id = p_user_id;
  delete from auth.users where id = p_user_id;   -- cascades to everything else
end $$;

-- Only the roles that need them.
revoke all on function public.charge_export(uuid, uuid, text, jsonb) from public;
revoke all on function public.refund_export(uuid) from public;
revoke all on function public.fulfill_checkout(uuid, text, text, text, text, integer, bigint, bigint, bigint, text) from public;
revoke all on function public.refund_order(text, bigint, boolean) from public;
revoke all on function public.erase_user(uuid) from public;
grant execute on function public.charge_export(uuid, uuid, text, jsonb) to service_role;
grant execute on function public.refund_export(uuid) to service_role;
grant execute on function public.fulfill_checkout(uuid, text, text, text, text, integer, bigint, bigint, bigint, text) to service_role;
grant execute on function public.refund_order(text, bigint, boolean) to service_role;
grant execute on function public.erase_user(uuid) to service_role;
grant execute on function public.credit_balance(uuid) to authenticated, service_role;
