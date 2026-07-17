-- Run once against the Postgres database at DATABASE_URL (Supabase SQL Editor, or `psql`).
--
-- Custom auth (our own users/sessions tables) + durable job/results storage. We do not use
-- Supabase Auth at all -- passwords are hashed and verified by us (app/auth.py), and
-- sessions are opaque tokens looked up in the `sessions` table (app/db.py), not JWTs.
--
-- No row-level security: RLS's `auth.uid()` is a Supabase-Auth-specific JWT claim that does
-- not exist in this setup (the backend connects directly as a superuser/pooler role, with
-- no per-request JWT context for Postgres to see) -- an RLS policy referencing it would
-- silently return nothing for everyone, which is worse than no policy at all since it looks
-- like protection while providing none. Enforcement is the explicit `user_id` filter in
-- every app/db.py query function instead.

drop table if exists jobs cascade;
drop table if exists sessions cascade;
drop table if exists users cascade;

create table users (
    id uuid primary key default gen_random_uuid(),
    email text not null unique,
    password_hash text not null,
    created_at timestamptz not null default now()
);

create table sessions (
    token text primary key,              -- opaque random token; sent as-is as the Bearer value
    user_id uuid not null references users(id) on delete cascade,
    created_at timestamptz not null default now(),
    expires_at timestamptz not null
);

create index sessions_user_id_idx on sessions (user_id);

create table jobs (
    id uuid primary key default gen_random_uuid(),
    batch_id uuid not null,
    user_id uuid not null references users(id) on delete cascade,
    filename text not null,
    rq_job_id text,
    status text not null default 'queued',   -- queued | started | finished | failed
    stage text,                              -- preprocessing | analyzing | done (while started)
    result jsonb,                            -- the structured result (see result.py) once finished
    error text,                              -- error message once failed
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index jobs_batch_id_idx on jobs (batch_id);
create index jobs_user_id_idx on jobs (user_id);

-- Keep `updated_at` current on every write.
create or replace function set_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

create trigger jobs_set_updated_at
    before update on jobs
    for each row
    execute function set_updated_at();
