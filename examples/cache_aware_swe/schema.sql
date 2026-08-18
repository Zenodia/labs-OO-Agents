CREATE TABLE IF NOT EXISTS authenticated_user (
    user_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    auth_subject TEXT NOT NULL,
    roles JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, auth_subject)
);

CREATE TABLE IF NOT EXISTS agent_identity (
    agent_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('orchestrator', 'planner', 'investigator', 'patcher', 'tester')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_run (
    run_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES authenticated_user(user_id),
    root_agent_id TEXT NOT NULL REFERENCES agent_identity(agent_id),
    repository TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    issue_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS context_segment (
    segment_id UUID PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES agent_run(run_id) ON DELETE CASCADE,
    parent_segment_id UUID REFERENCES context_segment(segment_id),
    scope TEXT NOT NULL CHECK (scope IN ('shared', 'tenant', 'run', 'ephemeral')),
    label TEXT NOT NULL,
    canonical_hash CHAR(64) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cache_reference (
    cache_reference_id UUID PRIMARY KEY,
    segment_id UUID NOT NULL REFERENCES context_segment(segment_id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    tokenizer_revision TEXT NOT NULL,
    runtime_revision TEXT NOT NULL,
    cache_scope TEXT NOT NULL CHECK (cache_scope IN ('shared', 'tenant', 'run', 'ephemeral')),
    salt_fingerprint CHAR(64) NOT NULL,
    prefix_hash CHAR(64) NOT NULL,
    priority SMALLINT NOT NULL CHECK (priority BETWEEN 0 AND 100),
    ttl_seconds INTEGER NOT NULL CHECK (ttl_seconds >= 0),
    status TEXT NOT NULL DEFAULT 'requested',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS cache_reference_lookup
    ON cache_reference (provider, model_revision, tokenizer_revision, runtime_revision, prefix_hash);
