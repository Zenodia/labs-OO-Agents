"""Postgres control plane for durable agent state and KV-cache metadata."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import psycopg

from state import CacheScope, ContextSegment, RunState


class StateRepository:
    """Stores authorization/state metadata; it never stores physical KV tensors."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def initialize(self) -> None:
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        with psycopg.connect(self.database_url) as connection:
            connection.execute(schema, prepare=False)

    def create_run(self, state: RunState, auth_subject: str, role: str = "orchestrator") -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                """INSERT INTO authenticated_user (user_id, tenant_id, auth_subject)
                   VALUES (%s, %s, %s) ON CONFLICT (user_id) DO NOTHING""",
                (state.user_id, state.tenant_id, auth_subject),
            )
            connection.execute(
                """INSERT INTO agent_identity (agent_id, tenant_id, role)
                   VALUES (%s, %s, %s) ON CONFLICT (agent_id) DO NOTHING""",
                (state.agent_id, state.tenant_id, role),
            )
            connection.execute(
                """INSERT INTO agent_run
                   (run_id, tenant_id, user_id, root_agent_id, repository, commit_sha, issue_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (run_id) DO NOTHING""",
                (
                    state.run_id,
                    state.tenant_id,
                    state.user_id,
                    state.agent_id,
                    state.repository,
                    state.commit_sha,
                    state.issue_id,
                ),
            )

    def append_segment(
        self,
        run_id: str,
        segment: ContextSegment,
        parent_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        segment_id = uuid.uuid4()
        canonical_hash = hashlib.sha256(segment.text.encode()).hexdigest()
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                """INSERT INTO context_segment
                   (segment_id, run_id, parent_segment_id, scope, label, canonical_hash, content)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    segment_id,
                    run_id,
                    parent_id,
                    segment.scope,
                    segment.label,
                    canonical_hash,
                    segment.text,
                ),
            )
        return segment_id

    def record_cache_reference(
        self,
        segment_id: uuid.UUID,
        *,
        provider: str,
        model_revision: str,
        tokenizer_revision: str,
        runtime_revision: str,
        cache_scope: CacheScope,
        cache_salt: str,
        prefix_hash: str,
        priority: int,
        ttl_seconds: int,
    ) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                """INSERT INTO cache_reference
                   (cache_reference_id, segment_id, provider, model_revision,
                    tokenizer_revision, runtime_revision, cache_scope, salt_fingerprint,
                    prefix_hash, priority, ttl_seconds)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    uuid.uuid4(),
                    segment_id,
                    provider,
                    model_revision,
                    tokenizer_revision,
                    runtime_revision,
                    cache_scope,
                    hashlib.sha256(cache_salt.encode()).hexdigest(),
                    prefix_hash,
                    priority,
                    ttl_seconds,
                ),
            )
