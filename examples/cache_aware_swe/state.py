"""Small, provider-neutral state and cache-policy primitives for a SWE agent.

This module intentionally does not persist physical KV tensors.  A KV cache is
runtime/model specific and ephemeral; this module records durable state and
produces a compatible-prefix key for a Dynamo-aware inference client.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum


class CacheScope(StrEnum):
    SHARED = "shared"
    TENANT = "tenant"
    RUN = "run"
    EPHEMERAL = "ephemeral"


@dataclass(frozen=True)
class ContextSegment:
    """An immutable, canonical prompt segment."""

    scope: CacheScope
    label: str
    text: str


@dataclass(frozen=True)
class CachePolicy:
    priority: int
    ttl_seconds: int
    retain: bool


POLICY = {
    CacheScope.SHARED: CachePolicy(priority=100, ttl_seconds=3600, retain=True),
    CacheScope.TENANT: CachePolicy(priority=80, ttl_seconds=1800, retain=True),
    CacheScope.RUN: CachePolicy(priority=60, ttl_seconds=900, retain=True),
    CacheScope.EPHEMERAL: CachePolicy(priority=1, ttl_seconds=0, retain=False),
}


@dataclass
class RunState:
    """Durable state. Persist ``to_json()`` in Postgres/object storage, not KV."""

    tenant_id: str
    user_id: str
    run_id: str
    agent_id: str
    repository: str
    commit_sha: str
    issue_id: str
    segments: list[ContextSegment] = field(default_factory=list)

    def append(self, scope: CacheScope, label: str, text: str) -> None:
        self.segments.append(ContextSegment(scope=scope, label=label, text=text))

    def prompt(self, include_ephemeral: bool = True) -> str:
        segments = (
            self.segments
            if include_ephemeral
            else [s for s in self.segments if s.scope != CacheScope.EPHEMERAL]
        )
        return "\n\n".join(f"## {segment.label}\n{segment.text.strip()}" for segment in segments)

    def prefix_key(self, *, model: str, tokenizer_revision: str, runtime_revision: str) -> str:
        """Hash the exact cache-compatibility boundary, never raw state alone."""
        payload = {
            "tenant": self.tenant_id,
            "model": model,
            "tokenizer_revision": tokenizer_revision,
            "runtime_revision": runtime_revision,
            "segments": [
                asdict(segment)
                for segment in self.segments
                if segment.scope != CacheScope.EPHEMERAL
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True)

    def cache_salt(self, secret: str, scope: CacheScope = CacheScope.TENANT) -> str:
        """Derive a non-reversible TRT-LLM cache salt; never store this raw value."""
        if scope == CacheScope.SHARED:
            identity = "deployment"
        elif scope == CacheScope.TENANT:
            identity = f"tenant:{self.tenant_id}"
        elif scope == CacheScope.RUN:
            identity = f"run:{self.tenant_id}:{self.run_id}"
        else:
            identity = f"agent:{self.tenant_id}:{self.run_id}:{self.agent_id}"
        return hmac.new(secret.encode(), identity.encode(), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class ModelTarget:
    name: str
    family: str
    quality_tier: int


@dataclass(frozen=True)
class RouteDecision:
    target: ModelTarget
    reuse_physical_kv: bool
    reason: str


def route(
    *,
    anchor: ModelTarget,
    candidate: ModelTarget,
    prefix_tokens: int,
    requires_high_quality: bool,
    kv_transfer_ready: bool = False,
) -> RouteDecision:
    """Prefer cache reuse for long contexts; only claim KV reuse within a family."""
    same_family = anchor.family == candidate.family
    if requires_high_quality or prefix_tokens >= 8_192:
        return RouteDecision(anchor, True, "keep anchor model: quality or cache value dominates")
    if same_family and kv_transfer_ready:
        return RouteDecision(
            candidate,
            True,
            "same-family route; compatible KV transfer is a measured extension",
        )
    if same_family:
        return RouteDecision(
            candidate,
            False,
            "same family without a validated KV mapper; send durable state",
        )
    return RouteDecision(
        candidate,
        False,
        "cross-provider route: hand off durable summary, not physical KV",
    )


def dynamo_hints(state: RunState, cache_salt_secret: str | None = None) -> dict[str, object]:
    """Provider-neutral intent to map to the installed Dynamo agent-hints API."""
    active_scopes = {segment.scope for segment in state.segments}
    if not active_scopes:
        hints: dict[str, object] = {
            "session_id": state.run_id,
            "tenant_cache_namespace": state.tenant_id,
            "cache_priority": 0,
            "cache_ttl_seconds": 0,
            "prefetch_on_tool_return": False,
        }
        if cache_salt_secret:
            hints["cache_salt"] = state.cache_salt(cache_salt_secret)
        return hints
    # A scratchpad must not demote the system/tool prefix in the same request.
    # Region-level retention is an engine concern; this request-level hint keeps
    # the most valuable reusable segment warm.
    policy = max((POLICY[scope] for scope in active_scopes), key=lambda item: item.priority)
    hints = {
        "session_id": state.run_id,
        "tenant_cache_namespace": state.tenant_id,
        "cache_priority": policy.priority,
        "cache_ttl_seconds": policy.ttl_seconds,
        "prefetch_on_tool_return": True,
    }
    if cache_salt_secret:
        hints["cache_salt"] = state.cache_salt(cache_salt_secret)
    return hints
