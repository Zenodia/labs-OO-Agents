"""No-model cache-awareness walkthrough for the SWE-agent context policy.

This reports expected reuse eligibility. It does not claim a physical KV hit:
only a running Dynamo/TRT-LLM deployment can report actual cache metrics.
"""

from __future__ import annotations

import hashlib

from context_engineering import AgentRole, ContextCompiler, Handoff
from state import CacheScope, RunState


SECRET = "demo-only"


def make_state(tenant_id: str = "acme") -> RunState:
    state = RunState(
        tenant_id,
        "user-001",
        "swe-001",
        "orchestrator-001",
        "owner/repo",
        "abc123",
        "#42",
    )
    state.append(CacheScope.SHARED, "Common tools", "Use git, tests, and artifact references.")
    state.append(CacheScope.RUN, "Issue", "Mitigate #42 and run focused tests.")
    state.append(CacheScope.EPHEMERAL, "Scratchpad", "Never hand this raw trace to another agent.")
    return state


def report(name: str, expectation: str, **values: object) -> None:
    print(f"\n=== {name} ===")
    print("Expected:", expectation)
    for key, value in values.items():
        print(f"{key}: {value}")


def logical_reuse_metrics(*, current_tokens: int, reusable_tokens: int) -> dict[str, object]:
    """Report no-model prefix eligibility in estimated tokens, not a GPU cache hit."""
    reusable_tokens = min(current_tokens, reusable_tokens)
    return {
        "full_prompt_tokens_estimate": current_tokens,
        "logical_reused_tokens_estimate": reusable_tokens,
        "new_prefill_tokens_estimate": current_tokens - reusable_tokens,
        "logical_prefix_reuse_pct_estimate": f"{100 * reusable_tokens / current_tokens:.1f}%",
        "physical_kv_hit": "not measured — requires live Dynamo/TRT-LLM metrics",
    }


compiler = ContextCompiler()
state = make_state()
handoff = Handoff("Mitigate #42", "A testable patch plan")
orchestrator = compiler.compile(state, AgentRole.ORCHESTRATOR, handoff)
orchestrator_resume = compiler.compile(state, AgentRole.ORCHESTRATOR, handoff)
planner = compiler.compile(state, AgentRole.PLANNER, handoff)
other_tenant_state = make_state("other-tenant")
other_tenant = compiler.compile(other_tenant_state, AgentRole.ORCHESTRATOR, handoff)

report(
    "1. Orchestrator first call",
    "Cold prefill; Dynamo may create reusable blocks.",
    shared_prefix_hash=orchestrator.shared_prefix_hash[:16],
    role_prefix_hash=orchestrator.role_prefix_hash[:16],
    full_prompt_hash=orchestrator.prompt_hash[:16],
    tenant_salt_fingerprint=hashlib.sha256(state.cache_salt(SECRET).encode()).hexdigest()[:16],
    **logical_reuse_metrics(
        current_tokens=orchestrator.estimated_tokens,
        reusable_tokens=0,
    ),
)
report(
    "2. Orchestrator resumes unchanged",
    "Full-prefix reuse is eligible: same canonical prompt and same tenant salt.",
    full_prefix_equal=orchestrator.prompt_hash == orchestrator_resume.prompt_hash,
    tenant_salt_equal=state.cache_salt(SECRET) == state.cache_salt(SECRET),
    **logical_reuse_metrics(
        current_tokens=orchestrator_resume.estimated_tokens,
        reusable_tokens=orchestrator.estimated_tokens,
    ),
)
report(
    "3. Planner handoff",
    "Shared prefix can reuse; planner role overlay and handoff suffix require new prefill.",
    shared_prefix_equal=orchestrator.shared_prefix_hash == planner.shared_prefix_hash,
    role_prefix_equal=orchestrator.role_prefix_hash == planner.role_prefix_hash,
    full_prefix_equal=orchestrator.prompt_hash == planner.prompt_hash,
    **logical_reuse_metrics(
        current_tokens=planner.estimated_tokens,
        reusable_tokens=planner.shared_prefix_tokens,
    ),
)
report(
    "4. Different tenant",
    "No physical reuse: the tenant-derived cache salt differs even if text overlaps.",
    shared_prefix_equal=orchestrator.shared_prefix_hash == other_tenant.shared_prefix_hash,
    tenant_salt_equal=state.cache_salt(SECRET) == other_tenant_state.cache_salt(SECRET),
    **logical_reuse_metrics(
        current_tokens=other_tenant.estimated_tokens,
        reusable_tokens=0,
    ),
)
report(
    "5. External provider escalation",
    "Semantic handoff only; provider-managed KV is not visible to local Dynamo.",
    handoff_fields="goal, expected output, artifact URIs, repository SHA",
)

small_budget = ContextCompiler(token_budget=32, compression_ratio=0.8)
long_state = make_state()
long_state.append(CacheScope.RUN, "Tool summary", "evidence " * 40)
compressed = small_budget.compile(long_state, AgentRole.INVESTIGATOR, handoff)
report(
    "6. Compression trigger",
    "Persist a compact summary and remove older raw trajectory before the next turn.",
    estimated_tokens=compressed.estimated_tokens,
    shared_prefix_tokens_estimate=compressed.shared_prefix_tokens,
    compression_required=compressed.compression_required,
)
