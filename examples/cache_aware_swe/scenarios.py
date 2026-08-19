"""No-model cache-awareness walkthrough for the SWE-agent context policy.

This reports logical prefix eligibility. It does not claim a physical KV hit:
only a running Dynamo/TRT-LLM deployment can emit physical cache metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from context_engineering import AgentRole, CompiledContext, ContextCompiler, Handoff
from state import CacheScope, RunState


SECRET = "demo-only"


@dataclass(frozen=True)
class TaskQuery:
    """Minimal SWE-bench-compatible task fields used by the agent harness."""

    source: str
    instance_id: str
    repository: str
    base_commit: str
    problem_statement: str


DEFAULT_TASK_QUERY = TaskQuery(
    source="synthetic SWE-bench-style POC (no benchmark record loaded)",
    instance_id="toy/owner__repo-42",
    repository="owner/repo",
    base_commit="abc123",
    problem_statement="Mitigate #42 and run focused tests.",
)


def load_task_query(record_path: str | None) -> TaskQuery:
    """Load one SWE-bench JSON or single-line JSONL instance when supplied."""
    if record_path is None:
        return DEFAULT_TASK_QUERY
    path = Path(record_path)
    if not path.is_file():
        raise SystemExit(
            f"SWE-bench record not found: {path}. "
            "Run without --swe-bench-record for the synthetic POC query, or copy "
            "/data/instance.json.example to /data/instance.json and replace it with "
            "a real SWE-bench record."
        )
    raw = path.read_text(encoding="utf-8").strip()
    payload = json.loads(raw.splitlines()[0])
    required = ("instance_id", "repo", "base_commit", "problem_statement")
    missing = [field for field in required if not payload.get(field)]
    if missing:
        raise ValueError(f"SWE-bench record is missing required fields: {', '.join(missing)}")
    return TaskQuery(
        source=f"SWE-bench record: {path}",
        instance_id=str(payload["instance_id"]),
        repository=str(payload["repo"]),
        base_commit=str(payload["base_commit"]),
        problem_statement=str(payload["problem_statement"]),
    )


def print_task_query(task: TaskQuery) -> None:
    print("\n=== SWE-bench task query ===")
    print("source:", task.source)
    print("instance_id:", task.instance_id)
    print("repository:", task.repository)
    print("base_commit:", task.base_commit)
    print("problem_statement:", task.problem_statement)


def make_state(tenant_id: str = "acme", task: TaskQuery = DEFAULT_TASK_QUERY) -> RunState:
    state = RunState(
        tenant_id,
        "user-001",
        "swe-001",
        "orchestrator-001",
        task.repository,
        task.base_commit,
        task.instance_id,
    )
    state.append(CacheScope.SHARED, "Common tools", "Use git, tests, and artifact references.")
    state.append(CacheScope.RUN, "Issue", task.problem_statement)
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


def print_prompt_boundary(
    *, label: str, current: CompiledContext, reusable_text: str, reuse_allowed: bool
) -> None:
    """Print the exact logical cache boundary for a human-readable demo."""
    reused = reusable_text if reuse_allowed else ""
    new_prefill = current.prompt[len(reusable_text) :] if reuse_allowed else current.prompt
    new_prefill = new_prefill.lstrip("\n")
    print(f"\n--- {label}: logical prompt boundary (whitespace-token estimate) ---")
    print(f"reused_prefix_tokens_estimate: {len(reused.split())}")
    print("reused_prefix_whitespace_tokens_estimate:")
    print(reused.split() if reused else "<none>")
    print("reused_prefix_text:")
    print(reused if reused else "<none>")
    print(f"new_prefill_tokens_estimate: {len(new_prefill.split())}")
    print("new_prefill_whitespace_tokens_estimate:")
    print(new_prefill.split() if new_prefill else "<none>")
    print("new_prefill_text:")
    print(new_prefill if new_prefill else "<none>")


def shared_prefix_text(compiler: ContextCompiler) -> str:
    return compiler.shared_prefix(["### Common tools\nUse git, tests, and artifact references."])


def main(show_prompts: bool, swe_bench_record: str | None) -> None:
    compiler = ContextCompiler()
    task = load_task_query(swe_bench_record)
    print_task_query(task)
    state = make_state(task=task)
    handoff = Handoff(task.problem_statement, "A testable patch plan")
    orchestrator = compiler.compile(state, AgentRole.ORCHESTRATOR, handoff)
    orchestrator_resume = compiler.compile(state, AgentRole.ORCHESTRATOR, handoff)
    planner = compiler.compile(state, AgentRole.PLANNER, handoff)
    other_tenant_state = make_state("other-tenant", task)
    other_tenant = compiler.compile(other_tenant_state, AgentRole.ORCHESTRATOR, handoff)

    report(
        "1. Orchestrator first call",
        "Cold prefill; Dynamo may create reusable blocks.",
        shared_prefix_hash=orchestrator.shared_prefix_hash[:16],
        role_prefix_hash=orchestrator.role_prefix_hash[:16],
        full_prompt_hash=orchestrator.prompt_hash[:16],
        tenant_salt_fingerprint=hashlib.sha256(state.cache_salt(SECRET).encode()).hexdigest()[:16],
        **logical_reuse_metrics(current_tokens=orchestrator.estimated_tokens, reusable_tokens=0),
    )
    if show_prompts:
        print_prompt_boundary(label="cold orchestrator call", current=orchestrator, reusable_text="", reuse_allowed=False)

    report(
        "2. Orchestrator resumes unchanged",
        "Full-prefix reuse is eligible: same canonical prompt and same tenant salt.",
        full_prefix_equal=orchestrator.prompt_hash == orchestrator_resume.prompt_hash,
        tenant_salt_equal=state.cache_salt(SECRET) == state.cache_salt(SECRET),
        **logical_reuse_metrics(current_tokens=orchestrator_resume.estimated_tokens, reusable_tokens=orchestrator.estimated_tokens),
    )
    if show_prompts:
        print_prompt_boundary(label="unchanged orchestrator resume", current=orchestrator_resume, reusable_text=orchestrator.prompt, reuse_allowed=True)

    report(
        "3. Planner handoff",
        "Shared prefix can reuse; planner role overlay and handoff suffix require new prefill.",
        shared_prefix_equal=orchestrator.shared_prefix_hash == planner.shared_prefix_hash,
        role_prefix_equal=orchestrator.role_prefix_hash == planner.role_prefix_hash,
        full_prefix_equal=orchestrator.prompt_hash == planner.prompt_hash,
        **logical_reuse_metrics(current_tokens=planner.estimated_tokens, reusable_tokens=planner.shared_prefix_tokens),
    )
    if show_prompts:
        print_prompt_boundary(label="planner handoff", current=planner, reusable_text=shared_prefix_text(compiler), reuse_allowed=True)

    report(
        "4. Different tenant",
        "No physical reuse: the tenant-derived cache salt differs even if text overlaps.",
        shared_prefix_equal=orchestrator.shared_prefix_hash == other_tenant.shared_prefix_hash,
        tenant_salt_equal=state.cache_salt(SECRET) == other_tenant_state.cache_salt(SECRET),
        **logical_reuse_metrics(current_tokens=other_tenant.estimated_tokens, reusable_tokens=0),
    )
    if show_prompts:
        print_prompt_boundary(label="different tenant", current=other_tenant, reusable_text=shared_prefix_text(compiler), reuse_allowed=False)

    report(
        "5. External provider escalation",
        "Semantic handoff only; provider-managed KV is not visible to local Dynamo.",
        handoff_fields="goal, expected output, artifact URIs, repository SHA",
    )

    # A small but internally consistent demo budget: the raw trace crosses the
    # 96-token threshold; the compact replacement fits below it.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="print the exact logical reused-prefix and new-prefill text boundaries",
    )
    parser.add_argument(
        "--swe-bench-record",
        help="path to one SWE-bench JSON record or a JSONL file whose first row is the instance",
    )
    args = parser.parse_args()
    main(show_prompts=args.show_prompts, swe_bench_record=args.swe_bench_record)
