"""Run without a model to inspect the durable state and cache-routing decision."""

from context_engineering import AgentRole, ContextCompiler, Handoff
from state import CacheScope, ModelTarget, RunState, dynamo_hints, route


state = RunState("acme", "user-001", "swe-001", "orchestrator-001", "owner/repo", "abc123", "#42")
state.append(CacheScope.SHARED, "System and tools", "Use the repository tools and return evidence.")
state.append(CacheScope.RUN, "Issue", "Mitigate #42 and run the focused tests.")
state.append(CacheScope.EPHEMERAL, "Scratchpad", "Inspect failing test output.")

anchor = ModelTarget("nemotron-anchor", "nemotron", quality_tier=2)
small = ModelTarget("nemotron-fast", "nemotron", quality_tier=1)
decision = route(anchor=anchor, candidate=small, prefix_tokens=12_000, requires_high_quality=False)
compiler = ContextCompiler()
artifact = compiler.tool_result(
    kind="test-output",
    summary="Focused test fails in parser_test.py:42.",
    uri="artifact://swe-001/test-output-001",
)
compiled = compiler.compile(
    state,
    AgentRole.PLANNER,
    Handoff("Mitigate #42", "A testable patch plan", (artifact,)),
)

print(compiled.prompt)
print(
    "\nprefix key:",
    state.prefix_key(model=anchor.name, tokenizer_revision="v1", runtime_revision="trtllm-v1"),
)
print("dynamo hints:", dynamo_hints(state, cache_salt_secret="demo-only"))
print("route:", decision)
print("compression required:", compiled.compression_required)
