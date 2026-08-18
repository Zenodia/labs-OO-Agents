"""A minimal NOOA harness surface for the cache-aware SWE state example.

Run this only in an OS-level sandbox. The model endpoint is intentionally
configured through environment variables so this example never embeds secrets.
"""

from __future__ import annotations

import os

from nooa import Agent
from nooa.unifiedllm import get_llm_client

from context_engineering import AgentRole, ContextCompiler, Handoff
from state import CacheScope, RunState, dynamo_hints


llm = get_llm_client(
    os.environ.get("SWEBENCH_MODEL", "hosted_vllm/Qwen/Qwen3-8B"),
    api_base=os.environ.get("SWEBENCH_MODEL_BASE_URL", "http://localhost:8000/v1"),
)


class CacheAwareSweAgent(Agent, llm=llm):
    """Investigate and mitigate one GitHub issue in a sandboxed checkout."""

    def __init__(self, state: RunState, role: AgentRole = AgentRole.ORCHESTRATOR) -> None:
        super().__init__()
        self.state = state
        self.role = role
        self.context_compiler = ContextCompiler()

    def inference_context(self, handoff: Handoff) -> str:
        """Return canonical context for the next model request."""
        return self.context_compiler.compile(self.state, self.role, handoff).prompt

    def inference_metadata(self) -> dict[str, object]:
        """Return cache intent for a Dynamo-aware OpenAI-compatible client."""
        return {
            **dynamo_hints(self.state, os.environ.get("CACHE_SALT_SECRET")),
            "prefix_key": self.state.prefix_key(
                model=os.environ.get("SWEBENCH_MODEL", "hosted_vllm/Qwen/Qwen3-8B"),
                tokenizer_revision=os.environ.get("TOKENIZER_REVISION", "unknown"),
                runtime_revision=os.environ.get("RUNTIME_REVISION", "unknown"),
            ),
        }

    async def propose_next_step(self) -> str:
        """Use the current issue context and tool evidence to propose one safe next step.

        Inspect evidence before asserting a diagnosis. Do not modify files outside
        the sandboxed repository.
        """
        ...


def example_state() -> RunState:
    state = RunState(
        "demo-tenant",
        "demo-user",
        "swe-001",
        "orchestrator-001",
        "owner/repo",
        "abc123",
        "#42",
    )
    state.append(
        CacheScope.SHARED,
        "System and tools",
        "Work only inside the sandboxed checkout.",
    )
    state.append(CacheScope.RUN, "Issue", "Investigate and mitigate #42.")
    return state
