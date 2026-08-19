"""Role-aware context compilation for a long-running SWE agent."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from state import CacheScope, ContextSegment, RunState


class AgentRole(StrEnum):
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    INVESTIGATOR = "investigator"
    PATCHER = "patcher"
    TESTER = "tester"


BASE_SYSTEM_PROMPT = """You are part of a sandboxed SWE agent team.
Work only in the checked-out repository. Prefer evidence over assertions.
Write large tool output to artifacts and pass references plus concise summaries."""

ROLE_OVERLAY = {
    AgentRole.ORCHESTRATOR: (
        "Own the task graph, delegate bounded work, and integrate verified results."
    ),
    AgentRole.PLANNER: "Produce an ordered, testable plan. Do not edit the repository.",
    AgentRole.INVESTIGATOR: (
        "Inspect code and artifacts. Return evidence, locations, and hypotheses."
    ),
    AgentRole.PATCHER: "Make the smallest justified patch and report the changed files.",
    AgentRole.TESTER: (
        "Run focused verification and report commands, outputs, and remaining failures."
    ),
}


@dataclass(frozen=True)
class Artifact:
    """A tool result stored outside the prompt; ``summary`` is the prompt budget."""

    artifact_id: str
    kind: str
    summary: str
    uri: str


@dataclass(frozen=True)
class Handoff:
    goal: str
    expected_output: str
    evidence: tuple[Artifact, ...] = ()


@dataclass(frozen=True)
class CompiledContext:
    prompt: str
    artifact_ids: tuple[str, ...]
    estimated_tokens: int
    shared_prefix_tokens: int
    role_prefix_tokens: int
    compression_required: bool
    shared_prefix_hash: str
    role_prefix_hash: str
    prompt_hash: str


class ContextCompiler:
    """Preserves stable cache prefixes and bounds agent-specific context."""

    def __init__(self, token_budget: int = 8_192, compression_ratio: float = 0.8) -> None:
        self.token_budget = token_budget
        self.compression_threshold = int(token_budget * compression_ratio)

    def compile(self, state: RunState, role: AgentRole, handoff: Handoff) -> CompiledContext:
        shared_segments = self._select_segments(state, CacheScope.SHARED)
        state_segments = self._select_segments(state, exclude_scope=CacheScope.SHARED)
        shared_prefix = self.shared_prefix(shared_segments)
        role_prefix = self.role_prefix(role)
        artifact_lines = [
            f"- {artifact.kind} {artifact.artifact_id}: {artifact.summary} ({artifact.uri})"
            for artifact in handoff.evidence
        ]
        sections = [
            shared_prefix,
            role_prefix,
            "## Durable state\n" + "\n\n".join(state_segments),
            "## Handoff\n"
            + f"Goal: {handoff.goal}\nExpected output: {handoff.expected_output}\n"
            + "Evidence:\n"
            + ("\n".join(artifact_lines) if artifact_lines else "- none"),
        ]
        prompt = "\n\n".join(section for section in sections if section.rstrip())
        estimated_tokens = self._estimate_tokens(prompt)
        return CompiledContext(
            prompt=prompt,
            artifact_ids=tuple(artifact.artifact_id for artifact in handoff.evidence),
            estimated_tokens=estimated_tokens,
            shared_prefix_tokens=self._estimate_tokens(shared_prefix),
            role_prefix_tokens=self._estimate_tokens(role_prefix),
            compression_required=estimated_tokens >= self.compression_threshold,
            shared_prefix_hash=self._hash(shared_prefix),
            role_prefix_hash=self._hash(role_prefix),
            prompt_hash=self._hash(prompt),
        )

    @staticmethod
    def shared_prefix(shared_segments: list[str] | None = None) -> str:
        suffix = "\n\n".join(shared_segments or [])
        prefix = "## Shared operating policy\n" + BASE_SYSTEM_PROMPT
        return prefix + ("\n\n" + suffix if suffix else "")

    @staticmethod
    def role_prefix(role: AgentRole) -> str:
        return "## Role\n" + ROLE_OVERLAY[role]

    def tool_result(self, *, kind: str, summary: str, uri: str) -> Artifact:
        """Create a stable artifact reference instead of injecting raw tool output."""
        artifact_id = hashlib.sha256(f"{kind}\0{uri}".encode()).hexdigest()[:16]
        return Artifact(artifact_id=artifact_id, kind=kind, summary=summary, uri=uri)

    def compression_segment(self, summary: str) -> ContextSegment:
        """Store the output of a summarizer when ``compression_required`` is true."""
        return ContextSegment(CacheScope.RUN, "Compressed run summary", summary)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text.split()))

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _select_segments(
        state: RunState,
        scope: CacheScope | None = None,
        exclude_scope: CacheScope | None = None,
    ) -> list[str]:
        """Keep durable state; never pass raw ephemeral scratchpads to another role."""
        return [
            f"### {segment.label}\n{segment.text.strip()}"
            for segment in state.segments
            if segment.scope != CacheScope.EPHEMERAL
            and (scope is None or segment.scope == scope)
            and (exclude_scope is None or segment.scope != exclude_scope)
        ]
