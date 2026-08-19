"""Second-turn and compaction walkthrough for the cache-aware SWE-agent POC.

This script is deliberately standalone: it reconstructs the same canonical
first-turn prompt used by ``scenarios.py``. A real harness would load that run
from Postgres rather than require one demo script to run before the other.
"""

from __future__ import annotations

import argparse
import hashlib

from context_engineering import AgentRole, ContextCompiler, Handoff
from scenarios import (
    TaskQuery,
    load_task_query,
    logical_reuse_metrics,
    make_state,
    print_prompt_boundary,
    print_task_query,
)


def token_estimate(prompt: str) -> int:
    return max(1, len(prompt.split()))


def digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def report(name: str, expectation: str, **values: object) -> None:
    print(f"\n=== {name} ===")
    print("Expected:", expectation)
    for key, value in values.items():
        print(f"{key}: {value}")


def prompt_boundary(*, label: str, prompt: str, reusable_prefix: str) -> None:
    """Use the same verbose format as scenarios.py for a video-friendly trace."""
    class PromptView:
        def __init__(self, value: str) -> None:
            self.prompt = value

    print_prompt_boundary(
        label=label,
        current=PromptView(prompt),  # type: ignore[arg-type]
        reusable_text=reusable_prefix,
        reuse_allowed=True,
    )


def main(show_prompts: bool, swe_bench_record: str | None) -> None:
    compiler = ContextCompiler()
    task: TaskQuery = load_task_query(swe_bench_record)
    print_task_query(task)
    state = make_state(task=task)
    handoff = Handoff(task.problem_statement, "A testable patch plan")
    turn_one = compiler.compile(state, AgentRole.ORCHESTRATOR, handoff)

    # This is the append-only turn-two event after the tool completes.
    tool_return = """## Tool return
test-output artifact://swe-001/test-output-001
Summary: parser_test.py:42 fails after the malformed empty-token input."""
    turn_two_prompt = turn_one.prompt + "\n\n" + tool_return
    turn_two_tokens = token_estimate(turn_two_prompt)

    report(
        "1. First turn establishes the reusable prefix",
        "Cold prefill creates cache blocks for the canonical first-turn prompt.",
        first_turn_full_prompt_hash=turn_one.prompt_hash[:16],
        first_turn_tokens_estimate=turn_one.estimated_tokens,
        shared_prefix_tokens_estimate=turn_one.shared_prefix_tokens,
        **logical_reuse_metrics(current_tokens=turn_one.estimated_tokens, reusable_tokens=0),
    )

    report(
        "2. Tool return becomes the second turn",
        "The exact first-turn prompt is a contiguous prefix; only the tool-return suffix needs prefill.",
        first_turn_prefix_of_second_turn=turn_two_prompt.startswith(turn_one.prompt),
        second_turn_full_prompt_hash=digest(turn_two_prompt)[:16],
        **logical_reuse_metrics(
            current_tokens=turn_two_tokens,
            reusable_tokens=turn_one.estimated_tokens,
        ),
    )
    if show_prompts:
        prompt_boundary(label="turn two after tool return", prompt=turn_two_prompt, reusable_prefix=turn_one.prompt)

    # A long autonomous loop grows past the demo policy budget.
    raw_trajectory = "\n".join(f"tool-observation-{index}: evidence" for index in range(1, 61))
    turn_three_prompt = turn_two_prompt + "\n\n## Raw trajectory\n" + raw_trajectory
    compression_threshold = 160
    turn_three_tokens = token_estimate(turn_three_prompt)

    # This is intentionally harness-owned. Dynamo does not make this summary.
    compact_summary = """## Compacted multi-turn state
Evidence: parser_test.py:42 fails on malformed empty-token input.
Next action: implement the smallest parser fix and run focused tests."""
    compact_turn_prompt = turn_one.prompt + "\n\n" + compact_summary
    compact_turn_tokens = token_estimate(compact_turn_prompt)

    report(
        "3. Multi-turn compression creates a new compact prefix",
        "The harness replaces obsolete trajectory with durable summary; full prompt identity changes, shared prefix survives.",
        raw_turn_tokens_estimate=turn_three_tokens,
        compression_threshold_tokens_estimate=compression_threshold,
        compression_required=turn_three_tokens >= compression_threshold,
        raw_turn_full_prompt_hash=digest(turn_three_prompt)[:16],
        compact_turn_tokens_estimate=compact_turn_tokens,
        compact_turn_full_prompt_hash=digest(compact_turn_prompt)[:16],
        full_prompt_hash_changed=digest(turn_three_prompt) != digest(compact_turn_prompt),
        first_turn_prefix_retained=compact_turn_prompt.startswith(turn_one.prompt),
        **logical_reuse_metrics(
            current_tokens=compact_turn_tokens,
            reusable_tokens=turn_one.estimated_tokens,
        ),
    )
    if show_prompts:
        prompt_boundary(label="first compacted turn", prompt=compact_turn_prompt, reusable_prefix=turn_one.prompt)

    report(
        "4. Unchanged compact turn resumes warm",
        "After the compacted prompt is prefetched once, its unchanged next turn is eligible for full-prefix reuse.",
        compact_prompt_hash_equal=digest(compact_turn_prompt) == digest(compact_turn_prompt),
        **logical_reuse_metrics(
            current_tokens=compact_turn_tokens,
            reusable_tokens=compact_turn_tokens,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="print exact logical reused-prefix and new-prefill prompt text boundaries",
    )
    parser.add_argument(
        "--swe-bench-record",
        help="path to one SWE-bench JSON record or a JSONL file whose first row is the instance",
    )
    args = parser.parse_args()
    main(show_prompts=args.show_prompts, swe_bench_record=args.swe_bench_record)
