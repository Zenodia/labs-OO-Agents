"""Make two real requests to a Dynamo/TensorRT-LLM OpenAI-compatible endpoint.

Run this only after ``scripts/launch_dynamo_trtllm_qwen3_8b.sh`` has started a
remote inference stack. It measures client-observed TTFT and server-reported
token usage. Physical KV hit/miss names are backend/version-specific, so fetch
them separately from the configured Prometheus endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from context_engineering import AgentRole, ContextCompiler, Handoff
from scenarios import load_task_query, make_state, print_task_query
from state import dynamo_hints


@dataclass(frozen=True)
class RequestResult:
    ttft_ms: float | None
    total_ms: float
    text: str
    usage: dict[str, Any] | None


def stream_chat(*, base_url: str, model: str, prompt: str, use_hints: bool) -> RequestResult:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if use_hints:
        # These are the published Dynamo agent-hints/cache-control fields. Do
        # not send the POC's cache_salt: it is local control-plane metadata,
        # not a portable public API field.
        payload["nvext"] = {
            "agent_hints": {"priority": 100, "osl": 128, "speculative_prefill": True},
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                value = line[6:]
                if value == "[DONE]":
                    break
                event = json.loads(value)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        text_parts.append(content)
    except urllib.error.URLError as error:
        raise SystemExit(
            f"Cannot reach Dynamo at {base_url}: {error}. Start the GPU runtime first, "
            "or pass --base-url with the reachable OpenAI-compatible endpoint."
        ) from error
    completed = time.perf_counter()
    return RequestResult(
        ttft_ms=None if first_token_at is None else (first_token_at - started) * 1000,
        total_ms=(completed - started) * 1000,
        text="".join(text_parts),
        usage=usage,
    )


def fetch_metric_lines(url: str | None) -> list[str]:
    if not url:
        return ["metrics: not requested (set --metrics-url when your Dynamo deployment exposes it)"]
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as error:
        return [f"metrics fetch failed: {error}"]
    lines = [line for line in body.splitlines() if "cache" in line.lower() or "kv" in line.lower()]
    return lines[:40] or ["metrics endpoint returned no cache/KV-named series"]


def print_result(name: str, result: RequestResult) -> None:
    print(f"\n=== {name} ===")
    print(f"client_ttft_ms: {result.ttft_ms:.1f}" if result.ttft_ms is not None else "client_ttft_ms: unavailable")
    print(f"client_total_ms: {result.total_ms:.1f}")
    print("server_usage:", result.usage or "not returned by endpoint")
    print("response_preview:", result.text[:400] or "<empty streamed content>")


def main(args: argparse.Namespace) -> None:
    task = load_task_query(args.swe_bench_record)
    print_task_query(task)
    compiler = ContextCompiler()
    state = make_state(task=task)
    handoff = Handoff(task.problem_statement, "A testable patch plan")
    turn_one = compiler.compile(state, AgentRole.ORCHESTRATOR, handoff)
    tool_return = (
        "\n\n## Tool return\n"
        "test-output artifact://swe-001/test-output-001\n"
        "Summary: parser_test.py:42 fails after the malformed empty-token input."
    )
    turn_two_prompt = turn_one.prompt + tool_return

    print("\n=== Local control-plane intent ===")
    print("model:", args.model)
    print("turn_one_logical_tokens_estimate:", turn_one.estimated_tokens)
    print("turn_two_logical_reused_tokens_estimate:", turn_one.estimated_tokens)
    print("turn_two_logical_new_prefill_tokens_estimate:", len(tool_return.split()))
    print("dynamo_hints_intent:", dynamo_hints(state, os.environ.get("CACHE_SALT_SECRET")))
    print("metrics_before:")
    print("\n".join(fetch_metric_lines(args.metrics_url)))

    first = stream_chat(base_url=args.base_url, model=args.model, prompt=turn_one.prompt, use_hints=not args.no_dynamo_hints)
    second = stream_chat(base_url=args.base_url, model=args.model, prompt=turn_two_prompt, use_hints=not args.no_dynamo_hints)
    print_result("1. Cold first turn", first)
    print_result("2. Warm second turn after tool return", second)
    print("\nmetrics_after:")
    print("\n".join(fetch_metric_lines(args.metrics_url)))
    print(
        "\nInterpretation: compare client_ttft_ms and server_usage across the two turns. "
        "Use the deployment's Prometheus cache/KV series—not this script—to assert a physical hit or miss."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("DYNAMO_BASE_URL", "http://host.docker.internal:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("DYNAMO_MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument("--metrics-url", default=os.environ.get("DYNAMO_METRICS_URL"))
    parser.add_argument("--swe-bench-record")
    parser.add_argument("--no-dynamo-hints", action="store_true", help="omit nvext agent-hints/cache-control for compatibility testing")
    main(parser.parse_args())
