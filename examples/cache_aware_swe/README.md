# Cache-Aware SWE Agent with NOOA

This is a local-first reference for a coding-copilot/SWE-bench agent. NOOA is
the harness; Postgres is the durable state and authorization control plane;
Dynamo/TensorRT-LLM is the inference and physical KV-cache data plane.

## What is covered

- User, run, and agent identities in Postgres.
- Durable context segments and KV metadata, never raw KV tensors.
- Tenant/run cache salts and prefix identities for local TensorRT-LLM calls.
- Role-aware context compilation for orchestrator, planner, investigator,
  patcher, and tester agents.
- Artifact offloading, bounded handoffs, verification gates, and compression
  triggers for long-running loops.
- Conservative local-versus-external model routing.

## Architecture and the KV boundary

```text
NOOA SWE harness
  orchestrator -> planner -> investigator / patcher / tester
       |                  |
       |                  +-- bounded handoff + artifact references
       v
Postgres: users, agents, runs, context DAG, cache metadata
       v
ContextCompiler: canonical prompt + cache policy
       v
Dynamo: cache-aware placement, routing, retention/prefetch decisions
       v
TensorRT-LLM: physical KV blocks, prefix matching, offload and eviction
```

Postgres does **not** contain KV tensors, and a database hash cannot load a KV
blob. It stores an authorized manifest saying which canonical prefix was
assembled for a tenant, run, model, tokenizer, runtime, salt, TTL, and
priority. The harness retrieves that manifest and compiles the exact prompt;
Dynamo/TRT-LLM performs compatible-prefix lookup and physical cache loading.

There is no application-level `fetch KV by prefix_hash` command. The hash is an
audit/index key that explains why a request should be compatible. Dynamo
ultimately confirms whether blocks exist, are accessible under the same salt,
and should be loaded, retained, or recomputed.

## Context engineering and loop policy

`context_engineering.py` implements the required policy:

- Stable common policy prefix followed by a stable role overlay.
- Artifact offloading: raw tool output stays outside the prompt; agents receive
  a stable URI and concise summary.
- Context isolation: `ephemeral` scratchpads never cross an agent handoff.
- Compression trigger: at 80% of token budget, persist a compact summary and
  stop carrying the older raw trajectory.

Each request uses the canonical order below. Stable content goes first to
maximize prefix reuse.

```text
1. Shared operating policy and common tool protocol
2. Role overlay
3. Authorized durable state
4. Handoff: goal, expected output, artifact summaries
```

The shared prefix can be reused across roles. Each role also reuses its own
overlay on later turns. Do not put timestamps, random IDs, volatile diagnostics,
raw tool output, or user-specific data in the shared prefix.

The loop is explicit:

```text
PLAN -> SELECT_CONTEXT -> EXECUTE_TOOL_OR_AGENT -> STORE_ARTIFACT
     -> VERIFY -> UPDATE_PLAN -> COMPLETE | REPLAN
```

Every plan step has expected evidence and a verification gate. A completed
subagent writes a compact result plus artifacts; its reasoning branch can then
be released as ephemeral.

## KV cache control: Dynamo and the harness

Dynamo can automatically make cache-aware placement and cache-management
decisions from tokenized prompt overlap, its block index, and worker load. The
harness controls the valuable inputs to that decision:

- canonical prompt order and model/runtime compatibility;
- session/run identity and `cache_salt` isolation;
- agent lifecycle, priority, retention TTL, and prefetch intent;
- anchor-model selection versus cache-miss cost;
- custom routing policy where the deployed Dynamo version exposes it.

The harness cannot force an arbitrary database hash to load a KV blob. It
controls the manifest and policy; Dynamo controls the physical cache decision.

### Cache-salt scopes

Use `RunState.cache_salt(CACHE_SALT_SECRET, CacheScope.TENANT)` for the first
POC. The same tenant can reuse a prefix while other tenants cannot. Do not make
the salt unique per request. A deployment-wide shared salt is safe only for a
request made entirely from non-sensitive shared prompt material; do not mix it
with user/run data in a single request.

Postgres stores only a fingerprint of the cache salt, never raw salt or raw KV.
The application derives the raw salt at its trusted boundary and sends it only
to the compatible local inference backend.

## Provider and model routing

| Target | Physical KV reuse | Handoff |
|---|---|---|
| Local Dynamo + TRT-LLM, same model/runtime | Yes | Exact compatible prefix |
| Local same-family models with validated mapper | Experimental | Mapped KV plus durable state |
| NVIDIA Build API, Anthropic, OpenAI | No raw provider-KV access | Summary, artifacts, plan, repo SHA |

Start with one local model. It proves context design, state restoration, tenant
isolation, and same-model KV reuse. Add NVIDIA Build as a quality-escalation
route later. Add a second local model only for the distinct cross-model
KV-transfer experiment.

### Selected local model and preferred compute

The checked-in NOOA adapter defaults to `Qwen/Qwen3-8B` through an
OpenAI-compatible local endpoint. It is intentionally small enough for the
first control-plane and cache-policy integration.

For the preferred one-model Dynamo POC, deploy
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` locally on **one H100 80 GB or
H200 141 GB**. TensorRT-LLM lists Nemotron-3 and Qwen3-family support; the
Nemotron 3.5 Nano 30B-A3B model card describes a 30B-total/3.5B-active hybrid
MoE intended for reasoning, coding, planning, and tool calling.

Keep the NVIDIA Build API model as an external route. It is useful for quality
comparison or escalation, but it does not make its provider-managed KV cache
visible to the local Dynamo deployment. Before using the supplied
`nvidia/nemotron-3.5-lightning-30b-a3b` identifier, validate that exact model
name against the Build catalog in the target account; this example does not
assume it is a locally deployable TensorRT-LLM target.

## Docker Compose control plane

The Compose stack starts Postgres and a health service. It needs no GPU, model,
or Dynamo deployment.

```bash
cd examples/cache_aware_swe
docker compose up --build
curl.exe http://localhost:8080/healthz
```

In PowerShell, this is equivalent:

```powershell
Invoke-RestMethod http://localhost:8080/healthz
```

Expect `{"ready":true}`. Use `docker compose down` to stop it. Add `--volumes`
only when you intentionally want to discard Postgres state. Replace the
development database password and `CACHE_SALT_SECRET` before shared use.

## Files and demo

| File | Purpose |
|---|---|
| `state.py` | Run identity, cache salt, prefix key, retention and routing policy |
| `context_engineering.py` | Role profiles, artifacts, handoffs, compression triggers |
| `postgres_repository.py` / `schema.sql` | Durable state and cache metadata control plane |
| `nooa_agent.py` | Minimal NOOA harness surface |
| `compose.yaml` | Postgres and health-service environment |
| `demo.py` | No-model policy/context-compilation demo |

```bash
uv run python examples/cache_aware_swe/demo.py
```

## GPU choices

| GPU | Local model for this example | Suitable first step |
|---|---|---|
| A10 24 GB | Quantized Qwen3-8B | One short-context agent and control-plane baseline |
| L40S 48 GB | Qwen3-8B or quantized 8-14B | Useful one-model cache-reuse baseline |
| A100/H100 80 GB | Nemotron-3 Nano 30B-A3B BF16 | Preferred cost-conscious Dynamo POC |
| H200 141 GB | Nemotron-3 Nano 30B-A3B BF16 | Long-context headroom and future second model |

Use a sandboxed container or VM for generated code and scope GitHub credentials
to the target repository. NOOA's in-process checks are not a containment
boundary.
