#!/usr/bin/env bash
# Start one aggregated Dynamo + TensorRT-LLM worker on a single A100 80 GB GPU.
# A single worker proves engine prefix reuse but cannot prove cross-worker routing.
set -euo pipefail

: "${DYNAMO_HOME:?Set DYNAMO_HOME to a local clone of https://github.com/ai-dynamo/dynamo}"
DYNAMO_VERSION="${DYNAMO_VERSION:-1.4.0}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-8B}"
DYNAMO_MODEL_CACHE="${DYNAMO_MODEL_CACHE:-$PWD/.dynamo-model-cache}"

mkdir -p "$DYNAMO_MODEL_CACHE"

# Dynamo's official local examples require the discovery infrastructure.
docker compose -f "$DYNAMO_HOME/dev/docker-compose.yml" up -d

docker pull "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:${DYNAMO_VERSION}"
exec docker run --rm --name cache-aware-swe-dynamo-trtllm --gpus all --network host --ipc host \
  -v "$DYNAMO_MODEL_CACHE:/root/.cache/huggingface" \
  -e MODEL_PATH -e SERVED_MODEL_NAME \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:${DYNAMO_VERSION}" \
  bash -lc 'cd "$DYNAMO_HOME/examples/backends/trtllm" && ./launch/agg.sh'
