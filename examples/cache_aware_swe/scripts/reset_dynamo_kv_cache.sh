#!/usr/bin/env bash
# Clear the *in-memory* KV cache for this example's Dynamo + TensorRT-LLM worker.
# This deliberately does not touch Postgres, the control-plane Compose stack,
# Dynamo discovery services, or the downloaded Hugging Face model cache.
set -euo pipefail

CONTAINER_NAME="cache-aware-swe-dynamo-trtllm"

if [[ "${1:-}" != "--yes" ]]; then
  cat <<EOF
This removes the running Dynamo/TensorRT-LLM worker container:
  ${CONTAINER_NAME}

Its model process and all of its in-memory KV-cache blocks will be discarded.
It does NOT delete Postgres state, Compose services, Dynamo discovery services,
or .dynamo-model-cache. Start the worker again with:
  bash scripts/launch_dynamo_trtllm_qwen3_8b.sh

Run again with --yes to continue.
EOF
  exit 2
fi

if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "No Dynamo/TensorRT-LLM worker named '$CONTAINER_NAME' is running. KV cache is already cold."
  exit 0
fi

echo "Removing $CONTAINER_NAME and clearing its in-memory KV cache..."
docker rm --force "$CONTAINER_NAME" >/dev/null
echo "Done. Start a new cold worker with: bash scripts/launch_dynamo_trtllm_qwen3_8b.sh"
