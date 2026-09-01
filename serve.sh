#!/usr/bin/env bash
# Serve Qwen3.8-27B (AutoRound INT4) on the Intel Arc Pro B70, single card (TP1),
# mirroring steveseguin/b70-optimization-lab's qwen38 TP1 nightly vLLM lane.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_TARGET_DEVICE=xpu
export TRITON_DEFAULT_BACKEND=intel
export VLLM_XPU_GDN_NATIVE_FALLBACK="${VLLM_XPU_GDN_NATIVE_FALLBACK:-1}"
export ZE_AFFINITY_MASK=0
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export VLLM_XPU_ENABLE_XPU_GRAPH="${VLLM_XPU_ENABLE_XPU_GRAPH:-1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CCL_ZE_IPC_EXCHANGE=sockets
export VLLM_NO_USAGE_STATS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_DIR="${MODEL_DIR:-/home/acm/work/models/Qwen3.8-27B-int4-AutoRound}"
PORT="${PORT:-8000}"

exec "$ROOT/.venv/bin/vllm" serve "$MODEL_DIR" \
  --host 0.0.0.0 --port "$PORT" --trust-remote-code \
  --served-model-name qwen38-tp1 \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-1024}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
  --dtype float16 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prompt-tokens-details \
  --no-enable-prefix-caching \
  ${EXTRA_VLLM_ARGS:-}