#!/usr/bin/env bash
# End-to-end setup for the Qwen3.8-27B B70 vLLM benchmark (Intel Arc Pro B70, TP1).
#
# Default: build the uv env and download the model weights, then STOP.
#   RUN_BENCH=1  additionally serve the model and run the realistic-suite benchmark.
#   SKIP_UV=1 / SKIP_DOWNLOAD=1  skip those stages.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---- Toggles / config ----
SKIP_UV="${SKIP_UV:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
RUN_BENCH="${RUN_BENCH:-0}"
PORT="${PORT:-8000}"

VLLM_COMMIT="e9d1398d9edfd90fcc1cf783805240e3effec013"
MODEL_ID="devan-carlin/Qwen3.8-27B-int4-AutoRound"
MODEL_REV="bce40cacab0a4535b92fb3d57615c2bea9adf3d1"
MODEL_DIR="${MODEL_DIR:-/home/acm/work/models/Qwen3.8-27B-int4-AutoRound}"
PATCHES=(
  patches/vllm-platform-xpu.diff
  patches/vllm-xpu-mem-snapshot.diff
  patches/vllm-xpu-skip-onccl-warmup.diff
  patches/vllm-xpu-triton-driver.diff
)

log() { printf '[setup] %s\n' "$*"; }

# ---- 1. Intel oneAPI (required to build vllm for XPU) ----
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    log "sourcing oneAPI"
    set +u
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force
    set -u
fi

# ---- 2. uv environment ----
if [ "$SKIP_UV" != "1" ]; then
    log "uv python 3.12"
    uv python install 3.12

    # Fresh checkout: clone vllm at the pinned nightly commit and re-apply the
    # host-specific patches (dual-GPU NVIDIA+Intel workarounds). When vendor/vllm
    # already exists (patched), leave it untouched.
    if [ ! -d vendor/vllm/vllm ]; then
        log "cloning vllm @ $VLLM_COMMIT"
        mkdir -p vendor
        git init vendor/vllm
        git -C vendor/vllm remote add origin https://github.com/vllm-project/vllm.git
        git -C vendor/vllm fetch --depth 1 origin "$VLLM_COMMIT"
        git -C vendor/vllm checkout FETCH_HEAD
        for p in "${PATCHES[@]}"; do
            log "applying $p"
            git -C vendor/vllm apply --check "$ROOT/$p"
            git -C vendor/vllm apply "$ROOT/$p"
        done
    fi

    log "uv sync (builds vllm for XPU; may take several minutes on first run)"
    uv sync

    log "sanity checks"
    .venv/bin/python -c "
import torch
assert torch.xpu.is_available(), 'torch XPU unavailable'
print('torch', torch.__version__, '| xpu:', torch.xpu.get_device_properties(0).name)
"
    .venv/bin/python -c "import vllm; print('vllm', vllm.__version__)"
fi

# ---- 3. Download model weights (resumable) ----
if [ "$SKIP_DOWNLOAD" != "1" ]; then
    log "model dir: $MODEL_DIR"
    mkdir -p "$MODEL_DIR"
    BASE="https://huggingface.co/$MODEL_ID/resolve/$MODEL_REV"
    TMP="$(mktemp)"
    curl -s "https://huggingface.co/api/models/$MODEL_ID/revision/$MODEL_REV" \
        | python3 -c '
import json, sys
for s in json.load(sys.stdin).get("siblings", []):
    n = s["rfilename"]
    if not n.startswith("."):
        print(n)
' > "$TMP"
    count=0
    while IFS= read -r f; do
        if [ -s "$MODEL_DIR/$f" ]; then
            log "present: $f"
        else
            log "download: $f"
            wget -c -q --tries=5 --timeout=60 -O "$MODEL_DIR/$f" "$BASE/$f"
        fi
        count=$((count + 1))
    done < "$TMP"
    rm -f "$TMP"

    # Verify all shards + key files are present and non-empty.
    missing=0
    for f in model-0000{1..7}-of-00007.safetensors model_extra_tensors.safetensors \
             model.safetensors.index.json config.json tokenizer.json quantization_config.json; do
        [ -s "$MODEL_DIR/$f" ] || { echo "MISSING: $f" >&2; missing=1; }
    done
    if [ "$missing" -ne 0 ]; then
        echo "ERROR: model files missing/incomplete" >&2
        exit 1
    fi
    log "weights OK ($count files)"
fi

log "DONE. Model ready at $MODEL_DIR"
log "Next: ./serve.sh   then   ./bench.sh   (or rerun with RUN_BENCH=1)"

# ---- 4. Optional: serve + benchmark (RUN_BENCH=1) ----
if [ "$RUN_BENCH" = "1" ]; then
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        log "server already healthy on port $PORT"
    else
        log "starting server"
        mkdir -p results
        nohup ./serve.sh > results/server.log 2>&1 &
        for i in $(seq 1 30); do
            sleep 20
            if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
                log "server healthy after ~$((i * 20))s"
                break
            fi
            if ! pgrep -f "bin/vllm serve" >/dev/null 2>&1; then
                echo "server died; see results/server.log" >&2
                exit 1
            fi
        done
        curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
            || { echo "health timeout on port $PORT" >&2; exit 1; }
    fi
    log "benchmarking"
    ./bench.sh
    log "benchmark done"
fi