#!/usr/bin/env bash
# Run the lab's realistic-suite benchmark against the local vLLM server and dump
# a summary. Mirrors the lab's qwen38 TP1 nightly invocation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8000}"
OUT_DIR="${OUT_DIR:-$ROOT/results}"
STAMP="$(date +%Y%m%dT%H%M%SZ)"
OUT="$OUT_DIR/$STAMP"
mkdir -p "$OUT"

"$ROOT/.venv/bin/python" "$ROOT/tools/bench-openai-realistic-suite.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --model "qwen38-tp1" \
  --api-mode chat \
  --suite "$ROOT/suites/validation-suite-v1.json" \
  --max-tokens 512 \
  --metric-tokens 100 \
  --seed 1 \
  --timeout 900 \
  --request-extra-json '{"chat_template_kwargs":{"enable_thinking":false},"ignore_eos":true}' \
  --out "$OUT/bench.json" > "$OUT/bench.stdout.log" 2>&1

"$ROOT/.venv/bin/python" - "$OUT/bench.json" <<'PY'
import json, statistics, sys
d = json.load(open(sys.argv[1]))
rows = d.get("rows", [])
summary = d.get("summary", {})
def med(x):
    x = sorted(x)
    return round(statistics.median(x), 4) if x else None
dec = [r.get("tok_s_1_100_intervals_after_ttft") for r in rows if r.get("tok_s_1_100_intervals_after_ttft")]
ttft = [r.get("ttft_s") for r in rows if r.get("ttft_s")]
pf = [r["prompt_tokens"]/r["ttft_s"] for r in rows if r.get("ttft_s") and r.get("prompt_tokens")]
print("valid rows:", len(rows))
print("decode_median_tok_s_1_100:", med(dec))
print("ttft_median_s:", med(ttft))
print("prefill_tok_s_median:", med(pf))
print("cached_tokens_all_zero:", all((r.get("cached_tokens") or 0) == 0 for r in rows))
PY