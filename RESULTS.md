# Qwen3.8-27B on Intel Arc Pro B70 — Single-Card vLLM Benchmark

**Date:** 2026-09-01 11:05 UTC
**Hardware:** 1x Intel Arc Pro B70 (`Battlemage G31`, PCI 0xe223, 32.5 GB / 30.3 GiB)
**Stack:** vLLM `e9d1398d9` (nightly, source-built) · torch `2.13.0+xpu` · triton-xpu `3.7.2` · vllm-xpu-kernels `0.1.13.2` · auto-round-lib `0.14.2`
**Model:** `devan-carlin/Qwen3.8-27B-int4-AutoRound` @ rev `bce40cacab0a4535b92fb3d57615c2bea9adf3d1` (~19 GB, AutoRound INT4 W4A16)

## Configuration (TP1, XPU graph ON, MTP off)

```
vllm serve /home/acm/work/models/Qwen3.8-27B-int4-AutoRound \
  --trust-remote-code --served-model-name qwen38-tp1 \
  --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 1 \
  --max-num-batched-tokens 1024 --gpu-memory-utilization 0.90 \
  --dtype float16 --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prompt-tokens-details --no-enable-prefix-caching
```

Env: `VLLM_TARGET_DEVICE=xpu`, `TRITON_DEFAULT_BACKEND=intel`, `ZE_AFFINITY_MASK=0`,
`ONEAPI_DEVICE_SELECTOR=level_zero:0`, `VLLM_XPU_ENABLE_XPU_GRAPH=1`, `CCL_ZE_IPC_EXCHANGE=sockets`.
Weights: 17.46 GiB loaded; XPU graph captured (+0.17 GiB).

## Results — realistic 25-prompt suite (`suites/validation-suite-v1.json`)

Suite: `qwen36-27b-int4-independent-validation-20260815-v1` · `--max-tokens 512` · `--metric-tokens 100` · `--seed 1` · cache-zero (prefix caching off) · 25/25 rows valid.

| Metric | median | min | p25 | p75 | max |
|---|---|---|---|---|---|
| **decode tok/s (tokens 1-100)** | **31.16** | 30.82 | 31.15 | 31.18 | 31.21 |
| TTFT (s) | 0.168 | 0.158 | 0.167 | 0.172 | 0.440 |
| prefill tok/s | 459.6 | 372.0 | 404.5 | 525.9 | 1931.4 |
| prompt tokens | 78 | 60 | 67 | 90 | 849 |
| completion tokens | 512 | 512 | 512 | 512 | 512 |

All 25 rows have cached tokens == 0 (no prefix/history acceleration).

## Comparison with steveseguin/b70-optimization-lab reference

Lab's single-card (TP1) vLLM reference — AutoRound INT4, XPU graph ON, MTP off, same 25-prompt
methodology (`experiments/qwen38-27b-b70/notes/2026-08-22-qwen38-tp1-vllm-nightly-bringup-finding.md`):

| Metric | This run | Lab reference |
|---|---|---|
| decode median tok/s (1-100) | **31.16** | 30.22 / 30.26 |
| TTFT | 0.168 s | ~0.275 s |
| prefill tok/s | ~460 | ~281 |

This run reproduces and slightly exceeds the lab's fastest single-card vLLM result
(llama.cpp Q4_K_M TP1 is 27.82 tok/s for comparison).

## Host-specific patches required

This host has **both an NVIDIA RTX 6000 Ada and the Intel B70**, which required four
local vLLM patches (baked into `vendor/vllm` + `.venv`, diffs in `patches/`):

1. `cuda_platform_plugin()` → `None` when `VLLM_TARGET_DEVICE=xpu` (dual-GPU platform conflict).
2. Skip oneCCL warm-up all_reduce when `world_size == 1` (vLLM issue #52386).
3. XPU memory snapshot uses `torch.xpu.mem_get_info` instead of `_C_cache_ops.getMemoryInfo`
   (reported `free=0` on a fresh allocator); plus `torch.zeros(1)` to force L0 context init.
4. Triton driver check keeps only the Intel driver when `VLLM_TARGET_DEVICE=xpu`
   (NVIDIA triton driver reported active despite `torch.cuda` unavailable).

**SYCL constraint:** torch `2.13.0+xpu` links the system oneAPI 2026.1 `libsycl.so.9`.
torch `2.12.0+xpu` bundles `libsycl.so.8`, which breaks Triton JIT
(`sycl::_V1::exception: Backends mismatch`) — do not downgrade torch or mix SYCL runtimes.

## Artifacts

- Full bench JSON: `bench.json` (this dir) · server log: `server.log` (repo root `results/`)
- Benchmark tool: `tools/bench-openai-realistic-suite.py` (from the lab repo)
- Setup docs: `AGENTS.md` in repo root