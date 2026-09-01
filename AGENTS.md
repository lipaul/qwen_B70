# AGENTS.md

## Status

- Scratch/test workspace for running **Qwen3.8-27B (AutoRound INT4) on the Intel Arc Pro B70 via vLLM (single card, TP1)**.
- This folder contains a uv project (`pyproject.toml`, `setup.sh`) with its own `.venv` — a minimal Intel XPU vLLM stack.

## The working stack (verified 2026-09-01)

- vllm built from source at upstream commit `e9d1398d9` (nightly, 0.26.1rc1-dev line) — local copy in `vendor/vllm/`.
- torch `2.13.0+xpu`, triton-xpu `3.7.2`, vllm-xpu-kernels `0.1.13.2`, auto-round-lib `0.14.2`.
- Model: `devan-carlin/Qwen3.8-27B-int4-AutoRound` @ revision `bce40cac` (~19 GB) at `/home/acm/work/models/Qwen3.8-27B-int4-AutoRound`.

## Result (lab reference: steveseguin/b70-optimization-lab TP1 vLLM nightly = 30.22/30.26 tok/s)

- Realistic 25-prompt suite (`suites/validation-suite-v1.json`): **decode median 31.16 tok/s** (tokens 1-100), TTFT 0.168 s, prefill ~460 tok/s, cache-zero. See `results/<stamp>/`.

## Hardware quirks this env must work around (patches baked into vendor + `.venv`)

This host has **both an NVIDIA RTX 6000 Ada and the Intel B70**, which breaks several things:

1. **vLLM platform detection** (`vllm/platforms/__init__.py`): NVML sees the NVIDIA GPU, so both CUDA and XPU plugins activate. Patched `cuda_platform_plugin()` to return `None` when `VLLM_TARGET_DEVICE=xpu`.
2. **oneCCL warm-up** (`vllm/v1/worker/xpu_worker.py`): the XPU worker's unconditional dummy all_reduce crashes (oneCCL can't init Level Zero comm on dual-GPU). Skipped when `world_size == 1` (upstream issue #52386).
3. **XPU memory snapshot** (`vllm/platforms/xpu.py`): `torch.ops._C_cache_ops.getMemoryInfo` reports `free=0` on a fresh allocator; patched to use `torch.xpu.mem_get_info`.
4. **Triton driver check** (`vllm/triton_utils/importing.py`): NVIDIA triton driver reports active alongside Intel; when `VLLM_TARGET_DEVICE=xpu`, keep only the Intel driver so Triton stays enabled.

Also applied: `torch.zeros(1)` before the memory snapshot in `xpu_worker.py` to force Level Zero context init.

## Critical build constraint: SYCL runtime must match

- torch `2.12.0+xpu` bundles `libsycl.so.8`, but Triton compiles its launcher shims with the system oneAPI 2026.1 (`libsycl.so.9`) → `sycl::_V1::exception: Backends mismatch` (all Triton kernels + torch.compile fail).
- torch `2.13.0+xpu` links the system `libsycl.so.9` → **consistent**, Triton works. Do NOT downgrade torch or mix SYCL runtimes.

## Commands

- `bash setup.sh` — end-to-end: oneAPI → `uv sync` (builds vllm from `vendor/vllm`) → download weights → **stop**. `RUN_BENCH=1` also serves + benchmarks; `SKIP_UV=1` / `SKIP_DOWNLOAD=1` skip stages.
- `./serve.sh` — serve on B70 TP1 (XPU graph on, MTP off, `--language-model-only` NOT used; `VLLM_XPU_ENABLE_XPU_GRAPH=1`). Requires env: `VLLM_TARGET_DEVICE=xpu`, `TRITON_DEFAULT_BACKEND=intel`, `ZE_AFFINITY_MASK=0`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`.
- `./bench.sh` — realistic-suite benchmark (25 prompts, median decode tok/s 1-100) → `results/<stamp>/`.
- Server health: `curl http://127.0.0.1:8000/health`.
- Re-patching a fresh vendor clone: `git -C vendor/vllm apply patches/*.diff` (4 host-specific diffs).

## Environment notes

- No system `pip`/`ensurepip` (Python 3.14 default); use `uv` (`/home/acm/.local/share/uv/...`, Python 3.12.13 managed).
- oneAPI 2026.1 is globally sourced (`ONEAPI_ROOT=/opt/intel/oneapi`); `icpx` on PATH.
- Level Zero dev header installed manually to `/usr/local/include/level_zero/ze_api.h` and `libze_loader.so` symlink at `/usr/lib/x86_64-linux-gnu/` (needed by Triton's JIT shim build; `level-zero-dev` pkg unavailable in this distro).
- Shared `../uv_scaffolding_xpu/vendor/vllm` is NOT used for this project's build — `vendor/vllm` here is a separate checkout at the nightly commit.