# Qwen3.8-27B on Intel Arc Pro B70 via vLLM (single card, TP1)

Run **Qwen3.8-27B (AutoRound INT4)** on one **Intel Arc Pro B70** GPU with **vLLM** and
reproduce the single-card result from [steveseguin/b70-optimization-lab].
This repo is a self-contained `uv` project: pinned environment, downloader,
server, benchmark, and the host-specific patches required on this dual-GPU box.

## Result

Realistic 25-prompt suite (`suites/validation-suite-v1.json`), AutoRound INT4,
XPU graph ON, MTP off, cache-zero:

| Metric | This repo | Lab reference |
|---|---|---|
| decode median tok/s (tokens 1-100) | **31.16** | 30.22 / 30.26 |
| TTFT | 0.168 s | ~0.275 s |
| prefill | ~460 tok/s | ~281 tok/s |

Full detail in [`RESULTS.md`](RESULTS.md).

## Stack

- **vLLM** built from source at upstream commit `e9d1398d9` (nightly, 0.26.1rc1 line)
- **torch** `2.13.0+xpu`, **triton-xpu** `3.7.2`, **vllm-xpu-kernels** `0.1.13.2`, **auto-round-lib** `0.14.2`
- **Model:** `devan-carlin/Qwen3.8-27B-int4-AutoRound` @ rev `bce40cac` (~19 GB, W4A16 INT4)
- Python 3.12 managed by `uv`; oneAPI 2026.1 (globally sourced)

## Reproduce

```bash
bash setup.sh              # oneAPI -> uv sync (builds vllm) -> download weights -> stop
./serve.sh                 # serve on the B70 (TP1, XPU graph on, port 8000)
./bench.sh                 # realistic-suite benchmark -> results/<stamp>/
```

Or one shot: `RUN_BENCH=1 bash setup.sh` (env + weights + serve + benchmark).
Toggles: `SKIP_UV=1`, `SKIP_DOWNLOAD=1`, `MODEL_DIR`, `PORT`.

## Host quirks (NVIDIA RTX 6000 Ada + Intel B70 in one box)

vLLM and Triton assume a single GPU vendor, so this host needs 4 local patches
(`patches/*.diff`, applied to `vendor/vllm` by `setup.sh` on a fresh clone):

1. **Platform detection** — NVML sees the NVIDIA GPU, so both CUDA and XPU
   plugins activate. `cuda_platform_plugin()` returns `None` when `VLLM_TARGET_DEVICE=xpu`.
2. **oneCCL warm-up** — the XPU worker's dummy all_reduce crashes on this host;
   skipped when `world_size == 1` (upstream issue #52386).
3. **XPU memory snapshot** — `torch.ops._C_cache_ops.getMemoryInfo` reports `free=0`
   on a fresh allocator; use `torch.xpu.mem_get_info` + a `torch.zeros(1)` to init L0.
4. **Triton driver check** — the NVIDIA Triton driver reports active; keep only the
   Intel driver when `VLLM_TARGET_DEVICE=xpu`.

### Critical constraint: SYCL runtime must match

torch `2.12.0+xpu` bundles `libsycl.so.8`, but Triton compiles its launcher shims
with the system oneAPI 2026.1 (`libsycl.so.9`) -> `sycl::exception: Backends mismatch`
(every Triton kernel and `torch.compile` fails). **torch `2.13.0+xpu` links the system
`libsycl.so.9` and is the minimum working version.** Do not downgrade torch or mix
SYCL runtimes. Triton's JIT also needs `level_zero/ze_api.h` + a `libze_loader.so`
symlink (installed manually; `level-zero-dev` is unavailable in this distro).

[steveseguin/b70-optimization-lab]: https://github.com/steveseguin/b70-optimization-lab