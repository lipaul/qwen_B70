# B70 Inference Performance Improvement Plan

Based on the investigation of Qwen3.8-27B (AutoRound INT4) running on a single Intel Arc
Pro B70 (XPU) and comparison with the NVIDIA RTX 6000 Ada baseline.

## Baseline Summary

| Metric | B70 (v0.1.13.2) | Ada6000 (v0.28.0) | Gap |
|---|---|---|---|
| decode (tok/s) | 31.16 | 45.25 | B70 ~31% slower |
| TTFT (s) | 0.168 | 0.054 | B70 ~3× slower |
| prefill (tok/s) | ~460 | ~1484 | B70 ~3.2× slower |

**Model:** Qwen3.8-27B, AutoRound INT4 (w4g128, ~18 GB), 64 layers:
- 48× GDN linear attention (key/value head_dim=128)
- 16× full attention (head_dim=256, GQA 24→4)

---

## Phase 1: Custom Kernel Rebuild — Result: No Improvement

### What was done
- Upgraded `vllm-xpu-kernels` from v0.1.13.2 (prebuilt wheel) to **v0.1.14 built from source**
- Added custom paged-decode kernel config: `8,256,16/32/64,true,false,false` (head_size=256 causal paged decode)
- Compiled ~18 min, installed, ran 25-prompt realistic suite

### Result
- **decode: 31.16 → 31.16 tok/s** (zero change)
- TTFT: 0.168 → 0.171 s (noise)
- prefill: 460 → 457 tok/s (noise)

### Why — the core hypothesis was wrong

The assumption was that full-attention decode was missing an optimized XE2 cutlass paged-decode
kernel for head_size=256. **This was false.** Server logs show:

```
[flash_attn.py:830] Using FlashAttention version 2
```

The full-attention layers on XPU dispatch to **FA2** (`_vllm_fa2_C.varlen_fwd`), **not** the
XE2 cutlass FMHA path (`cutlass_paged_decode_xe2`). The paged-decode kernel config that
was rebuilt applies to the XE2 cutlass path, which is a different attention backend. This
model uses the `FlashAttentionBackend` (FA2), so the new kernel was never dispatched.

---

## Phase 1B: GDN Triton Fallback Investigation — Result: False Alarm

### Observation
Server log showed:
```
[qwen_gdn_linear_attn.py:500] Falling back to the Triton GDN decode path:
  the fused CUDA kernel requires a BF16 GDN model ...
[qwen_gdn_linear_attn.py:505] GDN decode kernel: triton
```

### Root cause
`VLLM_GDN_DECODE_KERNEL` defaults to `"cuda"` (`vllm/envs.py:131`). The `__init__` of
`QwenGatedDeltaNetAttention` checks `_fused_gdn_decode_unsupported_reason()` which rejects
the XPU platform + float16 dtype → sets `gdn_decode_kernel = "triton"`.

**However**, this flag (`enable_fused_gdn_decode`) is only used inside `forward_cuda()`.
On XPU, the dispatch is:

```python
if current_platform.is_xpu():
    self._forward_method = self.forward_xpu    # ← XPU NEVER calls forward_cuda
```

And `forward_xpu` → `torch.ops.vllm.gdn_attention_core_xpu` → `torch.ops._xpu_C.gdn_attention`
(**fused XPU SYCL kernel**), which is independent of the `enable_fused_gdn_decode` flag.

**Conclusion:** The "Falling back to Triton" log is a misleading CUDA-path check. The 48 GDN
layers on XPU already use the fused XPU kernel. This is NOT a bottleneck.

---

## Corrected Bottleneck Analysis

### B70 is memory-bandwidth-bound for decode

B70 memory bandwidth ≈ 500 GB/s (estimated). INT4 weights = 18 GB. Each decode step reads
all weights:

- **Theoretical decode limit:** 18 GB / 500 GB/s ≈ 28 ms/step → ~36 tok/s
- **Measured:** 31 tok/s → **~85% of theoretical bandwidth**

This means decode is close to the memory bandwidth ceiling of the B70. **Kernel micro-optimizations
on the attention path alone cannot significantly improve throughput.** The ~14% gap between
31 and 36 tok/s is likely consumed by:
- KV cache reads (small — linear attention dominates)
- Activations / intermediate tensors
- Sampling overhead (PyTorch native fallback, per log: `topk_topp_sampler does not support
  per-request generators. Falling back to PyTorch-native implementation`)

### Attention breakdown (at decode)

| Component | Layers | Kernel | Bandwidth share |
|---|---|---|---|
| GDN linear attn | 48/64 | `_xpu_C.gdn_attention` (fused) | ~50% of weights |
| Full attn (FA2) | 16/64 | `_vllm_fa2_C.varlen_fwd` | ~17% of weights |
| Other (RMSNorm, proj, sampling) | all | native | ~33% of weights |

### Prefill gap (460 vs 1484 tok/s)

The large prefill gap (3.2×) is partly bandwidth and partly because the GDN prefill is
reported as using Triton/FLA (not the chunked XPU kernel):
```
Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)
```

This message is from the prefill backend selection — the Triton/FLA chunked prefill vs
the XPU chunked variant. This is a separate concern from the decode fallback.

---

## Revised Development Roadmap

### Phase 2: Speculative Decoding / MTP (Highest Impact)

The model has `mtp_num_hidden_layers: 1` and the weights are already quantized with MTP
support (the README says "MTP layers quantized to INT4 (model stays MTP-capable)").

**Expected gain:** +30-50% effective decode tok/s (each step produces 2 tokens instead of 1).

**Work items:**
1. Verify vLLM nightly `e9d1398d9` supports `--enable-mtp` on XPU
2. Check MTP kernel compatibility with vllm-xpu-kernels v0.1.14
3. Benchmark with MTP enabled (same 25-prompt suite)
4. If MTP not supported in this vLLM version, rebase vendor/vllm to a newer nightly

### Phase 3: Sampling Optimization (Medium Impact)

The topk-topp sampler falls back to PyTorch native:
```
topk_topp_sampler does not support per-request generators. Falling back to PyTorch-native
```

**Expected gain:** ~1-3 tok/s at decode (small but free).

**Work items:**
1. Remove `per-request generators` requirement (or fix the XPU kernel to support it)
2. Alternatively, cherry-pick the `topk_topp_sampler` CDF-outlier pivot optimization
   from vllm-xpu-kernels main (PR #561, post-v0.1.14) into the local build

### Phase 4: Prefill Optimization (Medium Impact)

The GDN prefill uses Triton/FLA chunked prefill rather than the XPU chunked variant.
This affects TTFT.

**Expected gain:** TTFT reduction from 0.168s toward 0.1s.

**Work items:**
1. Investigate why the XPU chunked GDN prefill is not selected
2. Compare Triton/FLA vs XPU chunked GDN prefill performance
3. If XPU chunked is faster, force its selection

### Phase 5: Additional Full-Attention Path Tuning (Low Impact)

Investigate whether the full-attention layers (head_dim=256) can be switched from FA2 to
the XE2 cutlass FMHA path. This is a speculative optimization — the gains may be small
since full attention is only 16/64 layers.

**Expected gain:** 0-3 tok/s (if any).

**Work items:**
1. Understand why the FlashAttentionBackend is selected for this model instead of
   a path that uses XE2 cutlass
2. Benchmark FA2 vs XE2 cutlass for head_dim=256 causal decode on B70
3. If XE2 is faster, add an attention backend fallback

---

## Phase 2: MTP / Speculative Decoding — Result: Not Viable on This Stack

### What was done
- Enabled MTP via `--spec-method mtp --spec-tokens 1` (model has `mtp_num_hidden_layers: 1`)
- MTP model loaded successfully as `Qwen3_5MTP`, drafter shared embedding/lm_head weights
- Probed decode speed and ran server logs

### Result
- **decode: 7.3 → 7.4 tok/s** (vs baseline 31.2 — **4.2× slower**)
- TTFT: 0.209 s (vs 0.168 s base)
- Mean acceptance length: 1.88, acceptance rate: 87.5%

### Root cause
The speculative decode pipeline uses Eagle-style Triton kernels (`eagle_prepare_inputs_padded_kernel`,
`eagle_prepare_next_token_padded_kernel`) that are **not optimized for XPU** — they JIT-compile
during inference (Triton, not vllm-xpu-kernels SYCL). The MTP drafter also runs the full 64-layer
model + MTP head per draft step, roughly doubling the effective compute per token. On XPU, the
Triton prepare kernels + doubled model forward overhead far outweigh the acceptance-length benefit
(1.88×), resulting in net 4× slowdown.

### Conclusion
**MTP is not viable with the current vLLM nightly + vllm-xpu-kernels v0.1.14 stack on B70.**
The Eagle-style speculative decode path lacks native XPU kernel support in vllm-xpu-kernels.
To make MTP viable, one would need to:
1. Implement XPU versions of `eagle_prepare_*_padded_kernel` in vllm-xpu-kernels
2. Or wait for a future vLLM + vllm-xpu-kernels release with native MTP support on XPU

---

## Phase 3: Sampling Optimization (Medium Impact, Next)

The topk-topp sampler falls back to PyTorch native:
```
topk_topp_sampler does not support per-request generators. Falling back to PyTorch-native
```

**Expected gain:** ~1-3 tok/s at decode (small but actionable).

**Work items:**
1. Remove `per-request generators` requirement (or fix the XPU kernel to support it)
2. Cherry-pick the `topk_topp_sampler` CDF-outlier pivot optimization from vllm-xpu-kernels
   main (PR #561, post-v0.1.14) into the local build

### Phase 4: GDN Prefill Optimization (Medium Impact)

The GDN prefill uses Triton/FLA chunked prefill rather than the XPU chunked variant.
This affects TTFT.

**Expected gain:** TTFT reduction from 0.168s toward 0.1s.

**Work items:**
1. Investigate why the XPU chunked GDN prefill is not selected
2. Compare Triton/FLA vs XPU chunked GDN prefill performance
3. If XPU chunked is faster, force its selection

### Phase 5: Additional Full-Attention Path Tuning (Low Impact)

Investigate whether the full-attention layers (head_dim=256) can be switched from FA2 to
the XE2 cutlass FMHA path.

**Expected gain:** 0-3 tok/s (if any).

**Work items:**
1. Understand why the FlashAttentionBackend is selected for this model instead of
   a path that uses XE2 cutlass
2. Benchmark FA2 vs XE2 cutlass for head_dim=256 causal decode on B70
3. If XE2 is faster, add an attention backend fallback

---

## Summary

| Phase | Focus | Result / Expected gain | Investment |
|---|---|---|---|
| **1** | Custom kernel rebuild (256 causal decode) | **0% gain** — wrong target (FA2 path) | 1 day |
| **1b** | GDN triton fallback investigation | **False alarm** — XPU uses fused kernel | 0.5 day |
| **2** | MTP / speculative decoding | **-76% gain** — overhead exceeds benefit | 1 day |
| **3** | Sampling (topk_topp fix) | **+1-3 t/s** | 1-2 days |
| **4** | GDN prefill (XPU chunked) | TTFT ↓ 30-40% | 2-3 days |
| **5** | Full-attn XE2 cutlass fallback | 0-3 t/s | 2-3 days |

**Recommended next step:** Phase 3 (sampling optimization) — the only remaining item with
a clear target and manageable effort. The topk_topp sampler fallback is logged, understood,
and can be fixed by cherry-picking the CDF-outlier pivot optimization from vllm-xpu-kernels
main (PR #561) into the local build, or by removing the `per-request generators` requirement.

## Artifacts

- Custom kernel config: `vendor/vllm-xpu-kernels/csrc/xpu/attn/kernel_configs/paged_decode_qwen38.conf`
- Custom kernel config: `vendor/vllm-xpu-kernels/csrc/xpu/attn/kernel_configs/chunk_prefill_qwen38.conf`
- vllm-xpu-kernels v0.1.14 source build: `vendor/vllm-xpu-kernels/`
- Phase 1 bench results: `results/20260903T130527Z/`
- Probe script (reusable): available in `../qwen_nv/probe.py`