# Metal Optimization Suggestions (derived from the CUDA speedup work)

This document translates the lessons from the CUDA decode optimization
(`docs/CUDA_OPTIMIZATION.md`, 4.6 → ~49 tok/s pure-GPU on a B300) into concrete,
prioritized suggestions for the **Metal** backend (`hy3.metal`, `hy3_metal.m`).

It is a design/review document. §§1 and 1b (online-softmax + split-KV attention)
are implemented as of this writing; the remaining items are unimplemented
suggestions.

---

## STATUS (measured on M2 Ultra, hy3_q4k_mixed, experts=8)

Following rule §7.1 ("profile first — do not assume it matches CUDA"), the
suggestions below were re-prioritized against real measurements:

- **§1 online-softmax attention — DONE.** Flash-style single pass, fp16-KV,
  no `scores[]`, no 8192 ceiling. Smoke tests pass.
- **§1b split-KV attention (FlashDecoding) — DONE.** `attention_split` +
  `attention_reduce` kernels partition the KV sequence across 16 chunks
  per head. Split kernel writes (m, l, acc[0..127]) partials; reduce
  kernel merges via online softmax. This cuts the per-head serial scan
  16×, removing the O(context) decode bottleneck. **Measured impact:**
  generation went from 10.1 → 22.2 tok/s (**+120%**), prefill from 19.9 →
  23.2 tok/s (+17%). See §1b below for design details.
- **§3 FP32 router — SKIP.** `blk.*.ffn_gate_inp.weight` is already `F32` in the
  GGUF, so `m_mul_mat` already dispatches `matmul_f32` for the router. No-op.
- **§2 ICB / pre-encoded command buffer — NOT WORTH IT here.** Measured CPU
  encode = **0.57 ms/token vs GPU exec = 43 ms/token (1.3%)**. The doc's premise
  ("at 30+ tok/s CPU encode becomes a real fraction") does not hold on this
  setup; an ICB would reclaim at most 1.3%.
- **§4 fusions — LOW VALUE here.** Launch/encode overhead is the 1.3% above, and
  in the concurrent encoder independent matmuls already overlap. QKV/gate+up
  fusion does not reduce weight bandwidth (same bytes read), only the tiny
  activation re-read. Not pursued.

**Pre-split-kv token breakdown (experts differential):** ~2.36 ms per routed
expert (3 Q4_K matmuls); routed experts ≈ 44% of the token, everything else
(attention + shared expert + dense + output GEMV + norms) ≈ 56%. Both halves
are memory-bandwidth-bound. Post split-kv, attention's O(N) scan no longer
grows with context, so the experts' share of per-token cost increases slightly
at long context — but the absolute token cost drops substantially.

**Final measured performance (C eval, 13 prompts, 256 max gen):**

| Phase | Before split-kv | After split-kv | Speedup |
|-------|----------------|----------------|---------|
| Prefill | 19.9 tok/s | 23.2 tok/s | +17% |
| Generation | 10.1 tok/s | 22.2 tok/s | **+120%** |

The remaining headroom is either lower precision (already Q4_K/Q8_0) or fewer
activated params (runtime `-experts`, already available), or multi-token decode
(speculative / the unused MTP layer 80) — a different class of change from this
document.

---

## 0. What Metal already has (parity with the CUDA wins)

Before listing opportunities, note the Metal fast path
(`metal_forward_model_fast`, `hy3_metal.m`) already implements most of the
high-impact CUDA techniques:

- **GPU-resident MoE routing** — `router_topk` kernel (`hy3.metal:707`) emits
  expert ids + renormalized combine weights on-device; no CPU round-trip.
- **Coalesced Q4_K expert matmul** — `matmul_q4_k_id` (`hy3.metal:748`) uses the
  llama.cpp-style layout (`ix = lane/8`, `ib += 4`, `kmask` unpack) so 32 lanes
  read consecutive weight words. This is the same coalescing idea that was the
  single biggest late CUDA win (`warp_row_dot_q4k`).
- **Batched multi-expert dispatch** — `m_mul_mat_id` (`hy3_metal.m:972`) computes
  all K routed experts in one dispatch (`tgpig.y = slot`, id + stride locate the
  expert), matching CUDA's `moe_matmul_q4k_id_kernel`.
- **FP16 KV cache** — `kv_cache_write` stores `half` (`hy3.metal:233`).
- **Single command buffer per token** — the whole 80-layer forward + final
  logits are encoded into one command buffer (one commit/wait per token).
- **Shared/routed overlap** — the fast path uses
  `computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent`
  (`hy3_metal.m:1108`) with `memoryBarrierWithScope:` between stages, so the
  shared-expert GEMVs and the routed-expert matmuls in the same stage run
  concurrently. This is actually a cleaner expression of the "second stream"
  overlap that was added to CUDA (`stream2` + events).

So the biggest CUDA levers — coalescing and shared/routed overlap — are already
present. The opportunities below are the remaining gaps.

---

## 1. ✅ DONE — Online-softmax (flash-style) attention

**CUDA did:** `attention_kernel_online` — single-pass online softmax, one warp
per head, float4-tiled, K and V each read once, running max/denominator, no
materialized score matrix.

**Metal now:** `attention` (`hy3.metal:310`) is a **single-pass online softmax**
kernel:
- Maintains running `m` (max), `l` (denominator), and an accumulator over V;
  updates them each timestep. No `scores[]` array.
- Each K and V vector read exactly once → halves KV global traffic vs old 2-pass.
- No threadgroup-memory score buffer → no 8192-token ceiling.
- One threadgroup per head, 128 threads; dot-product via `simd_sum` +
  threadgroup-reduction across 4 SIMD groups.

Implemented, tested, produces correct `66` for `11+22+33=?`.

---

## 1b. ✅ DONE — Split-KV attention (FlashDecoding-style)

The online-softmax kernel (§1) still has an O(context) bottleneck: **per head,
one threadgroup serially scans all `ntok` history tokens** — both for prefill
and per-token decode. At long context this serial SIMD scan dominates.

### CUDA precedent

`attention_split_kv_kernel` + `attention_reduce_kernel` (`hy3_gpu.cu`):
- Split pass: `n_heads × ATTN_SPLITS` warps (32 thr), each warp handles a chunk
  of the KV cache with online softmax, producing partial (m, l, acc[0..31]).
- Reduce pass: `n_heads` warps merge ATTN_SPLITS partials via another online
  softmax, writing the final output.
- Per-warp serial work cut 16×; graph phase flat at ~19 ms (was 75 ms at
  ctx=2000).

### Metal adaptation

Two kernels in `hy3.metal`, dispatched back-to-back with a mandatory
`memoryBarrierWithScope:MTLBarrierScopeBuffers` between them (regardless of
concurrent mode — the reduce pass depends on split partials being complete).

#### Split kernel (`attention_split`, buffer[0..11])

```metal
kernel void attention_split(
    device float       *partials   [[buffer(0)]],
    device const float *q          [[buffer(1)]],
    device const half  *k_cache    [[buffer(2)]],
    device const half  *v_cache    [[buffer(3)]],
    constant uint      &n_heads    [[buffer(4)]],
    constant uint      &n_kv_heads [[buffer(5)]],
    constant uint      &head_dim   [[buffer(6)]],
    constant int       &kv_len     [[buffer(7)]],
    constant uint      &kv_group   [[buffer(8)]],
    constant int       &layer_id   [[buffer(9)]],
    constant int       &n_layers   [[buffer(10)]],
    constant uint      &n_splits   [[buffer(11)]],
    threadgroup float  *red        [[threadgroup(0)]], ...)
```

Grid: `n_heads × ATTN_SPLITS` threadgroups, `head_dim` (128) threads each.
Chunk `[r0, r1)` computed from split index; same online-softmax as §1 but
restricted to the chunk. Each threadgroup writes `(m, l, acc[0..head_dim-1])`
to its exclusive region of `partials`.

#### Reduce kernel (`attention_reduce`, buffer[0..4])

```metal
kernel void attention_reduce(
    device float       *out       [[buffer(0)]],
    device const float *partials  [[buffer(1)]],
    constant uint      &n_heads   [[buffer(2)]],
    constant uint      &head_dim  [[buffer(3)]],
    constant uint      &n_splits  [[buffer(4)]], ...)
```

Grid: `n_heads` threadgroups, `head_dim` threads each. Each threadgroup merges
`n_splits` partial entries for its head via iterative online softmax:

```
for each split s:
  pm, pl = partials[off], partials[off+1]
  if pl <= 0: continue        // empty chunk
  m_new = max(m_global, pm)
  corr  = exp(m_global - m_new)
  rsc   = exp(pm - m_new)
  l_global = l_global * corr + pl * rsc
  acc = acc * corr + partials[off+2+tid] * rsc
  m_global = m_new
out = acc / l_global
```

#### Partials buffer

Allocated once in `hy3_metal_init`:

```c
#define METAL_ATTN_SPLITS 16
n = HY3_N_HEAD * METAL_ATTN_SPLITS * (2 + HY3_HEAD_DIM);  // 64×16×130 = 133120 floats
ctx->d_attn_partials = hy3_alloc(ctx->device, n);          // ≈ 520 KB, MTLResourceStorageModeShared
```

#### Dispatch in `m_attention()` (`hy3_metal.m`)

```objc
[enc setComputePipelineState:ctx->pipe_attention_split];
// ... set buffers 0–11 ...
[enc dispatchThreadgroups:MTLSizeMake(n_heads * n_splits, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];  // ALWAYS — not conditional on concurrent

[enc setComputePipelineState:ctx->pipe_attention_reduce];
// ... set buffers 0–4 ...
[enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
```

The barrier between split and reduce **must fire regardless of `ctx->concurrent`**
— without it, in concurrent-encoder mode, reduce threadgroups can start reading
partials before the split threadgroups have written them. This was the only
correctness bug found during implementation (caught as `11+22+33=?` producing
gibberish).

#### Numerical fidelity

The online-softmax merge is associative: each split partial uses the same
rescale-factor math as the original online-softmax, so the reduce's iterative
merge produces bit-identical output to the single-pass kernel. Verified by the
standard smoke test (`11+22+33=?` → `66` at temp=0, identical before/after).

#### Key differences from CUDA

| | CUDA | Metal |
|---|---|---|
| Threads per group | 32 (1 warp) | 128 (4 SIMD groups) |
| Dimensionality per thread | 4 floats (`float4`) | 1 float |
| Dot-product reduction | `__shfl_down_sync` | `simd_sum` + threadgroup `red[]` |
| Partial size per split | 2 + 128 = 130 floats | same (133120 floats / 520 KB total) |
| Grid | `<<<n_heads*ATTN_SPLITS, 32>>>` | `MTLSizeMake(64*16, 1, 1)` |
| Graph capture | CUDA graph | N/A (dispatch lists do the same) |

#### Measured impact

| Phase | Before (single-pass §1) | After (+ split-kv) | Speedup |
|-------|------------------------|-------------------|---------|
| Prefill | 19.9 tok/s (107s/2131 tok) | 23.2 tok/s (92s/2131 tok) | +17% |
| Generation | 10.1 tok/s (937s/9420 tok) | 22.2 tok/s (150s/3320 tok) | **+120%** |

Generation speed more than doubled. The split-kv cuts per-head serial work
~16×, removing the O(context) bottleneck that previously dominated decode.
The prefill improvement is smaller (~17%) because prefill already benefits
from parallelism across the prompt's many token positions; the split-kv adds
intra-head parallelism on top.

Tested via `tests/hy3_eval_metal.c` (13 prompts, model loaded once, 256 max
generation tokens per question). All 13 outputs are coherent and factually
correct at temp=0.

---

## 2. HIGH — Pre-encoded command buffer (`MTLIndirectCommandBuffer`): the CUDA-graph analog

**CUDA did:** captured the entire 80-layer decode into a **CUDA graph** and
replays it each token, with all position-dependent kernels reading the token
position from a device pointer `d_pos`. This collapsed ~1700 host-issued
launches per token into one graph launch.

**Metal today:** `metal_forward_model_fast` (`hy3_metal.m:1099`) **re-encodes the
whole token every step** — ~80 layers × ~15 dispatches ≈ **~1200
`setComputePipelineState` / `setBuffer` / `dispatchThreadgroups` calls per
token** on the CPU, plus per-token `setBytes` for positions.

**Recommendation:** encode the layer loop **once** into an
`MTLIndirectCommandBuffer` (ICB) and re-execute it every token:
- Refactor position-dependent kernels (`rope`, `kv_cache_write`, `attention`) to
  read the token position from a small **device buffer** (a `d_pos` analog)
  instead of `setBytes` each token — exactly the refactor CUDA needed for graph
  capture to keep the topology invariant.
- Bind weights via argument buffers so the ICB is static.
- Each token: update `d_pos` (tiny blit) and execute the ICB.

**Why it matters:** at 30+ tok/s the CPU encode time becomes a real fraction of
the token and serializes against the GPU. This reclaims it and, as on CUDA,
unlocks clean device-time profiling (a `HY3_GRAPH_BENCH`-style pure-GPU replay
timer).

---

## 3. MEDIUM — FP32 router GEMV for routing stability

**CUDA did:** moved the MoE router projection (`ffn_gate_inp`, a tiny
`[192 × 4096]` matrix) from an FP16 GEMM to a full **FP32** `cublasSgemv`. The
router's top-k `argmax` over near-equal expert scores is the model's most
precision-sensitive step; FP16 rounding there flips routes. Effect: the
identical 20-vs-80-layer greedy prefix extended 243 → 1055 chars, ~neutral
speed.

**Metal today:** the router matmul goes through `m_mul_mat(&l->ffn_gate_inp, …)`
(`hy3_metal.m:1012`), which selects a kernel by the weight's ggml type. If that
weight is F16 or quantized, the same routing-divergence risk applies.

**Recommendation:** keep an **F32 copy** of just this one small weight and
dispatch `matmul_f32` for the router only (everything else stays as-is). Cheap
(a few MB/layer) and improves determinism of routing across device/precision.

**Verify first:** check `l->ffn_gate_inp.t->ggml_type`. If it is already F32,
this is a no-op and can be skipped.

---

## 4. MEDIUM — Kernel fusion to cut dispatches and barriers

CUDA fused several ops that Metal still issues separately. Each fusion removes a
dispatch **and** a `BAR()` (`memoryBarrierWithScope:`); across 80 layers that is
several hundred fewer dispatches/barriers per token, and it compounds with the
ICB work in §2.

| Fusion | CUDA | Metal today |
|--------|------|-------------|
| QKV in one matmul | `d_layer_attn_qkv` (single GEMV) | 3 matmuls (`hy3_metal.m:945–947`) |
| qk-norm + RoPE | `qk_norm_rope_fused_kernel` | `rms_norm_heads` ×2 + `rope` |
| residual-add + RMSNorm | `add_rmsnorm_kernel` | `m_add` + `m_rms_norm` |
| shared gate + up | one GEMV | two matmuls (`:1036–1037`) |

**Recommendation:** start with QKV (largest, most obvious) and shared gate+up
(concatenate the weights at load time, one dispatch), then the two small fused
kernels.

---

## 5. LOWER — measure before spending effort

- **`simdgroup_matrix` (MMA) for dense/shared GEMVs.** `matmul_f16` is already
  vectorized (`half4`) + `simd_sum`, which is appropriate for batch-1 GEMV. MMA
  helps most for GEMM (prefill / batch > 1); likely marginal for decode. Worth a
  test only on the prefill path.
- **Tune `Q4K_N_ROWS` and threadgroup width** for the specific Apple GPU
  generation — a quick empirical sweep, analogous to picking `MOE_WPB = 8` on
  CUDA.
- **Barrier granularity** in `metal_encode_moe` — the `BAR()` after each stage
  serializes stages; confirm each is load-bearing (some disjoint-buffer stages
  may not need a full-buffer barrier).

---

## 6. Suggested order of work

1. ✅ **Online-softmax attention** (§1) — DONE.
2. ✅ **Split-KV attention (FlashDecoding)** (§1b) — DONE. 2.2× gen speedup.
3. **FP32 router** (§3) — SKIP (already F32 in the GGUF).
4. **Pre-encoded ICB + `d_pos` refactor** (§2) — NOT WORTH IT at current speeds
   (CPU encode < 2% of token time). Revisit if decode exceeds ~50 tok/s.
5. **Fusions** (§4) — LOW VALUE. Launch overhead is drowned by weight bandwidth.

---

## 7. Cross-cutting lessons from the CUDA work (apply directly)

1. **Profile first.** On CUDA, differential phase-skipping proved routed experts
   were 85% of the token. Add a Metal per-phase timer or use a GPU capture to
   confirm the experts-vs-attention split **before** optimizing — do not assume
   it matches CUDA (Metal already has coalesced experts + concurrency, so the
   balance may differ).
2. **Decode is latency/bandwidth-bound, not FLOP-bound.** The wins are fewer
   launches (ICB), smaller/half-precision reads, coalescing, and overlap — not
   raw arithmetic.
3. **Verify accuracy on every change.** Keep the two CUDA smoke tests:
   - `--gpu-layers`/Metal low-precision arithmetic → must reach `\boxed{66}`.
   - "The capital of France is" → must say **Paris** and stay coherent.
   Metal has its own reduction order and the router is equally sensitive, so a
   speed change can silently perturb routing.
4. **Commit each working state.** Small, verified, reversible steps.
