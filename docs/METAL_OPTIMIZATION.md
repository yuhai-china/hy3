# Metal Optimization Suggestions (derived from the CUDA speedup work)

This document translates the lessons from the CUDA decode optimization
(`docs/CUDA_OPTIMIZATION.md`, 4.6 → ~49 tok/s pure-GPU on a B300) into concrete,
prioritized suggestions for the **Metal** backend (`hy3.metal`, `hy3_metal.m`).

It is a design/review document, not a changelog: nothing here is implemented
yet. Each item states what CUDA did, what the Metal code does today, and the
recommended change.

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

## 1. HIGH — Online-softmax (flash-style) attention

**CUDA did:** `attention_kernel_online` — single-pass online softmax, one warp
per head, float4-tiled, K and V each read once, running max/denominator, no
materialized score matrix.

**Metal today:** `attention` (`hy3.metal:256`) is **two-pass**. It:
1. computes the full `scores[t]` vector into threadgroup memory,
2. does a max-reduce, then an `exp` pass over `scores[]`,
3. does a separate weighted-V pass that re-reads V from global memory.

It also clamps `ntok <= 8192` (`hy3.metal:279`) because `scores[]` lives in
threadgroup memory, capping context length.

**Recommendation:** rewrite as single-pass online softmax:
- Maintain running `m` (max) and `l` (denominator) and an accumulator over V;
  update them as each timestep's score is computed. No `scores[]` array.
- Read each K and V vector exactly once → roughly halves KV global traffic.
- Removes the threadgroup-memory score buffer and the `ntok <= 8192` ceiling.
- Fewer `threadgroup_barrier`s.

**Why it matters:** attention is a per-token, memory-bound pass over the growing
KV cache; halving its bandwidth and lifting the context cap is a clean win with
no math change (accumulate in float, as CUDA does).

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

1. **Online-softmax attention** (§1) — self-contained, correctness-neutral,
   clear bandwidth + context-limit win.
2. **FP32 router** (§3) — small, improves determinism, de-risks everything else.
3. **Pre-encoded ICB + `d_pos` refactor** (§2) — largest structural CPU-overhead
   win; also gives a pure-GPU replay timer.
4. **Fusions** (§4) — incremental, stack on top of the ICB.

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
