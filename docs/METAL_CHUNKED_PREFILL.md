# Metal chunked prefill — design & implementation guide

Status: **design + ready-to-integrate code, NOT yet verified on Apple hardware.**
This box is Linux/CUDA-only, so the CUDA version (`HY3_PREFILL_CHUNK`, in
`hy3_gpu.cu`) is the validated reference. This document ports that validated
design to the Metal backend. Anyone with an Apple-silicon Mac can follow it to
implement and verify; every step below has a concrete correctness check.

## 1. Why (same as CUDA)

Long-context prompts are **prefill-bound**: the prompt (haystack) dominates,
the answer is short. The default Metal path (`metal_forward_model_fast`)
processes prompt tokens **one at a time** — one command buffer per token, each
GEMV (`M=1`) and a split-KV attention that re-reads the whole KV cache. That
underutilizes the GPU and makes prefill `O(n²)` with a large constant.

Chunked prefill processes a chunk of `C` prompt tokens together so that:
- matmuls become `M=C` GEMM (Metal already ships `matmul_q8_0_mm` /
  `matmul_q4_k_mm` matrix-matrix kernels — reuse them);
- one **batched causal attention** kernel per layer amortizes the KV-cache read
  across the chunk (FlashInfer's operational-intensity argument: batching
  queries pushes prefill attention from IO-bound toward compute-bound).

Target: same ~1.9×-and-growing prefill speedup measured on CUDA at 8k, larger
at longer contexts. Decode (generation) is unaffected — it stays batch-1.

## 2. Component map: CUDA → Metal

| CUDA (`hy3_gpu.cu`, validated) | Metal (`hy3_metal.m` / `hy3.metal`) |
|---|---|
| `prefill_chunk_gpu()` | new `prefill_chunk_metal()` |
| `gpu_mul_mat_batched` (cuBLAS GEMM M=C) | existing `matmul_*_mm` kernels, dispatched with `n_cols=C` |
| `prefill_attn_int8/int4_kernel` | new MSL kernel `prefill_attn_q8` (§4) |
| `rms_norm_batched_kernel` | new MSL `rms_norm_batched` (one threadgroup/token) |
| `embed_batch_kernel` | loop existing `embed_f16/f32` over the chunk, or a batched variant |
| per-token RoPE (`qk_norm_rope_fused`) | existing `rms_norm_heads_rope`, looped per token in the chunk |
| per-token KV write (`kv_quantize_*`) | existing `kv_cache_write_q8`, looped per token |
| per-token FFN/MoE (`gpu_ffn_moe`) | existing `metal_encode_dense`/`metal_encode_moe` bodies, looped per token |

Metal's KV cache is **Q8**, stored as **separate arrays** (unlike CUDA's packed
layout): `d_k_cache_q8`/`d_v_cache_q8` hold `int8` values with
`base = (t*n_layers+layer_id)*n_kv_heads*head_dim + kv_h*head_dim`, and
`d_k_scales`/`d_v_scales` hold one FP32 scale per `(slot, kv_head)` at index
`slot*n_kv_heads + kv_h`. The batched attention kernel below matches this exact
layout (copied from `attention_split_q8`).

## 3. Two Metal-specific simplifications

1. **No CUDA-graph hazard.** The CUDA port had to disable the single-token
   decode graph for multi-token evals. Metal has no graph; it records a command
   buffer per call. So `prefill_chunk_metal` just records the chunk into one (or
   a few) command buffers and commits. Nothing to disable.

2. **No `h_pos` race.** The nastiest CUDA bug was a shared pinned position
   scalar overwritten by the host before the async copy consumed it (garbage for
   `chunk≥2`). Metal passes `pos` by value via `setBytes:` on each dispatch, so
   each dispatch gets its own copy — **the race cannot occur**. Just pass
   `pos = base_tok + c` per token in the RoPE/KV-write loop. (If you instead
   batch RoPE, pass positions as a small buffer, mirroring `d_pf_pos`.)

## 4. The batched causal attention kernel (MSL)

Add to `hy3.metal`. One threadgroup per `(head, query-tile)`; `PF_QB` queries
per tile; `head_dim` (128) threads; `simd_sum` score reduction — matching the
existing `attention_split_q8` structure. Query `i` (global position
`base_tok+q0+i`) attends keys `[0 .. base_tok+q0+i]` (causal).

```metal
// PF_QB queries share each KV-cache read. Queries packed in d_q with stride
// qstride (=n_heads*head_dim here, since d_pf_q is contiguous per token).
kernel void prefill_attn_q8(
    device float       *out        [[buffer(0)]],   // [nq, n_heads*head_dim]
    device const float *q          [[buffer(1)]],   // [nq, qstride]
    device const char  *k_q8       [[buffer(2)]],
    device const char  *v_q8       [[buffer(3)]],
    device const float *k_sc       [[buffer(4)]],
    device const float *v_sc       [[buffer(5)]],
    constant uint      &n_heads    [[buffer(6)]],
    constant uint      &n_kv_heads [[buffer(7)]],
    constant uint      &head_dim   [[buffer(8)]],
    constant uint      &kv_group   [[buffer(9)]],
    constant int       &layer_id   [[buffer(10)]],
    constant int       &n_layers   [[buffer(11)]],
    constant int       &base_tok   [[buffer(12)]],
    constant int       &nq         [[buffer(13)]],
    constant uint      &qstride    [[buffer(14)]],
    threadgroup float  *red        [[threadgroup(0)]],
    uint2 gid  [[threadgroup_position_in_grid]],     // (head, query-tile)
    uint tid   [[thread_index_in_threadgroup]],
    uint tgSz  [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_grp  [[simdgroup_index_in_threadgroup]])
{
    const uint PF_QB = 8;
    uint h = gid.x; if (h >= n_heads) return;
    uint kv_h = h / kv_group;
    int q0 = (int)gid.y * (int)PF_QB; if (q0 >= nq) return;
    int nqi = min((int)PF_QB, nq - q0);
    uint n_simd = tgSz / 32;

    // per-query running softmax state; each thread owns dim `tid`
    float m[8], l[8], acc[8];
    float qd[8];
    for (int i = 0; i < nqi; i++) {
        m[i] = -INFINITY; l[i] = 0.0f; acc[i] = 0.0f;
        device const float *qh = q + (size_t)(q0+i)*qstride + (size_t)h*head_dim;
        qd[i] = (tid < head_dim) ? qh[tid] : 0.0f;
    }
    float scale = rsqrt(float(head_dim));
    int maxkey = base_tok + q0 + nqi - 1;
    for (int t = 0; t <= maxkey; t++) {
        size_t base = (size_t)(t*n_layers+layer_id)*n_kv_heads*head_dim + (size_t)kv_h*head_dim;
        size_t slot = (size_t)(t*n_layers+layer_id);
        float ks = k_sc[slot*n_kv_heads + kv_h];
        float vs = v_sc[slot*n_kv_heads + kv_h];
        float kd = (tid < head_dim) ? ((float)k_q8[base+tid] * ks) : 0.0f;
        float vd = (tid < head_dim) ? ((float)v_q8[base+tid] * vs) : 0.0f;
        for (int i = 0; i < nqi; i++) {
            if (t > base_tok + q0 + i) continue;           // causal
            float part = simd_sum(qd[i] * kd);
            if (simd_lane == 0) red[simd_grp] = part;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float s = 0; for (uint j = 0; j < n_simd; j++) s += red[j];
            s *= scale;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float m_new = max(m[i], s);
            float corr = exp(m[i]-m_new), prob = exp(s-m_new);
            l[i] = l[i]*corr + prob; acc[i] = acc[i]*corr + prob*vd; m[i] = m_new;
        }
    }
    for (int i = 0; i < nqi; i++) {
        if (tid < head_dim) {
            device float *oh = out + (size_t)(q0+i)*(n_heads*head_dim) + (size_t)h*head_dim;
            oh[tid] = acc[i] / (l[i] + 1e-10f);
        }
    }
}
```

Notes:
- `PF_QB=8` keeps per-thread arrays small (Metal threadgroup has 128 threads →
  8 floats × 4 arrays = fine). Tune upward if occupancy allows.
- The inner `simd_sum`+barrier per (key, query) matches `attention_split_q8`;
  KV bytes are read once per key and reused across the `nqi` queries — that is
  the amortization win.
- This kernel does **not** split over the key dimension (no `n_splits`). With
  `n_heads * ceil(nq/PF_QB)` threadgroups there is already ample parallelism for
  a chunk; if a chunk sits at very high `base_tok` and you want more, add a
  split-K variant + a reduce pass (mirror `attention_reduce`).

## 5. Orchestration: `prefill_chunk_metal()`

```
static void prefill_chunk_metal(ctx, m, const int *toks, int nq) {
    int qs = HY3_N_HEAD*HY3_HEAD_DIM, kvd = HY3_N_KV_HEAD*HY3_HEAD_DIM;
    int t0 = m->cache_len / HY3_N_LAYER;
    hy3_metal_grow_kv_cache(ctx, (t0+nq)*HY3_N_LAYER);
    // allocate d_pf_x/d_pf_s[nq*EMBD], d_pf_qkv[nq*(qs+2kvd)] or separate
    //   d_pf_q[nq*qs], d_pf_k[nq*kvd], d_pf_v[nq*kvd], d_pf_ao[nq*qs] (once)
    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    // embed all nq tokens into d_pf_x  (loop embed_f16/f32 or batched kernel)
    for (int il = 0; il < HY3_N_LAYER; il++) {
        rms_norm_batched(enc, d_pf_s, d_pf_x, attn_norm[il], nq, EMBD);   // nq TGs
        // batched QKV via the _mm matmul kernels, n_cols = nq:
        m_mul_mat_mm(enc, &l->attn_q, d_pf_q, d_pf_s, qs,  EMBD, nq);
        m_mul_mat_mm(enc, &l->attn_k, d_pf_k, d_pf_s, kvd, EMBD, nq);
        m_mul_mat_mm(enc, &l->attn_v, d_pf_v, d_pf_s, kvd, EMBD, nq);
        for (int c = 0; c < nq; c++) {                   // per-token rope + kv write
            rms_norm_heads_rope(enc, d_pf_q+c*qs, d_pf_k+c*kvd, qnorm,knorm,
                                HEAD_DIM,N_HEAD,N_KV_HEAD, pos=t0+c);      // pos by value: no race
            kv_cache_write_q8(enc, slot=t0+c, d_pf_k+c*kvd, d_pf_v+c*kvd);
        }
        prefill_attn_q8(enc, d_pf_ao, d_pf_q, kcache,vcache,kscales,vscales,
                        ..., base_tok=t0, nq, qstride=qs);                 // batched attention
        m_mul_mat_mm(enc, &l->attn_output, d_pf_o, d_pf_ao, EMBD, qs, nq); // batched O-proj
        add(enc, d_pf_x, d_pf_o, nq*EMBD);                                 // x += o
        rms_norm_batched(enc, d_pf_s, d_pf_x, ffn_norm[il], nq, EMBD);
        for (int c = 0; c < nq; c++)                     // per-token FFN/MoE (reuse existing bodies,
            ffn_or_moe_one(enc, il, d_pf_s+c*EMBD, d_pf_x+c*EMBD);         // pointed at token c's slices)
    }
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    m->cache_len = (t0+nq)*HY3_N_LAYER;
}
```

Hook in `hy3_eval_metal`: mirror the CUDA hook — if `HY3_PREFILL_CHUNK>0` and
`tokens->len>1`, run `prefill_chunk_metal` over `tokens[0..len-2]` in chunks,
then run the final token through the existing `metal_forward_model_fast(...,
want_logits=1)`.

Because Metal encodes many dispatches into one command buffer, you may need to
split into several command buffers if a chunk×80-layers exceeds encoder limits;
committing every ~8–16 layers is safe.

## 6. Buffers to add (`hy3_metal_ctx_t`)

`d_pf_x, d_pf_s [C*EMBD]`, `d_pf_q, d_pf_ao, d_pf_o [C*qs]`,
`d_pf_k, d_pf_v [C*kvd]`, plus MoE per-token scratch (reuse `d_s2`/`d_moe_*` as
the decode path does — serial per token, so singletons are fine, exactly as the
CUDA `gpu_ffn_moe` reuse works). Allocate once for `C = HY3_PREFILL_CHUNK`.

## 7. Validation plan (do these in order, on a Mac)

1. **Bit-parity, chunk=1**: `HY3_PREFILL_CHUNK=1 ./run_metal.sh -m hy3.gguf -p
   "11+22+33=?" -n 24 -temp 0 --raw` must equal the default output (`…=66…`).
   (chunk=1 isolates per-token structure from batching, exactly as it caught the
   CUDA bug.)
2. **Batching parity, chunk=2 and 8**: same prompt, must still give `66`.
   If chunk≥2 diverges while chunk=1 is correct, suspect the batched attention
   or a per-query position mistake (the Metal analogue of the CUDA `h_pos` race).
3. **Scale**: `eval/hy3_needle.py` at 8k with `HY3_PREFILL_CHUNK=512` must PASS
   and be faster than default; compare `prompt … tok/s` in the timing line.
4. Keep it **off by default**; document in README once verified.

## 8. Expected result

Same shape as CUDA: correct (bit-identical greedy) and prefill-faster, with the
gain growing with prompt length. Metal's unified memory makes the KV read cheap
per byte, so the amortization still helps but the win may differ from CUDA;
measure on-device. A grouped-GEMM MoE (MegaBlocks style) is the follow-up to
also batch the routed-expert term.
