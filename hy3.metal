// hy3.metal - Metal Shading Language compute kernels for HYV3 inference.
//
// These deliberately mirror the already-validated CPU (hy3.c) and CUDA
// (hy3_gpu.cu) implementations formula-for-formula:
//   - RoPE uses the "rotate_half" pairing (dim d with dim d+head_dim/2), NOT
//     the interleaved (2i,2i+1) pairing -- verified against transformers'
//     apply_rotary_pos_emb().
//   - The KV cache is interleaved by layer: slot = token_idx*HY3_N_LAYER +
//     layer_id. attention() must read/write with that exact indexing or
//     layers silently attend over each other's keys/values.
//   - Q4_K dequant follows the same chunk-of-64 packing as
//     quantize_row_q4_K in hy3_convert.c (LOW nibble = sub-block 2c+0, HIGH
//     nibble = sub-block 2c+1, per 32-byte chunk c).
//   - MoE routing: sigmoid(router_logits), select top-k by (sigmoid+bias),
//     but weight the combination by the *unbiased* sigmoid values,
//     renormalized to sum 1 and scaled by router_scaling_factor.
//
// Unlike the CUDA backend (which must copy dequantized weights into
// discrete GPU VRAM), weights here are read directly from Metal buffers
// that wrap the mmap'd GGUF file with zero copy (see hy3_metal.m) -- Apple
// Silicon's unified memory means the GPU can dereference the same pages the
// CPU mmap'd, so quantized formats are dequantized inline per dot product
// rather than pre-expanded to F32.

#include <metal_stdlib>
using namespace metal;

constant float HY3_RMS_EPS_C = 1e-5f;

// ---------------------------------------------------------------------
// Elementwise / reduction helpers
// ---------------------------------------------------------------------

// One threadgroup per call; n can exceed threadgroup size.
kernel void rms_norm(device float       *out   [[buffer(0)]],
                      device const float *x     [[buffer(1)]],
                      device const float *w     [[buffer(2)]],
                      constant uint      &n     [[buffer(3)]],
                      uint tid   [[thread_index_in_threadgroup]],
                      uint tgSz  [[threads_per_threadgroup]])
{
    threadgroup float sdata[256];
    float sum = 0.0f;
    for (uint i = tid; i < n; i += tgSz) sum += x[i] * x[i];
    sdata[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = rsqrt(sdata[0] / float(n) + HY3_RMS_EPS_C);
    for (uint i = tid; i < n; i += tgSz) out[i] = x[i] * r * w[i];
}

// Same as rms_norm but reads/writes a sub-range starting at `off` (used for
// per-head Q/K norm where the weight is shared across heads but the input
// offset differs per head). n = head_dim.
kernel void rms_norm_offset(device float       *buf   [[buffer(0)]],
                             device const float *w     [[buffer(1)]],
                             constant uint      &n     [[buffer(2)]],
                             constant uint      &off   [[buffer(3)]],
                             uint tid   [[thread_index_in_threadgroup]],
                             uint tgSz  [[threads_per_threadgroup]])
{
    threadgroup float sdata[256];
    device float *x = buf + off;
    float sum = 0.0f;
    for (uint i = tid; i < n; i += tgSz) sum += x[i] * x[i];
    sdata[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = rsqrt(sdata[0] / float(n) + HY3_RMS_EPS_C);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n; i += tgSz) x[i] = x[i] * r * w[i];
}

// Fused per-head RMS norm: one threadgroup per head normalizes head_dim
// contiguous elements with the shared weight `w`. Replaces n_heads separate
// rms_norm_offset dispatches (was 72 tiny dispatches/layer). grid.x = n_heads.
kernel void rms_norm_heads(device float       *buf [[buffer(0)]],
                           device const float *w   [[buffer(1)]],
                           constant uint      &head_dim [[buffer(2)]],
                           uint head [[threadgroup_position_in_grid]],
                           uint tid  [[thread_index_in_threadgroup]],
                           uint tgSz [[threads_per_threadgroup]])
{
    threadgroup float sdata[256];
    device float *x = buf + (size_t)head * head_dim;
    float sum = 0.0f;
    for (uint i = tid; i < head_dim; i += tgSz) sum += x[i] * x[i];
    sdata[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = rsqrt(sdata[0] / float(head_dim) + HY3_RMS_EPS_C);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < head_dim; i += tgSz) x[i] = x[i] * r * w[i];
}

// Fused per-head Q/K RMSNorm + RoPE (rotate_half pairing). One threadgroup
// per head (Q heads then K heads). Replaces two rms_norm_heads dispatches
// plus the separate rope dispatch (3 -> 1) and drops the barrier between norm
// and rope. inv_freq[d] = 1/pow(theta, 2d/head_dim), precomputed on the host.
kernel void rms_norm_heads_rope(device float       *q   [[buffer(0)]],
                                device float       *k   [[buffer(1)]],
                                device const float *qw  [[buffer(2)]],
                                device const float *kw  [[buffer(3)]],
                                constant uint      &head_dim   [[buffer(4)]],
                                constant uint      &n_heads   [[buffer(5)]],
                                constant uint      &n_kv_heads [[buffer(6)]],
                                constant int       &pos       [[buffer(7)]],
                                device const float *inv_freq  [[buffer(8)]],
                                uint head [[threadgroup_position_in_grid]],
                                uint tid  [[thread_index_in_threadgroup]],
                                uint tgSz [[threads_per_threadgroup]])
{
    threadgroup float sdata[256];
    bool is_kv = (head >= n_heads);
    uint hh = is_kv ? (head - n_heads) : head;
    device float *buf = is_kv ? k : q;
    device const float *w = is_kv ? kw : qw;
    device float *x = buf + (size_t)hh * head_dim;
    float sum = 0.0f;
    for (uint i = tid; i < head_dim; i += tgSz) sum += x[i] * x[i];
    sdata[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = rsqrt(sdata[0] / float(head_dim) + HY3_RMS_EPS_C);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < head_dim; i += tgSz) x[i] = x[i] * r * w[i];
    // Barrier: rope reads x[d+half_dim] written by another thread's norm
    // above, so the norm writes must be visible before the rope reads.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // RoPE rotate_half: pair d with d+half_dim within this head.
    uint half_dim = head_dim / 2;
    for (uint d = tid; d < half_dim; d += tgSz) {
        float freq = float(pos) * inv_freq[d];
        float c = cos(freq), s = sin(freq);
        float v0 = x[d], v1 = x[d + half_dim];
        x[d]            = v0 * c - v1 * s;
        x[d + half_dim] = v1 * c + v0 * s;
    }
}

/* All of the small elementwise kernels below take an explicit element
 * count `n` and bounds-check, even though every current call site happens
 * to dispatch an exact multiple of the threadgroup size (so the check
 * never actually trips today): dispatchThreadgroups rounds up to whole
 * threadgroups, and without this check a future caller with a
 * non-multiple size would silently read/write past the buffer. */

kernel void silu_mul(device float *out [[buffer(0)]],
                      device const float *gate [[buffer(1)]],
                      device const float *up [[buffer(2)]],
                      constant uint &n [[buffer(3)]],
                      uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float x = gate[i];
    out[i] = (x / (1.0f + exp(-x))) * up[i];
}

kernel void sigmoid_inplace(device float *x [[buffer(0)]],
                             constant uint &n [[buffer(1)]],
                             uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    x[i] = 1.0f / (1.0f + exp(-x[i]));
}

kernel void add_inplace(device float *a [[buffer(0)]],
                         device const float *b [[buffer(1)]],
                         constant uint &n [[buffer(2)]],
                         uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    a[i] += b[i];
}

kernel void scale_add_inplace(device float *acc [[buffer(0)]],
                               device const float *b [[buffer(1)]],
                               constant float &scale [[buffer(2)]],
                               constant uint &n [[buffer(3)]],
                               uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    acc[i] += scale * b[i];
}

kernel void fill_zero(device float *x [[buffer(0)]],
                       constant uint &n [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    x[i] = 0.0f;
}

// Startup page-warming: read one byte per `stride` bytes across a model
// view and accumulate into a scratch sink. Forces the GPU driver to fault
// in / wire the mmap'd GGUF pages into GPU-accessible residency before
// timed inference, so per-token latency isn't dominated by first-touch VM
// faults on the zero-copy weight buffers. (Mirrors ds4_metal.m's
// kernel_touch_u8_stride.)
kernel void touch_u8_stride(device const uchar *data   [[buffer(0)]],
                             device atomic_uint *sink   [[buffer(1)]],
                             constant uint       &stride [[buffer(2)]],
                             constant uint       &n_steps[[buffer(3)]],
                             uint i [[thread_position_in_grid]])
{
    if (i >= n_steps) return;
    uint acc = data[(size_t)i * stride];
    if (acc == 0xFF)
        atomic_fetch_add_explicit(sink, acc, memory_order_relaxed);
}

kernel void embed_lookup_f16(device float *out [[buffer(0)]],
                              device const half *table [[buffer(1)]],
                              constant uint &token [[buffer(2)]],
                              constant uint &dim [[buffer(3)]],
                              uint i [[thread_position_in_grid]])
{
    if (i < dim) out[i] = float(table[(size_t)token * dim + i]);
}

kernel void embed_lookup_f32(device float *out [[buffer(0)]],
                              device const float *table [[buffer(1)]],
                              constant uint &token [[buffer(2)]],
                              constant uint &dim [[buffer(3)]],
                              uint i [[thread_position_in_grid]])
{
    if (i < dim) out[i] = table[(size_t)token * dim + i];
}

// ---------------------------------------------------------------------
// RoPE -- rotate_half convention (see file header comment).
// Dispatched with (n_heads+n_kv_heads)*(head_dim/2) threads.
// ---------------------------------------------------------------------
kernel void rope(device float *q [[buffer(0)]],
                   device float *k [[buffer(1)]],
                   constant int  &pos [[buffer(2)]],
                   constant uint &head_dim [[buffer(3)]],
                   constant uint &n_heads [[buffer(4)]],
                   constant uint &n_kv_heads [[buffer(5)]],
                   device const float *inv_freq [[buffer(6)]],
                   uint idx [[thread_position_in_grid]])
{
    uint half_dim = head_dim / 2;
    uint total = (n_heads + n_kv_heads) * half_dim;
    if (idx >= total) return;
    uint h, d; bool is_kv;
    if (idx < n_heads * half_dim) {
        h = idx / half_dim; d = idx % half_dim; is_kv = false;
    } else {
        uint off = idx - n_heads * half_dim;
        h = off / half_dim; d = off % half_dim; is_kv = true;
    }
    device float *buf = is_kv ? k : q;
    device float *base = buf + (size_t)h * head_dim;
    // inv_freq[d] = 1/pow(theta, 2d/head_dim) precomputed on the host, so a
    // multiply replaces a per-element pow()/exp() in this hot kernel.
    float freq = float(pos) * inv_freq[d];
    float c = cos(freq), s = sin(freq);
    float v0 = base[d], v1 = base[d + half_dim];
    base[d]            = v0 * c - v1 * s;
    base[d + half_dim] = v1 * c + v0 * s;
}

// Copy this token's K/V into the interleaved cache at slot dst_slot
// (= token_idx*HY3_N_LAYER + layer_id). Done as a kernel (rather than a
// CPU-side memcpy of the d_k/d_v buffer contents) so it stays correctly
// ordered, via Metal's automatic hazard tracking, after the rope kernel
// that just wrote d_k/d_v within the same command encoder -- a CPU memcpy
// at that point would race the GPU and could copy pre-RoPE (or stale)
// data, since nothing has been committed/waited on yet.
kernel void kv_cache_write(device half        *k_cache  [[buffer(0)]],
                            device half        *v_cache  [[buffer(1)]],
                            device const float *k        [[buffer(2)]],
                            device const float *v        [[buffer(3)]],
                            constant uint       &kv_size  [[buffer(4)]],
                            constant uint       &dst_slot [[buffer(5)]],
                            uint i [[thread_position_in_grid]])
{
    if (i >= kv_size) return;
    size_t off = (size_t)dst_slot * kv_size + i;
    k_cache[off] = (half)k[i];
    v_cache[off] = (half)v[i];
}

// ---------------------------------------------------------------------
// Q8 KV cache write: per-head-channel absmax → int8 quantization.
// One threadgroup per KV head, head_dim threads. Quantizes both K and V.
// ---------------------------------------------------------------------
kernel void kv_cache_write_q8(
    device char        *k_q8   [[buffer(0)]],
    device char        *v_q8   [[buffer(1)]],
    device float       *k_sc   [[buffer(2)]],
    device float       *v_sc   [[buffer(3)]],
    device const float *k      [[buffer(4)]],
    device const float *v      [[buffer(5)]],
    constant uint      &kv_sz  [[buffer(6)]],
    constant uint      &slot   [[buffer(7)]],
    constant uint      &nkv    [[buffer(8)]],
    threadgroup float  *red    [[threadgroup(0)]],
    uint h    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_grp  [[simdgroup_index_in_threadgroup]])
{
    if (h >= nkv) return;
    uint hdim = kv_sz / nkv, ho = h * hdim;
    float kv = (tid < hdim) ? k[ho + tid] : 0.0f;
    float vv = (tid < hdim) ? v[ho + tid] : 0.0f;
    float ak = fabs(kv), av = fabs(vv);
    ak = simd_max(ak); av = simd_max(av);
    if (simd_lane == 0) red[simd_grp] = ak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gk = 0; for (uint i = 0; i < 128/32; i++) gk = fmax(gk, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) red[simd_grp] = av;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gv = 0; for (uint i = 0; i < 128/32; i++) gv = fmax(gv, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ks = (gk > 0) ? (gk / 127.0f) : (1.0f / 127.0f);
    float vs = (gv > 0) ? (gv / 127.0f) : (1.0f / 127.0f);
    if (tid == 0) { k_sc[slot * nkv + h] = ks; v_sc[slot * nkv + h] = vs; }
    if (tid < hdim) {
        k_q8[(size_t)slot * kv_sz + ho + tid] = (char)(int)clamp(round(kv/ks), -127.f, 127.f);
        v_q8[(size_t)slot * kv_sz + ho + tid] = (char)(int)clamp(round(vv/vs), -127.f, 127.f);
    }
}

// ---------------------------------------------------------------------
// Online-softmax attention. One threadgroup per head, head_dim threads.
// each K and V vector is read exactly once, a running max `m` and
// denominator `l` are maintained, and the value accumulator is rescaled
// online. No materialized score array -> no context-length ceiling and
// ~half the KV global traffic of the old two-pass kernel.
//
// One threadgroup per head, head_dim (=128) threads. Thread `d` owns output
// dimension `d` and its accumulator `acc`. Per timestep the score q·k is
// computed by a cooperative threadgroup reduction, then every thread applies
// the online-softmax rescale and adds this timestep's contribution.
// slot(t, layer) = t*n_layers + layer.  Accumulate in float.
// ---------------------------------------------------------------------
kernel void attention(device float       *out      [[buffer(0)]],
                       device const float *q        [[buffer(1)]],
                       device const half  *k_cache  [[buffer(2)]],
                       device const half  *v_cache  [[buffer(3)]],
                       constant uint      &n_heads    [[buffer(4)]],
                       constant uint      &n_kv_heads [[buffer(5)]],
                       constant uint      &head_dim   [[buffer(6)]],
                       constant int       &kv_len     [[buffer(7)]],
                       constant uint      &kv_group   [[buffer(8)]],
                       constant int       &layer_id   [[buffer(9)]],
                       constant int       &n_layers   [[buffer(10)]],
                       threadgroup float  *red        [[threadgroup(0)]],
                       uint h    [[threadgroup_position_in_grid]],
                       uint tid  [[thread_index_in_threadgroup]],
                       uint tgSz [[threads_per_threadgroup]],
                       uint simd_lane [[thread_index_in_simdgroup]],
                       uint simd_grp  [[simdgroup_index_in_threadgroup]])
{
    if (h >= n_heads) return;
    uint kv_h = h / kv_group;
    device const float *q_h = q + (size_t)h * head_dim;
    float scale = rsqrt(float(head_dim));

    /* Number of tokens whose K/V are present for this layer: cache_len is in
     * units of (token*80 + layer), so the current token T contributes slots
     * 0..T (inclusive of its own K/V, written just above). That count is
     * (kv_len - layer_id)/n_layers + 1 = (kv_len - layer_id + n_layers)/n_layers.
     * The previous "- 1" excluded the current token's self-attention. */
    int ntok = (kv_len - layer_id + n_layers) / n_layers;
    if (ntok < 1) ntok = 1;

    /* q value this thread contributes to every dot product (thread d owns dim d). */
    float qd = (tid < head_dim) ? q_h[tid] : 0.0f;

    const uint n_simd = tgSz / 32;   /* number of simdgroups (128/32 = 4) */

    float m = -INFINITY;   /* running max score */
    float l = 0.0f;        /* running denominator */
    float acc = 0.0f;      /* running sum_t softmax_t * V[t][d] for this d */

    for (int t = 0; t < ntok; t++) {
        size_t base = (size_t)(t * n_layers + layer_id) * n_kv_heads * head_dim
                    + (size_t)kv_h * head_dim;
        /* score = scale * sum_d q[d]*k[d]; reduce partial products across threads. */
        float part = (tid < head_dim) ? qd * (float)k_cache[base + tid] : 0.0f;
        part = simd_sum(part);
        if (simd_lane == 0) red[simd_grp] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s = 0.0f;
        for (uint i = 0; i < n_simd; i++) s += red[i];
        s *= scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* online-softmax update */
        float m_new = max(m, s);
        float corr = exp(m - m_new);
        float p = exp(s - m_new);
        l = l * corr + p;
        float vd = (tid < head_dim) ? (float)v_cache[base + tid] : 0.0f;
        acc = acc * corr + p * vd;
        m = m_new;
    }

    if (tid < head_dim) {
        device float *out_h = out + (size_t)h * head_dim;
        out_h[tid] = acc / (l + 1e-10f);
    }
}

// ---------------------------------------------------------------------
// Split-KV attention (FlashDecoding). Partition the KV sequence
// across ATTN_SPLITS chunks; each chunk runs online-softmax independently
// and writes (m, l, acc[0..head_dim-1]) partials. A second reduce kernel
// merges partials via online-softmax. This cuts the per-head serial scan
// ~16x, removing the O(context) decode bottleneck.
// ---------------------------------------------------------------------

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
    threadgroup float  *red        [[threadgroup(0)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tgSz  [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_grp  [[simdgroup_index_in_threadgroup]])
{
    uint h     = gid / n_splits;
    uint split = gid % n_splits;
    if (h >= n_heads) return;

    uint kv_h = h / kv_group;
    int ntok  = (kv_len - layer_id + n_layers) / n_layers;
    if (ntok < 1) ntok = 1;

    int chunk  = (ntok + (int)n_splits - 1) / (int)n_splits;
    int r0     = (int)split * chunk;
    int r1     = r0 + chunk;
    if (r1 > ntok) r1 = ntok;

    device float *p = partials + (size_t)gid * (2 + head_dim);

    if (r0 >= r1) {
        if (tid == 0) { p[0] = -INFINITY; p[1] = 0.0f; }
        for (uint i = tid; i < head_dim; i += tgSz) p[2 + i] = 0.0f;
        return;
    }

    device const float *q_h = q + (size_t)h * head_dim;
    float qd = (tid < head_dim) ? q_h[tid] : 0.0f;
    float scale = rsqrt(float(head_dim));
    const uint n_simd = tgSz / 32;

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;

    for (int t = r0; t < r1; t++) {
        size_t base = (size_t)(t * n_layers + layer_id) * n_kv_heads * head_dim
                    + (size_t)kv_h * head_dim;

        float part = (tid < head_dim) ? qd * (float)k_cache[base + tid] : 0.0f;
        part = simd_sum(part);
        if (simd_lane == 0) red[simd_grp] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s = 0.0f;
        for (uint i = 0; i < n_simd; i++) s += red[i];
        s *= scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_new = max(m, s);
        float corr = exp(m - m_new);
        float prob = exp(s - m_new);
        l   = l   * corr + prob;
        acc = acc * corr + prob * ((tid < head_dim) ? (float)v_cache[base + tid] : 0.0f);
        m   = m_new;
    }

    if (tid == 0) { p[0] = m; p[1] = l; }
    p[2 + tid] = (tid < head_dim) ? acc : 0.0f;
}

kernel void attention_reduce(
    device float       *out       [[buffer(0)]],
    device const float *partials  [[buffer(1)]],
    constant uint      &n_heads   [[buffer(2)]],
    constant uint      &head_dim  [[buffer(3)]],
    constant uint      &n_splits  [[buffer(4)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (h >= n_heads || tid >= head_dim) return;

    float m_global = -INFINITY;
    float l_global = 0.0f;
    float acc = 0.0f;

    for (uint s = 0; s < n_splits; s++) {
        size_t off = (size_t)(h * n_splits + s) * (2 + head_dim);
        float pm = partials[off];
        float pl = partials[off + 1];
        if (pl <= 0.0f) continue;

        float m_new = max(m_global, pm);
        float corr  = exp(m_global - m_new);
        float rsc   = exp(pm - m_new);
        l_global = l_global * corr + pl * rsc;
        acc      = acc      * corr + partials[off + 2 + tid] * rsc;
        m_global = m_new;
    }

    float inv = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
    out[(size_t)h * head_dim + tid] = acc * inv;
}

// ---------------------------------------------------------------------
// Dense matmul kernels: dst[row] = dot(W[row,:], x). One threadgroup per
// output row. Vectorized loads (float4/half4) + SIMD-group reduction
// (simd_sum) instead of a full threadgroup barrier tree -- fewer barriers,
// coalesced memory. Threadgroup = 256 threads = 8 SIMD groups of 32.
// ---------------------------------------------------------------------

kernel void matmul_f32(device float       *dst [[buffer(0)]],
                        device const float *w   [[buffer(1)]],
                        device const float *x   [[buffer(2)]],
                        constant uint      &n   [[buffer(3)]],
                        uint row  [[threadgroup_position_in_grid]],
                        uint tid  [[thread_index_in_threadgroup]],
                        uint tgSz [[threads_per_threadgroup]],
                        uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_grp  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[8];
    device const float4 *wr = (device const float4 *)(w + (size_t)row * n);
    device const float4 *xv = (device const float4 *)x;
    uint n4 = n / 4;
    float sum = 0.0f;
    for (uint j = tid; j < n4; j += tgSz) {
        float4 a = wr[j], b = xv[j];
        sum += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) partial[simd_grp] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (uint i = 0; i < tgSz / 32; i++) t += partial[i];
        dst[row] = t;
    }
}

kernel void matmul_f16(device float      *dst [[buffer(0)]],
                        device const half *w   [[buffer(1)]],
                        device const float *x  [[buffer(2)]],
                        constant uint     &n   [[buffer(3)]],
                        uint row  [[threadgroup_position_in_grid]],
                        uint tid  [[thread_index_in_threadgroup]],
                        uint tgSz [[threads_per_threadgroup]],
                        uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_grp  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[8];
    device const half4  *wr = (device const half4 *)(w + (size_t)row * n);
    device const float4 *xv = (device const float4 *)x;
    uint n4 = n / 4;
    float sum = 0.0f;
    for (uint j = tid; j < n4; j += tgSz) {
        float4 a = float4(wr[j]), b = xv[j];
        sum += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) partial[simd_grp] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (uint i = 0; i < tgSz / 32; i++) t += partial[i];
        dst[row] = t;
    }
}

// Q8_0 block: 4-byte float scale + 32 signed int8 values = 36 bytes/32
// elems (this is hy3's own Q8_0 variant with an F32 scale, NOT upstream
// ggml's F16-scale Q8_0 -- see hy3_convert.c / hy3.c block_q8_0).
kernel void matmul_q8_0(device float       *dst [[buffer(0)]],
                         device const uchar *w   [[buffer(1)]],
                         device const float *x   [[buffer(2)]],
                         constant uint      &n   [[buffer(3)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint tgSz [[threads_per_threadgroup]],
                         uint simd_lane [[thread_index_in_simdgroup]],
                         uint simd_grp  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[8];
    uint nb = n / 32;
    device const uchar *wr = w + (size_t)row * nb * 36;
    float sum = 0.0f;
    for (uint j = tid; j < nb; j += tgSz) {
        device const uchar *blk = wr + (size_t)j * 36;
        float d = *(device const float *)blk;
        device const char4  *qs = (device const char4 *)(blk + 4);
        device const float4 *xv = (device const float4 *)(x + j * 32);
        float local = 0.0f;
        for (uint l = 0; l < 8; l++) {
            float4 qf = float4(qs[l]);
            float4 xf = xv[l];
            local += qf.x*xf.x + qf.y*xf.y + qf.z*xf.z + qf.w*xf.w;
        }
        sum += d * local;
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) partial[simd_grp] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (uint i = 0; i < tgSz / 32; i++) t += partial[i];
        dst[row] = t;
    }
}

// ---------------------------------------------------------------------
// Q8_0 matmul, SIMD-group vectorized variant (matmul_q8_0_mm).
//
// Same math as matmul_q8_0 above, restructured after llama.cpp's
// kernel_mul_mv_q8_0_f32 for higher memory-bandwidth utilization:
//   - One SIMD-group (32 lanes) cooperatively computes Q8_0_N_ROWS output
//     rows at once. The 32 lanes split so ix = lane/4 picks one of 8
//     blocks and il = lane%4 picks 8 of the 32 quants in that block;
//     consecutive lanes read consecutive words (coalesced loads).
//   - Each lane keeps a partial sum per row; a final simd_sum reduces the
//     32 lanes and lane 0 writes the result.
//
// Dispatch: threadgroups = ceil(m_rows / Q8_0_N_ROWS), 32 threads each.
// Buffer/arg layout identical to matmul_q8_0 (dst,w,x,n).
//
// hy3 block_q8_0 (36 bytes / 32 elems): float d, int8 qs[32].
// ---------------------------------------------------------------------

#define Q8_0_N_ROWS 4

kernel void matmul_q8_0_mm(device float       *dst [[buffer(0)]],
                            device const uchar *w   [[buffer(1)]],
                            device const float *x   [[buffer(2)]],
                            constant uint      &n   [[buffer(3)]],
                            uint  tgpig [[threadgroup_position_in_grid]],
                            ushort tiisg [[thread_index_in_simdgroup]])
{
    const short NQ = 8;           // quants handled per lane per block
    const short ix = tiisg / 4;   // 0..7  -> which block within a stride
    const short il = tiisg % 4;   // 0..3  -> which 8-quant slice of the block

    const uint nb = n / 32;       // blocks per row
    const uint first_row = (uint)tgpig * Q8_0_N_ROWS;
    const uint row_stride = nb * 36;   // bytes per weight row

    float sumf[Q8_0_N_ROWS] = {0.f};
    float yl[8];

    device const float *yb = x + ix * 32 + il * NQ;

    for (uint ib = ix; ib < nb; ib += 8) {
        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];

        device const uchar *blk0 = w + first_row * row_stride + (size_t)ib * 36;
        for (short row = 0; row < Q8_0_N_ROWS; row++) {
            device const uchar *blk = blk0 + (size_t)row * row_stride;
            float d = *(device const float *)blk;
            device const char *qs = (device const char *)(blk + 4) + il * NQ;
            float sumq = 0.f;
            for (short i = 0; i < NQ; ++i) sumq += (float)qs[i] * yl[i];
            sumf[row] += sumq * d;
        }

        yb += 8 * 32;
    }

    for (short row = 0; row < Q8_0_N_ROWS; row++) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0) dst[first_row + row] = tot;
    }
}
// but dequantizes inline instead of materializing a temporary buffer.
// Block layout (144 bytes / 256 elems): fp16 d, fp16 dmin, 12 bytes of
// packed 6-bit (scale,min) pairs for 8 sub-blocks, 128 bytes of 4-bit
// weights packed as 4 chunks of 64 (low nibble = sub-block 2c, high
// nibble = sub-block 2c+1 of each 32-byte chunk).
// ---------------------------------------------------------------------

inline void hy3_q4k_get_scale_min(uint j, device const uchar *q, thread uchar &sc, thread uchar &m) {
    if (j < 4) {
        sc = q[j] & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)  | ((q[j]      >> 6) << 4);
    }
}

// One block dot product against a contiguous 256-float slice of x.
// Vectorized: process the 32 nibble-bytes of each sub-block pair as uchar4
// against float4 activation lanes.
inline float hy3_q4k_dot_block(device const uchar *blk, device const float *xb) {
    half dh   = *(device const half *)blk;
    half dmh  = *(device const half *)(blk + 2);
    float d    = float(dh);
    float dmin = float(dmh);
    device const uchar *sc = blk + 4;
    device const uchar *q  = blk + 16;
    float acc = 0.0f;
    uint is = 0;
    for (uint j = 0; j < 256; j += 64) {
        uchar sc1, m1, sc2, m2;
        hy3_q4k_get_scale_min(is + 0, sc, sc1, m1);
        hy3_q4k_get_scale_min(is + 1, sc, sc2, m2);
        float d1 = d * float(sc1), dm1 = dmin * float(m1);
        float d2 = d * float(sc2), dm2 = dmin * float(m2);
        device const uchar4 *qq = (device const uchar4 *)(q + (j / 2));
        device const float4 *x1 = (device const float4 *)(xb + j);
        device const float4 *x2 = (device const float4 *)(xb + j + 32);
        for (uint l = 0; l < 8; l++) {
            uchar4 packed = qq[l];
            float4 lo = float4(packed & 0xF);
            float4 hi = float4(packed >> 4);
            float4 v1 = d1 * lo - dm1;
            float4 v2 = d2 * hi - dm2;
            float4 a1 = x1[l], a2 = x2[l];
            acc += v1.x*a1.x + v1.y*a1.y + v1.z*a1.z + v1.w*a1.w
                 + v2.x*a2.x + v2.y*a2.y + v2.z*a2.z + v2.w*a2.w;
        }
        is += 2;
    }
    return acc;
}

kernel void matmul_q4_k(device float       *dst [[buffer(0)]],
                         device const uchar *w   [[buffer(1)]],
                         device const float *x   [[buffer(2)]],
                         constant uint      &n   [[buffer(3)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint tgSz [[threads_per_threadgroup]],
                         uint simd_lane [[thread_index_in_simdgroup]],
                         uint simd_grp  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[8];
    uint nb = n / 256;
    device const uchar *wr = w + (size_t)row * nb * 144;
    float sum = 0.0f;
    for (uint bi = tid; bi < nb; bi += tgSz) {
        device const uchar *blk = wr + (size_t)bi * 144;
        sum += hy3_q4k_dot_block(blk, x + (size_t)bi * 256);
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) partial[simd_grp] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (uint i = 0; i < tgSz / 32; i++) t += partial[i];
        dst[row] = t;
    }
}

// ---------------------------------------------------------------------
// Q4_K matmul, SIMD-group vectorized variant (matmul_q4_k_mm).
//
// Same math as matmul_q4_k above, but restructured after llama.cpp's
// kernel_mul_mv_q4_K_f32 for far higher memory-bandwidth utilization:
//   - One SIMD-group (32 lanes) cooperatively computes Q4K_N_ROWS output
//     rows at once. The 32 lanes split into 4 groups of 8 (ix = lane/8),
//     each group striding over the super-blocks (ib += 4), so consecutive
//     lanes read consecutive weight/activation words (coalesced loads).
//   - The 6-bit (scale,min) pairs are unpacked with the same kmask trick
//     as ggml; the 4-bit weights are multiplied against the activations
//     without materializing a dequantized row.
//   - Each lane keeps a partial sum per row; a final simd_sum reduces the
//     32 lanes and lane 0 writes the result.
//
// Dispatch: threadgroups = ceil(m_rows / Q4K_N_ROWS), 32 threads each.
// Buffer/arg layout is identical to matmul_q4_k (dst,w,x,n) so the host
// only changes the pipeline + grid, not the bindings.
//
// hy3 block_q4_K (144 bytes / 256 elems): half d, half dmin, 12 bytes of
// packed 6-bit (scale,min) pairs, 128 bytes of 4-bit weights.
// ---------------------------------------------------------------------

#define Q4K_N_ROWS 4

kernel void matmul_q4_k_mm(device float       *dst [[buffer(0)]],
                            device const uchar *w   [[buffer(1)]],
                            device const float *x   [[buffer(2)]],
                            constant uint      &n   [[buffer(3)]],
                            uint  tgpig [[threadgroup_position_in_grid]],
                            ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;   // 0..3  -> which super-block within a stride
    const short it = tiisg % 8;   // 0..7
    const short iq = it / 4;      // 0 or 1
    const short ir = it % 4;      // 0..3

    const uint nb = n / 256;      // super-blocks per row
    const uint first_row = (uint)tgpig * Q4K_N_ROWS;
    const uint row_stride = nb * 144;   // bytes per weight row

    float yl[16];
    float yh[16];
    float sumf[Q4K_N_ROWS] = {0.f};

    device const float *y4 = x + ix * 256 + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uchar *blk0 = w + first_row * row_stride + (size_t)ib * 144;

        for (short row = 0; row < Q4K_N_ROWS; row++) {
            device const uchar *blk = blk0 + (size_t)row * row_stride;
            device const half     *dh = (device const half *)blk;
            device const uint16_t *sc = (device const uint16_t *)(blk + 4) + iq;
            device const uint16_t *q1 = (device const uint16_t *)(blk + 16) + 16 * iq + 4 * ir;

            sc16[0] =  sc[0] & kmask1;
            sc16[1] =  sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t *q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }

            float d    = (float)dh[0];
            float dmin = (float)dh[1];

            sumf[row] += d * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                              (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                              (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                              (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                 sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y4 += 4 * 256;
    }

    for (short row = 0; row < Q4K_N_ROWS; row++) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0) dst[first_row + row] = tot;
    }
}

// =====================================================================
// FAST MoE PATH (ds4-style, keeps the whole token on the GPU).
//
// router_topk : GPU-side top-k over the router logits. Emits the chosen
//               expert ids and their renormalized+scaled combine weights,
//               so the CPU never has to read router logits mid-token.
// q4k_mul_mv_id : one dispatch computes ALL K routed experts. tgpig.y is
//               the expert slot; the real expert id and combine weight come
//               from GPU buffers. Weights of a layer's experts are laid out
//               contiguously in the mmap with a fixed byte stride, so an
//               expert base pointer + id*stride locates any expert.
// =====================================================================

kernel void router_topk(device const float *logits [[buffer(0)]],  // NE
                        device const float *bias   [[buffer(1)]],  // NE (unused if has_bias==0)
                        device int         *out_ids [[buffer(2)]], // K
                        device float       *out_wts [[buffer(3)]], // K
                        constant uint      &NE      [[buffer(4)]],
                        constant uint      &K       [[buffer(5)]],
                        constant uint      &has_bias[[buffer(6)]],
                        constant float     &scaling [[buffer(7)]],
                        uint tid  [[thread_index_in_threadgroup]],
                        uint tgSz [[threads_per_threadgroup]])
{
    threadgroup float sig[256];    // sigmoid(logit)
    threadgroup float scored[256]; // sigmoid + bias (selection score)
    for (uint i = tid; i < NE; i += tgSz) {
        float s = 1.0f / (1.0f + exp(-logits[i]));
        sig[i] = s;
        scored[i] = s + (has_bias ? bias[i] : 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum_w = 0.0f;
        for (uint k = 0; k < K; k++) {
            int best = -1; float bv = -1e30f;
            for (uint i = 0; i < NE; i++) {
                float v = scored[i];
                if (v > bv) { bv = v; best = (int)i; }
            }
            out_ids[k] = best;
            out_wts[k] = sig[best];   // temporarily store unbiased sigmoid
            sum_w += sig[best];
            scored[best] = -1e30f;
        }
        float inv = 1.0f / (sum_w + 1e-20f);
        for (uint k = 0; k < K; k++)
            out_wts[k] = out_wts[k] * inv * scaling;
    }
}

// One dispatch = all K routed experts, Q4_K. grid = (M/4, K, 1), 32 threads.
// dst layout: [slot*M + row]. experts points at expert 0's weights; expert
// `ids[slot]` starts at experts + ids[slot]*expert_stride bytes.
kernel void matmul_q4_k_id(device float       *dst     [[buffer(0)]], // K*M
                           device const uchar *experts [[buffer(1)]],
                           device const float *x       [[buffer(2)]], // N or K*N (see x_per_slot)
                           constant uint      &n       [[buffer(3)]],
                           constant uint      &M       [[buffer(4)]],
                           constant uint      &expert_stride [[buffer(5)]],
                           device const int   *ids     [[buffer(6)]], // K
                           constant uint      &x_per_slot [[buffer(7)]], // 0: shared x, 1: x+slot*n
                           uint3  tgpig [[threadgroup_position_in_grid]],
                           ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const uint slot = tgpig.y;
    const int  eid  = ids[slot];
    device const uchar *w = experts + (size_t)eid * expert_stride;
    device const float *xs = x_per_slot ? (x + (size_t)slot * n) : x;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const uint nb = n / 256;
    const uint first_row = (uint)tgpig.x * 4;
    const uint row_stride = nb * 144;

    float yl[16];
    float yh[16];
    float sumf[4] = {0.f};

    device const float *y4 = xs + ix * 256 + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }
        device const uchar *blk0 = w + first_row * row_stride + (size_t)ib * 144;
        for (short row = 0; row < 4; row++) {
            device const uchar *blk = blk0 + (size_t)row * row_stride;
            device const half     *dh = (device const half *)blk;
            device const uint16_t *sc = (device const uint16_t *)(blk + 4) + iq;
            device const uint16_t *q1 = (device const uint16_t *)(blk + 16) + 16 * iq + 4 * ir;
            sc16[0] =  sc[0] & kmask1;
            sc16[1] =  sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const uint16_t *q2 = q1 + 32;
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }
            float d    = (float)dh[0];
            float dmin = (float)dh[1];
            sumf[row] += d * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                              (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                              (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                              (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                 sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }
        y4 += 4 * 256;
    }
    for (short row = 0; row < 4; row++) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0) dst[(size_t)slot * M + first_row + row] = tot;
    }
}

// SwiGLU for all K slots at once: mid[i] = silu(gate[i]) * up[i], i in [0,K*M).
kernel void moe_swiglu_id(device float       *mid  [[buffer(0)]], // K*M
                          device const float *gate [[buffer(1)]], // K*M
                          device const float *up   [[buffer(2)]], // K*M
                          constant uint      &total[[buffer(3)]], // K*M
                          uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    float g = gate[i];
    float s = g / (1.0f + exp(-g));
    mid[i] = s * up[i];
}

// Weighted combine of K routed-expert down-projections into the residual:
//   embed[r] += sum_slot  wt[slot] * down[slot*M + r]
kernel void moe_combine_id(device float       *embed [[buffer(0)]], // M (+=)
                           device const float *down  [[buffer(1)]], // K*M
                           device const float *wt    [[buffer(2)]], // K
                           constant uint      &M     [[buffer(3)]],
                           constant uint      &K     [[buffer(4)]],
                           uint r [[thread_position_in_grid]])
{
    if (r >= M) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; k++) acc += wt[k] * down[(size_t)k * M + r];
    embed[r] += acc;
}

// ---------------------------------------------------------------------
// Split-KV attention with Q8 KV cache. Same algorithm as attention_split
// but loads K/V from int8 cache with per-head-channel dequantization.
// ---------------------------------------------------------------------
kernel void attention_split_q8(
    device float       *partials   [[buffer(0)]],
    device const float *q          [[buffer(1)]],
    device const char  *k_q8       [[buffer(2)]],
    device const char  *v_q8       [[buffer(3)]],
    device const float *k_sc       [[buffer(12)]],
    device const float *v_sc       [[buffer(13)]],
    constant uint      &n_heads    [[buffer(4)]],
    constant uint      &n_kv_heads [[buffer(5)]],
    constant uint      &head_dim   [[buffer(6)]],
    constant int       &kv_len     [[buffer(7)]],
    constant uint      &kv_group   [[buffer(8)]],
    constant int       &layer_id   [[buffer(9)]],
    constant int       &n_layers   [[buffer(10)]],
    constant uint      &n_splits   [[buffer(11)]],
    threadgroup float  *red        [[threadgroup(0)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tgSz  [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_grp  [[simdgroup_index_in_threadgroup]])
{
    uint h = gid / n_splits, split = gid % n_splits;
    if (h >= n_heads) return;
    uint kv_h = h / kv_group;
    int ntok = (kv_len - layer_id + n_layers) / n_layers;
    if (ntok < 1) ntok = 1;
    int chunk = (ntok + (int)n_splits - 1) / (int)n_splits;
    int r0 = (int)split * chunk, r1 = r0 + chunk;
    if (r1 > ntok) r1 = ntok;
    device float *p = partials + (size_t)gid * (2 + head_dim);
    if (r0 >= r1) {
        if (tid == 0) { p[0] = -INFINITY; p[1] = 0.0f; }
        for (uint i = tid; i < head_dim; i += tgSz) p[2+i] = 0.0f;
        return;
    }
    device const float *q_h = q + (size_t)h * head_dim;
    float qd = (tid < head_dim) ? q_h[tid] : 0.0f;
    float scale = rsqrt(float(head_dim));
    uint n_simd = tgSz / 32;
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (int t = r0; t < r1; t++) {
        size_t base = (size_t)(t*n_layers+layer_id)*n_kv_heads*head_dim + (size_t)kv_h*head_dim;
        size_t slot = (size_t)(t*n_layers+layer_id);
        float ks = k_sc[slot * n_kv_heads + kv_h];
        float vs = v_sc[slot * n_kv_heads + kv_h];
        float part = (tid < head_dim) ? qd * ((float)k_q8[base+tid] * ks) : 0.0f;
        part = simd_sum(part);
        if (simd_lane == 0) red[simd_grp] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s = 0;
        for (uint i = 0; i < n_simd; i++) s += red[i];
        s *= scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float m_new = max(m, s);
        float corr = exp(m-m_new), prob = exp(s-m_new);
        float vd = (tid < head_dim) ? ((float)v_q8[base+tid] * vs) : 0.0f;
        l = l*corr+prob; acc = acc*corr+prob*vd; m = m_new;
    }
    if (tid == 0) { p[0] = m; p[1] = l; }
    if (tid < head_dim) p[2+tid] = acc;
}


