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

constant float HY3_ROPE_THETA_C = 11158840.0f;
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
    float freq = float(pos) / pow(HY3_ROPE_THETA_C, float(2 * d) / float(head_dim));
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
kernel void kv_cache_write(device float       *k_cache  [[buffer(0)]],
                            device float       *v_cache  [[buffer(1)]],
                            device const float *k        [[buffer(2)]],
                            device const float *v        [[buffer(3)]],
                            constant uint       &kv_size  [[buffer(4)]],
                            constant uint       &dst_slot [[buffer(5)]],
                            uint i [[thread_position_in_grid]])
{
    if (i >= kv_size) return;
    size_t off = (size_t)dst_slot * kv_size + i;
    k_cache[off] = k[i];
    v_cache[off] = v[i];
}

// ---------------------------------------------------------------------
// Attention over the layer-interleaved KV cache.
// slot(t, layer) = t * n_layers + layer.  Dispatched with n_heads
// threadgroups; every thread in a group redundantly computes the full
// score vector (cheap relative to the head_dim=128 dot products, and
// avoids a second synchronization point) then only the final
// value-weighted sum is split across threads. Mirrors attention_kernel in
// hy3_gpu.cu exactly.
// ---------------------------------------------------------------------
kernel void attention(device float       *out      [[buffer(0)]],
                       device const float *q        [[buffer(1)]],
                       device const float *k_cache  [[buffer(2)]],
                       device const float *v_cache  [[buffer(3)]],
                       constant uint      &n_heads    [[buffer(4)]],
                       constant uint      &n_kv_heads [[buffer(5)]],
                       constant uint      &head_dim   [[buffer(6)]],
                       constant int       &kv_len     [[buffer(7)]],
                       constant uint      &kv_group   [[buffer(8)]],
                       constant int       &layer_id   [[buffer(9)]],
                       constant int       &n_layers   [[buffer(10)]],
                       threadgroup float  *scores     [[threadgroup(0)]],
                       uint h    [[threadgroup_position_in_grid]],
                       uint tid  [[thread_index_in_threadgroup]],
                       uint tgSz [[threads_per_threadgroup]])
{
    if (h >= n_heads) return;
    uint kv_h = h / kv_group;
    device const float *q_h = q + (size_t)h * head_dim;
    float scale = rsqrt(float(head_dim));

    int ntok = (kv_len - layer_id + n_layers - 1) / n_layers;
    if (ntok < 1) ntok = 1;
    if (ntok > 8192) ntok = 8192;

    // Cooperatively compute the score vector into shared memory: each thread
    // handles a strided subset of the timesteps. `scores` is threadgroup
    // (shared) memory, so this MUST be split across threads with barriers --
    // having every thread redundantly write the whole array (as the CUDA
    // port's per-thread-local version did) races here and corrupts attention.
    threadgroup float shared_max[128];
    threadgroup float shared_sum[128];

    float local_max = -INFINITY;
    for (int t = (int)tid; t < ntok; t += (int)tgSz) {
        device const float *k_t = k_cache + (size_t)(t * n_layers + layer_id) * n_kv_heads * head_dim
                                           + (size_t)kv_h * head_dim;
        float s = 0.0f;
        for (uint d = 0; d < head_dim; d++) s += q_h[d] * k_t[d];
        s *= scale;
        scores[t] = s;
        if (s > local_max) local_max = s;
    }
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (tid < s) shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_score = shared_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (int t = (int)tid; t < ntok; t += (int)tgSz) {
        float e = exp(scores[t] - max_score);
        scores[t] = e;
        local_sum += e;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / (shared_sum[0] + 1e-10f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out_h = out + (size_t)h * head_dim;
    for (uint d = tid; d < head_dim; d += tgSz) {
        float val = 0.0f;
        for (int t = 0; t < ntok; t++) {
            device const float *v_t = v_cache + (size_t)(t * n_layers + layer_id) * n_kv_heads * head_dim
                                               + (size_t)kv_h * head_dim;
            val += scores[t] * inv * v_t[d];
        }
        out_h[d] = val;
    }
}

// ---------------------------------------------------------------------
// Dense matmul kernels: dst[row] = dot(W[row,:], x). One threadgroup per
// output row, parallel tree reduction across the input dimension.
// ---------------------------------------------------------------------

kernel void matmul_f32(device float       *dst [[buffer(0)]],
                        device const float *w   [[buffer(1)]],
                        device const float *x   [[buffer(2)]],
                        constant uint      &n   [[buffer(3)]],
                        uint row  [[threadgroup_position_in_grid]],
                        uint tid  [[thread_index_in_threadgroup]],
                        uint tgSz [[threads_per_threadgroup]])
{
    threadgroup float partial[256];
    device const float *wr = w + (size_t)row * n;
    float sum = 0.0f;
    for (uint j = tid; j < n; j += tgSz) sum += wr[j] * x[j];
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) dst[row] = partial[0];
}

kernel void matmul_f16(device float      *dst [[buffer(0)]],
                        device const half *w   [[buffer(1)]],
                        device const float *x  [[buffer(2)]],
                        constant uint     &n   [[buffer(3)]],
                        uint row  [[threadgroup_position_in_grid]],
                        uint tid  [[thread_index_in_threadgroup]],
                        uint tgSz [[threads_per_threadgroup]])
{
    threadgroup float partial[256];
    device const half *wr = w + (size_t)row * n;
    float sum = 0.0f;
    for (uint j = tid; j < n; j += tgSz) sum += float(wr[j]) * x[j];
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) dst[row] = partial[0];
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
                         uint tgSz [[threads_per_threadgroup]])
{
    threadgroup float partial[256];
    uint nb = n / 32;
    device const uchar *wr = w + (size_t)row * nb * 36;
    float sum = 0.0f;
    for (uint j = tid; j < nb; j += tgSz) {
        device const uchar *blk = wr + (size_t)j * 36;
        float d = *(device const float *)blk;
        device const char *qs = (device const char *)(blk + 4);
        float local = 0.0f;
        for (uint l = 0; l < 32; l++) local += float(qs[l]) * x[j * 32 + l];
        sum += d * local;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) dst[row] = partial[0];
}

// ---------------------------------------------------------------------
// Q4_K matmul -- mirrors dequantize_row_q4_K + the matmul loop in hy3.c,
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
        device const uchar *qq = q + (j / 2);
        for (uint l = 0; l < 32; l++) {
            float v1 = d1 * float(qq[l] & 0xF) - dm1;
            float v2 = d2 * float(qq[l] >> 4)  - dm2;
            acc += v1 * xb[j + l] + v2 * xb[j + 32 + l];
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
                         uint tgSz [[threads_per_threadgroup]])
{
    threadgroup float partial[256];
    uint nb = n / 256;
    device const uchar *wr = w + (size_t)row * nb * 144;
    float sum = 0.0f;
    for (uint bi = tid; bi < nb; bi += tgSz) {
        device const uchar *blk = wr + (size_t)bi * 144;
        sum += hy3_q4k_dot_block(blk, x + (size_t)bi * 256);
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) dst[row] = partial[0];
}
