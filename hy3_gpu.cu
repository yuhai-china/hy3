#ifdef __cplusplus
extern "C" {
#endif
#include "hy3.h"
#ifdef __cplusplus
}
#endif
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d (%s:%d)\n", (int)s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define BLOCK_DIM 256
#define QK_K 32
#define CUDA_QK_K 256

/* Q4_K block (144 bytes per 256 elements) -- same layout as ggml block_q4_K */
typedef struct {
    uint16_t d;          /* fp16 block scale */
    uint16_t dmin;       /* fp16 block min */
    uint8_t  scales[12]; /* 8×6-bit sub-block scales packed into 12 bytes */
    uint8_t  qs[CUDA_QK_K/2]; /* 128 bytes of 4-bit weights */
} cuda_block_q4_K;

/* Q8_K block (292 bytes per 256 elements) -- quantised activation */
typedef struct {
    float   d;                         /* block scale */
    int8_t  qs[CUDA_QK_K];             /* 256 int8 values */
    int16_t bsums[CUDA_QK_K/16];       /* 16 block-sums */
} cuda_block_q8_K;

/* ======================================================================
 * Device helper functions  (from ds4)
 * ====================================================================== */

__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ static void dev_q4_K_get_scale_min(
        uint32_t j, const uint8_t *scales, uint8_t *d_out, uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *x, const cuda_block_q8_K *y) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum = 0;
    int summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1)
        v += __shfl_down_sync(mask, v, offset, 8);
    (void)lane8;
    return v;
}

/* ======================================================================
 * Q8_K quantisation kernel  (from ds4)
 * ====================================================================== */
__global__ static void q8_K_quantize_kernel(cuda_block_q8_K *out, const float *x,
                                            uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out + (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int qv = (int)(iscale_s * xr[tid] + 0.5f);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

/* ======================================================================
 * MoE Q4_K×Q8_K matmul kernels  (adapted from ds4)
 * ====================================================================== */

/* Gate + Up projection for decode (single token, 1 slot per expert) */
__global__ static void moe_gate_up_q4K_qwarp32_kernel(
        float *gate_out, float *up_out,
        const uint8_t *gate_base, const uint8_t *up_base,
        const cuda_block_q8_K *xq,
        int32_t expert,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint32_t xq_blocks, uint32_t expert_mid_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    for (uint32_t rr = 0; rr < 16u; rr++) {
        uint32_t row = blockIdx.x * 512u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)
            (gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)
            (up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f, up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xq + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xq + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            gate_out[row] = gate;
            up_out[row] = up;
        }
    }
}

/* Down projection for decode — writes direct to out_row (no summing) */
__global__ static void moe_down_q4K_qwarp32_kernel(
        float *down_out,
        const uint8_t *down_base,
        const cuda_block_q8_K *midq,
        int32_t expert,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t midq_blocks, uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    for (uint32_t rr = 0; rr < 16u; rr++) {
        uint32_t row = blockIdx.x * 512u + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)
            (down_base + (uint64_t)(uint32_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u)
            acc += dev_dot_q4_K_q8_K_block(wr + b, midq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) down_out[row] = acc;
    }
}

/* MoE expert bias add (1D) */
__global__ void add_bias_kernel(float *x, const float *bias, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += bias[i];
}

/* ======================================================================
 * CUDA Kernels (existing)
 * ====================================================================== */
__global__ void rms_norm_kernel(float *out, const float *x, const float *w, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) sum += x[i] * x[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float r = rsqrtf(sdata[0] / (float)n + 1e-5f);
    for (int i = tid; i < n; i += blockDim.x) out[i] = x[i] * r * w[i];
}

__global__ void silu_mul_kernel(float *out, const float *gate, const float *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = gate[i];
        out[i] = (x / (1.0f + expf(-x))) * up[i];
    }
}

/* Must match the CPU rope() in hy3.c exactly: HYV3 uses the "rotate_half"
 * pairing (dim d with dim d+head_dim/2), not the interleaved (2i,2i+1)
 * pairing. This kernel is launched with (n_heads+n_kv_heads)*head_dim
 * threads (see gpu_rope below); only the first half of each head's dims
 * do work; the rest early-out. */
__global__ void rope_kernel(float *q, float *k, int pos, int head_dim, int n_heads, int n_kv_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total = (n_heads + n_kv_heads) * half;
    if (idx >= total) return;
    int h, d, is_kv;
    if (idx < n_heads * half) { h = idx / half; d = idx % half; is_kv = 0; }
    else { int off = idx - n_heads * half; h = off / half; d = off % half; is_kv = 1; }
    float *buf = is_kv ? k : q;
    float *base = buf + (size_t)h * head_dim;
    float freq = (float)pos / powf(11158840.0f, (float)(2 * d) / (float)head_dim);
    float c = cosf(freq), s = sinf(freq);
    float v0 = base[d], v1 = base[d + half];
    base[d]        = v0 * c - v1 * s;
    base[d + half] = v1 * c + v0 * s;
}
__global__ void attention_kernel(float *out, const float *q, const float *k_cache,
                                  const float *v_cache, int n_heads, int n_kv_heads,
                                  int head_dim, int kv_len, int kv_group, int layer_id,
                                  int n_layers) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int kv_h = h / kv_group;
    const float *q_h = q + (size_t)h * head_dim;
    extern __shared__ float scores[];
    float scale = rsqrtf((float)head_dim);
    int ntok = (kv_len - layer_id + n_layers - 1) / n_layers;
    if (ntok < 1) ntok = 1;
    if (ntok > 8192) ntok = 8192;
    float max_score = -FLT_MAX;
    for (int t = 0; t < ntok; t++) {
        const float *k_t = k_cache + (size_t)(t * n_layers + layer_id) * n_kv_heads * head_dim + (size_t)kv_h * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) s += q_h[d] * k_t[d];
        s *= scale;
        scores[t] = s;
        if (s > max_score) max_score = s;
    }
    float sum = 0.0f;
    for (int t = 0; t < ntok; t++) { scores[t] = expf(scores[t] - max_score); sum += scores[t]; }
    float inv = 1.0f / (sum + 1e-10f);
    float *out_h = out + (size_t)h * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t < ntok; t++) {
            const float *v_t = v_cache + (size_t)(t * n_layers + layer_id) * n_kv_heads * head_dim + (size_t)kv_h * head_dim;
            val += scores[t] * inv * v_t[d];
        }
        out_h[d] = val;
    }
}

__global__ void embed_lookup_kernel(float *out, const float *table, int token, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = table[(size_t)token * dim + i];
}

__global__ void softmax_topk_kernel(float *vals, int *inds, const float *logits, int n, int k) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    if (tid < n) { sdata[tid] = logits[tid]; ((int*)(sdata + 192))[tid] = tid; }
    __syncthreads();
    int ak = k < n ? k : n;
    for (int r = 0; r < ak; r++) {
        __syncthreads();
        if (tid == 0) {
            int best = r;
            for (int i = r + 1; i < n; i++) if (sdata[i] > sdata[best]) best = i;
            float tv = sdata[r]; sdata[r] = sdata[best]; sdata[best] = tv;
            int ti = ((int*)(sdata + 192))[r];
            ((int*)(sdata + 192))[r] = ((int*)(sdata + 192))[best];
            ((int*)(sdata + 192))[best] = ti;
        }
    }
    __syncthreads();
    float sum = 0.0f;
    for (int i = 0; i < ak; i++) sum += sdata[i];
    float inv = 1.0f / (sum + 1e-10f);
    if (tid < ak) { vals[tid] = sdata[tid] * inv; inds[tid] = ((int*)(sdata + 192))[tid]; }
}

__global__ void add_kernel(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void scale_add_kernel(float *out, const float *a, float s, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + s * b[i];
}

__global__ void memcpy_2d_kernel(float *dst, const float *src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void dequantize_bf16_kernel(float *dst, const uint16_t *src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t f32 = (uint32_t)src[i] << 16;
        memcpy(&dst[i], &f32, 4);
    }
}

__global__ void sigmoid_kernel(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + expf(-in[i]));
}

__global__ void fill_zero_kernel(float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = 0.0f;
}

/* ======================================================================
 * GPU Context
 * ====================================================================== */
/* Compressed Q4_K weight storage */
typedef struct {
    uint8_t *data;   /* raw Q4_K blocks on GPU */
    int bytes;       /* total bytes */
} q4k_buf_t;

typedef struct {
    cublasHandle_t cublas;

    /* Dense weight GPU pointers (F32) */
    float *d_token_embd;
    float *d_output_norm;
    float *d_output;

    float *d_layer_attn_q[81];
    float *d_layer_attn_k[81];
    float *d_layer_attn_v[81];
    float *d_layer_attn_output[81];
    float *d_layer_attn_q_norm[81];
    float *d_layer_attn_k_norm[81];
    float *d_layer_attn_norm[81];
    float *d_layer_ffn_norm[81];
    float *d_layer_ffn_gate_inp[81];
    float *d_layer_ffn_gate_shexp[81];
    float *d_layer_ffn_up_shexp[81];
    float *d_layer_ffn_down_shexp[81];
    float *d_layer_eh_proj[81];
    float *d_layer_enorm[81];
    float *d_layer_hnorm[81];
    float *d_layer_final_norm[81];
    float *d_layer_ffn_gate_exps_b[81];  /* 1D expert bias */

    /* Dense FFN (layer 0) */
    float *d_layer_dense_ffn_gate[81];
    float *d_layer_dense_ffn_up[81];
    float *d_layer_dense_ffn_down[81];

    /* Expert weights - compressed Q4_K for GPU MoE kernels */
    q4k_buf_t d_q4k_gate_exps[81][192];
    q4k_buf_t d_q4k_up_exps[81][192];
    q4k_buf_t d_q4k_down_exps[81][192];

    /* KV Cache */
    float *d_k_cache;
    float *d_v_cache;
    int ctx_cap;

    /* Scratch buffers */
    float *d_embed;
    float *d_scratch;
    float *d_scratch2;
    float *d_logits;

    /* Q8_K quantised activation buffer */
    cuda_block_q8_K *d_xq;    /* Q8_K quantised input for MoE */
    int d_xq_blocks;           /* number of blocks allocated */
} gpu_ctx_t;

/* ======================================================================
 * Weight Upload (compressed Q4_K for experts, F32 for everything else)
 * ====================================================================== */

static inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15);
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1f);
    uint32_t mant = (uint32_t)(h & 0x3ff);
    uint32_t f32;
    if (exp == 0) { f32 = (sign << 31) | ((0x7f - 15) << 23) | (mant << 13); }
    else if (exp == 31) { f32 = (sign << 31) | 0x7f800000 | (mant << 13); }
    else { f32 = (sign << 31) | ((exp + 0x70) << 23) | (mant << 13); }
    float r; memcpy(&r, &f32, 4); return r;
}

/* Upload F32 weight (type 0) */
static float *upload_f32(const uint8_t *data, uint64_t n) {
    float *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_buf, data, n * sizeof(float), cudaMemcpyHostToDevice));
    return d_buf;
}

/* Upload Q8_0 weight (type 8), dequant to F32 */
static float *upload_q8_0(const uint8_t *src, uint64_t n) {
    static const int QK8_0 = 32;
    float *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, n * sizeof(float)));
    float *h_buf = (float *)malloc(n * sizeof(float));
    uint64_t nb = n / QK8_0;
    for (uint64_t j = 0; j < nb; j++) {
        float d;
        memcpy(&d, src + j * 36, 4);
        const int8_t *qs = (const int8_t *)(src + j * 36 + 4);
        for (int k = 0; k < QK8_0; k++) h_buf[j * QK8_0 + k] = (float)qs[k] * d;
    }
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h_buf);
    return d_buf;
}

/* Upload F16 weight (type 1), dequant to F32 */
static float *upload_f16(const uint8_t *data, uint64_t n) {
    float *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, n * sizeof(float)));
    float *h_buf = (float *)malloc(n * sizeof(float));
    const uint16_t *src = (const uint16_t *)data;
    for (uint64_t i = 0; i < n; i++) h_buf[i] = fp16_to_float(src[i]);
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h_buf);
    return d_buf;
}

/* Upload Q4_K weight COMPRESSED (type 12) — keep as raw block on GPU */
static void upload_q4k_compressed(q4k_buf_t *out, const uint8_t *data, uint64_t n) {
    uint64_t nb = n / CUDA_QK_K;
    out->bytes = (int)(nb * sizeof(cuda_block_q4_K));
    CUDA_CHECK(cudaMalloc(&out->data, out->bytes));
    CUDA_CHECK(cudaMemcpy(out->data, data, out->bytes, cudaMemcpyHostToDevice));
}

/* Host-side Q4_K -> F32 dequant, mirroring dequantize_row_q4_K in hy3.c
 * exactly (must match quantize_row_q4_K in hy3_convert.c's bit layout). */
static void host_q4_k_get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

/* Upload Q4_K weight DEQUANTIZED to F32 (type 12) — used for "dense" (non
 * routed-expert) tensors such as attn_q/k/v/o and the dense/shared FFNs,
 * which go through cuBLAS on the GPU rather than the compressed Q4_K MoE
 * kernels. Previously there was no case for type 12 here, so these tensors
 * silently fell through to the zero-fill default -- attention and the
 * dense/shared FFN paths ran on all-zero weights for every GPU-resident
 * layer. */
static float *upload_q4k_dense(const uint8_t *src, uint64_t n) {
    static const int QK_K_LOCAL = 256;
    float *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, n * sizeof(float)));
    float *h_buf = (float *)malloc(n * sizeof(float));
    uint64_t nb = n / QK_K_LOCAL;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *blk = src + i * 144; /* sizeof(block_q4_K): 2+2+12+128 */
        uint16_t d16, dmin16;
        memcpy(&d16, blk, 2);
        memcpy(&dmin16, blk + 2, 2);
        const uint8_t *sc = blk + 4;
        const uint8_t *q = blk + 16;
        float d = fp16_to_float(d16);
        float dmin = fp16_to_float(dmin16);
        float *y = h_buf + i * QK_K_LOCAL;
        int is = 0;
        for (int j = 0; j < QK_K_LOCAL; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            host_q4_k_get_scale_min(is + 0, sc, &sc1, &m1);
            host_q4_k_get_scale_min(is + 1, sc, &sc2, &m2);
            float d1 = d * (float)sc1, dm1 = dmin * (float)m1;
            float d2 = d * (float)sc2, dm2 = dmin * (float)m2;
            for (int l = 0; l < 32; l++) y[l]      = d1 * (float)(q[l] & 0xF) - dm1;
            for (int l = 0; l < 32; l++) y[32 + l] = d2 * (float)(q[l] >> 4)  - dm2;
            y += 64; q += 32; is += 2;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h_buf);
    return d_buf;
}

/* Generic upload that picks the right path */
static float *upload_weight_dense(const hy3_weight *w, int layer, int expert) {
    if (!w || !w->data || !w->t) return NULL;
    const hy3_tensor_info *t = w->t;
    uint64_t n = t->elements;
    if (n == 0) return NULL;

    switch (t->ggml_type) {
    case 0:  return upload_f32(w->data, n);
    case 1:  return upload_f16(w->data, n);
    case 8:  return upload_q8_0(w->data, n);
    case 12: return upload_q4k_dense(w->data, n);
    default: {
        fprintf(stderr, "hy3_gpu: warning: unhandled ggml_type %u for dense weight, zero-filling\n",
                t->ggml_type);
        float *d_buf;
        CUDA_CHECK(cudaMalloc(&d_buf, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_buf, 0, n * sizeof(float)));
        return d_buf;
    }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

int hy3_gpu_init(hy3_model *m, int n_gpu_layers) {
    gpu_ctx_t *ctx = (gpu_ctx_t *)calloc(1, sizeof(gpu_ctx_t));
    if (!ctx) return -1;

    CUBLAS_CHECK(cublasCreate(&ctx->cublas));

    /* Upload global weights (always on GPU, F32) */
    ctx->d_token_embd = upload_weight_dense(&m->w.token_embd, -1, -1);
    ctx->d_output_norm = upload_weight_dense(&m->w.output_norm, -1, -1);
    ctx->d_output = upload_weight_dense(&m->w.output, -1, -1);

    if (n_gpu_layers <= 0) n_gpu_layers = 81;
    if (n_gpu_layers > 81) n_gpu_layers = 81;
    m->gpu_layers = n_gpu_layers;

    /* Allocate Q8_K quantisation buffer: enough for 1 row × max(embd, inter) / 256 blocks */
    int max_dim = HY3_N_EMBD > HY3_MOE_INTERMED ? HY3_N_EMBD : HY3_MOE_INTERMED;
    ctx->d_xq_blocks = max_dim / CUDA_QK_K;
    CUDA_CHECK(cudaMalloc(&ctx->d_xq, (size_t)ctx->d_xq_blocks * sizeof(cuda_block_q8_K)));

    for (int il = 0; il < n_gpu_layers; il++) {
        hy3_layer_weights *l = &m->w.layers[il];
        ctx->d_layer_attn_norm[il] = upload_weight_dense(&l->attn_norm, il, -1);
        ctx->d_layer_attn_q[il] = upload_weight_dense(&l->attn_q, il, -1);
        ctx->d_layer_attn_k[il] = upload_weight_dense(&l->attn_k, il, -1);
        ctx->d_layer_attn_v[il] = upload_weight_dense(&l->attn_v, il, -1);
        ctx->d_layer_attn_output[il] = upload_weight_dense(&l->attn_output, il, -1);
        ctx->d_layer_attn_q_norm[il] = upload_weight_dense(&l->attn_q_norm, il, -1);
        ctx->d_layer_attn_k_norm[il] = upload_weight_dense(&l->attn_k_norm, il, -1);
        ctx->d_layer_ffn_norm[il] = upload_weight_dense(&l->ffn_norm, il, -1);
        ctx->d_layer_eh_proj[il] = upload_weight_dense(&l->eh_proj, il, -1);
        ctx->d_layer_enorm[il] = upload_weight_dense(&l->enorm, il, -1);
        ctx->d_layer_hnorm[il] = upload_weight_dense(&l->hnorm, il, -1);
        ctx->d_layer_final_norm[il] = upload_weight_dense(&l->final_norm, il, -1);

        if (il < HY3_N_LAYER_DENSE) {
            ctx->d_layer_dense_ffn_gate[il] = upload_weight_dense(&l->ffn_gate, il, -1);
            ctx->d_layer_dense_ffn_up[il] = upload_weight_dense(&l->ffn_up, il, -1);
            ctx->d_layer_dense_ffn_down[il] = upload_weight_dense(&l->ffn_down, il, -1);
        } else {
            ctx->d_layer_ffn_gate_inp[il] = upload_weight_dense(&l->ffn_gate_inp, il, -1);
            ctx->d_layer_ffn_gate_shexp[il] = upload_weight_dense(&l->ffn_gate_shexp, il, -1);
            ctx->d_layer_ffn_up_shexp[il] = upload_weight_dense(&l->ffn_up_shexp, il, -1);
            ctx->d_layer_ffn_down_shexp[il] = upload_weight_dense(&l->ffn_down_shexp, il, -1);

            if (l->has_expert_bias) {
                CUDA_CHECK(cudaMalloc(&ctx->d_layer_ffn_gate_exps_b[il], HY3_N_EXPERT * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(ctx->d_layer_ffn_gate_exps_b[il], l->expert_bias,
                                      HY3_N_EXPERT * sizeof(float), cudaMemcpyHostToDevice));
            }

            for (int e = 0; e < HY3_N_EXPERT; e++) {
                upload_q4k_compressed(&ctx->d_q4k_gate_exps[il][e], l->ffn_gate_exps[e].data, l->ffn_gate_exps[e].t->elements);
                upload_q4k_compressed(&ctx->d_q4k_up_exps[il][e], l->ffn_up_exps[e].data, l->ffn_up_exps[e].t->elements);
                upload_q4k_compressed(&ctx->d_q4k_down_exps[il][e], l->ffn_down_exps[e].data, l->ffn_down_exps[e].t->elements);
            }
        }
    }

    /* Allocate GPU KV cache. The cache is interleaved by layer (slot =
     * token_idx * HY3_N_LAYER + layer_id, matching the CPU cache and
     * attention()/attention_kernel()), so holding N tokens of history
     * requires N * HY3_N_LAYER slots, not N slots. ctx_cap tracks capacity
     * in slots and grows on demand in hy3_eval_gpu() below; this initial
     * allocation covers max_context_tokens tokens of context. */
    int max_context_tokens = 8192;
    int kv_size = HY3_N_KV_HEAD * HY3_HEAD_DIM;
    size_t init_slots = (size_t)max_context_tokens * HY3_N_LAYER;
    CUDA_CHECK(cudaMalloc(&ctx->d_k_cache, init_slots * kv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_v_cache, init_slots * kv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_embed, HY3_N_EMBD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_scratch, (HY3_DENSE_INTERMED * 2 + HY3_N_EMBD * 4 + HY3_HEAD_DIM * 256) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_scratch2, (HY3_N_EXPERT * 4 + HY3_MOE_INTERMED * 8) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_logits, HY3_N_VOCAB * sizeof(float)));
    ctx->ctx_cap = (int)init_slots;

    size_t total = 0;
    for (int il = 0; il < n_gpu_layers; il++)
        for (int e = 0; e < HY3_N_EXPERT; e++)
            total += ctx->d_q4k_gate_exps[il][e].bytes +
                     ctx->d_q4k_up_exps[il][e].bytes +
                     ctx->d_q4k_down_exps[il][e].bytes;
    fprintf(stderr, "hy3_gpu: %d layers, expert Q4_K compressed: %.1f MB total\n",
            n_gpu_layers, total / 1e6);

    m->gpu_ctx = ctx;
    fprintf(stderr, "hy3_gpu: initialized\n");
    return 0;
}

void hy3_gpu_free(hy3_model *m) {
    gpu_ctx_t *ctx = (gpu_ctx_t *)m->gpu_ctx;
    if (!ctx) return;

    #define GPU_FREE(p) do { if (p) { cudaFree(p); p = NULL; } } while(0)

    GPU_FREE(ctx->d_token_embd);
    GPU_FREE(ctx->d_output_norm);
    GPU_FREE(ctx->d_output);
    GPU_FREE(ctx->d_xq);

    for (int il = 0; il < 81; il++) {
        GPU_FREE(ctx->d_layer_attn_norm[il]);
        GPU_FREE(ctx->d_layer_attn_q[il]);
        GPU_FREE(ctx->d_layer_attn_k[il]);
        GPU_FREE(ctx->d_layer_attn_v[il]);
        GPU_FREE(ctx->d_layer_attn_output[il]);
        GPU_FREE(ctx->d_layer_attn_q_norm[il]);
        GPU_FREE(ctx->d_layer_attn_k_norm[il]);
        GPU_FREE(ctx->d_layer_ffn_norm[il]);
        GPU_FREE(ctx->d_layer_ffn_gate_inp[il]);
        GPU_FREE(ctx->d_layer_ffn_gate_shexp[il]);
        GPU_FREE(ctx->d_layer_ffn_up_shexp[il]);
        GPU_FREE(ctx->d_layer_ffn_down_shexp[il]);
        GPU_FREE(ctx->d_layer_dense_ffn_gate[il]);
        GPU_FREE(ctx->d_layer_dense_ffn_up[il]);
        GPU_FREE(ctx->d_layer_dense_ffn_down[il]);
        GPU_FREE(ctx->d_layer_eh_proj[il]);
        GPU_FREE(ctx->d_layer_enorm[il]);
        GPU_FREE(ctx->d_layer_hnorm[il]);
        GPU_FREE(ctx->d_layer_final_norm[il]);
        GPU_FREE(ctx->d_layer_ffn_gate_exps_b[il]);

        for (int e = 0; e < HY3_N_EXPERT; e++) {
            GPU_FREE(ctx->d_q4k_gate_exps[il][e].data);
            GPU_FREE(ctx->d_q4k_up_exps[il][e].data);
            GPU_FREE(ctx->d_q4k_down_exps[il][e].data);
        }
    }

    GPU_FREE(ctx->d_k_cache);
    GPU_FREE(ctx->d_v_cache);
    GPU_FREE(ctx->d_embed);
    GPU_FREE(ctx->d_scratch);
    GPU_FREE(ctx->d_scratch2);
    GPU_FREE(ctx->d_logits);

    if (ctx->cublas) cublasDestroy(ctx->cublas);
    free(ctx);
    m->gpu_ctx = NULL;
}

/* ======================================================================
 * GPU Forward Pass
 * ====================================================================== */

/* cuBLAS matmul for dense (F32) weights */
static void gpu_mul_mat(gpu_ctx_t *ctx, const float *x, float *dst, const float *d_w, int m, int n) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                m, 1, n, &alpha, d_w, n, x, n, &beta, dst, m);
}

static void gpu_rms_norm(gpu_ctx_t *ctx, float *out, const float *x, const float *w, int n) {
    rms_norm_kernel<<<1, BLOCK_DIM, BLOCK_DIM * sizeof(float)>>>(out, x, w, n);
}

static void gpu_silu_mul(float *out, const float *gate, const float *up, int n) {
    int grid = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    silu_mul_kernel<<<grid, BLOCK_DIM>>>(out, gate, up, n);
}

static void gpu_rope(float *q, float *k, int pos, int head_dim, int n_heads, int n_kv_heads) {
    int total = (n_heads + n_kv_heads) * head_dim;
    int grid = (total + BLOCK_DIM - 1) / BLOCK_DIM;
    rope_kernel<<<grid, BLOCK_DIM>>>(q, k, pos, head_dim, n_heads, n_kv_heads);
}

static void gpu_attention(float *out, const float *q, const float *k_cache,
                           const float *v_cache, int n_heads, int n_kv_heads,
                           int head_dim, int kv_len, int layer_id) {
    int kv_group = n_heads / n_kv_heads;
    int ntok = (kv_len - layer_id + HY3_N_LAYER - 1) / HY3_N_LAYER;
    if (ntok < 1) ntok = 1;
    if (ntok > 8192) ntok = 8192;
    attention_kernel<<<n_heads, BLOCK_DIM, ntok * sizeof(float)>>>(
        out, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, kv_len, kv_group,
        layer_id, HY3_N_LAYER);
}

static void gpu_softmax_topk(float *vals, int *inds, const float *logits, int n, int k) {
    softmax_topk_kernel<<<1, 256, (192 * sizeof(float) + 192 * sizeof(int))>>>(
        vals, inds, logits, n, k);
}

/* Grow the GPU KV cache (in slots) if the interleaved-by-layer cache needs
 * more room than currently allocated. Without this, a fixed 8192-slot
 * buffer only covers 8192/HY3_N_LAYER ~= 102 tokens of context before
 * hy3_eval_gpu's cudaMemcpy into d_k_cache/d_v_cache goes out of bounds
 * (observed as "CUDA error: invalid argument" on any prompt longer than
 * that). */
static void gpu_ensure_kv_capacity(gpu_ctx_t *ctx, int needed_slots, int kv_dim) {
    if (needed_slots <= ctx->ctx_cap) return;
    size_t new_cap = (size_t)needed_slots + (size_t)8192 * HY3_N_LAYER; /* headroom: ~8192 more tokens */
    float *new_k, *new_v;
    CUDA_CHECK(cudaMalloc(&new_k, new_cap * (size_t)kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&new_v, new_cap * (size_t)kv_dim * sizeof(float)));
    cudaFree(ctx->d_k_cache);
    cudaFree(ctx->d_v_cache);
    ctx->d_k_cache = new_k;
    ctx->d_v_cache = new_v;
    ctx->ctx_cap = (int)new_cap;
    fprintf(stderr, "hy3_gpu: grew KV cache to %d slots (%.1f MB each)\n",
            ctx->ctx_cap, ctx->ctx_cap * (double)kv_dim * sizeof(float) / 1e6);
}

int hy3_eval_gpu(hy3_model *m, const hy3_tokens *tokens, float *logits, int *pos) {
    gpu_ctx_t *ctx = (gpu_ctx_t *)m->gpu_ctx;
    if (!ctx) return -1;

    int kv_size = HY3_N_KV_HEAD * HY3_HEAD_DIM;
    int n_gpu = m->gpu_layers;
    if (n_gpu <= 0) n_gpu = HY3_N_LAYER;
    if (n_gpu > HY3_N_LAYER) n_gpu = HY3_N_LAYER;

    for (int i = 0; i < tokens->len; i++) {
        int token = tokens->v[i];
        int cache_base = m->cache_len;

        size_t needed = (size_t)(cache_base + HY3_N_LAYER) * kv_size;
        if (needed > (size_t)m->ctx_size * kv_size) {
            size_t new_cap = cache_base + HY3_N_LAYER + 1024;
            m->cache_k = (float *)realloc(m->cache_k, new_cap * kv_size * sizeof(float));
            m->cache_v = (float *)realloc(m->cache_v, new_cap * kv_size * sizeof(float));
            m->ctx_size = (int)new_cap;
        }
        if (!m->cache_k) {
            m->ctx_size = 4096;
            m->cache_k = (float *)calloc((size_t)m->ctx_size * kv_size, sizeof(float));
            m->cache_v = (float *)calloc((size_t)m->ctx_size * kv_size, sizeof(float));
        }

        /* Embedding lookup on GPU */
        int grid = (HY3_N_EMBD + BLOCK_DIM - 1) / BLOCK_DIM;
        embed_lookup_kernel<<<grid, BLOCK_DIM>>>(ctx->d_embed, ctx->d_token_embd, token, HY3_N_EMBD);

        /* GPU layers (0..n_gpu-1) */
        for (int il = 0; il < n_gpu; il++) {
            if (!ctx->d_layer_attn_q[il]) break;

            int kv_len = cache_base + il;
            int kv_dim = HY3_N_KV_HEAD * HY3_HEAD_DIM;
            float *x = ctx->d_embed;
            float *s = ctx->d_scratch;
            float *s2 = ctx->d_scratch + HY3_N_EMBD * 2;
            float *attn_out = ctx->d_scratch + HY3_N_EMBD * 4;

            /* RMS norm before attention */
            gpu_rms_norm(ctx, s, x, ctx->d_layer_attn_norm[il], HY3_N_EMBD);

            /* QKV projections */
            int q_size = HY3_N_HEAD * HY3_HEAD_DIM;
            gpu_mul_mat(ctx, s, s2, ctx->d_layer_attn_q[il], q_size, HY3_N_EMBD);
            float *q_gpu = s2;
            float *k_gpu = s2 + q_size;
            float *v_gpu = s2 + q_size + kv_dim;
            gpu_mul_mat(ctx, s, k_gpu, ctx->d_layer_attn_k[il], kv_dim, HY3_N_EMBD);
            gpu_mul_mat(ctx, s, v_gpu, ctx->d_layer_attn_v[il], kv_dim, HY3_N_EMBD);

            /* QK norm (per-head) */
            if (ctx->d_layer_attn_q_norm[il])
                for (int h = 0; h < HY3_N_HEAD; h++)
                    gpu_rms_norm(ctx, q_gpu + h * HY3_HEAD_DIM, q_gpu + h * HY3_HEAD_DIM,
                                 ctx->d_layer_attn_q_norm[il], HY3_HEAD_DIM);
            if (ctx->d_layer_attn_k_norm[il])
                for (int h = 0; h < HY3_N_KV_HEAD; h++)
                    gpu_rms_norm(ctx, k_gpu + h * HY3_HEAD_DIM, k_gpu + h * HY3_HEAD_DIM,
                                 ctx->d_layer_attn_k_norm[il], HY3_HEAD_DIM);

            /* RoPE */
            int token_pos = cache_base / HY3_N_LAYER;
            gpu_rope(q_gpu, k_gpu, token_pos, HY3_HEAD_DIM, HY3_N_HEAD, HY3_N_KV_HEAD);

            /* Write K,V to CPU cache */
            {
                float *h_k = m->cache_k + (size_t)kv_len * kv_dim;
                float *h_v = m->cache_v + (size_t)kv_len * kv_dim;
                CUDA_CHECK(cudaMemcpy(h_k, k_gpu, kv_dim * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_v, v_gpu, kv_dim * sizeof(float), cudaMemcpyDeviceToHost));
            }

            /* Copy full CPU cache → GPU for attention */
            gpu_ensure_kv_capacity(ctx, kv_len + 1, kv_dim);
            CUDA_CHECK(cudaMemcpy(ctx->d_k_cache, m->cache_k,
                                  (size_t)(kv_len + 1) * kv_dim * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ctx->d_v_cache, m->cache_v,
                                  (size_t)(kv_len + 1) * kv_dim * sizeof(float), cudaMemcpyHostToDevice));

            /* Attention */
            gpu_attention(attn_out, q_gpu, ctx->d_k_cache, ctx->d_v_cache,
                         HY3_N_HEAD, HY3_N_KV_HEAD, HY3_HEAD_DIM, kv_len + 1, il);

            /* Output projection */
            gpu_mul_mat(ctx, attn_out, s, ctx->d_layer_attn_output[il], HY3_N_EMBD, q_size);
            add_kernel<<<(HY3_N_EMBD + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(x, x, s, HY3_N_EMBD);

            /* FFN norm */
            gpu_rms_norm(ctx, s, x, ctx->d_layer_ffn_norm[il], HY3_N_EMBD);

            if (il < HY3_N_LAYER_DENSE) {
                /* Dense FFN (layer 0) */
                float *gate = s2;
                float *up = s2 + HY3_DENSE_INTERMED;
                gpu_mul_mat(ctx, s, gate, ctx->d_layer_dense_ffn_gate[il], HY3_DENSE_INTERMED, HY3_N_EMBD);
                gpu_mul_mat(ctx, s, up, ctx->d_layer_dense_ffn_up[il], HY3_DENSE_INTERMED, HY3_N_EMBD);
                gpu_silu_mul(s2, gate, up, HY3_DENSE_INTERMED);
                gpu_mul_mat(ctx, s2, s, ctx->d_layer_dense_ffn_down[il], HY3_N_EMBD, HY3_DENSE_INTERMED);
                add_kernel<<<(HY3_N_EMBD + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(x, x, s, HY3_N_EMBD);
            } else {
                /* MoE FFN — GPU-accelerated Q4_K path */
                float *d_rlogits = (float *)ctx->d_scratch2;
                gpu_mul_mat(ctx, s, d_rlogits, ctx->d_layer_ffn_gate_inp[il], HY3_N_EXPERT, HY3_N_EMBD);

                /* Sigmoid on GPU */
                sigmoid_kernel<<<(HY3_N_EXPERT + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(
                    d_rlogits, d_rlogits, HY3_N_EXPERT);

                /* Bring sigmoid values + bias to CPU for routing */
                float h_sigmoid[HY3_N_EXPERT], h_scores[HY3_N_EXPERT];
                CUDA_CHECK(cudaMemcpy(h_sigmoid, d_rlogits, HY3_N_EXPERT * sizeof(float), cudaMemcpyDeviceToHost));
                memcpy(h_scores, h_sigmoid, sizeof(h_scores));
                if (ctx->d_layer_ffn_gate_exps_b[il]) {
                    float h_bias[HY3_N_EXPERT];
                    CUDA_CHECK(cudaMemcpy(h_bias, ctx->d_layer_ffn_gate_exps_b[il], HY3_N_EXPERT * sizeof(float), cudaMemcpyDeviceToHost));
                    for (int i = 0; i < HY3_N_EXPERT; i++) h_scores[i] += h_bias[i];
                }

                /* Topk on biased scores */
                float topk_vals[HY3_N_EXPERT_USED];
                int topk_inds[HY3_N_EXPERT_USED];
                {
                    float tv[HY3_N_EXPERT]; int ti[HY3_N_EXPERT];
                    for (int i = 0; i < HY3_N_EXPERT; i++) { ti[i] = i; tv[i] = h_scores[i]; }
                    for (int i = 0; i < HY3_N_EXPERT_USED; i++) {
                        int best = i;
                        for (int j = i+1; j < HY3_N_EXPERT; j++)
                            if (tv[j] > tv[best]) best = j;
                        float f = tv[i]; tv[i] = tv[best]; tv[best] = f;
                        int x = ti[i]; ti[i] = ti[best]; ti[best] = x;
                    }
                    for (int i = 0; i < HY3_N_EXPERT_USED; i++) { topk_vals[i] = 0; topk_inds[i] = ti[i]; }
                }

                /* Gather sigmoid-only weights, normalize, scale */
                float sum_w = 0;
                for (int e = 0; e < HY3_N_EXPERT_USED; e++) {
                    topk_vals[e] = h_sigmoid[topk_inds[e]];
                    sum_w += topk_vals[e];
                }
                float inv_sum = 1.0f / (sum_w + 1e-20f);
                for (int e = 0; e < HY3_N_EXPERT_USED; e++)
                    topk_vals[e] *= inv_sum * 2.826f;

                /* Shared expert (dense — use cuBLAS) */
                float *shared_gate = (float *)ctx->d_scratch2 + HY3_N_EXPERT * 2;
                float *shared_up = shared_gate + HY3_MOE_INTERMED;
                gpu_mul_mat(ctx, s, shared_gate, ctx->d_layer_ffn_gate_shexp[il], HY3_MOE_INTERMED, HY3_N_EMBD);
                gpu_mul_mat(ctx, s, shared_up, ctx->d_layer_ffn_up_shexp[il], HY3_MOE_INTERMED, HY3_N_EMBD);
                gpu_silu_mul(s2, shared_gate, shared_up, HY3_MOE_INTERMED);

                /* Routed experts — Q4_K GPU kernels */
                int expert_mid_dim = HY3_MOE_INTERMED;
                int expert_out_dim = HY3_N_EMBD;
                int in_dim = HY3_N_EMBD;
                int xq_blocks = in_dim / CUDA_QK_K;

                /* Compute exact bytes per expert (each expert weight is same size) */
                const hy3_tensor_info *t0 = m->w.layers[il].ffn_gate_exps[0].t;
                uint64_t expert_elems = t0 ? t0->elements : 0;
                int expert_bytes_gate = (int)((expert_elems / CUDA_QK_K) * sizeof(cuda_block_q4_K));
                int row_bytes_gate = (int)((in_dim / CUDA_QK_K) * sizeof(cuda_block_q4_K));

                const hy3_tensor_info *td = m->w.layers[il].ffn_down_exps[0].t;
                uint64_t down_elems = td ? td->elements : 0;
                int expert_bytes_down = (int)((down_elems / CUDA_QK_K) * sizeof(cuda_block_q4_K));
                int row_bytes_down = (int)((expert_mid_dim / CUDA_QK_K) * sizeof(cuda_block_q4_K));

                /* Output buffer for all experts, zero it */
                float *expert_out = (float *)ctx->d_scratch2 + HY3_N_EXPERT * 4;
                int out_fill_grid = (HY3_N_EMBD + BLOCK_DIM - 1) / BLOCK_DIM;
                fill_zero_kernel<<<out_fill_grid, BLOCK_DIM>>>(expert_out, HY3_N_EMBD);

                for (int e = 0; e < HY3_N_EXPERT_USED; e++) {
                    int ei = topk_inds[e];
                    float w = topk_vals[e];

                    float *gate_buf = (float *)ctx->d_scratch2 + HY3_N_EXPERT * 4 + HY3_N_EMBD;
                    float *up_buf = gate_buf + HY3_MOE_INTERMED;
                    float *mid_buf = up_buf + HY3_MOE_INTERMED;

                    /* Q8_K quantise input activation */
                    q8_K_quantize_kernel<<<dim3(xq_blocks, 1), CUDA_QK_K>>>(
                        ctx->d_xq, s, in_dim, 1);

                    /* Gate + Up: Q4_K-weighted projection */
                    int gate_grid = (expert_mid_dim + 511) / 512;
                    moe_gate_up_q4K_qwarp32_kernel<<<dim3(gate_grid, 1), 256>>>(
                        gate_buf, up_buf,
                        ctx->d_q4k_gate_exps[il][ei].data,
                        ctx->d_q4k_up_exps[il][ei].data,
                        ctx->d_xq, 0 /* expert_i always 0 since per-buffer */,
                        expert_bytes_gate, row_bytes_gate, xq_blocks, expert_mid_dim);
                    cudaDeviceSynchronize();


                    /* SiLU activation */
                    gpu_silu_mul(mid_buf, gate_buf, up_buf, expert_mid_dim);

                    /* Down: Q4_K-weighted projection */
                    /* Quantise mid activations */
                    int midq_blocks = expert_mid_dim / CUDA_QK_K;
                    int down_grid = (expert_out_dim + 511) / 512;
                    q8_K_quantize_kernel<<<dim3(midq_blocks, 1), CUDA_QK_K>>>(
                        ctx->d_xq, mid_buf, expert_mid_dim, 1);
                    moe_down_q4K_qwarp32_kernel<<<dim3(down_grid, 1), 256>>>(
                        gate_buf /* reuse as tmp */,
                        ctx->d_q4k_down_exps[il][ei].data,
                        ctx->d_xq, 0,
                        expert_bytes_down, row_bytes_down, midq_blocks, expert_out_dim);

                    /* Scale-add into accumulator */
                    scale_add_kernel<<<(HY3_N_EMBD + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(
                        expert_out, expert_out, w, gate_buf, HY3_N_EMBD);
                }

                /* Shared expert down */
                gpu_mul_mat(ctx, s2, s, ctx->d_layer_ffn_down_shexp[il], HY3_N_EMBD, HY3_MOE_INTERMED);

                /* Final combine: x += s(shared_down) + expert_out */
                add_kernel<<<(HY3_N_EMBD + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(s, s, expert_out, HY3_N_EMBD);
                add_kernel<<<(HY3_N_EMBD + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(x, x, s, HY3_N_EMBD);
            }
        }

        m->cache_len = cache_base + n_gpu;

        /* Sync embed GPU→CPU for CPU layers */
        CUDA_CHECK(cudaMemcpy(m->embed, ctx->d_embed, HY3_N_EMBD * sizeof(float), cudaMemcpyDeviceToHost));

        /* CPU layers (n_gpu..HY3_N_LAYER-1) */
        for (int il = n_gpu; il < HY3_N_LAYER; il++) {
            int pos = cache_base / HY3_N_LAYER;
            if (il < HY3_N_LAYER_DENSE)
                forward_layer_dense(m, il, pos);
            else
                forward_layer_moe(m, il, pos);
        }

        CUDA_CHECK(cudaMemcpy(ctx->d_embed, m->embed, HY3_N_EMBD * sizeof(float), cudaMemcpyHostToDevice));
        m->cache_len = cache_base + HY3_N_LAYER;
    }

    if (ctx->d_output_norm)
        gpu_rms_norm(ctx, ctx->d_embed, ctx->d_embed, ctx->d_output_norm, HY3_N_EMBD);

    gpu_mul_mat(ctx, ctx->d_embed, ctx->d_logits, ctx->d_output, HY3_N_VOCAB, HY3_N_EMBD);
    CUDA_CHECK(cudaMemcpy(logits, ctx->d_logits, HY3_N_VOCAB * sizeof(float), cudaMemcpyDeviceToHost));

    *pos = m->cache_len;
    fprintf(stderr,"  GPU logit[0..9]:");
    for(int i=0;i<10;i++) fprintf(stderr," %.2f",logits[i]);
    fprintf(stderr,"\n");
    return 0;
}

#ifdef __cplusplus
}
#endif
