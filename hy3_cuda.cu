#include "hy3_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#define CUDA_CHECK_VOID(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)stat); \
        return -1; \
    } \
} while(0)

#define BLOCK_DIM 256
#define QK_K 32

static cublasHandle_t cublas_handle = NULL;
static int cuda_initialized = 0;

int hy3_cuda_init(void) {
    if (cuda_initialized) return 0;
    
    CUDA_CHECK(cudaSetDevice(0));
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    cuda_initialized = 1;
    fprintf(stderr, "hy3_cuda: initialized on device 0\n");
    return 0;
}

void hy3_cuda_free(void) {
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = NULL;
    }
    cuda_initialized = 0;
}

void hy3_cuda_sync(void) {
    cudaDeviceSynchronize();
}

/* ========================================================================
 * RMS Norm Kernel
 * ======================================================================== */
__global__ void rms_norm_kernel(float *out, const float *x, const float *w, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = bid * n;
    x += offset;
    out += offset;

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += x[i] * x[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float r = rsqrtf(sdata[0] / (float)n + 1e-5f);
    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = x[i] * r * w[i];
    }
}

void hy3_cuda_rms_norm(float *out, const float *x, const float *w, int n) {
    int block = BLOCK_DIM;
    rms_norm_kernel<<<1, block, block * sizeof(float)>>>(out, x, w, n);
}

/* ========================================================================
 * SiLU + element-wise multiply kernel
 * ======================================================================== */
__global__ void silu_mul_kernel(float *out, const float *gate, const float *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = gate[i];
        float silu = x / (1.0f + expf(-x));
        out[i] = silu * up[i];
    }
}

void hy3_cuda_silu_mul(float *out, const float *gate, const float *up, int n) {
    int block = BLOCK_DIM;
    int grid = (n + block - 1) / block;
    silu_mul_kernel<<<grid, block>>>(out, gate, up, n);
}

/* ========================================================================
 * RoPE Kernel
 * ======================================================================== */
__global__ void rope_kernel(float *q, float *k, int pos, int head_dim, int n_heads, int n_kv_heads, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * head_dim + n_kv_heads * head_dim;
    if (idx >= total) return;

    int h, d, is_kv;
    if (idx < n_heads * head_dim) {
        h = idx / head_dim;
        d = idx % head_dim;
        is_kv = 0;
    } else {
        int off = idx - n_heads * head_dim;
        h = off / head_dim;
        d = off % head_dim;
        is_kv = 1;
    }

    if (d % 2 == 1) return;
    float *buf = is_kv ? k : q;
    float *base = buf + (size_t)h * head_dim;
    float freq = (float)pos / powf(theta, (float)d / (float)head_dim);
    float cos_val = cosf(freq);
    float sin_val = sinf(freq);
    float v0 = base[d];
    float v1 = base[d + 1];
    base[d]     = v0 * cos_val - v1 * sin_val;
    base[d + 1] = v0 * sin_val + v1 * cos_val;
}

void hy3_cuda_rope(float *q, float *k, int pos, int head_dim, int n_heads, int n_kv_heads) {
    int total = (n_heads + n_kv_heads) * head_dim;
    int block = BLOCK_DIM;
    int grid = (total + block - 1) / block;
    float theta = 11158840.0f;
    rope_kernel<<<grid, block>>>(q, k, pos, head_dim, n_heads, n_kv_heads, theta);
}

/* ========================================================================
 * Softmax Top-K Kernel
 * ======================================================================== */
__global__ void softmax_topk_kernel(float *vals, int *inds, const float *logits, int n, int k) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    if (tid < n) {
        sdata[tid * 2] = logits[tid];
        ((int*)(sdata + 192*2))[tid] = tid;
    } else if (tid < 256) {
        sdata[tid * 2] = -INFINITY;
        ((int*)(sdata + 192*2))[tid] = -1;
    }
    __syncthreads();

    int actual_k = k < n ? k : n;
    for (int r = 0; r < actual_k; r++) {
        __syncthreads();
        if (tid == 0) {
            int best = r;
            for (int i = r + 1; i < n; i++) {
                if (sdata[i * 2] > sdata[best * 2]) best = i;
            }
            float tv = sdata[r * 2]; sdata[r * 2] = sdata[best * 2]; sdata[best * 2] = tv;
            int ti = ((int*)(sdata + 192*2))[r];
            ((int*)(sdata + 192*2))[r] = ((int*)(sdata + 192*2))[best];
            ((int*)(sdata + 192*2))[best] = ti;
            if (sdata[r * 2] < 0) sdata[r * 2] = 0;
        }
    }
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < actual_k; i++) sum += sdata[i * 2];
    float inv = 1.0f / (sum + 1e-10f);
    if (tid < actual_k) {
        vals[tid] = sdata[tid * 2] * inv;
        inds[tid] = ((int*)(sdata + 192*2))[tid];
    }
}

void hy3_cuda_softmax_topk(float *vals, int *inds, const float *logits, int n, int k) {
    int block = 256;
    softmax_topk_kernel<<<1, block, (192 * 2 * sizeof(float) + 192 * sizeof(int))>>>(vals, inds, logits, n, k);
}

/* ========================================================================
 * Q8_0 Matrix-Vector Multiply Kernel
 * ======================================================================== */
typedef struct {
    float   d;
    int8_t  qs[32];
} block_q8_0;

__global__ void mul_mat_q8_0_kernel(float *dst, const block_q8_0 *w, const float *x, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    int nb = n / QK_K;
    float sum = 0.0f;
    const block_q8_0 *b = w + (size_t)i * nb;
    for (int j = 0; j < nb; j++) {
        float d = b[j].d;
        int acc = 0;
        for (int l = 0; l < QK_K; l++)
            acc += (int)b[j].qs[l] * (int)(x[(size_t)j * QK_K + l] * 8.0f);
        sum += d * (float)acc;
    }
    dst[i] = sum / 8.0f;
}

__global__ void mul_mat_f32_kernel(float *dst, const float *w, const float *x, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    float sum = 0.0f;
    for (int j = 0; j < n; j++)
        sum += w[(size_t)i * n + j] * x[j];
    dst[i] = sum;
}

void hy3_cuda_mul_mat_f32(float *dst, const uint8_t *w_data, uint32_t w_type,
                         const float *x, int m, int n) {
    int block = BLOCK_DIM;
    int grid = (m + block - 1) / block;
    
    switch (w_type) {
    case 0:
        mul_mat_f32_kernel<<<grid, block>>>(dst, (const float *)w_data, x, m, n);
        break;
    case 8:
        mul_mat_q8_0_kernel<<<grid, block>>>(dst, (const block_q8_0 *)w_data, x, m, n);
        break;
    default:
        fprintf(stderr, "hy3_cuda: unsupported type %u for mul_mat\n", w_type);
        cudaMemset(dst, 0, m * sizeof(float));
        break;
    }
}

/* ========================================================================
 * Attention Kernel (Flash Attention style)
 * ======================================================================== */
__global__ void attention_kernel(float *out, const float *q, const float *k_cache,
                                  const float *v_cache, int n_kv_heads, int n_heads,
                                  int head_dim, int kv_len, int kv_group) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int kv_h = h / kv_group;
    const float *q_h = q + (size_t)h * head_dim;

    extern __shared__ float sdata[];

    float max_score = -FLT_MAX;
    float sum = 0.0f;
    float scale = rsqrtf((float)head_dim);

    for (int t = 0; t < kv_len; t++) {
        const float *k_t = k_cache + (size_t)t * n_kv_heads * head_dim + (size_t)kv_h * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) s += q_h[d] * k_t[d];
        s *= scale;
        sdata[t] = s;
        if (s > max_score) max_score = s;
    }

    for (int t = 0; t < kv_len; t++) {
        sdata[t] = expf(sdata[t] - max_score);
        sum += sdata[t];
    }

    float inv_sum = 1.0f / (sum + 1e-10f);
    float *out_h = out + (size_t)h * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t < kv_len; t++) {
            float a = sdata[t] * inv_sum;
            const float *v_t = v_cache + (size_t)t * n_kv_heads * head_dim + (size_t)kv_h * head_dim;
            val += a * v_t[d];
        }
        out_h[d] = val;
    }
}

void hy3_cuda_attention(float *out, const float *q, float *k_cache, float *v_cache,
                       int n_kv_heads, int n_heads, int head_dim, int kv_len, int kv_group) {
    if (kv_len > 8192) kv_len = 8192;
    attention_kernel<<<n_heads, BLOCK_DIM, kv_len * sizeof(float)>>>(out, q, k_cache, v_cache, n_kv_heads, n_heads, head_dim, kv_len, kv_group);
}

/* ========================================================================
 * KV Cache Copy Kernel
 * ======================================================================== */
__global__ void kv_cache_copy_kernel(float *k_cache, float *v_cache, const float *k, const float *v, int kv_size, int pos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_size) return;
    k_cache[(size_t)pos * kv_size + i] = k[i];
    v_cache[(size_t)pos * kv_size + i] = v[i];
}

void hy3_cuda_copy_to_kv_cache(float *k_cache, float *v_cache, const float *k, const float *v, int kv_size, int pos) {
    int block = BLOCK_DIM;
    int grid = (kv_size + block - 1) / block;
    kv_cache_copy_kernel<<<grid, block>>>(k_cache, v_cache, k, v, kv_size, pos);
}

/* ========================================================================
 * Embedding Lookup Kernel
 * ======================================================================== */
__global__ void embed_kernel(float *out, const float *table, int token, int embd_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= embd_dim) return;
    out[i] = table[(size_t)token * embd_dim + i];
}

void hy3_cuda_embed(float *out, const float *table, int token, int embd_dim) {
    int block = BLOCK_DIM;
    int grid = (embd_dim + block - 1) / block;
    embed_kernel<<<grid, block>>>(out, table, token, embd_dim);
}
