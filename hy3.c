#include "hy3.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pthread.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define HY3_GGUF_MAGIC 0x46554747u
#define HY3_MAX_DIMS 8

#define HY3_NEG_INF (-1.0e30f)
#define HY3_RMS_EPS 1e-5f
#define HY3_ROPE_THETA 11158840.0f

static void die(const char *msg) {
    fprintf(stderr, "hy3: %s\n", msg);
    exit(1);
}

static void die_errno(const char *what, const char *path) {
    fprintf(stderr, "hy3: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void *xmalloc(size_t sz) {
    void *p = malloc(sz ? sz : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t sz) {
    p = realloc(p, sz);
    if (!p) die("out of memory");
    return p;
}

/* =========================================================================
 * GGUF Format
 * ========================================================================= */

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

/* hy3_kv and hy3_gguf_model are defined in hy3.h */

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} hy3_cursor;

static hy3_cursor cursor_at(const hy3_gguf_model *m, uint64_t pos) {
    hy3_cursor c;
    c.base = m->map;
    c.size = m->size;
    c.pos = pos;
    c.error[0] = 0;
    return c;
}

static bool cursor_u8(hy3_cursor *c, uint8_t *v) {
    if (c->pos + 1 > c->size) { snprintf(c->error, sizeof(c->error), "EOF reading u8"); return false; }
    *v = c->base[c->pos];
    c->pos += 1;
    return true;
}

static bool cursor_u16(hy3_cursor *c, uint16_t *v) {
    if (c->pos + 2 > c->size) { snprintf(c->error, sizeof(c->error), "EOF reading u16"); return false; }
    *v = (uint16_t)c->base[c->pos] | ((uint16_t)c->base[c->pos+1] << 8);
    c->pos += 2;
    return true;
}

static bool cursor_u32(hy3_cursor *c, uint32_t *v) {
    if (c->pos + 4 > c->size) { snprintf(c->error, sizeof(c->error), "EOF reading u32"); return false; }
    *v = (uint32_t)c->base[c->pos] | ((uint32_t)c->base[c->pos+1] << 8) |
         ((uint32_t)c->base[c->pos+2] << 16) | ((uint32_t)c->base[c->pos+3] << 24);
    c->pos += 4;
    return true;
}

static bool cursor_u64(hy3_cursor *c, uint64_t *v) {
    if (c->pos + 8 > c->size) { snprintf(c->error, sizeof(c->error), "EOF reading u64"); return false; }
    *v = (uint64_t)c->base[c->pos] | ((uint64_t)c->base[c->pos+1] << 8) |
         ((uint64_t)c->base[c->pos+2] << 16) | ((uint64_t)c->base[c->pos+3] << 24) |
         ((uint64_t)c->base[c->pos+4] << 32) | ((uint64_t)c->base[c->pos+5] << 40) |
         ((uint64_t)c->base[c->pos+6] << 48) | ((uint64_t)c->base[c->pos+7] << 56);
    c->pos += 8;
    return true;
}

static bool cursor_float32(hy3_cursor *c, float *v) {
    uint32_t tmp;
    if (!cursor_u32(c, &tmp)) return false;
    memcpy(v, &tmp, 4);
    return true;
}

static bool cursor_string(hy3_cursor *c, hy3_str *s) {
    uint64_t len;
    if (!cursor_u64(c, &len)) return false;
    if (c->pos + len > c->size) { snprintf(c->error, sizeof(c->error), "EOF reading string"); return false; }
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

static bool skip_value(hy3_cursor *c, uint32_t type, int depth) {
    if (depth > 16) { snprintf(c->error, sizeof(c->error), "metadata nesting too deep"); return false; }
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:    c->pos += 1; return true;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:   c->pos += 2; return true;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32: c->pos += 4; return true;
    case GGUF_VALUE_BOOL:    c->pos += 1; return true;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64: c->pos += 8; return true;
    case GGUF_VALUE_STRING: {
        uint64_t len;
        if (!cursor_u64(c, &len)) return false;
        c->pos += len;
        return true;
    }
    case GGUF_VALUE_ARRAY: {
        uint32_t atype;
        uint64_t an;
        if (!cursor_u32(c, &atype)) return false;
        if (!cursor_u64(c, &an)) return false;
        for (uint64_t i = 0; i < an; i++)
            if (!skip_value(c, atype, depth + 1)) return false;
        return true;
    }
    default:
        snprintf(c->error, sizeof(c->error), "unknown GGUF value type %u", type);
        return false;
    }
}

static uint64_t align_up(uint64_t v, uint64_t align) {
    return (v + align - 1) & ~(align - 1);
}

typedef struct {
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

static const gguf_type_info gguf_types[] = {
    [0]  = {1,   4},
    [1]  = {1,   2},
    [2]  = {32,  18},
    [3]  = {32,  20},
    [6]  = {32,  22},
    [7]  = {32,  24},
    [8]  = {32,  36},  /* hy3's Q8_0: F32 scale (4) + 32xint8 = 36 bytes/block, NOT ggml's 34 */
    [9]  = {32,  40},
    [10] = {256, 84},
    [11] = {256, 110},
    [12] = {256, 144},
    [13] = {256, 176},
    [14] = {256, 210},
    [15] = {256, 292},
    [16] = {256, 66},
    [17] = {256, 74},
    [18] = {256, 98},
    [19] = {256, 110},
    [20] = {256, 50},
    [21] = {256, 110},
    [22] = {256, 82},
    [23] = {256, 136},
    [24] = {1, 1},
    [25] = {1, 2},
    [26] = {1, 4},
    [27] = {1, 8},
    [28] = {1, 8},
    [29] = {256, 56},
    [30] = {1, 2},
};

static bool tensor_nbytes(uint32_t type, uint64_t elems, uint64_t *out) {
    if (type >= sizeof(gguf_types)/sizeof(gguf_types[0])) return false;
    const gguf_type_info *ti = &gguf_types[type];
    if (ti->block_bytes == 0 && type != 0) return false;
    if (ti->block_elems == 0) return false;
    *out = (elems / ti->block_elems) * ti->block_bytes;
    return true;
}

static const uint8_t *tensor_data(const hy3_gguf_model *m, const hy3_tensor_info *t) {
    return m->map + t->abs_offset;
}

static void parse_metadata(hy3_gguf_model *m, hy3_cursor *c) {
    m->kv = xcalloc((size_t)m->n_kv, sizeof(m->kv[0]));
    m->alignment = 32;
    for (uint64_t i = 0; i < m->n_kv; i++) {
        hy3_kv *kv = &m->kv[i];
        if (!cursor_string(c, &kv->key)) die(c->error);
        if (!cursor_u32(c, &kv->type)) die(c->error);
        kv->value_pos = c->pos;
        if (kv->key.len == 18 && memcmp(kv->key.ptr, "general.alignment", 18) == 0 && kv->type == GGUF_VALUE_UINT32) {
            hy3_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t align;
            if (cursor_u32(&tmp, &align) && align != 0) m->alignment = align;
        }
        if (!skip_value(c, kv->type, 0)) die(c->error);
    }
}

static void parse_tensors(hy3_gguf_model *m, hy3_cursor *c) {
    m->tensors = xcalloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        hy3_tensor_info *t = &m->tensors[i];
        if (!cursor_string(c, &t->name)) die(c->error);
        if (!cursor_u32(c, &t->ndim)) die(c->error);
        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) die(c->error);
            t->elements *= t->dim[d];
        }
        if (!cursor_u32(c, &t->ggml_type)) die(c->error);
        if (!cursor_u64(c, &t->rel_offset)) die(c->error);
        if (!tensor_nbytes(t->ggml_type, t->elements, &t->bytes)) {
            fprintf(stderr, "hy3: warning: tensor %.*s unsupported type %u\n",
                    (int)t->name.len, t->name.ptr, t->ggml_type);
        }
    }
    m->tensor_data_pos = align_up(c->pos, m->alignment);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        hy3_tensor_info *t = &m->tensors[i];
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->abs_offset + t->bytes > m->size)
            die("GGUF tensor data extends past end of file");
    }
}

static bool streq(hy3_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && memcmp(s.ptr, z, n) == 0;
}

static hy3_str make_str(const char *z) {
    hy3_str s;
    s.ptr = z;
    s.len = strlen(z);
    return s;
}

static hy3_kv *find_kv(const hy3_gguf_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++)
        if (streq(m->kv[i].key, key)) return &m->kv[i];
    return NULL;
}

static hy3_tensor_info *find_tensor(const hy3_gguf_model *m, const char *name) {
    size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++)
        if (m->tensors[i].name.len == len && memcmp(m->tensors[i].name.ptr, name, len) == 0)
            return &m->tensors[i];
    return NULL;
}

static hy3_tensor_info *required_tensor(const hy3_gguf_model *m, const char *name) {
    hy3_tensor_info *t = find_tensor(m, name);
    if (!t) { fprintf(stderr, "hy3: missing required tensor '%s'\n", name); exit(1); }
    return t;
}

static uint32_t get_u32(const hy3_gguf_model *m, const char *key, uint32_t def) {
    hy3_kv *kv = find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return def;
    hy3_cursor c = cursor_at(m, kv->value_pos);
    uint32_t v;
    cursor_u32(&c, &v);
    return v;
}

static float get_f32(const hy3_gguf_model *m, const char *key, float def) {
    hy3_kv *kv = find_kv(m, key);
    if (!kv) return def;
    hy3_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_FLOAT32) { float v; cursor_float32(&c, &v); return v; }
    if (kv->type == GGUF_VALUE_FLOAT64) {
        uint64_t tmp; cursor_u64(&c, &tmp); double v; memcpy(&v, &tmp, 8); return (float)v;
    }
    return def;
}

/* =========================================================================
 * Hy3 Model Constants
 * ========================================================================= */

#define HY3_N_EMBD         4096
#define HY3_N_VOCAB        120832
#define HY3_N_HEAD         64
#define HY3_N_KV_HEAD      8
#define HY3_HEAD_DIM       128
#define HY3_N_EXPERT       192
#define HY3_N_EXPERT_USED  8
#define HY3_N_SHARED       1
#define HY3_MOE_INTERMED   1536
#define HY3_DENSE_INTERMED 13312
#define HY3_N_LAYER_DENSE  1

/* =========================================================================
 * FP16 Utilities
 * ========================================================================= */

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15);
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1f);
    uint32_t mant = (uint32_t)(h & 0x3ff);
    uint32_t f32;
    if (exp == 0) {
        f32 = (sign << 31) | ((0x7f - 15) << 23) | (mant << 13);
    } else if (exp == 31) {
        f32 = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        f32 = (sign << 31) | ((exp + 0x70) << 23) | (mant << 13);
    }
    float r;
    memcpy(&r, &f32, 4);
    return r;
}

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 1;
    uint32_t exp  = (x >> 23) & 0xff;
    uint32_t mant = x & 0x7fffff;
    if (exp == 0) {
        return (uint16_t)(sign << 15);
    } else if (exp == 0xff) {
        uint16_t h = (uint16_t)((sign << 15) | 0x7c00);
        if (mant) h |= 1;
        return h;
    } else {
        int nexp = (int)exp - 127 + 15;
        if (nexp >= 31) return (uint16_t)((sign << 15) | 0x7c00);
        if (nexp <= 0) {
            mant = (mant | 0x800000) >> (1 - nexp);
            return (uint16_t)((sign << 15) | (mant >> 13));
        }
        return (uint16_t)((sign << 15) | ((uint32_t)nexp << 10) | (mant >> 13));
    }
}

/* =========================================================================
 * Quantized Block Types (Q8_0 for CPU inference)
 * ========================================================================= */

#define QK_K 256

typedef struct {
    float   d;
    int8_t  qs[32];
} block_q8_0;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K/2];
} block_q4_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  qs[QK_K/4];
    uint8_t  scales[QK_K/16];
} block_q2_K;

/* =========================================================================
 * Hy3 Weight Pointers
 * ========================================================================= */

/* hy3_weight, hy3_layer_weights, hy3_weights are defined in hy3.h */

/* =========================================================================
 * Hy3 Model
 * ========================================================================= */

/* hy3_model struct is defined in hy3.h */

/* =========================================================================
 * CPU Kernels
 * ========================================================================= */

static inline float rms_norm_val(const float *x, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    return 1.0f / sqrtf(ss / (float)n + eps);
}

static void rms_norm(float *out, const float *x, const float *w, int n, float eps) {
    float r = rms_norm_val(x, n, eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

__attribute__((unused)) static void matmul_f32(float *dst, const float *w, const float *x, int m, int n, int k) {
    (void)n;
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < k; j++)
            sum += w[(size_t)i * k + j] * x[j];
        dst[i] = sum;
    }
}

static void matmul_q8_0(float *dst, const block_q8_0 *w, const float *x, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        const block_q8_0 *b = w + (size_t)i * (k / QK_K);
        for (int j = 0; j < k; j += QK_K) {
            float d = b->d;
            int32_t s = 0;
            for (int l = 0; l < QK_K; l++) s += (int32_t)b->qs[l] * (int32_t)(x[j + l] * 8.0f);
            sum += d * (float)s;
            b++;
        }
        dst[i] = sum / 8.0f;
    }
}

/* Rotary position embedding. HYV3 (like Llama/Mistral) uses the "rotate_half"
 * convention: the head is split into two HALVES and dim d is paired with
 * dim d+half (NOT the interleaved (2i, 2i+1) pairing from the original RoPE
 * paper / GPT-NeoX). Verified against transformers' apply_rotary_pos_emb():
 *   rotate_half(x) = cat(-x[d/2:], x[:d/2])
 *   x_embed = x*cos + rotate_half(x)*sin
 * which expands to the per-pair update below. Getting this pairing wrong
 * silently produces a different (wrong) rotation for every position except
 * position 0, corrupting attention for any sequence longer than one token. */
static void rope(float *q, float *k, int pos, int head_dim, int n_heads, int n_kv_heads) {
    float theta = HY3_ROPE_THETA;
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *q_h = q + (size_t)h * head_dim;
        for (int d = 0; d < half; d++) {
            float freq = (float)pos / powf(theta, (float)(2 * d) / (float)head_dim);
            float cos_val = cosf(freq);
            float sin_val = sinf(freq);
            float q0 = q_h[d];
            float q1 = q_h[d + half];
            q_h[d]        = q0 * cos_val - q1 * sin_val;
            q_h[d + half] = q1 * cos_val + q0 * sin_val;
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float *k_h = k + (size_t)h * head_dim;
        for (int d = 0; d < half; d++) {
            float freq = (float)pos / powf(theta, (float)(2 * d) / (float)head_dim);
            float cos_val = cosf(freq);
            float sin_val = sinf(freq);
            float k0 = k_h[d];
            float k1 = k_h[d + half];
            k_h[d]        = k0 * cos_val - k1 * sin_val;
            k_h[d + half] = k1 * cos_val + k0 * sin_val;
        }
    }
}

static void attention(float *out, const float *q, const float *k_cache, const float *v_cache,
                      int n_kv_heads, int n_heads, int head_dim, int cache_len, int kv_group,
                      int layer_id) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int n_layers = HY3_N_LAYER;
    int ntok = (cache_len - layer_id + n_layers - 1) / n_layers;
    if (ntok < 1) ntok = 1;
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / kv_group;
        const float *q_h = q + (size_t)h * head_dim;
        float max_score = -FLT_MAX;
        float scores[4096];
        int safe_n = ntok < 4096 ? ntok : 4096;
        for (int t = 0; t < safe_n; t++) {
            const float *k_t = k_cache + (size_t)(t * n_layers + layer_id) * n_kv_heads * head_dim + (size_t)kv_h * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++) s += q_h[d] * k_t[d];
            s *= scale;
            scores[t] = s;
            if (s > max_score) max_score = s;
        }
        float sum = 0.0f;
        for (int t = 0; t < safe_n; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }
        float inv_sum = 1.0f / (sum + 1e-10f);
        float *out_h = out + (size_t)h * head_dim;
        memset(out_h, 0, head_dim * sizeof(float));
        for (int t = 0; t < safe_n; t++) {
            float a = scores[t] * inv_sum;
            const float *v_t = v_cache + (size_t)(t * n_layers + layer_id) * n_kv_heads * head_dim + (size_t)kv_h * head_dim;
            for (int d = 0; d < head_dim; d++) out_h[d] += a * v_t[d];
        }
    }
}

static void topk_select(float *vals, int *inds, const float *logits, int n, int k) {
    int idx[192];
    float val[192];
    for (int i = 0; i < n; i++) { idx[i] = i; val[i] = logits[i]; }
    for (int i = 0; i < k && i < n; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++)
            if (val[j] > val[best]) best = j;
        float tv = val[i]; val[i] = val[best]; val[best] = tv;
        int ti = idx[i]; idx[i] = idx[best]; idx[best] = ti;
    }
    for (int i = 0; i < k && i < n; i++) { vals[i] = val[i]; inds[i] = idx[i]; }
}

static void softmax_topk(float *vals, int *inds, const float *logits, int n, int k) {
    int idx[192];
    float val[192];
    for (int i = 0; i < n; i++) {
        idx[i] = i;
        val[i] = logits[i];
    }
    for (int i = 0; i < k && i < n; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++)
            if (val[j] > val[best]) best = j;
        float tv = val[i]; val[i] = val[best]; val[best] = tv;
        int ti = idx[i]; idx[i] = idx[best]; idx[best] = ti;
        vals[i] = val[i] > 0 ? val[i] : 0;
        inds[i] = idx[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < k && i < n; i++) sum += vals[i];
    float inv = 1.0f / (sum + 1e-10f);
    for (int i = 0; i < k && i < n; i++) vals[i] *= inv;
}

/* =========================================================================
 * Weight Loading
 * ========================================================================= */

static void load_weight(hy3_weight *w, const hy3_gguf_model *m, const hy3_tensor_info *t) {
    w->t = t;
    w->data = t ? tensor_data(m, t) : NULL;
}

static void load_weights(hy3_model *model, const hy3_gguf_model *m) {
    hy3_weights *w = &model->w;
    w->n_layer = HY3_N_LAYER;

    load_weight(&w->token_embd, m, find_tensor(m, "token_embd.weight"));
    load_weight(&w->output_norm, m, find_tensor(m, "output_norm.weight"));
    load_weight(&w->output, m, find_tensor(m, "output.weight"));

    for (int il = 0; il < HY3_N_LAYER; il++) {
        hy3_layer_weights *l = &w->layers[il];
        char name[128];

        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", il);
        load_weight(&l->attn_norm, m, find_tensor(m, name));
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", il);
        load_weight(&l->attn_q, m, find_tensor(m, name));
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", il);
        load_weight(&l->attn_k, m, find_tensor(m, name));
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", il);
        load_weight(&l->attn_v, m, find_tensor(m, name));
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", il);
        load_weight(&l->attn_output, m, find_tensor(m, name));
        snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", il);
        load_weight(&l->attn_q_norm, m, find_tensor(m, name));
        snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", il);
        load_weight(&l->attn_k_norm, m, find_tensor(m, name));

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", il);
        load_weight(&l->ffn_norm, m, find_tensor(m, name));

        if (il < HY3_N_LAYER_DENSE) {
            snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", il);
            load_weight(&l->ffn_gate, m, find_tensor(m, name));
            snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", il);
            load_weight(&l->ffn_up, m, find_tensor(m, name));
            snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", il);
            load_weight(&l->ffn_down, m, find_tensor(m, name));
        } else {
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", il);
            load_weight(&l->ffn_gate_inp, m, find_tensor(m, name));

            const char *exp_prefixes[] = {
                "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"
            };
            const char *exp_suffixes[] = {
                ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"
            };
            hy3_weight *exp_arrays[] = {
                l->ffn_gate_exps, l->ffn_up_exps, l->ffn_down_exps
            };
            for (int ei = 0; ei < 3; ei++) {
                char prefix[64];
                snprintf(prefix, sizeof(prefix), "blk.%d.%s.", il, exp_prefixes[ei]);
                size_t plen = strlen(prefix);
                size_t slen = strlen(exp_suffixes[ei]);
                for (int e = 0; e < HY3_N_EXPERT; e++) {
                    memset(&exp_arrays[ei][e], 0, sizeof(hy3_weight));
                }
                for (uint64_t ti = 0; ti < m->n_tensors; ti++) {
                    const hy3_tensor_info *t = &m->tensors[ti];
                    if (t->name.len > plen + 5) {
                        if (memcmp(t->name.ptr, prefix, plen) == 0) {
                            const char *rest = t->name.ptr + plen;
                            char *end;
                            long eid = strtol(rest, &end, 10);
                            if (end && eid >= 0 && eid < HY3_N_EXPERT &&
                                (size_t)(t->name.ptr + t->name.len - end) >= slen &&
                                memcmp(end, exp_suffixes[ei], slen) == 0) {
                                load_weight(&exp_arrays[ei][(int)eid], m, t);
                            }
                        }
                    }
                }
            }

            snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", il);
            load_weight(&l->ffn_gate_shexp, m, find_tensor(m, name));
            snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", il);
            load_weight(&l->ffn_up_shexp, m, find_tensor(m, name));
            snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", il);
            load_weight(&l->ffn_down_shexp, m, find_tensor(m, name));

            l->has_expert_bias = false;
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps_b.bias", il);
            hy3_tensor_info *bias_t = find_tensor(m, name);
            if (bias_t) {
                l->has_expert_bias = true;
                memcpy(l->expert_bias, tensor_data(m, bias_t), HY3_N_EXPERT * sizeof(float));
            }
        }

        snprintf(name, sizeof(name), "blk.%d.eh_proj.weight", il);
        load_weight(&l->eh_proj, m, find_tensor(m, name));
        snprintf(name, sizeof(name), "blk.%d.enorm.weight", il);
        load_weight(&l->enorm, m, find_tensor(m, name));
        snprintf(name, sizeof(name), "blk.%d.hnorm.weight", il);
        load_weight(&l->hnorm, m, find_tensor(m, name));
        snprintf(name, sizeof(name), "blk.%d.final_norm.weight", il);
        load_weight(&l->final_norm, m, find_tensor(m, name));
    }
}

/* =========================================================================
 * Model Loading (from GGUF)
 * ========================================================================= */

int hy3_model_load(hy3_model **out, const char *path, int n_threads) {
    double t0 = now_sec();
    hy3_model *    m = xcalloc(1, sizeof(hy3_model));
    m->n_threads = n_threads > 0 ? n_threads : 4;
#ifdef _OPENMP
    omp_set_num_threads(m->n_threads); /* -t was previously stored but never applied: every
                                         * mul_mat_f32/dequantize_row_* call ran on a single
                                         * core regardless of this value. Silently a no-op when
                                         * built without OpenMP (e.g. plain macOS clang without
                                         * libomp) -- the #pragma omp lines below just compile
                                         * away in that case rather than failing the build. */
#endif
    m->gpu_layers = 81;
    m->rng_state = (uint64_t)(uintptr_t)m ^ (uint64_t)time(NULL);

    /* Runtime MoE top-k. Defaults to the model's native HY3_N_EXPERT_USED (8)
     * but can be lowered (e.g. 4 or 2) via HY3_TOP_K_EXPERTS or the CLI to
     * trade quality for speed. Clamped to [1, HY3_N_EXPERT_USED] so the
     * fixed-size top-k arrays are never overrun. */
    m->n_expert_used = HY3_N_EXPERT_USED;
    {
        const char *env = getenv("HY3_TOP_K_EXPERTS");
        if (env && env[0]) {
            int k = atoi(env);
            if (k >= 1 && k <= HY3_N_EXPERT_USED) m->n_expert_used = k;
            else fprintf(stderr, "hy3: ignoring HY3_TOP_K_EXPERTS=%s (must be 1..%d)\n",
                         env, HY3_N_EXPERT_USED);
        }
    }

    hy3_gguf_model *g = &m->gguf;
    g->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd == -1) die_errno("cannot open model", path);
    struct stat st;
    if (fstat(fd, &st) == -1) die_errno("cannot stat model", path);
    if (st.st_size < 32) die("model file too small");

    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) die_errno("cannot mmap model", path);

    /* The Metal backend wraps this mmap as zero-copy MTLBuffers. If a page
     * isn't resident when the GPU dereferences it, the GPU reads ZERO rather
     * than faulting it in -- silently corrupting weights (observed: the
     * output.weight matmul returned all-zero logits on the first token). We
     * therefore force the whole model resident up front. mlock() would be
     * ideal but macOS caps user-wired memory (vm.user_wire_limit) and it
     * fails with EAGAIN near the cap; a plain CPU read-touch of every page is
     * more robust (model < RAM here) and achieves residency without wiring.
     * Skip with HY3_NO_PREFAULT=1. */
    const char *no_pf = getenv("HY3_NO_PREFAULT");
    if (!(no_pf && no_pf[0] == '1')) {
        double tl0 = now_sec();
#ifdef MADV_WILLNEED
        madvise(map, (size_t)st.st_size, MADV_WILLNEED);
#endif
        size_t page = (size_t)getpagesize();
        volatile uint8_t sink = 0;
        const uint8_t *p = (const uint8_t *)map;
        for (size_t off = 0; off < (size_t)st.st_size; off += page) sink ^= p[off];
        (void)sink;
        fprintf(stderr, "hy3: prefaulted %.1f GiB resident in %.1fs\n",
                (double)st.st_size / (1024.0*1024.0*1024.0), now_sec() - tl0);
    }

    g->fd = fd;
    g->map = map;
    g->size = (uint64_t)st.st_size;

    hy3_cursor c = cursor_at(g, 0);
    uint32_t magic;
    if (!cursor_u32(&c, &magic)) die(c.error);
    if (magic != HY3_GGUF_MAGIC) die("not a GGUF file");
    if (!cursor_u32(&c, &g->version)) die(c.error);
    if (!cursor_u64(&c, &g->n_tensors)) die(c.error);
    if (!cursor_u64(&c, &g->n_kv)) die(c.error);
    if (g->version != 3) die("only GGUF v3 supported");

    parse_metadata(g, &c);
    parse_tensors(g, &c);

    load_weights(m, g);

    if (!m->w.token_embd.data) die("missing token_embd.weight");
    if (!m->w.output.data) die("missing output.weight");

    const hy3_tensor_info *embd = m->w.token_embd.t;
    if (embd->ndim != 2) die("token_embd.weight has unexpected ndim");
    if (embd->dim[0] != HY3_N_VOCAB || embd->dim[1] != HY3_N_EMBD) {
        fprintf(stderr, "hy3: token_embd.weight shape: [%llu, %llu], expected [%d, %d]\n",
                (unsigned long long)embd->dim[0], (unsigned long long)embd->dim[1],
                HY3_N_VOCAB, HY3_N_EMBD);
        die("token_embd.weight has unexpected shape");
    }

    m->ctx_size = HY3_N_VOCAB;
    m->embed = xcalloc(HY3_N_EMBD, sizeof(float));
    m->scratch = xcalloc(HY3_DENSE_INTERMED * 2 + HY3_N_EMBD * 4 + HY3_N_HEAD * HY3_HEAD_DIM * 4, sizeof(float));
    m->scratch2 = xcalloc(HY3_N_EXPERT + HY3_MOE_INTERMED * 4 + HY3_N_EMBD * 2, sizeof(float));
    
    
    m->t_load = now_sec() - t0;
    fprintf(stderr, "hy3: loaded model in %.2f seconds\n", m->t_load);
    fprintf(stderr, "hy3: %s | layers=%d embd=%d heads=%d kv_heads=%d head_dim=%d\n",
            "HYV3ForCausalLM", HY3_N_LAYER, HY3_N_EMBD, HY3_N_HEAD, HY3_N_KV_HEAD, HY3_HEAD_DIM);

    *out = m;
    return 0;
}

#ifdef HY3_CUDA
void hy3_gpu_free(hy3_model *m);
#endif
#ifdef HY3_METAL
void hy3_metal_free(hy3_model *m);
#endif

void hy3_model_free(hy3_model *m) {
    if (!m) return;
#ifdef HY3_CUDA
    if (m->gpu_ctx) {
        hy3_gpu_free(m);
    }
#endif
#ifdef HY3_METAL
    if (m->metal_ctx) {
        hy3_metal_free(m);
    }
#endif
    if (m->gguf.map && m->gguf.fd >= 0) {
        munmap((void *)m->gguf.map, (size_t)m->gguf.size);
        close(m->gguf.fd);
    }
    free(m->gguf.kv);
    free(m->gguf.tensors);
    free(m->embed);
    free(m->scratch);
    free(m->scratch2);
    free(m->cache_k);
    free(m->cache_v);
    free(m);
}

/* Expose weights pointer for GPU integration */
const hy3_weights *hy3_get_weights(const hy3_model *m) { return &m->w; }
void hy3_set_gpu_ctx(hy3_model *m, void *ctx) { m->gpu_ctx = ctx; }

/* Reset conversational state so the next hy3_generate() starts a fresh context.
 * The KV cache is keyed by m->cache_len (slot = token_idx*HY3_N_LAYER+layer),
 * so zeroing it is sufficient for both the CPU and Metal paths -- old slots are
 * simply overwritten from the start. Used by batch mode to reuse one loaded
 * model across many independent prompts. */
void hy3_reset_context(hy3_model *m) { m->cache_len = 0; }

int hy3_model_vocab_size(hy3_model *m) { return HY3_N_VOCAB; }
const char *hy3_model_name(hy3_model *m) { return "HYV3ForCausalLM"; }
int hy3_model_ctx_size(hy3_model *m) { return m->ctx_size; }

/* =========================================================================
 * Dequantization Helpers
 * ========================================================================= */

static void dequantize_row_f32(float *dst, const uint8_t *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

static void dequantize_row_q8_0(float *dst, const uint8_t *src, int n) {
    const block_q8_0 *b = (const block_q8_0 *)src;
    int nb = n / QK_K;
    for (int i = 0; i < nb; i++) {
        float d = b[i].d;
        for (int j = 0; j < QK_K; j++)
            dst[i * QK_K + j] = (float)b[i].qs[j] * d;
    }
}

static void q4_k_get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

/* Q4_K packs 256 values as 4 chunks of 64. Each chunk stores 32 bytes; the
 * LOW nibble of byte l is sub-block (2*chunk+0)'s value l, and the HIGH
 * nibble is sub-block (2*chunk+1)'s value l (see quantize_row_q4_K in
 * hy3_convert.c, which packs `q[l] = L[j+l] | (L[j+l+32]<<4)`). The previous
 * version of this function read only qs[0..15] for every sub-block (missing
 * the per-chunk byte offset and using the wrong nibble/byte split), so it
 * ignored 7/8 of the quantized weight bytes -- this must mirror the packer
 * exactly or every Q4_K tensor dequantizes to near-random noise. */
static void dequantize_row_q4_K(float *dst, const uint8_t *src, int n) {
    const block_q4_K *b = (const block_q4_K *)src;
    int nb = n / QK_K;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(b[i].d);
        float dmin = fp16_to_fp32(b[i].dmin);
        const uint8_t *sc = b[i].scales;
        const uint8_t *q = b[i].qs;
        float *y = dst + (size_t)i * QK_K;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            q4_k_get_scale_min(is + 0, sc, &sc1, &m1);
            q4_k_get_scale_min(is + 1, sc, &sc2, &m2);
            float d1 = d * (float)sc1, dm1 = dmin * (float)m1;
            float d2 = d * (float)sc2, dm2 = dmin * (float)m2;
            for (int l = 0; l < 32; l++) y[l]      = d1 * (float)(q[l] & 0xF) - dm1;
            for (int l = 0; l < 32; l++) y[32 + l] = d2 * (float)(q[l] >> 4)  - dm2;
            y += 64;
            q += 32;
            is += 2;
        }
    }
}

static void dequantize_row_q2_K(float *dst, const uint8_t *src, int n) {
    const block_q2_K *b = (const block_q2_K *)src;
    int nb = n / QK_K;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nb; i++) {
        float d = (float)b[i].d;
        float dmin = (float)b[i].dmin;
        for (int j = 0; j < 256; j++) {
            int s = j / 16;
            int l = j % 16;
            int v = (b[i].qs[l] >> (2 * (j / 16 % 4))) & 3;
            int sc = (b[i].scales[s] >> (j / 64 * 4)) & 0xf;
            dst[i * QK_K + j] = (float)v * d * (float)sc + dmin;
        }
    }
}

/* =========================================================================
 * Matrix Multiply with Dequantization
 * ========================================================================= */

/* Every case below computes one output row per `i`, reading only the
 * (read-only) activation vector `x` -- rows are fully independent, so this
 * parallelizes safely across CPU cores. This was previously fully serial
 * despite the build linking OpenMP and hy3_cli.c accepting a `-t` thread
 * count: the CLI flag was stored but never applied, so any layer running on
 * CPU (the whole model with no --gpu-layers, or the tail layers beyond
 * --gpu-layers N) used exactly one core no matter how many were requested. */
static void mul_mat_f32(float *dst, const uint8_t *w_data, uint32_t w_type,
                        const float *x, int m, int n) {
    switch (w_type) {
    case 0: { // f32
        const float *wf = (const float *)w_data;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++)
                sum += wf[(size_t)i * n + j] * x[j];
            dst[i] = sum;
        }
        break;
    }
    case 1: { // f16
        const uint16_t *hf = (const uint16_t *)w_data;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++)
                sum += fp16_to_fp32(hf[(size_t)i * n + j]) * x[j];
            dst[i] = sum;
        }
        break;
    }
    case 8: { // q8_0
        /* Dequantize weight (d * qs[l]) and multiply directly against the
         * float activation. The previous version scaled x by a fixed factor
         * of 8 and truncated to int32 before the dot product -- a crude
         * ~3-bit fixed-point encoding of the activation that measured ~70%
         * relative error per row on typical post-RMSNorm magnitudes. Q8_0
         * is now used for attention q/k/v/o, output, and shared/dense FFN
         * weights specifically to preserve precision, so the activation
         * side must not be the bottleneck. */
        const block_q8_0 *wq = (const block_q8_0 *)w_data;
        int nb = n / 32;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            const block_q8_0 *b = wq + (size_t)i * nb;
            for (int j = 0; j < nb; j++) {
                float d = b[j].d;
                for (int l = 0; l < 32; l++)
                    sum += d * (float)b[j].qs[l] * x[(size_t)j * 32 + l];
            }
            dst[i] = sum;
        }
        break;
    }
    case 12: { // q4_K
        float *tmp = xmalloc((size_t)m * n * sizeof(float));
        dequantize_row_q4_K(tmp, w_data, m * n);
        const float *wf = tmp;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++)
                sum += wf[(size_t)i * n + j] * x[j];
            dst[i] = sum;
        }
        free(tmp);
        break;
    }
    case 10: { // q2_K
        float *tmp = xmalloc((size_t)m * n * sizeof(float));
        dequantize_row_q2_K(tmp, w_data, m * n);
        const float *wf = tmp;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++)
                sum += wf[(size_t)i * n + j] * x[j];
            dst[i] = sum;
        }
        free(tmp);
        break;
    }
    default:
        fprintf(stderr, "hy3: unsupported ggml_type %u for matrix multiply\n", w_type);
        memset(dst, 0, m * sizeof(float));
        break;
    }
}

/* =========================================================================
 * Forward Pass
 * ========================================================================= */

void forward_layer_dense(hy3_model *m, int il, int pos) {
    hy3_layer_weights *l = &m->w.layers[il];
    float *x = m->embed;
    float *s = m->scratch;
    float *s2 = m->scratch + HY3_N_EMBD * 2;
    float *attn_out = m->scratch + HY3_N_EMBD * 4;

    rms_norm(s, x, (const float *)l->attn_norm.data, HY3_N_EMBD, HY3_RMS_EPS);

    int q_size = HY3_N_HEAD * HY3_HEAD_DIM;
    int kv_size = HY3_N_KV_HEAD * HY3_HEAD_DIM;

    mul_mat_f32(s2, l->attn_q.data, l->attn_q.t->ggml_type, s, q_size, HY3_N_EMBD);
    float *q = s2;
    float *k = s2 + q_size;
    float *v = s2 + q_size + kv_size;
    mul_mat_f32(k, l->attn_k.data, l->attn_k.t->ggml_type, s, kv_size, HY3_N_EMBD);
    mul_mat_f32(v, l->attn_v.data, l->attn_v.t->ggml_type, s, kv_size, HY3_N_EMBD);

    if (l->attn_q_norm.data) {
        const float *wn = (const float *)l->attn_q_norm.data;
        for (int h = 0; h < HY3_N_HEAD; h++)
            rms_norm(q + h * HY3_HEAD_DIM, q + h * HY3_HEAD_DIM, wn, HY3_HEAD_DIM, HY3_RMS_EPS);
    }
    if (l->attn_k_norm.data) {
        const float *wn = (const float *)l->attn_k_norm.data;
        for (int h = 0; h < HY3_N_KV_HEAD; h++)
            rms_norm(k + h * HY3_HEAD_DIM, k + h * HY3_HEAD_DIM, wn, HY3_HEAD_DIM, HY3_RMS_EPS);
    }

    rope(q, k, pos, HY3_HEAD_DIM, HY3_N_HEAD, HY3_N_KV_HEAD);

    int kv_len = m->cache_len;
    float *k_cache = m->cache_k + (size_t)kv_len * kv_size;
    float *v_cache = m->cache_v + (size_t)kv_len * kv_size;
    memcpy(k_cache, k, kv_size * sizeof(float));
    memcpy(v_cache, v, kv_size * sizeof(float));
    m->cache_len = kv_len + 1;

    attention(attn_out, q, m->cache_k, m->cache_v,
              HY3_N_KV_HEAD, HY3_N_HEAD, HY3_HEAD_DIM, m->cache_len,
              HY3_N_HEAD / HY3_N_KV_HEAD, il);

    float *o_proj_out = s2;
    mul_mat_f32(o_proj_out, l->attn_output.data, l->attn_output.t->ggml_type,
                attn_out, HY3_N_EMBD, q_size);
    for (int i = 0; i < HY3_N_EMBD; i++) x[i] += o_proj_out[i];

    rms_norm(s, x, (const float *)l->ffn_norm.data, HY3_N_EMBD, HY3_RMS_EPS);

    int inter = HY3_DENSE_INTERMED;
    mul_mat_f32(s2, l->ffn_gate.data, l->ffn_gate.t->ggml_type, s, inter, HY3_N_EMBD);
    mul_mat_f32(s2 + inter, l->ffn_up.data, l->ffn_up.t->ggml_type, s, inter, HY3_N_EMBD);
    for (int i = 0; i < inter; i++) s2[i] = silu(s2[i]) * s2[inter + i];
    mul_mat_f32(s, l->ffn_down.data, l->ffn_down.t->ggml_type, s2, HY3_N_EMBD, inter);
    for (int i = 0; i < HY3_N_EMBD; i++) x[i] += s[i];
}

void forward_layer_moe(hy3_model *m, int il, int pos) {
    hy3_layer_weights *l = &m->w.layers[il];
    float *x = m->embed;
    float *s = m->scratch;
    float *s2 = m->scratch + HY3_N_EMBD * 2;
    float *attn_out = m->scratch + HY3_N_EMBD * 4;

    rms_norm(s, x, (const float *)l->attn_norm.data, HY3_N_EMBD, HY3_RMS_EPS);

    int q_size = HY3_N_HEAD * HY3_HEAD_DIM;
    int kv_size = HY3_N_KV_HEAD * HY3_HEAD_DIM;

    float *q = s2;
    float *k = s2 + q_size;
    float *v = s2 + q_size + kv_size;
    mul_mat_f32(q, l->attn_q.data, l->attn_q.t->ggml_type, s, q_size, HY3_N_EMBD);
    mul_mat_f32(k, l->attn_k.data, l->attn_k.t->ggml_type, s, kv_size, HY3_N_EMBD);
    mul_mat_f32(v, l->attn_v.data, l->attn_v.t->ggml_type, s, kv_size, HY3_N_EMBD);

    if (l->attn_q_norm.data) {
        const float *wn = (const float *)l->attn_q_norm.data;
        for (int h = 0; h < HY3_N_HEAD; h++)
            rms_norm(q + h * HY3_HEAD_DIM, q + h * HY3_HEAD_DIM, wn, HY3_HEAD_DIM, HY3_RMS_EPS);
    }
    if (l->attn_k_norm.data) {
        const float *wn = (const float *)l->attn_k_norm.data;
        for (int h = 0; h < HY3_N_KV_HEAD; h++)
            rms_norm(k + h * HY3_HEAD_DIM, k + h * HY3_HEAD_DIM, wn, HY3_HEAD_DIM, HY3_RMS_EPS);
    }

    rope(q, k, pos, HY3_HEAD_DIM, HY3_N_HEAD, HY3_N_KV_HEAD);

    int kv_len = m->cache_len;
    float *k_cache = m->cache_k + (size_t)kv_len * kv_size;
    float *v_cache = m->cache_v + (size_t)kv_len * kv_size;
    memcpy(k_cache, k, kv_size * sizeof(float));
    memcpy(v_cache, v, kv_size * sizeof(float));
    m->cache_len = kv_len + 1;

    attention(attn_out, q, m->cache_k, m->cache_v,
              HY3_N_KV_HEAD, HY3_N_HEAD, HY3_HEAD_DIM, m->cache_len,
              HY3_N_HEAD / HY3_N_KV_HEAD, il);

    mul_mat_f32(s2, l->attn_output.data, l->attn_output.t->ggml_type,
                attn_out, HY3_N_EMBD, q_size);
    for (int i = 0; i < HY3_N_EMBD; i++) x[i] += s2[i];

    rms_norm(s, x, (const float *)l->ffn_norm.data, HY3_N_EMBD, HY3_RMS_EPS);

    float *router_logits = m->scratch2;
    mul_mat_f32(router_logits, l->ffn_gate_inp.data, l->ffn_gate_inp.t->ggml_type,
                s, HY3_N_EXPERT, HY3_N_EMBD);

    float sigmoid_vals[HY3_N_EXPERT];
    for (int i = 0; i < HY3_N_EXPERT; i++) {
        sigmoid_vals[i] = 1.0f / (1.0f + expf(-router_logits[i]));
        router_logits[i] = sigmoid_vals[i];
        if (l->has_expert_bias) router_logits[i] += l->expert_bias[i];
    }

    float topk_vals[HY3_N_EXPERT_USED];
    int topk_inds[HY3_N_EXPERT_USED];
    int n_used = m->n_expert_used;
    topk_select(topk_vals, topk_inds, router_logits, HY3_N_EXPERT, n_used);
    float sum_w = 0;
    for (int i = 0; i < n_used; i++) {
        topk_vals[i] = sigmoid_vals[topk_inds[i]];
        sum_w += topk_vals[i];
    }
    float inv_sum = 1.0f / (sum_w + 1e-20f);
    float router_scaling = 2.826f;
    for (int i = 0; i < n_used; i++)
        topk_vals[i] = topk_vals[i] * inv_sum * router_scaling;

    float *shared_gate = m->scratch2 + HY3_N_EXPERT;
    float *shared_up = shared_gate + HY3_MOE_INTERMED;
    mul_mat_f32(shared_gate, l->ffn_gate_shexp.data, l->ffn_gate_shexp.t->ggml_type,
                s, HY3_MOE_INTERMED, HY3_N_EMBD);
    mul_mat_f32(shared_up, l->ffn_up_shexp.data, l->ffn_up_shexp.t->ggml_type,
                s, HY3_MOE_INTERMED, HY3_N_EMBD);
    for (int i = 0; i < HY3_MOE_INTERMED; i++)
        shared_gate[i] = silu(shared_gate[i]) * shared_up[i];

    float *expert_out = m->scratch2 + HY3_N_EXPERT + HY3_MOE_INTERMED * 2;
    memset(expert_out, 0, HY3_N_EMBD * sizeof(float));
    float *gate_buf = m->scratch2 + HY3_N_EXPERT + HY3_MOE_INTERMED * 2 + HY3_N_EMBD;
    float *up_buf = gate_buf + HY3_MOE_INTERMED;
    float *down_buf = up_buf + HY3_MOE_INTERMED;
    for (int e = 0; e < n_used; e++) {
        int ei = topk_inds[e];
        mul_mat_f32(gate_buf, l->ffn_gate_exps[ei].data, l->ffn_gate_exps[ei].t->ggml_type,
                    s, HY3_MOE_INTERMED, HY3_N_EMBD);
        mul_mat_f32(up_buf, l->ffn_up_exps[ei].data, l->ffn_up_exps[ei].t->ggml_type,
                    s, HY3_MOE_INTERMED, HY3_N_EMBD);
        for (int i = 0; i < HY3_MOE_INTERMED; i++)
            gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
        mul_mat_f32(down_buf, l->ffn_down_exps[ei].data, l->ffn_down_exps[ei].t->ggml_type,
                    gate_buf, HY3_N_EMBD, HY3_MOE_INTERMED);
        float w = topk_vals[e];
        for (int i = 0; i < HY3_N_EMBD; i++)
            expert_out[i] += w * down_buf[i];
    }

    mul_mat_f32(s, l->ffn_down_shexp.data, l->ffn_down_shexp.t->ggml_type,
                shared_gate, HY3_N_EMBD, HY3_MOE_INTERMED);
    for (int i = 0; i < HY3_N_EMBD; i++)
        x[i] += s[i] + expert_out[i];
}

static void forward_model(hy3_model *m, int token) {
    /* token_embd.weight may be F32 or F16 (the converter now writes it as
     * F16 to halve its size); dispatch on the tensor's actual type rather
     * than assuming a raw float layout, or an F16 table would be
     * reinterpreted as garbage F32 data. */
    if (m->w.token_embd.t->ggml_type == 1) {
        const uint16_t *embd_table = (const uint16_t *)m->w.token_embd.data;
        const uint16_t *row = embd_table + (size_t)token * HY3_N_EMBD;
        for (int i = 0; i < HY3_N_EMBD; i++)
            m->embed[i] = fp16_to_fp32(row[i]);
    } else {
        const float *embd_table = (const float *)m->w.token_embd.data;
        const float *row = embd_table + (size_t)token * HY3_N_EMBD;
        for (int i = 0; i < HY3_N_EMBD; i++)
            m->embed[i] = row[i];
    }

    for (int il = 0; il < HY3_N_LAYER; il++) {
        int pos = m->cache_len / HY3_N_LAYER;
        if (il < HY3_N_LAYER_DENSE)
            forward_layer_dense(m, il, pos);
        else
            forward_layer_moe(m, il, pos);
    }
}

/* =========================================================================
 * Public API
 * ========================================================================= */

#ifdef HY3_CUDA
int hy3_gpu_init(hy3_model *m);
void hy3_gpu_free(hy3_model *m);
int hy3_gpu_eval(hy3_model *m, const hy3_tokens *tokens, float *logits, int *pos);
#endif
#ifdef HY3_METAL
void hy3_metal_free(hy3_model *m);
#endif

int hy3_eval(hy3_model *m, const hy3_tokens *tokens, float *logits, int *pos) {
#ifdef HY3_CUDA
    if (m->gpu_ctx) {
        return hy3_eval_gpu(m, tokens, logits, pos);
    }
#endif
#ifdef HY3_METAL
    if (m->metal_ctx) {
        return hy3_eval_metal(m, tokens, logits, pos);
    }
#endif
    
    for (int i = 0; i < tokens->len; i++) {
        int token = tokens->v[i];

        /* The KV cache is laid out interleaved by layer (slot = token_idx *
         * HY3_N_LAYER + layer_id, see attention()/forward_layer_*), so a
         * single forward_model() call writes HY3_N_LAYER new slots, not 1.
         * m->ctx_size tracks capacity in SLOTS (not floats/bytes) once
         * allocated; comparing "needed slots" against it directly (instead
         * of multiplying only one side by kv_size, as the old code did)
         * keeps the units consistent and avoids reallocating on almost
         * every token. */
        int kv_size = HY3_N_KV_HEAD * HY3_HEAD_DIM;
        int needed = m->cache_len + HY3_N_LAYER;

        if (!m->cache_k) {
            m->ctx_size = needed + 1024;
            m->cache_k = xcalloc((size_t)m->ctx_size * kv_size, sizeof(float));
            m->cache_v = xcalloc((size_t)m->ctx_size * kv_size, sizeof(float));
        } else if (needed > m->ctx_size) {
            size_t new_cap = (size_t)needed + 1024;
            m->cache_k = xrealloc(m->cache_k, new_cap * kv_size * sizeof(float));
            m->cache_v = xrealloc(m->cache_v, new_cap * kv_size * sizeof(float));
            m->ctx_size = (int)new_cap;
        }

        forward_model(m, token);
    }

    const float *output_norm_w = (const float *)m->w.output_norm.data;
    rms_norm(m->embed, m->embed, output_norm_w, HY3_N_EMBD, HY3_RMS_EPS);

    const hy3_tensor_info *out_t = m->w.output.t;
    int n_vocab = (int)out_t->dim[0];
    mul_mat_f32(logits, m->w.output.data, out_t->ggml_type, m->embed, n_vocab, HY3_N_EMBD);

    *pos = m->cache_len;
    return 0;
}

int hy3_sample(hy3_model *m, const float *logits, float temperature, int top_k, float top_p) {
    int n = HY3_N_VOCAB_VALID;

    /* Greedy (argmax) when temperature is ~0. A true argmax avoids the
     * degenerate softmax with inv_temp=1000 used before, which amplified
     * floating-point noise among near-tied logits (common inside a reasoning
     * block) and could underflow the whole distribution to zero/NaN, producing
     * garbage tokens. */
    if (temperature < 0.001f) {
        int best = 0;
        float best_v = -FLT_MAX;
        for (int i = 0; i < n; i++)
            if (logits[i] > best_v) { best_v = logits[i]; best = i; }
        return best;
    }

    float *probs = xmalloc(n * sizeof(float));
    memcpy(probs, logits, n * sizeof(float));

    float inv_temp = 1.0f / temperature;

    float max_val = -FLT_MAX;
    for (int i = 0; i < n; i++)
        if (probs[i] > max_val) max_val = probs[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((probs[i] - max_val) * inv_temp);
        sum += probs[i];
    }
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int i = 0; i < n; i++) probs[i] *= inv_sum;

    if (top_k > 0 && top_k < n) {
        float threshold = 0.0f;
        float sorted[120832];
        memcpy(sorted, probs, n * sizeof(float));
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (sorted[j] > sorted[i]) {
                    float t = sorted[i];
                    sorted[i] = sorted[j];
                    sorted[j] = t;
                }
            }
        }
        threshold = sorted[top_k - 1];
        float new_sum = 0.0f;
        for (int i = 0; i < n; i++) {
            if (probs[i] < threshold) probs[i] = 0.0f;
            new_sum += probs[i];
        }
        if (new_sum > 0) {
            inv_sum = 1.0f / new_sum;
            for (int i = 0; i < n; i++) probs[i] *= inv_sum;
        }
    }

    m->rng_state = m->rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    float r = (float)(m->rng_state >> 33) / (float)(1ULL << 31);
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        cum += probs[i];
        if (r <= cum) { free(probs); return i; }
    }
    free(probs);
    return n - 1;
}

int hy3_generate(hy3_model *m, const hy3_tokens *prompt,
                 int n_predict, hy3_params *params,
                 int (*emit)(void *ud, int token),
                 void *emit_ud)
{
    float *logits = xmalloc(HY3_N_VOCAB * sizeof(float));
    int pos = 0;

    hy3_tokens input;
    input.v = NULL;
    input.len = 0;
    input.cap = 0;

    hy3_tokens_push(&input, prompt->v[0]);
    double t_prompt0 = now_sec();
    hy3_eval(m, &input, logits, &pos);

    for (int i = 1; i < prompt->len; i++) {
        hy3_tokens single;
        single.v = &prompt->v[i];
        single.len = 1;
        single.cap = 1;
        hy3_eval(m, &single, logits, &pos);
    }
    double t_prompt = now_sec() - t_prompt0;

    int n_generated = 0;
    double t_gen0 = now_sec();

    for (int i = 0; i < n_predict; i++) {
        int token = hy3_sample(m, logits, params->temperature, params->top_k, params->top_p);
        /* Stop on any Hunyuan V3 end-of-turn / end-of-sequence marker:
         * eos(120025), endofsentence(120001), EOT(120008). */
        if (token == hy3_token_eos(m) || token == 120001 || token == 120008) {
            break;
        }

        if (emit && emit(emit_ud, token)) break;

        hy3_tokens single;
        single.v = &token;
        single.len = 1;
        single.cap = 1;
        hy3_eval(m, &single, logits, &pos);
        n_generated++;
    }
    double t_gen = now_sec() - t_gen0;

    fprintf(stderr,
            "\nhy3: timing | prompt %d tok in %.3fs (%.2f tok/s) | gen %d tok in %.3fs (%.2f tok/s)\n",
            prompt->len, t_prompt, prompt->len / (t_prompt + 1e-9),
            n_generated, t_gen, n_generated / (t_gen + 1e-9));

    hy3_tokens_free(&input);
    free(logits);
    return n_generated;
}

void hy3_tokens_push(hy3_tokens *tv, int token) {
    if (tv->len >= tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = xrealloc(tv->v, tv->cap * sizeof(int));
    }
    tv->v[tv->len++] = token;
}

void hy3_tokens_unshift(hy3_tokens *tv, int token) {
    hy3_tokens_push(tv, 0);
    for (int i = tv->len - 1; i > 0; i--) tv->v[i] = tv->v[i-1];
    tv->v[0] = token;
}

void hy3_tokens_free(hy3_tokens *tv) {
    free(tv->v);
    tv->v = NULL;
    tv->len = tv->cap = 0;
}

/* =========================================================================
 * Simple BPE Tokenizer (from tokenizer.json GGUF metadata)
 * ========================================================================= */

typedef struct {
    int id;
    hy3_str text;
    float score;
} hy3_vocab_entry;

static hy3_vocab_entry *vocab = NULL;
static int vocab_size = 0;
static int prev_vocab_size = 0;
static int byte_to_token[256];

static const hy3_vocab_entry *find_vocab_entry(int id) {
    for (int i = 0; i < vocab_size; i++)
        if (vocab[i].id == id) return &vocab[i];
    return NULL;
}

static int utf8_decode(const uint8_t *s, size_t n, uint32_t *cp);
static int unicode_to_byte(uint32_t cp);

/* Inverse of unicode_to_byte: map a raw byte (0..255) to the GPT-2 byte-level
 * BPE "surface" codepoint used in the vocab. Printable ASCII / Latin-1 map to
 * themselves; the control/space bytes are shifted into the 0x100.. range. */
static uint32_t byte_to_unicode(uint8_t b) {
    if (b >= 0x21 && b <= 0x7E) return b;
    if (b >= 0xA1 && b <= 0xAC) return b;
    if (b >= 0xAE && b <= 0xFF) return b;
    /* bytes 0..32, 127, 128..160, 173 -> 0x100 + running index */
    uint32_t idx = 0;
    for (uint32_t x = 0; x < b; x++) {
        if ((x >= 0x21 && x <= 0x7E) || (x >= 0xA1 && x <= 0xAC) || (x >= 0xAE && x <= 0xFF))
            continue;
        idx++;
    }
    return 0x100 + idx;
}

/* Encode a codepoint as UTF-8 into buf (must hold >=4 bytes). Returns length. */
static size_t utf8_encode(uint32_t cp, char *buf) {
    if (cp < 0x80) { buf[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    buf[0] = (char)(0xF0 | (cp >> 18));
    buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    buf[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

void hy3_tokenize(hy3_model *m, const char *text, hy3_tokens *out) {
    const hy3_gguf_model *g = &m->gguf;

    if (!vocab) {
        if (vocab) free(vocab);
        hy3_kv *tokens_kv = find_kv(g, "tokenizer.ggml.tokens");
        hy3_kv *scores_kv = find_kv(g, "tokenizer.ggml.scores");
        if (!tokens_kv) die("no tokenizer.ggml.tokens in GGUF");

        hy3_cursor c = cursor_at(g, tokens_kv->value_pos);
        uint32_t atype;
        uint64_t an;
        cursor_u32(&c, &atype);
        cursor_u64(&c, &an);

        if (atype == GGUF_VALUE_STRING) {
            /* v2 format: tokens array is array of strings, use index as token ID */
            vocab_size = (int)an;
            vocab = xcalloc(vocab_size, sizeof(hy3_vocab_entry));

            /* Read all token texts */
            hy3_cursor tc = cursor_at(g, tokens_kv->value_pos + 4 + 8);
            tc.pos = tokens_kv->value_pos;
            cursor_u32(&tc, &atype);
            cursor_u64(&tc, &an);
            for (int i = 0; i < vocab_size; i++) {
                cursor_string(&tc, &vocab[i].text);
                vocab[i].id = i;
            }
            /* Read scores if available (using their own cursor) */
            if (scores_kv) {
                hy3_cursor sc = cursor_at(g, scores_kv->value_pos + 4 + 8);
                sc.pos = scores_kv->value_pos;
                uint32_t satype;
                cursor_u32(&sc, &satype);
                uint64_t sn;
                cursor_u64(&sc, &sn);
                for (int i = 0; i < vocab_size && i < (int)sn; i++)
                    cursor_float32(&sc, &vocab[i].score);
            }
        } else if (atype == GGUF_VALUE_INT32 || atype == GGUF_VALUE_UINT32) {
            /* v1 format: tokens array is array of int32, text in separate entries */
            vocab_size = (int)an;
            vocab = xcalloc(vocab_size, sizeof(hy3_vocab_entry));

            hy3_cursor sc = {0};
            uint64_t sn = 0;
            if (scores_kv) {
                sc = cursor_at(g, scores_kv->value_pos);
                uint32_t satype;
                cursor_u32(&sc, &satype);
                cursor_u64(&sc, &sn);
            }

            for (int i = 0; i < vocab_size; i++) {
                hy3_cursor tc = cursor_at(g, tokens_kv->value_pos + 4 + 8);
                tc.pos = tokens_kv->value_pos;
                cursor_u32(&tc, &atype);
                cursor_u64(&tc, &an);
                for (int j = 0; j <= i; j++) {
                    if (atype == GGUF_VALUE_INT32) {
                        int32_t val;
                        cursor_u32(&tc, (uint32_t *)&val);
                        if (j == i) vocab[i].id = val;
                    } else {
                        uint32_t val;
                        cursor_u32(&tc, &val);
                        if (j == i) vocab[i].id = (int)val;
                    }
                }

                int32_t tid = vocab[i].id;
                char key[64];
                snprintf(key, sizeof(key), "tokenizer.ggml.token_%d.text", tid);
                hy3_kv *tk = find_kv(g, key);
                if (tk) {
                    hy3_cursor tc2 = cursor_at(g, tk->value_pos);
                    cursor_string(&tc2, &vocab[i].text);
                } else {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "<token_%d>", tid);
                    vocab[i].text.ptr = strdup(buf);
                    vocab[i].text.len = strlen(buf);
                }
            }
        } else {
            die("unexpected token array type");
        }
        prev_vocab_size = vocab_size;
        
        for (int i = 0; i < 256; i++) byte_to_token[i] = -1;
        for (int i = 0; i < vocab_size; i++) {
            const uint8_t *src = (const uint8_t *)vocab[i].text.ptr;
            size_t slen = vocab[i].text.len;
            if (slen > 0) {
                uint32_t cp;
                size_t consumed = (size_t)utf8_decode(src, slen, &cp);
                if (consumed == slen) {
                    int b = unicode_to_byte((int)cp);
                    if (b >= 0 && b < 256 && byte_to_token[b] < 0)
                        byte_to_token[b] = vocab[i].id;
                }
            }
        }
    }

    int raw_len = (int)strlen(text);

    /* Convert the raw UTF-8 input into the GPT-2 byte-level "surface" string:
     * every raw byte is remapped to its surface codepoint and re-encoded as
     * UTF-8. The vocab stores tokens in exactly this surface form (e.g. the
     * Chinese char "\xE4\xBD\xA0" is stored as three surface codepoints), so
     * matching must happen in surface space, not against the raw bytes. */
    char *surf = xmalloc((size_t)raw_len * 4 + 1);
    int surf_len = 0;
    for (int i = 0; i < raw_len; i++) {
        uint32_t cp = byte_to_unicode((uint8_t)text[i]);
        surf_len += (int)utf8_encode(cp, surf + surf_len);
    }
    surf[surf_len] = 0;

    int len = surf_len;
    const char *stext = surf;
    for (int i = 0; i < len; ) {
        int best_id = -1;
        int best_len = 1;
        for (int j = 0; j < vocab_size; j++) {
            hy3_str t = vocab[j].text;
            if (t.len > 0 && (size_t)(len - i) >= t.len &&
                memcmp(stext + i, t.ptr, t.len) == 0) {
                if ((int)t.len > best_len) {
                    best_len = (int)t.len;
                    best_id = vocab[j].id;
                }
            }
        }
        if (best_id < 0 && i < len) {
            unsigned char b = (unsigned char)stext[i];
            best_id = byte_to_token[b];
            if (best_id < 0) best_id = (int)b;
            best_len = 1;
        }
        i += best_len;
        if (best_id >= 0)
            hy3_tokens_push(out, best_id);
    }

    free(surf);
}

/* Hunyuan V3 special token IDs (verified against the GGUF vocab). These are
 * stored in the vocab as raw UTF-8 (with the fullwidth bar U+FF5C), NOT in the
 * byte-level surface encoding that hy3_tokenize applies, so they must be
 * injected by ID rather than tokenized from text. */
#define HY3_TOK_BOS            120000  /* <|hy_begin_of_sentence:opensource|> */
#define HY3_TOK_USER           120006  /* <|hy_User:opensource|> */
#define HY3_TOK_ASSISTANT      120007  /* <|hy_Assistant:opensource|> */
#define HY3_TOK_THINK_BEGIN    120029  /* <think:opensource> */
#define HY3_TOK_THINK_END      120030  /* </think:opensource> */
#define HY3_TOK_REASONING_MODE 120044  /* <|reasoning_mode:opensource|> */

/* Build the chat-formatted prompt for a single user turn, matching
 * tencent/Hy3 chat_template.jinja with the defaults used for inference:
 * no system prompt, no tools, add_generation_prompt=true, and
 * reasoning_effort="no_think" (so an empty <think></think> block is emitted
 * and the model answers directly).
 *
 * Rendered sequence:
 *   BOS REASONING_MODE "reasoning_effort:no_think" USER <prompt>
 *   ASSISTANT THINK_BEGIN THINK_END
 *
 * think controls the reasoning block: 0 = no_think (empty think block, direct
 * answer), 1 = low reasoning effort, 2 = high reasoning effort. For 1/2 only
 * THINK_BEGIN is emitted so generation starts inside the reasoning block. */
void hy3_tokenize_chat(hy3_model *m, const char *user_text, int think, hy3_tokens *out) {
    const char *effort = (think >= 2) ? "reasoning_effort:high"
                       : (think == 1) ? "reasoning_effort:low"
                                      : "reasoning_effort:no_think";
    hy3_tokens_push(out, HY3_TOK_BOS);
    hy3_tokens_push(out, HY3_TOK_REASONING_MODE);
    hy3_tokenize(m, effort, out);
    hy3_tokens_push(out, HY3_TOK_USER);
    hy3_tokenize(m, user_text, out);
    hy3_tokens_push(out, HY3_TOK_ASSISTANT);
    hy3_tokens_push(out, HY3_TOK_THINK_BEGIN);
    if (think == 0) hy3_tokens_push(out, HY3_TOK_THINK_END);
}


static int utf8_decode(const uint8_t *s, size_t n, uint32_t *cp) {
    if (n == 0) return 0;
    if (s[0] < 0x80) { *cp = s[0]; return 1; }
    if (s[0] < 0xC0) return 0; /* continuation byte as start */
    if (s[0] < 0xE0 && n >= 2) { *cp = (uint32_t)(s[0] & 0x1F) << 6 | (s[1] & 0x3F); return 2; }
    if (s[0] < 0xF0 && n >= 3) { *cp = (uint32_t)(s[0] & 0x0F) << 12 | (uint32_t)(s[1] & 0x3F) << 6 | (s[2] & 0x3F); return 3; }
    if (n >= 4) { *cp = (uint32_t)(s[0] & 0x07) << 18 | (uint32_t)(s[1] & 0x3F) << 12 | (uint32_t)(s[2] & 0x3F) << 6 | (s[3] & 0x3F); return 4; }
    return 0;
}

static int unicode_to_byte(uint32_t cp) {
    if (cp >= 0x21 && cp <= 0x7E) return (int)cp;
    if (cp >= 0xA1 && cp <= 0xAC) return (int)cp;
    if (cp >= 0xAE && cp <= 0xFF) return (int)cp;
    if (cp >= 0x100 && cp <= 0x143) {
        /* GPT-2 byte-level BPE mapping for bytes 0-32, 127, 128-159, 160, 173 */
        static const uint8_t byte_map[68] = {
            0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
            127,
            128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,
            151,152,153,154,155,156,157,158,159,
            160,
            173
        };
        return byte_map[cp - 0x100];
    }
    return -1;
}

int hy3_detokenize(hy3_model *m, int token, char *buf, size_t cap) {
    const hy3_vocab_entry *e = find_vocab_entry(token);
    if (e && e->text.ptr && cap > 0) {
        const uint8_t *src = (const uint8_t *)e->text.ptr;
        size_t slen = e->text.len;
        size_t spos = 0;
        size_t dpos = 0;

        while (spos < slen && dpos < cap - 1) {
            uint32_t cp;
            size_t consumed = (size_t)utf8_decode(src + spos, slen - spos, &cp);
            if (consumed == 0) { spos++; continue; }
            spos += consumed;

            int b = unicode_to_byte(cp);
            if (b >= 0) {
                if (dpos < cap - 1) buf[dpos++] = (char)b;
            } else {
                size_t i;
                for (i = 0; i < consumed && dpos < cap - 1; i++)
                    buf[dpos++] = (char)src[spos - consumed + i];
            }
        }
        buf[dpos] = 0;
        return (int)dpos;
    }
    if (cap > 0) buf[0] = 0;
    return 0;
}

int hy3_token_eos(hy3_model *m) { return 120025; }
int hy3_token_bos(hy3_model *m) { return 120000; }
