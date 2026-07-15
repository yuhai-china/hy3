#ifndef HY3_H
#define HY3_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __CUDACC__
#define HY3_CUDA_ENABLED 1
#else
#define HY3_CUDA_ENABLED 0
#endif

#ifdef HY3_METAL
#define HY3_METAL_ENABLED 1
#else
#define HY3_METAL_ENABLED 0
#endif

typedef struct hy3_model hy3_model;

typedef struct {
    int *v;
    int len;
    int cap;
} hy3_tokens;

#define HY3_DEFAULT_TEMPERATURE 0.9f
#define HY3_DEFAULT_TOP_P 1.0f

#define HY3_N_LAYER 80
#define HY3_N_EMBD 4096
#define HY3_N_VOCAB 120832
#define HY3_N_VOCAB_VALID 120818
#define HY3_N_HEAD 64
#define HY3_N_KV_HEAD 8
#define HY3_HEAD_DIM 128
#define HY3_N_EXPERT 192
#define HY3_N_EXPERT_USED 8
#define HY3_N_SHARED 1
#define HY3_MOE_INTERMED 1536
#define HY3_DENSE_INTERMED 13312
#define HY3_N_LAYER_DENSE 1

typedef struct {
    const char *model_path;
    int n_threads;
    int ctx_size;
    float temperature;
    float top_p;
    int top_k;
    bool verbose;
    bool use_gpu;
    int gpu_layers;
} hy3_params;

int hy3_model_load(hy3_model **out, const char *path, int n_threads);
void hy3_model_free(hy3_model *m);
int hy3_model_vocab_size(hy3_model *m);
const char *hy3_model_name(hy3_model *m);
int hy3_model_ctx_size(hy3_model *m);

void hy3_tokenize(hy3_model *m, const char *text, hy3_tokens *out);
void hy3_tokenize_chat(hy3_model *m, const char *user_text, int think, hy3_tokens *out);
void hy3_chat_append_user(hy3_model *m, hy3_tokens *conv, const char *user_text, int think, int is_first);
int hy3_detokenize(hy3_model *m, int token, char *buf, size_t cap);
int hy3_token_eos(hy3_model *m);
int hy3_token_bos(hy3_model *m);

void hy3_tokens_push(hy3_tokens *tv, int token);
void hy3_tokens_unshift(hy3_tokens *tv, int token);
void hy3_tokens_free(hy3_tokens *tv);

void hy3_rope_init(hy3_model *m);
void hy3_rope_get_params(const hy3_model *m, float *inv_freq_out, float *attn_factor_out);

int hy3_eval(hy3_model *m, const hy3_tokens *tokens, float *logits, int *pos);
int hy3_eval_gpu(hy3_model *m, const hy3_tokens *tokens, float *logits, int *pos);
int hy3_eval_metal(hy3_model *m, const hy3_tokens *tokens, float *logits, int *pos);
int hy3_sample(hy3_model *m, const float *logits, float temperature, int top_k, float top_p);
void hy3_reset_context(hy3_model *m);

void forward_layer_dense(hy3_model *m, int il, int pos);
void forward_layer_moe(hy3_model *m, int il, int pos);

int hy3_generate(hy3_model *m, const hy3_tokens *prompt,
                 int n_predict, hy3_params *params,
                 int (*emit)(void *ud, int token),
                 void *emit_ud);

/* Weight / tensor info exposed for GPU integration */
typedef struct {
    const char *ptr;
    uint64_t len;
} hy3_str;

typedef struct {
    hy3_str key;
    uint32_t type;
    uint64_t value_pos;
} hy3_kv;

typedef struct {
    hy3_str name;
    uint32_t ndim;
    uint64_t dim[8];
    uint32_t ggml_type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} hy3_tensor_info;

typedef struct {
    const hy3_tensor_info *t;
    const uint8_t *data;
} hy3_weight;

typedef struct {
    hy3_weight token_embd;

    hy3_weight attn_norm;
    hy3_weight attn_q;
    hy3_weight attn_k;
    hy3_weight attn_v;
    hy3_weight attn_output;
    hy3_weight attn_q_norm;
    hy3_weight attn_k_norm;

    hy3_weight ffn_norm;
    hy3_weight ffn_gate;    // dense
    hy3_weight ffn_up;
    hy3_weight ffn_down;

    hy3_weight ffn_gate_inp;  // router
    hy3_weight ffn_gate_exps[192];
    hy3_weight ffn_up_exps[192];
    hy3_weight ffn_down_exps[192];
    hy3_weight ffn_gate_shexp;
    hy3_weight ffn_up_shexp;
    hy3_weight ffn_down_shexp;
    float      expert_bias[192];
    bool       has_expert_bias;

    hy3_weight eh_proj;
    hy3_weight enorm;
    hy3_weight hnorm;
    hy3_weight final_norm;
} hy3_layer_weights;

typedef struct {
    hy3_weight token_embd;
    hy3_weight output_norm;
    hy3_weight output;
    hy3_layer_weights layers[82];
    int n_layer;
} hy3_weights;

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    hy3_kv *kv;
    hy3_tensor_info *tensors;
} hy3_gguf_model;

struct hy3_model {
    hy3_gguf_model gguf;
    hy3_weights w;
    int n_threads;
    uint64_t rng_state;
    double t_load;
    int ctx_size;
    int n_expert_used;   /* runtime top-k for MoE routing; default HY3_N_EXPERT_USED, clamped to [1, HY3_N_EXPERT_USED] */

    /* RoPE frequency table + YaRN long-context extrapolation.
     * rope_inv_freq[d] = 1/theta^(2d/head_dim) for the default rope_type, or
     * the YaRN-interpolated per-dim frequency when long-context scaling is
     * enabled. rope_attn_factor is YaRN's mscale (1.0 when disabled). Resolved
     * once in hy3_rope_init() and shared with the CUDA/Metal backends via
     * hy3_rope_get_params(). Default (no env/metadata scaling) reproduces the
     * model's native rope exactly. */
    float rope_inv_freq[HY3_HEAD_DIM / 2];
    float rope_attn_factor;
    int   rope_yarn;
    int   rope_orig_ctx;
    float rope_factor;
    int   rope_pos_stride;   /* HY3_POS_STRIDE: RoPE position multiplier (1=normal) */

    float *embed;
    float *cache_k;
    float *cache_v;
    int cache_len;

    /* Prefix caching (multi-turn): token ids whose KV is currently resident, in
     * order. cache_ntok == cache_len / HY3_N_LAYER. On each generate() the
     * longest common prefix with the new prompt is reused (cache rolled back to
     * it) so only the divergent suffix is prefilled. */
    int *cache_tokens;
    int  cache_ntok;
    int  cache_tok_cap;

    float *scratch;
    float *scratch2;

    void *gpu_ctx;
    int gpu_layers;

    void *metal_ctx;
};

#endif
