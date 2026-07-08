#ifndef HY3_CUDA_H
#define HY3_CUDA_H

#include "hy3.h"

typedef struct hy3_model hy3_model;

typedef struct {
    float *d;
    int size;
    int cap;
} gpu_buf;

typedef struct {
    float *token_embd;
    float *output_norm;
    float *output;

    float *attn_q;
    float *attn_k;
    float *attn_v;
    float *attn_output;
    float *attn_q_norm;
    float *attn_k_norm;
    float *attn_norm;
    float *ffn_norm;
    float *ffn_gate;
    float *ffn_up;
    float *ffn_down;
    float *ffn_gate_inp;
    float *ffn_gate_shexp;
    float *ffn_up_shexp;
    float *ffn_down_shexp;
    float *ffn_gate_exps[192];
    float *ffn_up_exps[192];
    float *ffn_down_exps[192];
    float *eh_proj;
    float *enorm;
    float *hnorm;
    float *final_norm;
} gpu_weights;

typedef struct {
    gpu_weights w;
    gpu_buf embed;
    gpu_buf scratch;
    gpu_buf scratch2;
    gpu_buf k_cache;
    gpu_buf v_cache;
    gpu_buf logits;
    int cache_len;
    int ctx_cap;
} gpu_ctx;

int hy3_gpu_init(hy3_model *m);
void hy3_gpu_free(hy3_model *m);
int hy3_gpu_eval(hy3_model *m, int token, float *logits, int *pos);

#endif
