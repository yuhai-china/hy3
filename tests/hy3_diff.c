/* Diff harness: run ONE token (pos=0) through CPU then Metal, compare logits.
 * At pos=0 attention is trivial (1 token, softmax=1), so any divergence
 * isolates the bug to matmul / rms_norm / embed / MoE routing rather than
 * attention/rope. Usage: hy3_diff <model.gguf> */
#include "../hy3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HY3_METAL
int hy3_metal_init(hy3_model *m);
int hy3_eval_metal(hy3_model *m, const hy3_tokens *tokens, float *logits, int *pos);
#endif

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 1; }
    hy3_model *m = NULL;
    if (hy3_model_load(&m, argv[1], 8) != 0) { fprintf(stderr, "load failed\n"); return 1; }

    int V = HY3_N_VOCAB;
    float *lc = malloc(V * sizeof(float));
    float *lm = malloc(V * sizeof(float));

    int tok = 1013; /* first token of "11+22+33=?" */
    hy3_tokens single = { .v = &tok, .len = 1, .cap = 1 };
    int pos = 0;

    /* Optional per-layer CPU probe: replicate forward_model up to layer N. */
    const char *sl = getenv("HY3_STOP_LAYER");
    if (sl) {
        int stop = atoi(sl);
        /* embed lookup (F16 table) */
        extern float fp16_to_fp32_pub(unsigned short h);
        /* Manually do embed + N layers using public API. cache must be sized. */
        m->cache_len = 0;
        if (!m->cache_k) {
            int kv = HY3_N_KV_HEAD * HY3_HEAD_DIM;
            m->ctx_size = HY3_N_LAYER + 1024;
            m->cache_k = calloc((size_t)m->ctx_size * kv, sizeof(float));
            m->cache_v = calloc((size_t)m->ctx_size * kv, sizeof(float));
        }
        /* embed */
        {
            const unsigned short *tbl = (const unsigned short *)m->w.token_embd.data;
            const unsigned short *row = tbl + (size_t)tok * HY3_N_EMBD;
            for (int i = 0; i < HY3_N_EMBD; i++) {
                unsigned short h = row[i];
                unsigned int sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
                unsigned int f;
                if (exp == 0) { if (man == 0) f = sign << 31; else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3ff; f = (sign<<31)|(exp<<23)|(man<<13); } }
                else if (exp == 0x1f) f = (sign<<31)|(0xff<<23)|(man<<13);
                else f = (sign<<31)|((exp-15+127)<<23)|(man<<13);
                memcpy(&m->embed[i], &f, 4);
            }
        }
        for (int il = 0; il <= stop; il++) {
            if (il < HY3_N_LAYER_DENSE) forward_layer_dense(m, il, 0);
            else forward_layer_moe(m, il, 0);
        }
        printf("CPU  after layer %d embed[0..3]: %.4f %.4f %.4f %.4f\n",
               stop, m->embed[0], m->embed[1], m->embed[2], m->embed[3]);
    }

    /* CPU path (metal_ctx is NULL so hy3_eval runs on CPU) */
    m->cache_len = 0;
    hy3_eval(m, &single, lc, &pos);
    printf("CPU  post-norm embed[0..3]: %.4f %.4f %.4f %.4f\n",
           m->embed[0], m->embed[1], m->embed[2], m->embed[3]);
    printf("CPU  logits[0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.4f", lc[i]);
    printf("\n");

#ifdef HY3_METAL
    if (hy3_metal_init(m) != 0) { fprintf(stderr, "metal init failed\n"); return 1; }
    m->cache_len = 0;
    hy3_eval_metal(m, &single, lm, &pos);
    printf("META logits[0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.4f", lm[i]);
    printf("\n");

    double max_abs = 0, sum_sq = 0;
    int argmax_c = 0, argmax_m = 0;
    for (int i = 0; i < V; i++) {
        double d = fabs((double)lc[i] - (double)lm[i]);
        if (d > max_abs) max_abs = d;
        sum_sq += d * d;
        if (lc[i] > lc[argmax_c]) argmax_c = i;
        if (lm[i] > lm[argmax_m]) argmax_m = i;
    }
    printf("max|Δ|=%.5f  rmse=%.5f  argmax cpu=%d meta=%d\n",
           max_abs, sqrt(sum_sq / V), argmax_c, argmax_m);
#endif
    return 0;
}
