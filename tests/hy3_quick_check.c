/* Fast, minimal quality check: 3 short prompts, greedy decode, small token
 * budgets. Not a substitute for the full hy3_eval_gpu suite -- just a quick
 * signal that a GGUF conversion / inference-engine change didn't break
 * basic capability, without waiting on long reasoning questions.
 * Usage: hy3_quick_check <model.gguf> [n_threads]
 */
#include "../hy3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ask(hy3_model *m, const char *prompt, int n_predict) {
    hy3_tokens input = {0};
    hy3_tokenize(m, prompt, &input);
    float *logits = malloc(120832 * sizeof(float));
    int pos = 0;
    hy3_tokens single;
    single.v = &input.v[0]; single.len = 1; single.cap = 1;
    hy3_eval(m, &single, logits, &pos);
    for (int i = 1; i < input.len; i++) { single.v = &input.v[i]; hy3_eval(m, &single, logits, &pos); }

    char out[4096]; int opos = 0; out[0] = 0;
    for (int i = 0; i < n_predict; i++) {
        int best = 0; float bv = -1e30f;
        for (int v = 0; v < 120818; v++) if (logits[v] > bv) { bv = logits[v]; best = v; }
        if (best == hy3_token_eos(m)) break;
        char buf[256];
        int tlen = hy3_detokenize(m, best, buf, sizeof(buf));
        if (opos + tlen < (int)sizeof(out) - 1) { memcpy(out+opos, buf, tlen); opos += tlen; out[opos] = 0; }
        single.v = &best;
        hy3_eval(m, &single, logits, &pos);
    }
    printf(">>> %s\n<<< %s\n\n", prompt, out);
    fflush(stdout);
    free(logits);
    hy3_tokens_free(&input);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [n_threads]\n", argv[0]); return 1; }
    int n_threads = argc > 2 ? atoi(argv[2]) : 64;
    hy3_model *m = NULL;
    if (hy3_model_load(&m, argv[1], n_threads) != 0) { fprintf(stderr, "load failed\n"); return 1; }

    ask(m, "11+22+33=?", 12);
    m->cache_len = 0;
    ask(m, "The capital of France is", 8);
    m->cache_len = 0;
    ask(m, "Find the sum of all integer bases b>9 for which 17_b is a divisor of 97_b.\n"
           "Solve. End with exactly: Answer: <integer>", 200);
    m->cache_len = 0;
    ask(m, "Q: What is 7 * 8?\nA:", 8);

    hy3_model_free(m);
    return 0;
}
