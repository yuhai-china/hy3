#include "../hy3.h"
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

static char answer_letter(const char *out, int nchoices) {
    const char *ans = strstr(out, "Answer:");
    if (ans) {
        const char *p = ans + 7;
        while (*p == ' ') p++;
        char c = toupper((unsigned char)*p);
        if (c >= 'A' && c <= (char)('A'+nchoices-1)) return c;
    }
    for (ans = out + strlen(out); ans > out; ans--)
        if (toupper((unsigned char)ans[-1]) >= 'A' && toupper((unsigned char)ans[-1]) <= (char)('A'+nchoices-1))
            return (char)toupper((unsigned char)ans[-1]);
    return '?';
}

int main() {
    hy3_model *m = NULL;
    if (hy3_model_load(&m, "/root/models/Hy3/hy3_q8.gguf", 4)) return 1;
    
    /* Question 1: AIME integer answer */
    const char *q1 = "Find the sum of all integer bases b>9 for which 17_b is a divisor of 97_b.\n\nSolve the problem. At the end, write exactly one final line: Answer: <integer>";
    hy3_tokens input = {0};
    hy3_tokenize(m, q1, &input);
    fprintf(stderr, "Q1 prompt: %d tokens\n", input.len);
    
    float *logits = malloc(120832 * sizeof(float));
    int pos = 0;
    hy3_tokens single = { .v = &input.v[0], .len = 1, .cap = 1 };
    hy3_eval(m, &single, logits, &pos);
    for (int i = 1; i < input.len; i++) { single.v = &input.v[i]; hy3_eval(m, &single, logits, &pos); }
    
    char out[2048]; int opos = 0;
    for (int i = 0; i < 30; i++) {
        int best = 0;
        float bv = -1e30f;
        for (int v = 0; v < 120818; v++) if (logits[v] > bv) { bv = logits[v]; best = v; }
        if (best == hy3_token_eos(m)) break;
        char buf[256];
        int tlen = hy3_detokenize(m, best, buf, sizeof(buf));
        if (opos + tlen < 2047) { memcpy(out+opos, buf, tlen); opos += tlen; out[opos] = 0; }
        single.v = &best;
        hy3_eval(m, &single, logits, &pos);
    }
    fprintf(stderr, "Q1 output: ");
    for (int i = 0; out[i]; i++) fputc(out[i] >= 32 ? out[i] : '.', stderr);
    fputc('\n', stderr);
    
    /* Check */
    char *a = strstr(out, "Answer:");
    int q1_pass = a && strstr(a+7, "70");
    fprintf(stderr, "Q1: %s (expected: 70)\n", q1_pass ? "PASS" : "FAIL");
    
    /* Question 2: Multiple choice */
    m->cache_len = 0;
    hy3_tokens_free(&input);
    
    const char *q2 = "An intelligent civilization travels at 0.99999987*c. "
        "How long in the astronaut's frame to reach Earth? "
        "Choices: A. 0.22 years B. 7 days C. 35 days D. 2 months E. 0.5 year\n"
        "Solve. End with: Answer: <letter>";
    hy3_tokenize(m, q2, &input);
    fprintf(stderr, "\nQ2 prompt: %d tokens\n", input.len);
    
    pos = 0;
    single.v = &input.v[0]; hy3_eval(m, &single, logits, &pos);
    for (int i = 1; i < input.len; i++) { single.v = &input.v[i]; hy3_eval(m, &single, logits, &pos); }
    
    opos = 0; out[0] = 0;
    for (int i = 0; i < 20; i++) {
        int best = 0; float bv = -1e30f;
        for (int v = 0; v < 120818; v++) if (logits[v] > bv) { bv = logits[v]; best = v; }
        if (best == hy3_token_eos(m)) break;
        char buf[256];
        int tlen = hy3_detokenize(m, best, buf, sizeof(buf));
        if (opos + tlen < 2047) { memcpy(out+opos, buf, tlen); opos += tlen; out[opos] = 0; }
        single.v = &best;
        hy3_eval(m, &single, logits, &pos);
    }
    fprintf(stderr, "Q2 output: ");
    for (int i = 0; out[i]; i++) fputc(out[i] >= 32 ? out[i] : '.', stderr);
    fputc('\n', stderr);
    
    char got = answer_letter(out, 5);
    int q2_pass = got == 'B';
    fprintf(stderr, "Q2: %s (got=%c, expected=B)\n", q2_pass ? "PASS" : "FAIL", got);
    
    fprintf(stderr, "\nSummary: Q1=%s Q2=%s\n", q1_pass ? "PASS" : "FAIL", q2_pass ? "PASS" : "FAIL");
    
    free(logits);
    hy3_tokens_free(&input);
    hy3_model_free(m);
    return (q1_pass && q2_pass) ? 0 : 1;
}
