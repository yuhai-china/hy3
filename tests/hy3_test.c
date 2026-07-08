#include "../hy3.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int n_pass = 0, n_fail = 0;
static int screen_width = 80;

#define MODEL_PATH "/root/models/Hy3/hy3_q8.gguf"

static void draw_line(char c) {
    for (int i = 0; i < screen_width; i++) fputc(c, stderr);
    fputc('\n', stderr);
}

#define TEST(fmt, ...) do { \
    draw_line('='); \
    fprintf(stderr, "  TEST: " fmt "\n", ##__VA_ARGS__); \
    draw_line('='); \
} while(0)

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  \x1b[31mFAIL\x1b[0m %s\n", msg); \
        n_fail++; \
    } else { \
        fprintf(stderr, "  \x1b[32mPASS\x1b[0m %s\n", msg); \
        n_pass++; \
    } \
} while(0)

static int count_words(const char *s) {
    int n = 0, in = 0;
    for (; *s; s++) {
        if (*s > ' ' && !in) { n++; in = 1; }
        else if (*s <= ' ') in = 0;
    }
    return n;
}

static int all_same_token(const int *toks, int n) {
    if (n <= 1) return 0;
    for (int i = 1; i < n; i++)
        if (toks[i] != toks[0]) return 0;
    return 1;
}

static void test_deterministic(hy3_model *m) {
    TEST("Deterministic output at temperature 0.0");
    
    float *logits1 = malloc(120832 * sizeof(float));
    float *logits2 = malloc(120832 * sizeof(float));
    hy3_tokens input = {0};
    hy3_tokens_push(&input, hy3_token_bos(m));
    
    int pos = 0;
    hy3_eval(m, &input, logits1, &pos);
    
    m->cache_len = 0;
    pos = 0;
    hy3_eval(m, &input, logits2, &pos);
    
    int match = 1;
    float max_diff = 0;
    for (int i = 0; i < 120832; i++) {
        float d = fabsf(logits1[i] - logits2[i]);
        if (d > max_diff) max_diff = d;
        if (fabsf(logits1[i] - logits2[i]) > 1e-4f) match = 0;
    }
    CHECK(match, "Logits deterministic across runs");
    fprintf(stderr, "  max diff between runs: %.6f\n", max_diff);
    
    // Check no NaN/inf in logits
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < 120832; i++) {
        if (isnan(logits1[i])) nan_count++;
        if (isinf(logits1[i])) inf_count++;
    }
    CHECK(nan_count == 0, "No NaN in logits");
    CHECK(inf_count == 0, "No Inf in logits");
    
    // Check logits have reasonable range
    float min_l = FLT_MAX, max_l = -FLT_MAX;
    for (int i = 0; i < 120832; i++) {
        if (logits1[i] < min_l) min_l = logits1[i];
        if (logits1[i] > max_l) max_l = logits1[i];
    }
    fprintf(stderr, "  logit range: [%.2f, %.2f]\n", min_l, max_l);
    CHECK(max_l > -50 && max_l < 100, "Logits in reasonable range");
    
    // Check top tokens are varied
    int top5[5]; float top5v[5];
    for (int j = 0; j < 5; j++) { top5v[j] = -1e30f; top5[j] = 0; }
    for (int i = 0; i < 120818; i++) {
        float v = logits1[i];
        for (int j = 0; j < 5; j++) {
            if (v > top5v[j]) {
                for (int k = 4; k > j; k--) { top5v[k] = top5v[k-1]; top5[k] = top5[k-1]; }
                top5v[j] = v; top5[j] = i; break;
            }
        }
    }
    fprintf(stderr, "  top5 tokens: ");
    for (int j = 0; j < 5; j++) {
        char buf[256];
        hy3_detokenize(m, top5[j], buf, sizeof(buf));
        fprintf(stderr, "%d:'%s'(%.2f)%s", top5[j], buf, top5v[j], j < 4 ? ", " : "");
    }
    fprintf(stderr, "\n");
    
    // Check top5 tokens are different from each other
    int all_diff = 1;
    for (int i = 0; i < 5; i++)
        for (int j = i+1; j < 5; j++)
            if (top5[i] == top5[j]) all_diff = 0;
    CHECK(all_diff, "Top-5 tokens are distinct");
    CHECK(top5v[0] > top5v[4] - 30, "Top-1 not unreasonably far from top-5");
    
    free(logits1);
    free(logits2);
    hy3_tokens_free(&input);
}

static void test_generation(hy3_model *m, const char *prompt, int n_tokens) {
    fprintf(stderr, "\n");
    TEST("Generation: \"%s\"", prompt);
    
    hy3_tokens input = {0};
    hy3_tokenize(m, prompt, &input);
    fprintf(stderr, "  prompt tokens: %d\n", input.len);
    
    float *logits = malloc(120832 * sizeof(float));
    int pos = 0;
    
    // Prefill all prompt tokens
    hy3_tokens first;
    first.v = &input.v[0];
    first.len = 1;
    first.cap = 1;
    hy3_eval(m, &first, logits, &pos);
    
    for (int i = 1; i < input.len; i++) {
        hy3_tokens single;
        single.v = &input.v[i];
        single.len = 1;
        single.cap = 1;
        hy3_eval(m, &single, logits, &pos);
    }
    
    // Generate tokens greedily
    int generated[256];
    int n_gen = n_tokens > 256 ? 256 : n_tokens;
    
    for (int i = 0; i < n_gen; i++) {
        // greedy: pick top-1
        int best = 0;
        float best_v = -FLT_MAX;
        for (int v = 0; v < 120818; v++) {
            if (logits[v] > best_v) { best_v = logits[v]; best = v; }
        }
        generated[i] = best;
        
        if (best == hy3_token_eos(m)) {
            n_gen = i + 1;
            break;
        }
        
        hy3_tokens single;
        single.v = &best;
        single.len = 1;
        single.cap = 1;
        hy3_eval(m, &single, logits, &pos);
    }
    
    // Print generated text
    char line[4096];
    int line_pos = 0;
    line[0] = 0;
    for (int i = 0; i < n_gen; i++) {
        char buf[256];
        hy3_detokenize(m, generated[i], buf, sizeof(buf));
        int blen = strlen(buf);
        if (line_pos + blen >= (int)sizeof(line) - 1) break;
        memcpy(line + line_pos, buf, blen);
        line_pos += blen;
        line[line_pos] = 0;
    }
    fprintf(stderr, "  output (%d tokens): \"%s\"\n", n_gen, line);
    
    // Check output validity
    CHECK(n_gen > 0, "Generated at least one token");
    CHECK(!all_same_token(generated, n_gen), "Output not stuck in repetition loop");
    
    // Check top-5 logits on last step
    int top5[5]; float top5v[5];
    for (int j = 0; j < 5; j++) { top5v[j] = -1e30f; top5[j] = 0; }
    for (int i = 0; i < 120818; i++) {
        float v = logits[i];
        for (int j = 0; j < 5; j++) {
            if (v > top5v[j]) {
                for (int k = 4; k > j; k--) { top5v[k] = top5v[k-1]; top5[k] = top5[k-1]; }
                top5v[j] = v; top5[j] = i; break;
            }
        }
    }
    fprintf(stderr, "  last-step top5: ");
    for (int j = 0; j < 5; j++) {
        char buf[256];
        hy3_detokenize(m, top5[j], buf, sizeof(buf));
        fprintf(stderr, "%d:'%s'(%.1f)%s", top5[j], buf, top5v[j], j < 4 ? ", " : "");
        if (j == 2) fprintf(stderr, "\n    ");
    }
    fprintf(stderr, "\n");
    
    free(logits);
    hy3_tokens_free(&input);
}

static void test_repetition_loop(hy3_model *m) {
    TEST("Repetition loop detection");
    
    // Generate 30 tokens and check no repeated pattern
    float *logits = malloc(120832 * sizeof(float));
    int pos = 0;
    
    hy3_tokens input;
    input.v = NULL; input.len = 0; input.cap = 0;
    hy3_tokens_push(&input, hy3_token_bos(m));
    
    hy3_tokens first;
    first.v = &input.v[0];
    first.len = 1;
    first.cap = 1;
    hy3_eval(m, &first, logits, &pos);
    
    int generated[64];
    for (int i = 0; i < 30; i++) {
        int best = 0;
        float best_v = -FLT_MAX;
        for (int v = 0; v < 120818; v++) {
            if (logits[v] > best_v) { best_v = logits[v]; best = v; }
        }
        generated[i] = best;
        if (best == hy3_token_eos(m)) break;
        hy3_tokens single;
        single.v = &best;
        single.len = 1;
        single.cap = 1;
        hy3_eval(m, &single, logits, &pos);
    }
    
    // Check no 3+ consecutive identical tokens
    int rep = 0;
    for (int i = 2; i < 30; i++)
        if (generated[i] == generated[i-1] && generated[i] == generated[i-2]) rep = 1;
    CHECK(!rep, "No triple-repeated tokens (repetition loop)");
    
    free(logits);
    hy3_tokens_free(&input);
}

int main() {
    screen_width = 80;
    fprintf(stderr, "\n");
    draw_line('#');
    fprintf(stderr, "  HY3 Model Verification Tests\n");
    fprintf(stderr, "  Model: %s\n", MODEL_PATH);
    draw_line('#');
    fprintf(stderr, "\n");
    
    hy3_model *m = NULL;
    int ret = hy3_model_load(&m, MODEL_PATH, 4);
    CHECK(ret == 0, "Model loaded successfully");
    if (!m || ret != 0) {
        fprintf(stderr, "FATAL: cannot load model, aborting tests\n");
        return 1;
    }
    
    // Deterministic output test
    test_deterministic(m);
    
    // Reset model state
    m->cache_len = 0;
    
    // Repetition loop test
    test_repetition_loop(m);
    
    // Reset model state
    m->cache_len = 0;
    
    // Generation tests on diverse prompts
    test_generation(m, "Hello", 20);
    m->cache_len = 0;
    
    test_generation(m, "The meaning of life is", 20);
    m->cache_len = 0;
    
    test_generation(m, "def fibonacci(n):", 20);
    m->cache_len = 0;
    
    test_generation(m, "What is 2+2?", 20);
    m->cache_len = 0;
    
    test_generation(m, "Python is a", 20);
    m->cache_len = 0;
    
    test_generation(m, "Once upon a time", 20);
    m->cache_len = 0;
    
    test_generation(m, "自然界中", 20);
    m->cache_len = 0;
    
    test_generation(m, "今天天气真", 20);
    m->cache_len = 0;
    
    // Summary
    draw_line('=');
    fprintf(stderr, "  RESULTS: %d PASS, %d FAIL out of %d tests\n",
            n_pass, n_fail, n_pass + n_fail);
    draw_line('=');
    
    hy3_model_free(m);
    return n_fail > 0 ? 1 : 0;
}
