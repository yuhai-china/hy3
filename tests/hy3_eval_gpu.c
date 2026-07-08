/* hy3_eval_gpu.c - Quick GPU-accelerated quality eval, adapted from
 * ds4-eval's approach: a handful of embedded reasoning questions (GPQA
 * Diamond / SuperGPQA / AIME 2025), greedy-decoded, answer-extracted and
 * graded against a known key. Intended as a fast smoke test that a GGUF
 * conversion / inference-engine change didn't break model quality --
 * not a substitute for a full benchmark run.
 *
 * Usage: hy3_eval_gpu [gpu_layers] [model.gguf]
 *   gpu_layers  number of layers to offload to GPU (0 = CPU only). Default 0.
 *   model.gguf  path to the GGUF to test. Default /root/models/Hy3/hy3_q8.gguf.
 */
#include "../hy3.h"
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

int hy3_gpu_init(hy3_model *m, int n_gpu_layers);

#define EVAL_MAX_CHOICES 10

typedef struct {
    const char *source;
    const char *domain;
    const char *question;
    const char *choice[EVAL_MAX_CHOICES];
    const char *answer;
    int nchoices;
} eval_case;

static const eval_case eval_cases[] = {
    {
        .source = "GPQA Diamond", .domain = "Physics",
        .question = "An intelligent civilization in the Large Magellanic Cloud has "
            "engineered an extraordinary spacecraft capable of traveling at a "
            "substantial fraction of the speed of light. The average lifetime of "
            "these aliens is roughly 150 solar years. Now, having Earth as their "
            "destination in mind, they are determined to travel with this spacecraft "
            "at a constant speed of 0.99999987*c, where c is the speed of light. "
            "Approximately, how long will it take for their 22 years old astronaut "
            "(from the point of view of the astronaut) to reach the Earth using this "
            "incredibly fast spacecraft?",
        .choice = {"0.22 years", "7 days", "35 days", "2 months", "0.5 year"},
        .nchoices = 5, .answer = "B",
    },
    {
        .source = "SuperGPQA", .domain = "Engineering",
        .question = "Given a brick masonry compression member with e/h=0.18, "
            "height-to-thickness ratio beta=17, seismic fortification intensity of "
            "7 degrees, and Class II site soil, please select a masonry structural "
            "component plan that is safe and reliable, economically reasonable, "
            "and convenient for construction.",
        .choice = {"MU10 brick, M5 mortar", "MU15 brick, M5 mortar", "MU20 brick, M5 mortar",
                   "MU10 brick, M7.5 mortar", "MU15 brick, M7.5 mortar", "MU20 brick, M7.5 mortar",
                   "MU10 brick, M10 mortar", "MU15 brick, M10 mortar", "MU20 brick, M10 mortar",
                   "None of the above"},
        .nchoices = 10, .answer = "J",
    },
    {
        .source = "AIME 2025", .domain = "Math",
        .question = "Find the sum of all integer bases b>9 for which 17_b is a divisor of 97_b.",
        .nchoices = 0, .answer = "70",
    },
    {
        .source = "AIME 2025", .domain = "Math",
        .question = "Six points A, B, C, D, E and F lie in a straight line in that order. "
            "Suppose that G is a point not on the line and that AC=26, BD=22, CE=31, "
            "DF=33, AF=73, CG=40, and DG=30. Find the area of triangle BGE.",
        .nchoices = 0, .answer = "468",
    },
    {
        .source = "AIME 2025", .domain = "Math",
        .question = "The 9 members of a baseball team went to an ice-cream parlor after "
            "their game. Each player had a single-scoop cone of chocolate, vanilla, or "
            "strawberry ice cream. At least one player chose each flavor, and the number "
            "of players who chose chocolate was greater than the number of players who "
            "chose vanilla, which was greater than the number of players who chose "
            "strawberry. Let N be the number of different assignments of flavors to "
            "players that meet these conditions. Find the remainder when N is divided by 1000.",
        .nchoices = 0, .answer = "16",
    },
    {
        .source = "Arithmetic", .domain = "Math",
        .question = "11+22+33=?",
        .nchoices = 0, .answer = "66",
    },
};

static int ncases = sizeof(eval_cases) / sizeof(eval_cases[0]);
static int n_pass = 0, n_fail = 0, n_skip = 0;

static void draw_line(char c) { for (int i = 0; i < 72; i++) fputc(c, stderr); fputc('\n', stderr); }

static char *build_question_prompt(const eval_case *tc) {
    size_t cap = 4096;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    int pos = snprintf(buf, cap, "%s\n", tc->question);
    if (tc->nchoices > 0) {
        pos += snprintf(buf + pos, cap - (size_t)pos, "\nChoices:\n");
        for (int i = 0; i < tc->nchoices && i < EVAL_MAX_CHOICES; i++)
            pos += snprintf(buf + pos, cap - (size_t)pos, "%c. %s\n", 'A' + i, tc->choice[i]);
        pos += snprintf(buf + pos, cap - (size_t)pos,
            "\nSolve the question. At the end, write exactly one final line in this "
            "format and do not write anything after it:\nAnswer: <letter>");
    } else {
        pos += snprintf(buf + pos, cap - (size_t)pos,
            "\nSolve the problem. At the end, write exactly one final line in this "
            "format and do not write anything after it:\nAnswer: <integer>");
    }
    return buf;
}

static char find_answer_letter(const char *generated, int nchoices) {
    if (nchoices <= 0) return '?';
    char max_answer = (char)('A' + nchoices - 1);
    const char *ans = strstr(generated, "Answer:");
    if (ans) {
        const char *p = ans + 7;
        while (*p == ' ' || *p == '\t') p++;
        char c = (char)toupper((unsigned char)*p);
        if (c >= 'A' && c <= max_answer) return c;
    }
    ans = strstr(generated, "answer is");
    if (ans) {
        const char *p = ans + 9;
        while (*p == ' ' || *p == '\t') p++;
        char c = (char)toupper((unsigned char)*p);
        if (c >= 'A' && c <= max_answer) return c;
    }
    char last = '?';
    for (const char *p = generated; *p; p++) {
        char c = (char)toupper((unsigned char)*p);
        if (c >= 'A' && c <= max_answer) last = c;
    }
    return last;
}

static void normalize_integer_answer(const char *p, size_t len, char *dst, size_t dstlen) {
    while (len > 1 && *p == '0') { p++; len--; }
    if (dstlen == 0) return;
    size_t n = len < dstlen - 1 ? len : dstlen - 1;
    memcpy(dst, p, n);
    dst[n] = '\0';
}

static void find_integer_answer(const char *generated, char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    snprintf(dst, dstlen, "?");
    const char *ans = strstr(generated, "Answer:");
    if (ans) {
        const char *p = ans + 7;
        while (*p == ' ' || *p == '\t') p++;
        if (isdigit((unsigned char)*p) || (*p == '-' && isdigit((unsigned char)p[1]))) {
            const char *start = p, *end = p;
            if (*end == '-') end++;
            while (isdigit((unsigned char)*end)) end++;
            normalize_integer_answer(start, (size_t)(end - start), dst, dstlen);
            return;
        }
    }
    const char *last_start = NULL, *last_end = NULL;
    for (const char *p = generated; *p; p++) {
        if (isdigit((unsigned char)*p)) {
            last_start = p;
            while (isdigit((unsigned char)*p)) p++;
            last_end = p;
            p--;
        }
    }
    if (last_start && last_end) normalize_integer_answer(last_start, (size_t)(last_end - last_start), dst, dstlen);
}

static bool answer_matches(const eval_case *tc, const char *got) {
    if (tc->nchoices > 0) return got && got[0] && tc->answer && got[0] == tc->answer[0];
    char expected[64];
    normalize_integer_answer(tc->answer, strlen(tc->answer), expected, sizeof(expected));
    return got && strcmp(got, expected) == 0;
}

static void run_case(hy3_model *m, const eval_case *tc, int idx) {
    draw_line('-');
    fprintf(stderr, "  [%d/%d] %s | %s\n", idx + 1, ncases, tc->source, tc->domain);
    draw_line('-');

    char *prompt = build_question_prompt(tc);
    if (!prompt) { fprintf(stderr, "  OOM\n"); n_skip++; return; }

    hy3_tokens input = {0};
    hy3_tokenize(m, prompt, &input);
    fprintf(stderr, "  prompt: %d tokens\n", input.len);

    float *logits = malloc(120832 * sizeof(float));
    int pos = 0;
    hy3_tokens single;
    single.v = &input.v[0]; single.len = 1; single.cap = 1;
    hy3_eval(m, &single, logits, &pos);
    for (int i = 1; i < input.len; i++) { single.v = &input.v[i]; hy3_eval(m, &single, logits, &pos); }

    char output[8192];
    int opos = 0;
    output[0] = '\0';

    double t0 = 0;
    for (int i = 0; i < 256; i++) {
        int best = 0;
        float best_v = -1e30f;
        for (int v = 0; v < 120818; v++) if (logits[v] > best_v) { best_v = logits[v]; best = v; }
        if (best == hy3_token_eos(m)) break;

        char tok[256];
        int tlen = hy3_detokenize(m, best, tok, sizeof(tok));
        if (opos + tlen < (int)sizeof(output) - 1) {
            memcpy(output + opos, tok, (size_t)tlen);
            opos += tlen;
            output[opos] = '\0';
        }

        if (tc->nchoices > 0) {
            const char *a = strstr(output, "Answer:");
            if (a) { const char *p = a + 7; while (*p == ' ') p++; if (isalpha((unsigned char)*p) || *p == '\n') break; }
        } else {
            if (strstr(output, "Answer:")) {
                const char *a = strstr(output, "Answer:");
                const char *p = a + 7; while (*p == ' ') p++;
                if (isdigit((unsigned char)*p)) break;
            }
        }
        single.v = &best;
        hy3_eval(m, &single, logits, &pos);
    }
    (void)t0;

    fprintf(stderr, "  output: ");
    for (int i = 0; output[i]; i++) {
        fputc(output[i] >= 32 && output[i] < 127 ? output[i] : '.', stderr);
        if (i >= 200) { fprintf(stderr, "..."); break; }
    }
    fputc('\n', stderr);

    char got[64] = {0};
    if (tc->nchoices > 0) { got[0] = find_answer_letter(output, tc->nchoices); got[1] = '\0'; }
    else find_integer_answer(output, got, sizeof(got));

    bool pass = answer_matches(tc, got);
    if (pass) { fprintf(stderr, "  PASS answer: %s (expected: %s)\n", got, tc->answer); n_pass++; }
    else { fprintf(stderr, "  FAIL answer: %s (expected: %s)\n", got, tc->answer); n_fail++; }

    free(logits);
    hy3_tokens_free(&input);
    free(prompt);
}

int main(int argc, char **argv) {
    int gpu_layers = 0;
    const char *model_path = "/root/models/Hy3/hy3_q8.gguf";
    int n_threads = 8;
    if (argc > 1) gpu_layers = atoi(argv[1]);
    if (gpu_layers < 0 || gpu_layers > 81) gpu_layers = 0;
    if (argc > 2) model_path = argv[2];
    if (argc > 3) n_threads = atoi(argv[3]);

    fprintf(stderr, "\n");
    draw_line('#');
    fprintf(stderr, "  HY3 Evaluation (%d questions)\n", ncases);
    fprintf(stderr, "  Model: %s | gpu_layers=%d\n", model_path, gpu_layers);
    draw_line('#');
    fprintf(stderr, "\n");

    hy3_model *m = NULL;
    if (hy3_model_load(&m, model_path, n_threads) != 0) {
        fprintf(stderr, "FATAL: cannot load model\n");
        return 1;
    }

    if (gpu_layers > 0) {
        fprintf(stderr, "hy3_eval_gpu: initializing GPU with %d layers\n", gpu_layers);
        if (hy3_gpu_init(m, gpu_layers)) {
            fprintf(stderr, "hy3_eval_gpu: GPU init failed\n");
            hy3_model_free(m);
            return 1;
        }
    }

    for (int i = 0; i < ncases; i++) {
        m->cache_len = 0;
        run_case(m, &eval_cases[i], i);
        fprintf(stderr, "\n");
    }

    draw_line('=');
    fprintf(stderr, "  Results: %d FAIL, %d PASS, %d SKIP  (total: %d)\n",
            n_fail, n_pass, n_skip, n_pass + n_fail + n_skip);
    draw_line('=');

    hy3_model_free(m);
    return n_fail > 0 ? 1 : 0;
}
