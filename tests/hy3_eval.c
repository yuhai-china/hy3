#include "../hy3.h"
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define ANSI_RESET "\x1b[0m"
#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_YELLOW "\x1b[33m"
#define ANSI_BOLD "\x1b[1m"
#define ANSI_DIM "\x1b[90m"

#define EVAL_MAX_CHOICES 10
#define EVAL_MAX_CONTEXT 4096

typedef struct {
    const char *source;
    const char *id;
    const char *domain;
    const char *question;
    const char *choice[EVAL_MAX_CHOICES];
    const char *answer;
    int nchoices;
} eval_case;

/* Pick questions from GPQA Diamond / SuperGPQA / AIME */
static const eval_case eval_cases[] = {
    /* GPQA Physics: multiple choice */
    {
        .source = "GPQA Diamond",
        .id = "recNu3MXkvWUzHZr9",
        .domain = "Physics",
        .question = "An intelligent civilization in the Large Magellanic Cloud has "
            "engineered an extraordinary spacecraft capable of traveling at a "
            "substantial fraction of the speed of light. The average lifetime of "
            "these aliens is roughly 150 solar years. Now, having Earth as their "
            "destination in mind, they are determined to travel with this spacecraft "
            "at a constant speed of 0.99999987*c, where c is the speed of light. "
            "Approximately, how long will it take for their 22 years old astronaut "
            "(from the point of view of the astronaut) to reach the Earth using this "
            "incredibly fast spacecraft?",
        .choice[0] = "0.22 years",
        .choice[1] = "7 days",
        .choice[2] = "35 days",
        .choice[3] = "2 months",
        .choice[4] = "0.5 year",
        .nchoices = 5,
        .answer = "B",
    },
    /* SuperGPQA: multiple choice */
    {
        .source = "SuperGPQA",
        .id = "s01",
        .domain = "Engineering",
        .question = "Given a brick masonry compression member with e/h=0.18, "
            "height-to-thickness ratio beta=17, seismic fortification intensity of "
            "7 degrees, and Class II site soil, please select a masonry structural "
            "component plan that is safe and reliable, economically reasonable, "
            "and convenient for construction.",
        .choice[0] = "MU10 brick, M5 mortar",
        .choice[1] = "MU15 brick, M5 mortar",
        .choice[2] = "MU20 brick, M5 mortar",
        .choice[3] = "MU10 brick, M7.5 mortar",
        .choice[4] = "MU15 brick, M7.5 mortar",
        .choice[5] = "MU20 brick, M7.5 mortar",
        .choice[6] = "MU10 brick, M10 mortar",
        .choice[7] = "MU15 brick, M10 mortar",
        .choice[8] = "MU20 brick, M10 mortar",
        .choice[9] = "None of the above",
        .nchoices = 10,
        .answer = "J",
    },
    /* AIME 2025: integer math answer */
    {
        .source = "AIME 2025",
        .id = "aime_problem_1",
        .domain = "Math",
        .question = "Find the sum of all integer bases b>9 for which 17_b is a divisor of 97_b.",
        .nchoices = 0,
        .answer = "70",
    },
    /* Integer answer from GPQA context */
    {
        .source = "AIME 2025",
        .id = "aime_problem_2",
        .domain = "Math",
        .question = "Six points A, B, C, D, E and F lie in a straight line in that order. "
            "Suppose that G is a point not on the line and that AC=26, BD=22, CE=31, "
            "DF=33, AF=73, CG=40, and DG=30. Find the area of triangle BGE.",
        .nchoices = 0,
        .answer = "468",
    },
    /* Another AIME problem */
    {
        .source = "AIME 2025",
        .id = "aime_problem_3",
        .domain = "Math",
        .question = "The 9 members of a baseball team went to an ice-cream parlor after "
            "their game. Each player had a single-scoop cone of chocolate, vanilla, or "
            "strawberry ice cream. At least one player chose each flavor, and the number "
            "of players who chose chocolate was greater than the number of players who "
            "chose vanilla, which was greater than the number of players who chose "
            "strawberry. Let N be the number of different assignments of flavors to "
            "players that meet these conditions. Find the remainder when N is divided by 1000.",
        .nchoices = 0,
        .answer = "16",
    },
    /* Chinese-language question */
    {
        .source = "SuperGPQA",
        .id = "s02",
        .domain = "Engineering",
        .question = "What is the key factor in selecting a reasonable amount of "
            "blasting charge in mining blasting operations?",
        .choice[0] = "rock hardness",
        .choice[1] = "blast hole diameter",
        .choice[2] = "charge structure",
        .choice[3] = "minimum resistance line",
        .choice[4] = "ore grade",
        .choice[5] = "charge coefficient",
        .nchoices = 6,
        .answer = "E",
    },
};

static int ncases = sizeof(eval_cases) / sizeof(eval_cases[0]);
static int n_pass = 0, n_fail = 0, n_skip = 0;

static void draw_line(char c) {
    for (int i = 0; i < 72; i++) fputc(c, stderr);
    fputc('\n', stderr);
}

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
            "format and do not write anything after it:\n"
            "Answer: <letter>");
    } else {
        pos += snprintf(buf + pos, cap - (size_t)pos,
            "\nSolve the problem. At the end, write exactly one final line in this "
            "format and do not write anything after it:\n"
            "Answer: <integer>");
    }
    return buf;
}

static char find_answer_letter(const char *generated, int nchoices) {
    if (nchoices <= 0) return '?';
    char max_answer = (char)('A' + nchoices - 1);
    /* Search for "Answer: X" pattern */
    const char *ans = strstr(generated, "Answer:");
    if (ans) {
        const char *p = ans + 7;
        while (*p == ' ' || *p == '\t') p++;
        char c = (char)toupper((unsigned char)*p);
        if (c >= 'A' && c <= max_answer) return c;
    }
    /* Fallback: scan for "answer is X" */
    ans = strstr(generated, "answer is");
    if (ans) {
        const char *p = ans + 9;
        while (*p == ' ' || *p == '\t') p++;
        char c = (char)toupper((unsigned char)*p);
        if (c >= 'A' && c <= max_answer) return c;
    }
    /* Last resort: find the last letter in range */
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
    
    /* Try "Answer: <number>" first */
    const char *ans = strstr(generated, "Answer:");
    if (ans) {
        const char *p = ans + 7;
        while (*p == ' ' || *p == '\t') p++;
        if (isdigit((unsigned char)*p) || (*p == '-' && isdigit((unsigned char)p[1]))) {
            const char *start = p;
            const char *end = p;
            if (*end == '-') end++;
            while (isdigit((unsigned char)*end)) end++;
            normalize_integer_answer(start, (size_t)(end - start), dst, dstlen);
            return;
        }
    }
    
    /* Fallback: last number in text */
    const char *last_start = NULL;
    const char *last_end = NULL;
    for (const char *p = generated; *p; p++) {
        if (isdigit((unsigned char)*p)) {
            last_start = p;
            while (isdigit((unsigned char)*p)) p++;
            last_end = p;
            p--;
        }
    }
    if (last_start && last_end) {
        normalize_integer_answer(last_start, (size_t)(last_end - last_start), dst, dstlen);
    }
}

static bool answer_matches(const eval_case *tc, const char *got) {
    if (tc->nchoices > 0) {
        return got && got[0] && tc->answer && got[0] == tc->answer[0];
    }
    char expected[64];
    normalize_integer_answer(tc->answer, strlen(tc->answer), expected, sizeof(expected));
    return got && strcmp(got, expected) == 0;
}

static int screen_width = 72;

static void run_case(hy3_model *m, const eval_case *tc, int idx) {
    draw_line('-');
    fprintf(stderr, "  \x1b[1m[%d/%d]\x1b[0m %s | %s | %s\n",
            idx + 1, ncases, tc->source, tc->domain, tc->id);
    draw_line('-');

    char *prompt = build_question_prompt(tc);
    if (!prompt) { fprintf(stderr, "  OOM\n"); n_skip++; return; }
    
    /* Tokenize */
    hy3_tokens input = {0};
    hy3_tokenize(m, prompt, &input);
    fprintf(stderr, "  \x1b[90mprompt:\x1b[0m %d tokens\n", input.len);
    
    if (input.len > EVAL_MAX_CONTEXT - 128) {
        fprintf(stderr, "  \x1b[33mSKIP\x1b[0m prompt too long (%d tokens)\n", input.len);
        hy3_tokens_free(&input);
        free(prompt);
        n_skip++;
        return;
    }
    
    float *logits = malloc(120832 * sizeof(float));
    int pos = 0;
    
    /* Prefill */
    hy3_tokens single;
    single.v = &input.v[0];
    single.len = 1;
    single.cap = 1;
    hy3_eval(m, &single, logits, &pos);
    for (int i = 1; i < input.len; i++) {
        single.v = &input.v[i];
        hy3_eval(m, &single, logits, &pos);
    }
    
    /* Generate answer (max 256 tokens) */
    char output[8192];
    int opos = 0;
    output[0] = '\0';
    
    for (int i = 0; i < 256; i++) {
        int best = 0;
        float best_v = -1e30f;
        for (int v = 0; v < 120818; v++) {
            if (logits[v] > best_v) { best_v = logits[v]; best = v; }
        }
        if (best == hy3_token_eos(m)) break;
        
        char tok[256];
        int tlen = hy3_detokenize(m, best, tok, sizeof(tok));
        if (opos + tlen < (int)sizeof(output) - 1) {
            memcpy(output + opos, tok, (size_t)tlen);
            opos += tlen;
            output[opos] = '\0';
        }
        
        /* Stop if we see "Answer: X" with a clear answer */
        if (tc->nchoices > 0) {
            const char *a = strstr(output, "Answer:");
            if (a) {
                const char *p = a + 7;
                while (*p == ' ') p++;
                if (isalpha((unsigned char)*p) || *p == '\n') break;
            }
        } else {
            if (strstr(output, "Answer:")) {
                /* Check if there's a number after "Answer:" */
                const char *a = strstr(output, "Answer:");
                const char *p = a + 7;
                while (*p == ' ') p++;
                if (isdigit((unsigned char)*p)) break;
            }
        }
        
        single.v = &best;
        hy3_eval(m, &single, logits, &pos);
    }
    
    /* Display output (first 512 chars) */
    fprintf(stderr, "  \x1b[90moutput:\x1b[0m ");
    for (int i = 0; output[i]; i++) {
        fputc(output[i] >= 32 && output[i] < 127 ? output[i] : '.', stderr);
        if (i >= 200) { fputc('.', stderr); fputc('.', stderr); fputc('.', stderr); break; }
    }
    fputc('\n', stderr);
    
    /* Extract and grade answer */
    char got[64] = {0};
    if (tc->nchoices > 0) {
        got[0] = find_answer_letter(output, tc->nchoices);
        got[1] = '\0';
    } else {
        find_integer_answer(output, got, sizeof(got));
    }
    
    bool pass = answer_matches(tc, got);
    if (pass) {
        fprintf(stderr, "  \x1b[32mPASS\x1b[0m answer: %s (expected: %s)\n",
                got, tc->answer);
        n_pass++;
    } else {
        fprintf(stderr, "  \x1b[31mFAIL\x1b[0m answer: %s (expected: %s)\n",
                got, tc->answer);
        n_fail++;
    }
    
    free(logits);
    hy3_tokens_free(&input);
    free(prompt);
}

int main() {
    fprintf(stderr, "\n");
    draw_line('#');
    fprintf(stderr, "  \x1b[1mHY3 Evaluation\x1b[0m  (%d questions)\n", ncases);
    fprintf(stderr, "  Model: /root/models/Hy3/hy3_q8.gguf\n");
    draw_line('#');
    fprintf(stderr, "\n");
    
    hy3_model *m = NULL;
    if (hy3_model_load(&m, "/root/models/Hy3/hy3_q8.gguf", 4) != 0) {
        fprintf(stderr, "FATAL: cannot load model\n");
        return 1;
    }
    
    for (int i = 0; i < ncases; i++) {
        m->cache_len = 0;
        run_case(m, &eval_cases[i], i);
        fprintf(stderr, "\n");
    }
    
    draw_line('=');
    fprintf(stderr, "  \x1b[1mResults:\x1b[0m ");
    if (n_fail > 0) fprintf(stderr, "\x1b[31m");
    fprintf(stderr, "%d FAIL\x1b[0m, ", n_fail);
    if (n_pass > 0) fprintf(stderr, "\x1b[32m");
    fprintf(stderr, "%d PASS\x1b[0m, ", n_pass);
    fprintf(stderr, "%d SKIP", n_skip);
    fprintf(stderr, "  (total: %d)\n", n_pass + n_fail + n_skip);
    draw_line('=');
    
    hy3_model_free(m);
    return n_fail > 0 ? 1 : 0;
}
