#include "hy3.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef HY3_CUDA
int hy3_gpu_init(hy3_model *m, int n_gpu_layers);
#endif
#ifdef HY3_METAL
int hy3_metal_init(hy3_model *m);
#endif

static int emit_token(void *ud, int token) {
    hy3_model *m = (hy3_model *)ud;
    static int color_checked = 0, use_color = 0;
    if (!color_checked) { color_checked = 1; use_color = isatty(STDOUT_FILENO); }
    if (token == 120029) { /* <think:opensource> */
        if (use_color) fwrite("\033[32m", 1, 5, stdout);
        fflush(stdout);
        return 0;
    }
    if (token == 120030) { /* </think:opensource> */
        if (use_color) fwrite("\033[0m\n", 1, 5, stdout);
        fflush(stdout);
        return 0;
    }
    char buf[256];
    int n = hy3_detokenize(m, token, buf, sizeof(buf));
    if (n > 0) fwrite(buf, 1, n, stdout);
    fflush(stdout);
    return 0;
}

/* Unescape a batch-file line in place: turn the two-character sequence "\n"
 * into a real newline and "\\" into a single backslash, so multi-line prompts
 * can be stored one-per-line. Returns the new length. */
static size_t unescape_inplace(char *s) {
    char *w = s;
    for (char *r = s; *r; r++) {
        if (r[0] == '\\' && r[1] == 'n')      { *w++ = '\n'; r++; }
        else if (r[0] == '\\' && r[1] == 't') { *w++ = '\t'; r++; }
        else if (r[0] == '\\' && r[1] == '\\'){ *w++ = '\\'; r++; }
        else                                   { *w++ = *r; }
    }
    *w = 0;
    return (size_t)(w - s);
}

/* Batch mode: read one prompt per line from `path`, run each through a fresh
 * context on the already-loaded model, and frame every answer with
 * <<<HY3_BEGIN i>>> / <<<HY3_END>>> markers so a harness can split them.
 * Returns 0 on success, non-zero if the file can't be opened. */
static int run_batch(hy3_model *model, const char *path, int n_predict,
                     hy3_params *params, int raw_prompt, int think) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "hy3: cannot open batch file '%s'\n", path);
        return 1;
    }
    /* Lines may be long (code prompts); grow the buffer as needed. */
    size_t cap = 1 << 16;
    char *line = malloc(cap);
    if (!line) { fclose(f); return 1; }

    int idx = 0;
    while (fgets(line, (int)cap, f)) {
        /* Extend the buffer if the line didn't fit (no trailing newline and
         * not EOF-terminated). */
        size_t len = strlen(line);
        while (len == cap - 1 && line[len - 1] != '\n') {
            char *bigger = realloc(line, cap * 2);
            if (!bigger) break;
            line = bigger;
            if (!fgets(line + len, (int)(cap + 1), f)) break;
            cap *= 2;
            len = strlen(line);
        }
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = 0;
        if (len == 0) continue;

        unescape_inplace(line);

        hy3_reset_context(model);
        hy3_tokens input;
        memset(&input, 0, sizeof(input));
        if (raw_prompt) hy3_tokenize(model, line, &input);
        else            hy3_tokenize_chat(model, line, think, &input);

        printf("<<<HY3_BEGIN %d>>>\n", idx);
        fflush(stdout);
        hy3_generate(model, &input, n_predict, params, emit_token, model);
        printf("\n<<<HY3_END>>>\n");
        fflush(stdout);

        hy3_tokens_free(&input);
        idx++;
    }
    free(line);
    fclose(f);
    fprintf(stderr, "hy3: batch complete, %d prompt(s) processed\n", idx);
    return 0;
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options] -m <model.gguf> [prompt]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -m <file>     Path to GGUF model file\n");
    fprintf(stderr, "  -t <n>        Number of threads (default: 4)\n");
    fprintf(stderr, "  -n <n>        Number of tokens to generate (default: 512)\n");
    fprintf(stderr, "  -p <prompt>   Input prompt\n");
    fprintf(stderr, "  -temp <f>     Temperature (default: 0.9)\n");
    fprintf(stderr, "  -top_k <n>    Top-k sampling (default: 0 = off)\n");
    fprintf(stderr, "  -top_p <f>    Top-p sampling (default: 1.0)\n");
    fprintf(stderr, "  -experts <n>  MoE experts per token (3..8, default: 8 recommended -- native\n");
    fprintf(stderr, "                top-8. Lower values save a little GPU work but degrade quality;\n");
    fprintf(stderr, "                do NOT use 1-2: incoherent output.)\n");
    fprintf(stderr, "  --raw         Feed the prompt as raw text (no chat template)\n");
    fprintf(stderr, "  --think       Enable high reasoning effort\n");
    fprintf(stderr, "  --think-low   Enable low reasoning effort (shorter chain-of-thought)\n");
    fprintf(stderr, "  --batch <f>   Batch mode: run every prompt line in file <f> (model loads once).\n");
    fprintf(stderr, "                Each line is one prompt with \\n escaped; blank lines skipped.\n");
    fprintf(stderr, "                Output is framed by <<<HY3_BEGIN i>>> ... <<<HY3_END>>> markers.\n");
    fprintf(stderr, "  --rope-yarn   Enable YaRN RoPE scaling for context beyond the native 262144\n");
    fprintf(stderr, "                (EXPERIMENTAL extrapolation; the base model is rope_type default).\n");
    fprintf(stderr, "  --rope-factor <f> YaRN scale factor (e.g. 4 -> ~1M ctx). Implies --rope-yarn.\n");
    fprintf(stderr, "  --rope-ctx <n>    Target context length; derives the YaRN factor. Implies --rope-yarn.\n");
#ifdef HY3_CUDA
    fprintf(stderr, "  --gpu         Use GPU acceleration (CUDA)\n");
    fprintf(stderr, "  --gpu-layers <n>  Number of layers to offload to CUDA GPU\n");
#endif
#ifdef HY3_METAL
    fprintf(stderr, "  --metal       Use Metal acceleration (macOS/Apple Silicon, all layers)\n");
#endif
    fprintf(stderr, "  -h            Show this help\n");
}

int main(int argc, char **argv) {
    hy3_params params;
    memset(&params, 0, sizeof(params));
    params.temperature = HY3_DEFAULT_TEMPERATURE;
    params.top_p = HY3_DEFAULT_TOP_P;
    params.top_k = 0;
    params.n_threads = 4;
    params.ctx_size = 2048;
    params.use_gpu = HY3_CUDA_ENABLED;

    const char *model_path = NULL;
    const char *prompt = NULL;
    int n_predict = 512;
    int use_metal = 0;
    int n_experts = 0;   /* 0 = leave model default (from HY3_TOP_K_EXPERTS or 8) */
    int raw_prompt = 0;  /* 0 = apply chat template, 1 = feed raw text */
    int think = 0;       /* reasoning block: 0 = no_think, 1 = low, 2 = high */
    const char *batch_file = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            params.n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-temp") == 0 && i + 1 < argc) {
            params.temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "-top_k") == 0 && i + 1 < argc) {
            params.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-top_p") == 0 && i + 1 < argc) {
            params.top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "-experts") == 0 && i + 1 < argc) {
            n_experts = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--raw") == 0) {
            raw_prompt = 1;
        } else if (strcmp(argv[i], "--think") == 0) {
            think = 2;
        } else if (strcmp(argv[i], "--think-low") == 0) {
            think = 1;
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_file = argv[++i];
        } else if (strcmp(argv[i], "--rope-yarn") == 0) {
            setenv("HY3_ROPE_YARN", "1", 1);
        } else if (strcmp(argv[i], "--rope-factor") == 0 && i + 1 < argc) {
            setenv("HY3_ROPE_FACTOR", argv[++i], 1);
        } else if (strcmp(argv[i], "--rope-ctx") == 0 && i + 1 < argc) {
            setenv("HY3_CTX", argv[++i], 1);
#ifdef HY3_CUDA
        } else if (strcmp(argv[i], "--gpu") == 0) {
            params.use_gpu = 1;
        } else if (strcmp(argv[i], "--gpu-layers") == 0 && i + 1 < argc) {
            params.gpu_layers = atoi(argv[++i]);
            params.use_gpu = 1;
#endif
#ifdef HY3_METAL
        } else if (strcmp(argv[i], "--metal") == 0) {
            use_metal = 1;
            setenv("HY3_METAL", "1", 1);
#endif
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }

    hy3_model *model = NULL;
    if (hy3_model_load(&model, model_path, params.n_threads) != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    if (n_experts != 0) {
        if (n_experts >= 1 && n_experts <= HY3_N_EXPERT_USED) {
            model->n_expert_used = n_experts;
        } else {
            fprintf(stderr, "hy3: -experts %d out of range (1..%d), using %d\n",
                    n_experts, HY3_N_EXPERT_USED, model->n_expert_used);
        }
    }
    fprintf(stderr, "hy3: MoE experts per token = %d\n", model->n_expert_used);

#ifdef HY3_CUDA
    if (params.use_gpu) {
        if (hy3_gpu_init(model, params.gpu_layers) != 0) {
            fprintf(stderr, "hy3: failed to initialize GPU, falling back to CPU\n");
        } else {
            fprintf(stderr, "hy3: GPU acceleration enabled\n");
        }
    }
#endif
#ifdef HY3_METAL
    if (use_metal) {
        if (hy3_metal_init(model) != 0) {
            fprintf(stderr, "hy3: failed to initialize Metal, falling back to CPU\n");
        } else {
            fprintf(stderr, "hy3: Metal acceleration enabled (all layers)\n");
        }
    }
#else
    (void)use_metal;
#endif

    if (batch_file) {
        int rc = run_batch(model, batch_file, n_predict, &params, raw_prompt, think);
        hy3_model_free(model);
        return rc;
    }

    if (prompt && prompt[0]) {
        hy3_tokens input;
        memset(&input, 0, sizeof(input));
        if (raw_prompt) hy3_tokenize(model, prompt, &input);
        else            hy3_tokenize_chat(model, prompt, think, &input);

        hy3_generate(model, &input, n_predict, &params, emit_token, model);
        printf("\n");
    } else {
        fprintf(stderr, "hy3: interactive mode (type 'exit' to quit)\n");

        char line[4096];
        while (1) {
            fprintf(stderr, "> ");
            fflush(stderr);
            if (!fgets(line, sizeof(line), stdin)) break;
            size_t len = strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
            if (len == 0) continue;
            if (strcmp(line, "exit") == 0 || strcmp(line, "quit") == 0) break;

            hy3_tokens input;
            memset(&input, 0, sizeof(input));
            if (raw_prompt) hy3_tokenize(model, line, &input);
            else            hy3_tokenize_chat(model, line, think, &input);

            printf("\n");
            hy3_generate(model, &input, n_predict, &params, emit_token, model);
            printf("\n\n");
            hy3_tokens_free(&input);
        }
    }

    hy3_model_free(model);
    return 0;
}
