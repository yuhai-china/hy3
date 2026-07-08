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
    char buf[256];
    int n = hy3_detokenize(m, token, buf, sizeof(buf));
    if (n > 0) fwrite(buf, 1, n, stdout);
    fflush(stdout);
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
    fprintf(stderr, "  --gpu         Use GPU acceleration (CUDA)\n");
    fprintf(stderr, "  --gpu-layers <n>  Number of layers to offload to CUDA GPU\n");
    fprintf(stderr, "  --metal       Use Metal acceleration (macOS/Apple Silicon, all layers)\n");
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
        } else if (strcmp(argv[i], "--gpu") == 0) {
            params.use_gpu = 1;
        } else if (strcmp(argv[i], "--gpu-layers") == 0 && i + 1 < argc) {
            params.gpu_layers = atoi(argv[++i]);
            params.use_gpu = 1;
        } else if (strcmp(argv[i], "--metal") == 0) {
            use_metal = 1;
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

    if (prompt && prompt[0]) {
        hy3_tokens input;
        memset(&input, 0, sizeof(input));
        hy3_tokenize(model, prompt, &input);
        fprintf(stderr, "hy3: prompt tokens (%d):", input.len);
        for (int i = 0; i < input.len && i < 10; i++) fprintf(stderr, " %d", input.v[i]);
        fprintf(stderr, "\n");

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
            hy3_tokenize(model, line, &input);

            printf("\n");
            hy3_generate(model, &input, n_predict, &params, emit_token, model);
            printf("\n\n");
            hy3_tokens_free(&input);
        }
    }

    hy3_model_free(model);
    return 0;
}
