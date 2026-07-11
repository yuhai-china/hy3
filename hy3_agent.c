/* hy3_agent.c — native coding agent for tencent/Hy3.
 *
 * GPU-accelerated interactive agent that loads one Hy3 GGUF, accepts user
 * requests, calls tools (bash, read, write, edit, grep, list) via
 * prompt-engineering, and feeds tool results back to the model in a turn loop.
 *
 * Usage: hy3-agent -m <model.gguf> [--gpu-layers N]
 */

#include "hy3.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

int hy3_gpu_init(hy3_model *m, int n_gpu_layers);

/* ============================================================================
 * Configuration
 * ============================================================================ */

#define AGENT_MAX_TOKENS    4096
#define AGENT_TEMP           0.7f
#define AGENT_READ_CHUNK     500
#define AGENT_BASH_TIMEOUT   60
#define AGENT_MAX_TURNS      16

/* ============================================================================
 * Growable string buffer
 * ============================================================================ */

typedef struct {
    char  *buf;
    size_t cap;
    size_t len;
} strbuf_t;

static void sb_init(strbuf_t *b, size_t cap) {
    b->cap = cap ? cap : 4096;
    b->buf = malloc(b->cap);
    b->len = 0;
    if (b->buf) b->buf[0] = '\0';
}

static void sb_free(strbuf_t *b) {
    free(b->buf); b->buf = NULL; b->cap = b->len = 0;
}

static void sb_grow(strbuf_t *b, size_t need) {
    if (!b->buf || b->len + need + 1 <= b->cap) return;
    b->cap = (b->len + need + 1) * 2;
    b->buf = realloc(b->buf, b->cap);
}

static void sb_append(strbuf_t *b, const char *s, size_t n) {
    sb_grow(b, n);
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = '\0';
}

static void sb_append_str(strbuf_t *b, const char *s) {
    if (s) sb_append(b, s, strlen(s));
}

static void sb_append_char(strbuf_t *b, char c) {
    char tmp = c; sb_append(b, &tmp, 1);
}

/* ============================================================================
 * System prompt with tool schemas (JSON, prompt-engineering based)
 * ============================================================================ */

static const char *agent_system_prompt(void) {
    return
    "You are hy3-agent, a coding agent running inside a local inference engine. "
    "Use tools to read, search, edit, and run code. Keep answers terse.\n\n"
    "### Rules\n"
    "- Explore before editing: list, grep, then read files.\n"
    "- One change at a time.\n"
    "- If a tool fails, diagnose the error and retry.\n"
    "- Run tests/lint after making changes.\n"
    "- Do not print large file contents inline — use write or edit.\n\n"
    "### Tool format\n"
    "Emit exactly ONE tool call per message:\n\n"
    "<tool_call>{\"name\":\"<tool>\",\"arguments\":{<params>}}</tool_call>\n\n"
    "When done with all tool work, write your final answer with NO <tool_call> block.\n\n"
    "### Tools\n\n"
    "bash: Run a shell command.\n"
    "  {\"command\":\"...\",\"timeout\":<seconds>}\n\n"
    "read: Read a file (default 500 lines from start).\n"
    "  {\"path\":\"...\",\"offset\":<line>,\"limit\":<lines>}\n\n"
    "write: Create or overwrite a file.\n"
    "  {\"path\":\"...\",\"content\":\"...\"}\n\n"
    "edit: Replace old text with new (old must match exactly once).\n"
    "  {\"path\":\"...\",\"old\":\"...\",\"new\":\"...\"}\n\n"
    "grep: Search with regex pattern.\n"
    "  {\"pattern\":\"...\",\"path\":\"...\",\"include\":\"*.c\"}\n\n"
    "list: List a directory.\n"
    "  {\"path\":\"...\"}";
}

/* ============================================================================
 * Minimal JSON helper for tool-call parsing
 * ============================================================================ */

static const char *json_skip_ws(const char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
    return s;
}

static const char *json_get_string(const char *s, strbuf_t *out) {
    if (*s != '"') return NULL;
    s++;
    while (*s && *s != '"') {
        if (*s == '\\' && s[1]) { s++; sb_append_char(out, *s); s++; }
        else { sb_append_char(out, *s); s++; }
    }
    if (*s != '"') return NULL;
    return s + 1;
}

/* Find value for key in a flat JSON object. Returns pointer to value start. */
static const char *json_find_key(const char *s, const char *key) {
    s = json_skip_ws(s);
    if (*s != '{') return NULL;
    s++;
    for (;;) {
        s = json_skip_ws(s);
        if (*s == '}') break;
        if (*s == ',') { s++; continue; }
        strbuf_t k = {0}; sb_init(&k, 64);
        s = json_get_string(s, &k);
        if (!s) { sb_free(&k); return NULL; }
        s = json_skip_ws(s);
        if (*s != ':') { sb_free(&k); return NULL; }
        s = json_skip_ws(s + 1);
        if (strcmp(k.buf, key) == 0) { sb_free(&k); return s; }
        /* skip value */
        if (*s == '"') { strbuf_t dummy = {0}; sb_init(&dummy, 64);
            s = json_get_string(s, &dummy); sb_free(&dummy); }
        else if (*s == '{' || *s == '[') {
            int depth = 1; s++; while (*s && depth > 0) {
                if (*s == '{' || *s == '[') depth++;
                else if (*s == '}' || *s == ']') depth--;
                s++;
            }
        } else { while (*s && *s != ',' && *s != '}') s++; }
        sb_free(&k);
    }
    return NULL;
}

/* ============================================================================
 * Tool implementations
 * ============================================================================ */

static void tool_bash(strbuf_t *result, const char *command, int timeout_sec) {
    if (!command || !command[0]) { sb_append_str(result, "error: no command"); return; }
    if (timeout_sec <= 0) timeout_sec = AGENT_BASH_TIMEOUT;

    int pipefd[2];
    if (pipe(pipefd) < 0) { sb_append_str(result, "error: pipe() failed"); return; }

    pid_t pid = fork();
    if (pid < 0) { sb_append_str(result, "error: fork() failed");
        close(pipefd[0]); close(pipefd[1]); return; }

    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execl("/bin/bash", "bash", "-c", command, (char*)NULL);
        _exit(127);
    }

    close(pipefd[1]);
    time_t deadline = time(NULL) + timeout_sec;
    char tmp[4096];
    for (;;) {
        if (time(NULL) >= deadline) {
            kill(pid, SIGKILL);
            char nb[32]; snprintf(nb, sizeof(nb), "\n[timeout %ds]", timeout_sec);
            sb_append_str(result, nb);
            break;
        }
        ssize_t n = read(pipefd[0], tmp, sizeof(tmp));
        if (n < 0) { if (errno == EINTR) continue; break; }
        if (n == 0) break;
        sb_append(result, tmp, (size_t)n);
    }
    close(pipefd[0]);

    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        char nb[64]; snprintf(nb, sizeof(nb), "\n[exit:%d]", WEXITSTATUS(status));
        sb_append_str(result, nb);
    }
}

static void tool_read_file(strbuf_t *result, const char *path, int offset, int limit) {
    if (!path || !path[0]) { sb_append_str(result, "error: no path"); return; }
    if (limit <= 0) limit = AGENT_READ_CHUNK;
    if (offset < 1) offset = 1;

    FILE *f = fopen(path, "r");
    if (!f) { sb_append_str(result, "error: cannot open "); sb_append_str(result, path); return; }

    char line[4096];
    int lineno = 0, written = 0;
    while (fgets(line, sizeof(line), f) && written < limit) {
        lineno++;
        if (lineno < offset) continue;
        char tmp[4128];
        int tl = snprintf(tmp, sizeof(tmp), "%d: %s", lineno, line);
        sb_append(result, tmp, (size_t)tl);
        written++;
    }

    long pos = ftell(f);
    fseek(f, 0, SEEK_END);
    long endpos = ftell(f);
    fclose(f);

    if (pos < endpos) {
        char tail[64];
        snprintf(tail, sizeof(tail), "[more after offset=%d]", offset + limit);
        sb_append_str(result, tail);
    }
}

static void tool_write_file(strbuf_t *result, const char *path, const char *content) {
    if (!path || !path[0]) { sb_append_str(result, "error: no path"); return; }
    if (!content) content = "";

    FILE *f = fopen(path, "w");
    if (!f) { sb_append_str(result, "error: cannot write "); sb_append_str(result, path); return; }
    fputs(content, f);
    fclose(f);
    sb_append_str(result, "wrote "); sb_append_str(result, path);
}

static void tool_edit_file(strbuf_t *result, const char *path,
                            const char *old_str, const char *new_str) {
    if (!path || !path[0]) { sb_append_str(result, "error: no path"); return; }
    if (!old_str) old_str = "";
    if (!new_str) new_str = "";

    FILE *f = fopen(path, "r");
    if (!f) { sb_append_str(result, "error: cannot open "); sb_append_str(result, path); return; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc((size_t)fsize + 1);
    if (!text) { fclose(f); sb_append_str(result, "error: oom"); return; }
    if (fread(text, 1, (size_t)fsize, f) != (size_t)fsize && fsize > 0) {
        free(text); fclose(f); sb_append_str(result, "error: read failed"); return;
    }
    text[fsize] = '\0';
    fclose(f);

    char *pos = strstr(text, old_str);
    if (!pos) { free(text); sb_append_str(result, "error: old text not found"); return; }
    if (strstr(pos + strlen(old_str), old_str)) {
        free(text); sb_append_str(result, "error: old matches multiple times"); return;
    }

    f = fopen(path, "w");
    if (!f) { free(text); sb_append_str(result, "error: cannot write"); return; }
    *pos = '\0';
    fputs(text, f); fputs(new_str, f); fputs(pos + strlen(old_str), f);
    fclose(f); free(text);
    sb_append_str(result, "edited "); sb_append_str(result, path);
}

static void tool_grep(strbuf_t *result, const char *pattern, const char *path,
                       const char *include) {
    if (!pattern || !pattern[0]) { sb_append_str(result, "error: no pattern"); return; }

    strbuf_t cmd; sb_init(&cmd, 4096);
    sb_append_str(&cmd, "grep -rn --color=never ");
    if (include) { sb_append_str(&cmd, "--include="); sb_append_str(&cmd, include);
                   sb_append_char(&cmd, ' '); }
    sb_append_char(&cmd, '"'); sb_append_str(&cmd, pattern); sb_append_char(&cmd, '"');
    sb_append_char(&cmd, ' ');
    sb_append_str(&cmd, path ? path : ".");
    sb_append_str(&cmd, " 2>/dev/null | head -80");

    tool_bash(result, cmd.buf, 30);
    sb_free(&cmd);
}

static void tool_list(strbuf_t *result, const char *path) {
    if (!path || !path[0]) path = ".";
    char cmd[4096];
    snprintf(cmd, sizeof(cmd), "ls -la '%s' 2>&1 | head -200", path);
    tool_bash(result, cmd, 10);
}

/* ============================================================================
 * Tool call parser
 * ============================================================================ */

typedef struct {
    char name[64];
    char args_buf[16384];
    int  args_len;
} parsed_tool_call_t;

static const char *parse_tool_call(const char *text, parsed_tool_call_t *tc) {
    const char *tag = "<tool_call>";
    const char *end_tag = "</tool_call>";
    const char *start = strstr(text, tag);
    if (!start) return NULL;
    start += strlen(tag);
    const char *end = strstr(start, end_tag);
    if (!end) return NULL;

    size_t jslen = (size_t)(end - start);
    char *json = malloc(jslen + 1);
    if (!json) return NULL;
    memcpy(json, start, jslen);
    json[jslen] = '\0';

    memset(tc, 0, sizeof(*tc));

    const char *nv = json_find_key(json, "name");
    if (nv) {
        nv = json_skip_ws(nv);
        strbuf_t n = {0}; sb_init(&n, 64);
        if (*nv == '"') { nv = json_get_string(nv, &n);
            strncpy(tc->name, n.buf, sizeof(tc->name) - 1); }
        sb_free(&n);
    }

    const char *av = json_find_key(json, "arguments");
    if (av) {
        av = json_skip_ws(av);
        if (*av == '{') {
            int depth = 0;
            const char *p = av;
            while (*p && tc->args_len < (int)sizeof(tc->args_buf) - 1) {
                tc->args_buf[tc->args_len++] = *p;
                if (*p == '{') depth++;
                else if (*p == '}') { depth--; if (depth == 0) { p++; break; } }
                p++;
            }
            tc->args_buf[tc->args_len] = '\0';
        }
    }

    free(json);
    return end + strlen(end_tag);
}

static char *tc_get_string(const parsed_tool_call_t *tc, const char *key) {
    const char *v = json_find_key(tc->args_buf, key);
    if (!v) return NULL;
    v = json_skip_ws(v);
    if (*v != '"') return NULL;
    strbuf_t out = {0}; sb_init(&out, 4096);
    json_get_string(v, &out);
    if (out.len == 0) { sb_free(&out); return NULL; }
    return out.buf;
}

static double tc_get_number(const parsed_tool_call_t *tc, const char *key,
                             double def) {
    const char *v = json_find_key(tc->args_buf, key);
    if (!v) return def;
    v = json_skip_ws(v);
    char *end;
    double val = strtod(v, &end);
    return (end > v) ? val : def;
}

/* ============================================================================
 * Agent conversation + model wrapper
 * ============================================================================ */

typedef struct {
    hy3_model *model;
    float     *logits;
    int        pos;
} agent_session_t;

static void session_init(agent_session_t *s, hy3_model *m) {
    s->model  = m;
    s->logits = malloc((size_t)HY3_N_VOCAB * sizeof(float));
    s->pos    = 0;
}

static void session_free(agent_session_t *s) {
    free(s->logits);
}

/* Tokenize a chat-formatted string and run the full prefill + generation.
 * Returns heap-allocated string of decoded assistant text. */
static char *session_generate(agent_session_t *s, const char *chat_text,
                               float temp, int max_tokens) {
    hy3_model *m = s->model;
    hy3_tokens input = {0};
    hy3_tokenize(m, chat_text, &input);
    if (input.len == 0) { free(input.v); return strdup(""); }

    /* Prefill all prompt tokens */
    { hy3_tokens single = { .len = 1, .v = &input.v[0] };
      hy3_eval(m, &single, s->logits, &s->pos); }
    for (int i = 1; i < input.len; i++) {
        hy3_tokens single = { .len = 1, .v = &input.v[i] };
        hy3_eval(m, &single, s->logits, &s->pos);
    }
    free(input.v);

    /* Generate */
    strbuf_t out; sb_init(&out, 4096);
    int eos = hy3_token_eos(m);

    for (int step = 0; step < max_tokens; step++) {
        int token = hy3_sample(m, s->logits, temp, -1, 1.0f);
        if (token == eos || token == 120001 || token == 120008) break;

        char piece[128];
        if (hy3_detokenize(m, token, piece, sizeof(piece)) <= 0) break;

        /* Stop if model starts a new turn */
        if (strstr(piece, "<｜hy_User") || strstr(piece, "<｜hy_Assistant")) break;

        sb_append_str(&out, piece);

        hy3_tokens single = { .len = 1, .v = &token };
        hy3_eval(m, &single, s->logits, &s->pos);
    }
    return out.buf;
}

/* ============================================================================
 * Agent loop
 * ============================================================================ */

static int toolbuf_truncate(strbuf_t *b, size_t max_len) {
    if (b->len <= max_len) return 0;
    b->buf[max_len] = '\0';
    b->len = max_len;
    sb_append_str(b, "\n... [truncated]");
    return 1;
}

static void agent_run(agent_session_t *s, const char *user_msg) {
    /* Build conversation: system + tools + user message */
    strbuf_t conv; sb_init(&conv, 65536);
    sb_append_str(&conv, agent_system_prompt());
    sb_append_str(&conv, "\n\nUser request: ");
    sb_append_str(&conv, user_msg);
    sb_append_str(&conv, "\n\nYou may use tools. When done, give your final answer.");

    for (int turn = 0; turn < AGENT_MAX_TURNS; turn++) {
        fprintf(stderr, "\r[agent] turn %d/%d thinking...", turn + 1, AGENT_MAX_TURNS);
        fflush(stderr);

        char *raw = session_generate(s, conv.buf, AGENT_TEMP, AGENT_MAX_TOKENS);
        if (!raw || !*raw) { free(raw); break; }

        /* Check for tool call */
        parsed_tool_call_t tc;
        const char *after = parse_tool_call(raw, &tc);
        (void)after;

        if (tc.name[0]) {
            fprintf(stderr, "\n[agent] tool: %s\n", tc.name);

            strbuf_t result; sb_init(&result, 4096);

            if (strcmp(tc.name, "bash") == 0) {
                char *cmd = tc_get_string(&tc, "command");
                int timeout = (int)tc_get_number(&tc, "timeout", AGENT_BASH_TIMEOUT);
                tool_bash(&result, cmd ? cmd : "", timeout);
                free(cmd);
            } else if (strcmp(tc.name, "read") == 0) {
                char *path = tc_get_string(&tc, "path");
                int offset = (int)tc_get_number(&tc, "offset", 1);
                int limit  = (int)tc_get_number(&tc, "limit", AGENT_READ_CHUNK);
                tool_read_file(&result, path ? path : "", offset, limit);
                free(path);
            } else if (strcmp(tc.name, "write") == 0) {
                char *path = tc_get_string(&tc, "path");
                char *content = tc_get_string(&tc, "content");
                tool_write_file(&result, path ? path : "", content ? content : "");
                free(path); free(content);
            } else if (strcmp(tc.name, "edit") == 0) {
                char *path = tc_get_string(&tc, "path");
                char *o = tc_get_string(&tc, "old");
                char *n = tc_get_string(&tc, "new");
                tool_edit_file(&result, path ? path : "", o ? o : "", n ? n : "");
                free(path); free(o); free(n);
            } else if (strcmp(tc.name, "grep") == 0) {
                char *pattern = tc_get_string(&tc, "pattern");
                char *path = tc_get_string(&tc, "path");
                char *include = tc_get_string(&tc, "include");
                tool_grep(&result, pattern ? pattern : "",
                          path ? path : ".", include ? include : "");
                free(pattern); free(path); free(include);
            } else if (strcmp(tc.name, "list") == 0) {
                char *path = tc_get_string(&tc, "path");
                tool_list(&result, path ? path : ".");
                free(path);
            } else {
                sb_append_str(&result, "error: unknown tool "); sb_append_str(&result, tc.name);
            }

            toolbuf_truncate(&result, 32000);

            /* Append tool result + ask model to continue */
            sb_append_str(&conv, "\n\n<tool_result>\n");
            sb_append_str(&conv, result.buf);
            sb_append_str(&conv, "\n</tool_result>\nContinue. Give your final answer or use another tool.");
            free(raw);
            continue;
        }

        /* No tool call → final answer */
        fprintf(stderr, "\n");
        printf("\n%s\n", raw);
        free(raw);
        break;
    }

    sb_free(&conv);
}

/* ============================================================================
 * Entry point
 * ============================================================================ */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -m <model.gguf> [--gpu-layers N]\n"
        "Interactive coding agent for tencent/Hy3.\n", prog);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    int gpu_layers = 0;
    int n_threads = 4;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--gpu-layers") == 0 && i + 1 < argc) gpu_layers = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    if (!model_path) { usage(argv[0]); return 1; }

    /* Load model */
    fprintf(stderr, "[agent] loading model...\n");
    hy3_model *model = NULL;
    if (hy3_model_load(&model, model_path, n_threads) != 0 || !model) {
        fprintf(stderr, "[agent] failed to load model\n"); return 1;
    }

    if (gpu_layers > 0) {
        if (hy3_gpu_init(model, gpu_layers) != 0)
            fprintf(stderr, "[agent] GPU init failed, using CPU\n");
    }

    agent_session_t session;
    session_init(&session, model);

    fprintf(stderr, "[agent] ready. Type a request (Ctrl+D to exit).\n\n");

    /* REPL */
    char *line = NULL;
    size_t linecap = 0;
    ssize_t len;

    while (1) {
        printf("> "); fflush(stdout);
        len = getline(&line, &linecap, stdin);
        if (len < 0) { printf("\n"); break; }
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        if (strcmp(line, "exit") == 0 || strcmp(line, "quit") == 0) break;

        agent_run(&session, line);
        fflush(stdout);
    }

    free(line);
    session_free(&session);
    hy3_model_free(model);
    return 0;
}
