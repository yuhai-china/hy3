/* hy3_agent.c — native coding agent for tencent/Hy3.
 *
 * GPU-accelerated interactive coding agent.  Loads a Hy3 GGUF, accepts user
 * requests in a REPL, streams model output in real-time with in-flight
 * <tool_call> detection, executes tools (bash / read / more / write / edit /
 * grep / list), feeds results back, and loops until the model produces a
 * final answer.
 *
 * Usage: hy3-agent -m <model.gguf> [--gpu-layers N]
 */

#include "hy3.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
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
 * Config
 * ============================================================================ */

#define AGENT_MAX_TOKENS   4096
#define AGENT_TEMP          0.0f    /* greedy: deterministic, correct for tool calls */
#define AGENT_READ_CHUNK    500
#define AGENT_BASH_TIMEOUT   60
#define AGENT_MAX_TURNS      16

/* ============================================================================
 * Growable string
 * ============================================================================ */

typedef struct { char *buf; size_t cap, len; } strbuf;

static void sb_init(strbuf *b, size_t cap) {
    b->cap = cap ? cap : 4096; b->buf = malloc(b->cap); b->len = 0;
    if (b->buf) b->buf[0] = '\0';
}
static void sb_free(strbuf *b) { free(b->buf); b->buf = NULL; b->cap = b->len = 0; }
static void sb_grow(strbuf *b, size_t need) {
    if (!b->buf || b->len + need + 1 <= b->cap) return;
    b->cap = (b->len + need + 1) * 2; b->buf = realloc(b->buf, b->cap);
}
static void sb_add(strbuf *b, const char *s, size_t n) {
    sb_grow(b, n); memcpy(b->buf + b->len, s, n); b->len += n; b->buf[b->len] = '\0';
}
static void sb_puts(strbuf *b, const char *s) { if (s) sb_add(b, s, strlen(s)); }
static void sb_putc(strbuf *b, char c)  { sb_add(b, &c, 1); }

/* ============================================================================
 * System prompt  (ds4-agent style)
 * ============================================================================ */

static const char *agent_system_prompt(void) {
    return
    "You are a function-calling coding assistant running in a local workspace. "
    "Use tools for local file and system work. Avoid printing large file contents "
    "as answers; create or edit files with tools, then summarise briefly.\n\n"
    "You can call the following tools (described as JSON schemas):\n\n"
    "[\n"
    "  {\n"
    "    \"name\": \"bash\",\n"
    "    \"description\": \"Run a shell command.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"command\": {\"type\": \"string\"},\n"
    "        \"timeout\": {\"type\": \"number\"}\n"
    "      },\n"
    "      \"required\": [\"command\"]\n"
    "    }\n"
    "  },\n"
    "  {\n"
    "    \"name\": \"read\",\n"
    "    \"description\": \"Read a text file or a range of lines.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"offset\": {\"type\": \"number\"},\n"
    "        \"limit\": {\"type\": \"number\"}\n"
    "      },\n"
    "      \"required\": [\"path\"]\n"
    "    }\n"
    "  },\n"
    "  {\n"
    "    \"name\": \"more\",\n"
    "    \"description\": \"Continue the previous read for more lines.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"lines\": {\"type\": \"number\"}\n"
    "      }\n"
    "    }\n"
    "  },\n"
    "  {\n"
    "    \"name\": \"write\",\n"
    "    \"description\": \"Create or overwrite a text file.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"content\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"path\", \"content\"]\n"
    "    }\n"
    "  },\n"
    "  {\n"
    "    \"name\": \"edit\",\n"
    "    \"description\": \"Replace old text with new in a file. old must match exactly once.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"old\": {\"type\": \"string\"},\n"
    "        \"new\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"path\", \"old\", \"new\"]\n"
    "    }\n"
    "  },\n"
    "  {\n"
    "    \"name\": \"grep\",\n"
    "    \"description\": \"Search with a regex pattern.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"pattern\": {\"type\": \"string\"},\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"include\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"pattern\"]\n"
    "    }\n"
    "  },\n"
    "  {\n"
    "    \"name\": \"list\",\n"
    "    \"description\": \"List a directory.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"path\"]\n"
    "    }\n"
    "  }\n"
    "]\n\n"
    "Instructions:\n"
    "- Explore before editing: list, grep, then read.\n"
    "- One logical change at a time. Do not re-read files after editing.\n"
    "- If the user's request should be handled by one of the tools, respond "
    "with EXACTLY one tool call and nothing else, in this format:\n"
    "  <tool_call>{\"name\": \"<tool_name>\", \"arguments\": {<args>}}</tool_call>\n"
    "- The arguments must be valid JSON and match the tool's schema.\n"
    "- If none of the tools are appropriate, respond with:\n"
    "  <tool_call>none</tool_call>\n"
    "  followed by a brief direct answer.\n"
    "- After receiving a tool result, either call another tool or give your final answer.\n"
    "- When you are done with all tool work, give your final answer with no <tool_call> tag.";
}

/* ============================================================================
 * Minimal JSON helpers
 * ============================================================================ */

static const char *json_skip_ws(const char *s) {
    while (*s && (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r')) s++;
    return s;
}
static const char *json_get_string(const char *s, strbuf *out) {
    if (*s != '"') return NULL;
    s++;
    while (*s && *s != '"') {
        if (*s == '\\' && s[1]) { s++; sb_putc(out, *s); s++; }
        else { sb_putc(out, *s); s++; }
    }
    return (*s == '"') ? s + 1 : NULL;
}
static const char *json_find_key(const char *s, const char *key) {
    s = json_skip_ws(s); if (*s != '{') return NULL; s++;
    for (;;) {
        s = json_skip_ws(s); if (*s == '}') break;
        if (*s == ',') { s++; continue; }
        strbuf k = {0}; sb_init(&k, 64);
        s = json_get_string(s, &k); if (!s) { sb_free(&k); return NULL; }
        s = json_skip_ws(s); if (*s != ':') { sb_free(&k); return NULL; }
        s = json_skip_ws(s + 1);
        if (strcmp(k.buf, key) == 0) { sb_free(&k); return s; }
        /* skip value */
        if (*s == '"')       { strbuf d = {0}; sb_init(&d, 64); s = json_get_string(s, &d); sb_free(&d); }
        else if (*s == '{' || *s == '[') { int d = 1; s++; while (*s && d) { if (*s == '{' || *s == '[') d++; else if (*s == '}' || *s == ']') d--; s++; } }
        else { while (*s && *s != ',' && *s != '}') s++; }
        sb_free(&k);
    }
    return NULL;
}

/* ============================================================================
 * Tool implementations
 * ============================================================================ */

static void tool_bash(strbuf *r, const char *cmd, int to) {
    if (!cmd || !*cmd) { sb_puts(r, "error: no command"); return; }
    if (to <= 0) to = AGENT_BASH_TIMEOUT;
    int fd[2]; if (pipe(fd) < 0) { sb_puts(r, "error: pipe"); return; }
    pid_t pid = fork();
    if (pid < 0) { sb_puts(r, "error: fork"); close(fd[0]); close(fd[1]); return; }
    if (pid == 0) { close(fd[0]); dup2(fd[1], 1); dup2(fd[1], 2); close(fd[1]);
        execl("/bin/bash", "bash", "-c", cmd, (char*)NULL); _exit(127); }
    close(fd[1]);
    time_t dl = time(NULL) + to; char tmp[4096];
    for (;;) { if (time(NULL) >= dl) { kill(pid, SIGKILL);
        char nb[32]; snprintf(nb, sizeof(nb), "\n[timeout %ds]", to); sb_puts(r, nb); break; }
        ssize_t n = read(fd[0], tmp, sizeof(tmp)); if (n < 0) { if (errno == EINTR) continue; break; }
        if (!n) break;
        sb_add(r, tmp, (size_t)n); }
    close(fd[0]);
    int st; waitpid(pid, &st, 0);
    if (WIFEXITED(st) && WEXITSTATUS(st)) { char nb[64]; snprintf(nb, sizeof(nb), "\n[exit:%d]", WEXITSTATUS(st)); sb_puts(r, nb); }
}

static void tool_read_file(strbuf *r, const char *path, int off, int lim,
                            char **out_path, int *out_off) {
    if (!path || !*path) { sb_puts(r, "error: no path"); return; }
    if (lim <= 0) lim = AGENT_READ_CHUNK;
    if (off < 1) off = 1;
    FILE *f = fopen(path, "r");
    if (!f) { sb_puts(r, "error: cannot open "); sb_puts(r, path); return; }
    char line[4096]; int lno = 0, wr = 0;
    while (fgets(line, sizeof(line), f) && wr < lim) { lno++;
        if (lno < off) continue;
        char out[4128]; int nl = snprintf(out, sizeof(out), "%d: %s", lno, line);
        sb_add(r, out, (size_t)nl); wr++; }
    long pos = ftell(f); fseek(f, 0, SEEK_END); long endp = ftell(f); fclose(f);
    if (pos < endp) { char t[64]; snprintf(t, sizeof(t), "\n[more, use more lines=<N>]"); sb_puts(r, t); }
    /* remember read state for 'more' */
    if (out_path) { free(*out_path); *out_path = strdup(path); }
    if (out_off) *out_off = off + wr;
}

static void tool_write_file(strbuf *r, const char *path, const char *content) {
    if (!path || !*path) { sb_puts(r, "error: no path"); return; }
    if (!content) content = "";
    FILE *f = fopen(path, "w");
    if (!f) { sb_puts(r, "error: cannot write "); sb_puts(r, path); return; }
    fputs(content, f); fclose(f);
    sb_puts(r, "wrote "); sb_puts(r, path);
}

static void tool_edit_file(strbuf *r, const char *path, const char *old_s,
                            const char *new_s) {
    if (!path || !*path) { sb_puts(r, "error: no path"); return; }
    if (!old_s) old_s = "";
    if (!new_s) new_s = "";

    /* handle [upto] anchor: replace everything between head and tail */
    char *upto_pos = strstr(old_s, "[upto]");

    FILE *f = fopen(path, "r"); if (!f) { sb_puts(r, "error: cannot open "); sb_puts(r, path); return; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = malloc((size_t)fsize + 1);
    if (!text) { fclose(f); sb_puts(r, "error: oom"); return; }
    if (fread(text, 1, (size_t)fsize, f) != (size_t)fsize && fsize > 0)
        { free(text); fclose(f); sb_puts(r, "error: read"); return; }
    text[fsize] = '\0'; fclose(f);

    char *match_start, *match_end;
    if (upto_pos) {
        /* anchored: head is text before [upto], tail is text after */
        size_t head_len = (size_t)(upto_pos - old_s);
        const char *tail_str = upto_pos + 6; /* skip "[upto]" */
        /* trim trailing newline after [upto] */
        while (*tail_str == '\n' || *tail_str == '\r') tail_str++;
        size_t tail_len = strlen(tail_str);

        match_start = strstr(text, old_s); /* try full string first (no anchor) */
        if (!match_start) {
            /* find head, then find tail after it */
            char *h = text;
            while (1) {
                /* manual strnstr: find first head_len chars of old_s in h */
                char *found = NULL;
                for (char *p = h; *p && !found; p++)
                    if (strncmp(p, old_s, head_len) == 0) found = p;
                h = found;
                if (!h) break;
                char *t = h + head_len;
                char *tl = strstr(t, tail_str);
                if (tl) { match_start = h; match_end = tl + tail_len; break; }
                h++;
            }
            if (!match_start) { free(text); sb_puts(r, "error: anchored old not found"); return; }
        } else {
            match_end = match_start + strlen(old_s);
        }
    } else {
        match_start = strstr(text, old_s);
        if (!match_start) { free(text); sb_puts(r, "error: old text not found"); return; }
        if (strstr(match_start + strlen(old_s), old_s))
            { free(text); sb_puts(r, "error: old matches multiple times"); return; }
        match_end = match_start + strlen(old_s);
    }

    f = fopen(path, "w"); if (!f) { free(text); sb_puts(r, "error: cannot write"); return; }
    *match_start = '\0';
    fputs(text, f); fputs(new_s, f); fputs(match_end, f);
    fclose(f); free(text);
    sb_puts(r, "edited "); sb_puts(r, path);
}

static void tool_grep(strbuf *r, const char *pat, const char *path, const char *inc) {
    if (!pat || !*pat) { sb_puts(r, "error: no pattern"); return; }
    strbuf cmd; sb_init(&cmd, 4096);
    sb_puts(&cmd, "grep -rn --color=never ");
    if (inc) { sb_puts(&cmd, "--include="); sb_puts(&cmd, inc); sb_putc(&cmd, ' '); }
    sb_putc(&cmd, '"'); sb_puts(&cmd, pat); sb_putc(&cmd, '"');
    sb_putc(&cmd, ' '); sb_puts(&cmd, path ? path : ".");
    sb_puts(&cmd, " 2>/dev/null | head -80");
    tool_bash(r, cmd.buf, 30); sb_free(&cmd);
}

static void tool_list(strbuf *r, const char *path) {
    if (!path || !*path) path = ".";
    char cmd[4096]; snprintf(cmd, sizeof(cmd), "ls -la '%s' 2>&1 | head -200", path);
    tool_bash(r, cmd, 10);
}

/* ============================================================================
 * Tool call parser  (in-flight, scans buffer for <tool_call>...</tool_call>)
 * ============================================================================ */

typedef struct { char name[64]; char args_buf[16384]; int args_len; int is_none; } tool_call;

/* Returns pointer past </tool_call> if a complete call found, NULL otherwise.
 * tc->is_none = 1 if the model chose not to call a tool. */
static const char *parse_tool_call(const char *text, tool_call *tc) {
    const char *tag = "<tool_call>", *endt = "</tool_call>";
    const char *s = strstr(text, tag);
    if (!s) return NULL;
    s += strlen(tag);
    const char *e = strstr(s, endt);
    if (!e) return NULL;
    memset(tc, 0, sizeof(*tc));
    /* check for "none" */
    { const char *p = s; while (*p == ' ') p++;
      if (strncmp(p, "none", 4) == 0) { tc->is_none = 1; return e + strlen(endt); } }
    size_t jl = (size_t)(e - s);
    char *js = malloc(jl + 1); if (!js) return NULL;
    memcpy(js, s, jl); js[jl] = '\0';
    const char *nv = json_find_key(js, "name");
    if (nv) { nv = json_skip_ws(nv); strbuf n = {0}; sb_init(&n, 64);
        if (*nv == '"') { nv = json_get_string(nv, &n); strncpy(tc->name, n.buf, sizeof(tc->name)-1); }
        sb_free(&n); }
    const char *av = json_find_key(js, "arguments");
    if (av) { av = json_skip_ws(av);
        if (*av == '{') { int d = 0; const char *p = av;
            while (*p && tc->args_len < (int)sizeof(tc->args_buf)-1)
                { tc->args_buf[tc->args_len++] = *p; if (*p=='{') d++; else if (*p=='}') { d--; if (!d) { p++; break; } } p++; }
            tc->args_buf[tc->args_len] = '\0'; } }
    free(js);
    return e + strlen(endt);
}

static char *tc_str(const tool_call *tc, const char *k) {
    const char *v = json_find_key(tc->args_buf, k); if (!v) return NULL;
    v = json_skip_ws(v); if (*v != '"') return NULL;
    strbuf o = {0}; sb_init(&o, 4096); json_get_string(v, &o);
    if (!o.len) { sb_free(&o); return NULL; } return o.buf;
}
static double tc_num(const tool_call *tc, const char *k, double def) {
    const char *v = json_find_key(tc->args_buf, k); if (!v) return def;
    v = json_skip_ws(v); char *end; double val = strtod(v, &end);
    return (end > v) ? val : def;
}

/* ============================================================================
 * Agent session  (model handle + read state for 'more')
 * ============================================================================ */

typedef struct {
    hy3_model *model;
    float     *logits;
    int        pos;
    /* read state for 'more' tool */
    char      *last_read_path;
    int        last_read_off;
} agent_session;

static void sess_init(agent_session *s, hy3_model *m) {
    s->model = m; s->logits = malloc((size_t)HY3_N_VOCAB * sizeof(float));
    s->pos = 0; s->last_read_path = NULL; s->last_read_off = 0;
}
static void sess_free(agent_session *s) { free(s->logits); free(s->last_read_path); }

/* ============================================================================
 * Streaming generation with in-flight tool-call detection
 * ============================================================================ */

/* Evaluate a chat-formatted prompt, streaming tokens. Returns the complete
 * assistant text (heap-allocated). If a full <tool_call>...</tool_call> is
 * detected mid-stream, stops early and returns the text. */
static char *sess_generate(agent_session *s, const char *chat_text,
                            float temp, int max_tok) {
    hy3_model *m = s->model;
    hy3_tokens input = {0};
    hy3_tokenize(m, chat_text, &input);
    if (!input.len) { free(input.v); return strdup(""); }

    /* prefill */
    { hy3_tokens one = {.len=1,.v=&input.v[0]}; hy3_eval(m, &one, s->logits, &s->pos); }
    for (int i = 1; i < input.len; i++)
        { hy3_tokens one = {.len=1,.v=&input.v[i]}; hy3_eval(m, &one, s->logits, &s->pos); }
    free(input.v);

    /* generate */
    strbuf out; sb_init(&out, 4096);
    int eos = hy3_token_eos(m);
    int tool_detected = 0;

    for (int step = 0; step < max_tok; step++) {
        int tok = hy3_sample(m, s->logits, temp, -1, 1.0f);
        if (tok == eos || tok == 120001 || tok == 120008) break;

        char piece[128];
        if (hy3_detokenize(m, tok, piece, sizeof(piece)) <= 0) break;
        if (strstr(piece, "<｜hy_User") || strstr(piece, "<｜hy_Assistant")) break;

        printf("%s", piece); fflush(stdout);
        sb_puts(&out, piece);

        /* in-flight tool call detection */
        tool_call tc;
        if (parse_tool_call(out.buf, &tc)) { tool_detected = 1; break; }

        hy3_tokens one = {.len=1,.v=&tok};
        hy3_eval(m, &one, s->logits, &s->pos);
    }

    if (!tool_detected) printf("\n");
    return out.buf;
}

/* ============================================================================
 * Tool dispatch
 * ============================================================================ */

static char *dispatch_tool(agent_session *s, tool_call *tc) {
    strbuf r; sb_init(&r, 4096);

    if (strcmp(tc->name, "bash") == 0) {
        char *cmd = tc_str(tc, "command"); int to = (int)tc_num(tc, "timeout", AGENT_BASH_TIMEOUT);
        tool_bash(&r, cmd ? cmd : "", to); free(cmd);
    } else if (strcmp(tc->name, "read") == 0) {
        char *p = tc_str(tc, "path"); int off = (int)tc_num(tc, "offset", 1);
        int lim = (int)tc_num(tc, "limit", AGENT_READ_CHUNK);
        tool_read_file(&r, p ? p : "", off, lim, &s->last_read_path, &s->last_read_off);
        free(p);
    } else if (strcmp(tc->name, "more") == 0) {
        int lines = (int)tc_num(tc, "lines", AGENT_READ_CHUNK);
        if (!s->last_read_path) sb_puts(&r, "error: no previous read to continue");
        else tool_read_file(&r, s->last_read_path, s->last_read_off, lines,
                            &s->last_read_path, &s->last_read_off);
    } else if (strcmp(tc->name, "write") == 0) {
        char *p = tc_str(tc, "path"), *c = tc_str(tc, "content");
        tool_write_file(&r, p ? p : "", c ? c : ""); free(p); free(c);
    } else if (strcmp(tc->name, "edit") == 0) {
        char *p = tc_str(tc, "path"), *o = tc_str(tc, "old"), *n = tc_str(tc, "new");
        tool_edit_file(&r, p ? p : "", o ? o : "", n ? n : ""); free(p); free(o); free(n);
    } else if (strcmp(tc->name, "grep") == 0) {
        char *pt = tc_str(tc, "pattern"), *pa = tc_str(tc, "path"), *inc = tc_str(tc, "include");
        tool_grep(&r, pt ? pt : "", pa ? pa : ".", inc ? inc : "");
        free(pt); free(pa); free(inc);
    } else if (strcmp(tc->name, "list") == 0) {
        char *p = tc_str(tc, "path"); tool_list(&r, p ? p : "."); free(p);
    } else {
        sb_puts(&r, "error: unknown tool "); sb_puts(&r, tc->name);
    }
    return r.buf;
}

/* ============================================================================
 * Agent loop
 * ============================================================================ */

static void agent_run(agent_session *s, const char *user_msg) {
    /* Reset read state for this request */
    free(s->last_read_path); s->last_read_path = NULL; s->last_read_off = 0;

    /* Build conversation: system prompt + user message + assistant preamble */
    strbuf conv; sb_init(&conv, 65536);
    sb_puts(&conv, agent_system_prompt());
    sb_puts(&conv, "\n---\nUser: ");
    sb_puts(&conv, user_msg);
    sb_puts(&conv, "\nAssistant: ");

    for (int turn = 0; turn < AGENT_MAX_TURNS; turn++) {
        fprintf(stderr, "\n[agent] turn %d/%d ", turn + 1, AGENT_MAX_TURNS);
        fflush(stderr);

        char *raw = sess_generate(s, conv.buf, AGENT_TEMP, AGENT_MAX_TOKENS);
        if (!raw || !*raw) { free(raw); break; }

        tool_call tc;
        parse_tool_call(raw, &tc);

        if (tc.is_none || !tc.name[0]) {
            break;
        }

        /* Tool call detected — execute it */
        char *result = dispatch_tool(s, &tc);
        printf("[tool: %s]\n%.450s%s\n\n", tc.name, result,
               strlen(result) > 450 ? "..." : "");

        /* Append tool result to conversation */
        sb_puts(&conv, "\n<Tool result for "); sb_puts(&conv, tc.name);
        sb_puts(&conv, ">\n"); sb_puts(&conv, result);
        sb_puts(&conv, "\n</Tool result>\nAssistant: ");

        free(raw); free(result);
    }

    sb_free(&conv);
}

/* ============================================================================
 * Entry point
 * ============================================================================ */

static void usage(const char *p) {
    fprintf(stderr, "Usage: %s -m <model.gguf> [--gpu-layers N] [-t threads]\n", p);
}

int main(int argc, char **argv) {
    const char *mp = NULL; int gl = 0, nt = 4;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i+1<argc) mp = argv[++i];
        else if (!strcmp(argv[i], "--gpu-layers") && i+1<argc) gl = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i+1<argc) nt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (!mp) { usage(argv[0]); return 1; }

    hy3_model *m = NULL;
    if (hy3_model_load(&m, mp, nt) || !m)
        { fprintf(stderr, "[agent] load failed\n"); return 1; }
    if (gl > 0 && hy3_gpu_init(m, gl))
        fprintf(stderr, "[agent] GPU init failed, using CPU\n");

    agent_session s; sess_init(&s, m);
    fprintf(stderr, "[agent] ready. Type a request (Ctrl+D to exit).\n\n");

    char *ln = NULL; size_t lc = 0; ssize_t ll;
    while (1) {
        printf("\n> "); fflush(stdout);
        ll = getline(&ln, &lc, stdin); if (ll < 0) { printf("\n"); break; }
        while (ll > 0 && (ln[ll-1]=='\n' || ln[ll-1]=='\r')) ln[--ll] = '\0';
        if (!ll) continue;
        if (!strcmp(ln, "exit") || !strcmp(ln, "quit")) break;
        agent_run(&s, ln); fflush(stdout);
    }
    free(ln); sess_free(&s); hy3_model_free(m);
    return 0;
}
