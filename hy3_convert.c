/* hy3_convert.c - Convert HuggingFace safetensors to GGUF format.
 *
 * Reads a Hy3 model directory containing config.json, model.safetensors.index.json,
 * and the safetensors shards, and writes a single GGUF file.
 *
 * Usage: hy3_convert -i <input_dir> -o <output.gguf> [-t f32|q8_0|q4_k]
 *
 * -t f32 stores everything as F32 (reference/debug). -t q8_0 and -t q4_k are
 * equivalent and select a mixed-precision layout, see select_ggml_type().
 */

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* =========================================================================
 * GGUF Writer
 * ========================================================================= */

#define GGUF_MAGIC 0x46554747u

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} buf_t;

static void buf_init(buf_t *b, size_t cap) {
    b->ptr = malloc(cap);
    b->len = 0;
    b->cap = cap;
}

static void buf_write(buf_t *b, const void *data, size_t n) {
    while (b->len + n > b->cap) {
        b->cap = b->cap ? b->cap * 2 : 65536;
        b->ptr = realloc(b->ptr, b->cap);
    }
    memcpy(b->ptr + b->len, data, n);
    b->len += n;
}

static void buf_u8(buf_t *b, uint8_t v) { buf_write(b, &v, 1); }
static void buf_u16(buf_t *b, uint16_t v) { buf_write(b, &v, 2); }
static void buf_u32(buf_t *b, uint32_t v) { buf_write(b, &v, 4); }
static void buf_u64(buf_t *b, uint64_t v) { buf_write(b, &v, 8); }
static void buf_f32(buf_t *b, float v) { buf_write(b, &v, 4); }
static void buf_str(buf_t *b, const char *s) {
    size_t n = strlen(s);
    buf_u64(b, n);
    buf_write(b, s, n);
}

/* GGML type IDs matching GGUF */
#define GGML_TYPE_F32   0
#define GGML_TYPE_F16   1
#define GGML_TYPE_Q8_0  8
#define GGML_TYPE_Q4_K  12

typedef struct {
    uint32_t block_elems;
    uint32_t block_bytes;
} type_info;

static const type_info types[] = {
    [GGML_TYPE_F32]   = {1, 4},
    [GGML_TYPE_F16]   = {1, 2},
    [GGML_TYPE_Q8_0]  = {32, 36},
    [GGML_TYPE_Q4_K]  = {256, 144},
};

static uint64_t type_size(uint32_t t, uint64_t elems) {
    if (t >= sizeof(types)/sizeof(types[0])) return 0;
    return (elems / types[t].block_elems) * types[t].block_bytes;
}

static uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) & ~(a - 1);
}

/* =========================================================================
 * Safetensors Reader
 * ========================================================================= */

typedef struct {
    uint64_t offset;
    uint64_t size;
} st_entry;

typedef struct {
    char *name;
    int dtype;
    uint64_t shape[4];
    int ndim;
    st_entry entry;
    int shard_idx;
} st_tensor;

typedef struct {
    char **shard_paths;
    int n_shards;
    st_tensor *tensors;
    int n_tensors;
} st_db;

static void st_db_free(st_db *db) {
    for (int i = 0; i < db->n_tensors; i++) free(db->tensors[i].name);
    free(db->tensors);
    for (int i = 0; i < db->n_shards; i++) free(db->shard_paths[i]);
    free(db->shard_paths);
}

/* Safetensors dtype codes */
#define ST_DTYPE_F32  0
#define ST_DTYPE_F16  1
#define ST_DTYPE_BF16 2
#define ST_DTYPE_I64  3
#define ST_DTYPE_I32  4
#define ST_DTYPE_I16  5
#define ST_DTYPE_I8   6
#define ST_DTYPE_U8   7
#define ST_DTYPE_F8_E5M2 10
#define ST_DTYPE_F8_E4M3 11

static int st_dtype_bytes(int dtype) {
    switch (dtype) {
        case ST_DTYPE_F32:  return 4;
        case ST_DTYPE_F16:
        case ST_DTYPE_BF16: return 2;
        case ST_DTYPE_I64:  return 8;
        case ST_DTYPE_I32:  return 4;
        case ST_DTYPE_I16:  return 2;
        case ST_DTYPE_I8:
        case ST_DTYPE_U8:   return 1;
        case ST_DTYPE_F8_E5M2:
        case ST_DTYPE_F8_E4M3: return 1;
        default: return 0;
    }
}

static int st_dtype_to_ggml(int st_dtype) {
    switch (st_dtype) {
        case ST_DTYPE_F32:  return GGML_TYPE_F32;
        case ST_DTYPE_F16:
        case ST_DTYPE_BF16: return GGML_TYPE_F16;
        case ST_DTYPE_F8_E5M2:
        case ST_DTYPE_F8_E4M3: return GGML_TYPE_F32;
        case ST_DTYPE_I32:  return GGML_TYPE_F32;
        default: return -1;
    }
}

static char *read_file(const char *path, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *data = malloc(len + 1);
    if (fread(data, 1, len, fp) != (size_t)len) { free(data); fclose(fp); return NULL; }
    fclose(fp);
    data[len] = 0;
    *out_len = (size_t)len;
    return data;
}

static char *json_extract_value(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p && (isspace(*p) || *p == ':')) p++;
    if (!*p) return NULL;

    const char *start = p;
    const char *end;
    if (*p == '"') {
        p++;
        end = p;
        while (*end && *end != '"') {
            if (*end == '\\') end++;
            if (*end) end++;
        }
        end++;
    } else if (*p == '[') {
        end = p + 1;
        int depth = 1;
        while (*end && depth > 0) {
            if (*end == '\\') { end += 2; continue; }
            if (*end == '"') {
                end++;
                while (*end && *end != '"') {
                    if (*end == '\\') end++;
                    if (*end) end++;
                }
                if (*end) end++;
                continue;
            }
            if (*end == '[') depth++;
            if (*end == ']') depth--;
            end++;
        }
    } else if (*p == '{') {
        end = p + 1;
        int depth = 1;
        while (*end && depth > 0) {
            if (*end == '\\') { end += 2; continue; }
            if (*end == '"') {
                end++;
                while (*end && *end != '"') {
                    if (*end == '\\') end++;
                    if (*end) end++;
                }
                if (*end) end++;
                continue;
            }
            if (*end == '{') depth++;
            if (*end == '}') depth--;
            end++;
        }
    } else {
        end = p;
        while (*end && !isspace(*end) && *end != ',' && *end != '}') end++;
    }

    size_t n = (size_t)(end - start);
    char *s = malloc(n + 1);
    memcpy(s, start, n);
    s[n] = 0;
    return s;
}

static int st_db_open(st_db *db, const char *dir) {
    memset(db, 0, sizeof(*db));

    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir);
    size_t json_len;
    char *json = read_file(path, &json_len);
    if (!json) {
        fprintf(stderr, "error: cannot read %s\n", path);
        return -1;
    }

    char *wm = json_extract_value(json, "weight_map");
    if (!wm) { free(json); fprintf(stderr, "error: no weight_map in index\n"); return -1; }

    int n_entries = 0;
    const char *p = wm;
    while (*p) {
        if (*p == '"') n_entries++;
        p++;
    }
    n_entries /= 4;

    db->n_tensors = n_entries;
    db->tensors = calloc(n_entries, sizeof(st_tensor));

    char **unique_shards = calloc(n_entries, sizeof(char *));
    int n_unique = 0;

    p = wm + 1;
    for (int i = 0; i < n_entries; i++) {
        while (*p && *p != '"') p++;
        if (!*p) break;
        p++;
        const char *name_start = p;
        while (*p && *p != '"') p++;
        size_t nlen = (size_t)(p - name_start);
        db->tensors[i].name = malloc(nlen + 1);
        memcpy(db->tensors[i].name, name_start, nlen);
        db->tensors[i].name[nlen] = 0;

        p++;
        while (*p && isspace(*p)) p++;
        if (*p == ':') p++;
        while (*p && isspace(*p)) p++;
        if (*p != '"') break;
        p++;
        const char *val_start = p;
        while (*p && *p != '"') p++;
        size_t vlen = (size_t)(p - val_start);
        char shard[256];
        size_t slen = vlen < 255 ? vlen : 255;
        memcpy(shard, val_start, slen);
        shard[slen] = 0;

        p++;
        int found = -1;
        for (int j = 0; j < n_unique; j++)
            if (strcmp(unique_shards[j], shard) == 0) { found = j; break; }
        if (found < 0) {
            unique_shards[n_unique] = strdup(shard);
            found = n_unique;
            n_unique++;
        }

        db->tensors[i].shard_idx = found;
        if (*p == ',') p++;
    }
    free(wm);
    free(json);

    db->n_shards = n_unique;
    db->shard_paths = calloc(n_unique, sizeof(char *));
    for (int i = 0; i < n_unique; i++) {
        char spath[1024];
        snprintf(spath, sizeof(spath), "%s/%s", dir, unique_shards[i]);
        db->shard_paths[i] = strdup(spath);
        free(unique_shards[i]);
    }
    free(unique_shards);

    for (int i = 0; i < n_unique; i++) {
        int fd = open(db->shard_paths[i], O_RDONLY);
        if (fd == -1) {
            fprintf(stderr, "error: cannot open shard %s\n", db->shard_paths[i]);
            st_db_free(db);
            return -1;
        }

        uint64_t hdr_size;
        if (read(fd, &hdr_size, 8) != 8) {
            close(fd);
            fprintf(stderr, "error: cannot read shard header size\n");
            st_db_free(db);
            return -1;
        }

        if (hdr_size > 100 * 1024 * 1024) {
            fprintf(stderr, "error: shard header too large (%llu)\n", (unsigned long long)hdr_size);
            close(fd);
            st_db_free(db);
            return -1;
        }

        char *sjson = malloc((size_t)hdr_size + 1);
        if (read(fd, sjson, (size_t)hdr_size) != (ssize_t)hdr_size) {
            free(sjson);
            close(fd);
            fprintf(stderr, "error: cannot read shard header\n");
            st_db_free(db);
            return -1;
        }
        sjson[hdr_size] = 0;
        uint64_t data_offset = 8 + hdr_size;
        close(fd);

        for (int j = 0; j < db->n_tensors; j++) {
            if (db->tensors[j].shard_idx != i) continue;

            char *ts = json_extract_value(sjson, db->tensors[j].name);
            if (!ts) continue;

            char *dtype_str = json_extract_value(ts, "dtype");
            char *shape_str = json_extract_value(ts, "shape");
            char *offsets_str = json_extract_value(ts, "data_offsets");

            if (dtype_str) {
                if (strcmp(dtype_str, "\"F32\"") == 0) db->tensors[j].dtype = ST_DTYPE_F32;
                else if (strcmp(dtype_str, "\"F16\"") == 0) db->tensors[j].dtype = ST_DTYPE_F16;
                else if (strcmp(dtype_str, "\"BF16\"") == 0) db->tensors[j].dtype = ST_DTYPE_BF16;
                else if (strcmp(dtype_str, "\"I32\"") == 0) db->tensors[j].dtype = ST_DTYPE_I32;
                else if (strcmp(dtype_str, "\"F8_E5M2\"") == 0) db->tensors[j].dtype = ST_DTYPE_F8_E5M2;
                else if (strcmp(dtype_str, "\"F8_E4M3\"") == 0) db->tensors[j].dtype = ST_DTYPE_F8_E4M3;
                free(dtype_str);
            }

            if (shape_str) {
                db->tensors[j].ndim = 0;
                const char *sp = shape_str;
                while (*sp) {
                    if (*sp >= '0' && *sp <= '9') {
                        db->tensors[j].shape[db->tensors[j].ndim++] = strtoull(sp, (char **)&sp, 10);
                    } else sp++;
                }
                free(shape_str);
            }

            if (offsets_str) {
                sscanf(offsets_str + 1, "%llu", (unsigned long long *)&db->tensors[j].entry.offset);
                const char *comma = strchr(offsets_str, ',');
                if (comma) {
                    uint64_t end;
                    sscanf(comma + 1, "%llu", (unsigned long long *)&end);
                    db->tensors[j].entry.size = end - db->tensors[j].entry.offset;
                }
                db->tensors[j].entry.offset += data_offset;
                free(offsets_str);
            }
            free(ts);
        }
        free(sjson);
    }

    return 0;
}

static int st_read_tensor(st_db *db, int idx, float *out) {
    st_tensor *t = &db->tensors[idx];
    int shard_idx = t->shard_idx;

    static uint8_t *shard_data[256];
    static int shard_loaded[256];

    if (shard_idx >= 256) return -1;

    if (!shard_loaded[shard_idx]) {
        int fd = open(db->shard_paths[shard_idx], O_RDONLY);
        if (fd == -1) return -1;
        off_t sz = lseek(fd, 0, SEEK_END);
        shard_data[shard_idx] = mmap(NULL, (size_t)sz, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (shard_data[shard_idx] == MAP_FAILED) return -1;
        shard_loaded[shard_idx] = 1;
    }

    const uint8_t *data = shard_data[shard_idx];
    uint64_t data_off = t->entry.offset;

    uint64_t elems = 1;
    for (int d = 0; d < t->ndim; d++) elems *= t->shape[d];

    if (t->dtype == ST_DTYPE_F32) {
        memcpy(out, data + data_off, elems * sizeof(float));
    } else if (t->dtype == ST_DTYPE_F16) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)elems; i++) {
            uint16_t fp16;
            memcpy(&fp16, data + data_off + i * 2, 2);
            uint32_t sign = (uint32_t)(fp16 >> 15);
            uint32_t exp = (uint32_t)((fp16 >> 10) & 0x1f);
            uint32_t mant = (uint32_t)(fp16 & 0x3ff);
            uint32_t f32;
            if (exp == 0) {
                f32 = (sign << 31) | ((0x7f - 15) << 23) | (mant << 13);
            } else if (exp == 31) {
                f32 = (sign << 31) | 0x7f800000 | (mant << 13);
            } else {
                f32 = (sign << 31) | ((exp + 0x70) << 23) | (mant << 13);
            }
            memcpy(&out[i], &f32, 4);
        }
    } else if (t->dtype == ST_DTYPE_BF16) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)elems; i++) {
            uint16_t bf16;
            memcpy(&bf16, data + data_off + i * 2, 2);
            uint32_t f32 = (uint32_t)bf16 << 16;
            memcpy(&out[i], &f32, 4);
        }
    } else if (t->dtype == ST_DTYPE_F8_E5M2) {
        for (int64_t i = 0; i < (int64_t)elems; i++) {
            uint8_t fp8;
            memcpy(&fp8, data + data_off + i, 1);
            uint32_t sign = (uint32_t)(fp8 >> 7);
            uint32_t exp = (uint32_t)((fp8 >> 2) & 0x1f);
            uint32_t mant = (uint32_t)(fp8 & 0x3);
            uint32_t f32;
            if (exp == 0) {
                f32 = (sign << 31) | ((0x7f - 15) << 23) | (mant << 21);
            } else if (exp == 31) {
                f32 = (sign << 31) | 0x7f800000 | (mant << 21);
            } else {
                f32 = (sign << 31) | ((exp + 0x70) << 23) | (mant << 21);
            }
            memcpy(&out[i], &f32, 4);
        }
    } else if (t->dtype == ST_DTYPE_F8_E4M3) {
        for (uint64_t i = 0; i < elems; i++) {
            uint8_t fp8;
            memcpy(&fp8, data + data_off + i, 1);
            uint32_t sign = (uint32_t)(fp8 >> 7);
            uint32_t exp = (uint32_t)((fp8 >> 3) & 0xf);
            uint32_t mant = (uint32_t)(fp8 & 0x7);
            uint32_t f32;
            if (exp == 0) {
                f32 = (sign << 31) | ((0x7f - 7) << 23) | (mant << 20);
            } else if (exp == 15) {
                f32 = (sign << 31) | 0x7f800000 | (mant << 20);
            } else {
                f32 = (sign << 31) | ((exp + 0x70) << 23) | (mant << 20);
            }
            memcpy(&out[i], &f32, 4);
        }
    } else {
        return -1;
    }
    return 0;
}

#define QK_K 256

/* =========================================================================
 * FP16 + Q4_K Quantization
 * ========================================================================= */

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 1;
    uint32_t exp  = (x >> 23) & 0xff;
    uint32_t mant = x & 0x7fffff;
    if (exp == 0) return (uint16_t)(sign << 15);
    if (exp == 0xff) {
        uint16_t h = (uint16_t)((sign << 15) | 0x7c00);
        if (mant) h |= 1;
        return h;
    }
    int nexp = (int)exp - 127 + 15;
    if (nexp >= 31) return (uint16_t)((sign << 15) | 0x7c00);
    if (nexp <= 0) {
        mant = (mant | 0x800000) >> (1 - nexp);
        return (uint16_t)((sign << 15) | (mant >> 13));
    }
    return (uint16_t)((sign << 15) | ((uint32_t)nexp << 10) | (mant >> 13));
}

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15);
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1f);
    uint32_t mant = (uint32_t)(h & 0x3ff);
    uint32_t f32;
    if (exp == 0) f32 = (sign << 31) | ((0x7f - 15) << 23) | (mant << 13);
    else if (exp == 31) f32 = (sign << 31) | 0x7f800000 | (mant << 13);
    else f32 = (sign << 31) | ((exp + 0x70) << 23) | (mant << 13);
    float r; memcpy(&r, &f32, 4); return r;
}

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float make_qkx2_quants(int n, int nmax, const float *restrict x, const float *restrict weights,
        uint8_t *restrict L, float *restrict the_min, uint8_t *restrict Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0], max = x[0];
    float sum_w = weights[0], sum_x = sum_w * x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i]; sum_w += w; sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; i++) L[i] = 0;
        *the_min = -min; return 0.f;
    }
    float iscale = (float)nmax/(max - min), scale = 1/iscale;
    float best_error = 0;
    for (int i = 0; i < n; i++) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = (uint8_t)(l < 0 ? 0 : (l > nmax ? nmax : l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        best_error += weights[i] * diff;
    }
    for (int i = 0; i < n; i++) Laux[i] = L[i];
    float min_orig = min;
    for (int is = 0; is < nstep; is++) {
        if (is > 0) {
            min = rmin*(float)is/nstep + min_orig*(1 - (float)is/nstep);
            if (min > 0) min = 0;
        }
        float sumlx = 0; int suml2 = 0;
        for (int i = 0; i < n; i++) {
            int l = nearest_int(iscale*(x[i] - min));
            l = l < 0 ? 0 : (l > nmax ? nmax : l);
            L[i] = (uint8_t)l; sumlx += (x[i] - min)*l; suml2 += l*l;
        }
        scale = sumlx/(suml2 + 1e-10f);
        float error = 0;
        for (int i = 0; i < n; i++) {
            float diff = scale*L[i] + min - x[i];
            diff = use_mad ? fabsf(diff) : diff*diff;
            error += weights[i] * diff;
        }
        if (error < best_error) {
            best_error = error;
            for (int i = 0; i < n; i++) Laux[i] = L[i];
            min_orig = min;
        }
    }
    for (int i = 0; i < n; i++) L[i] = Laux[i];
    *the_min = -min_orig;
    return scale;
}

static void quantize_row_q4_K(const float *x, uint8_t *vy, int k) {
    typedef struct { uint16_t d,dmin; uint8_t scales[12]; uint8_t qs[QK_K/2]; } block_q4_K;
    block_q4_K *y = (block_q4_K *)vy;
    int nb = k / QK_K;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nb; i++) {
        uint8_t *L = (uint8_t *)malloc(QK_K);
        const float *xb = x + (size_t)i * QK_K;
        float sub_scale[8], sub_min[8];
        for (int j = 0; j < 8; j++) {
            float smin = FLT_MAX, smax = -FLT_MAX;
            for (int l = 0; l < 32; l++) {
                float v = xb[32*j + l];
                if (v < smin) smin = v; if (v > smax) smax = v;
            }
            sub_scale[j] = (smax - smin) / 15.0f;
            sub_min[j] = smin < 0 ? -smin : 0;
        }
        float max_s = 0, max_m = 0;
        for (int j = 0; j < 8; j++) { if (sub_scale[j] > max_s) max_s = sub_scale[j]; if (sub_min[j] > max_m) max_m = sub_min[j]; }
        if (max_s < 1e-10f) max_s = 1e-10f;
        if (max_m < 1e-10f) max_m = 1e-10f;
        float d = max_s / 63.0f, dmin = max_m / 63.0f;
        y[i].d = fp32_to_fp16(d); y[i].dmin = fp32_to_fp16(dmin);
        for (int j = 0; j < 8; j++) {
            int ls = (int)(sub_scale[j] / d + 0.5f), lm = (int)(sub_min[j] / dmin + 0.5f);
            if (ls < 0) ls = 0; if (ls > 63) ls = 63;
            if (lm < 0) lm = 0; if (lm > 63) lm = 63;
            uint8_t sc = (uint8_t)ls, m = (uint8_t)lm;
            if (j < 4) { y[i].scales[j] = sc; y[i].scales[j+4] = m; }
            else { y[i].scales[j+4] = (sc & 0xF) | ((m & 0xF) << 4); y[i].scales[j-4] |= ((sc >> 4) << 6); y[i].scales[j-0] |= ((m >> 4) << 6); }
            float eff_d = d * (float)ls, eff_dm = dmin * (float)lm;
            for (int ii = 0; ii < 32; ii++) {
                int q = (int)((xb[32*j + ii] + eff_dm) / (eff_d > 1e-10f ? eff_d : 1e-10f) + 0.5f);
                if (q < 0) q = 0; if (q > 15) q = 15;
                L[32*j + ii] = (uint8_t)q;
            }
        }
        uint8_t *q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; l++) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }
        free(L);
    }
}

/* =========================================================================
 * Per-tensor dtype selection
 *
 * Mixed precision scheme (used whenever -t is not f32):
 *   - routed experts (ffn_{gate,up,down}_exps)      -> Q4_K (bulk of the
 *     model; always Q4_K regardless of -t, same as before this change)
 *   - token_embd.weight                             -> F16
 *   - output.weight, attention q/k/v/o projections  -> Q8_0
 *   - shared-expert FFN + the single dense layer's
 *     FFN (both are *always active*, unlike routed
 *     experts, so they get the same protection as
 *     attention/output)                             -> Q8_0
 *   - norms, router gate, expert bias, unused MTP
 *     tensors (eh_proj/enorm/hnorm/final)            -> F32 (tiny; keep
 *     full precision)
 * `-t f32` is a debug escape hatch that keeps everything F32 except the
 * always-forced Q4_K experts (unchanged from before this change).
 * This function is the single source of truth for tensor dtype and is
 * called identically from both the tensor-info pass and the data-write
 * pass so they can never drift out of sync. */

static bool name_is_expert_weight(const char *name) {
    return (strstr(name, "ffn_gate_exps.") || strstr(name, "ffn_up_exps.") || strstr(name, "ffn_down_exps.")) &&
           !strstr(name, "_b.bias") && strstr(name, ".weight");
}

static bool name_is_f32_forced(const char *name) {
    return strstr(name, "_norm") || strstr(name, "norm.") || strstr(name, "_b.bias") ||
           strstr(name, "enorm") || strstr(name, "hnorm") || strstr(name, "final") ||
           strstr(name, "gate_inp") || strstr(name, "eh_proj");
}

static bool name_is_q8_0_forced(const char *name) {
    /* output projection + attention q/k/v/o */
    if (strcmp(name, "output.weight") == 0) return true;
    if (strstr(name, "attn_q.weight") || strstr(name, "attn_k.weight") ||
        strstr(name, "attn_v.weight") || strstr(name, "attn_output.weight")) return true;
    /* shared-expert FFN (MoE layers) and dense FFN (layer 0) -- always
     * active every token, so treated like attention rather than like the
     * sparsely-routed experts. */
    if (strstr(name, "ffn_gate_shexp.weight") || strstr(name, "ffn_up_shexp.weight") ||
        strstr(name, "ffn_down_shexp.weight")) return true;
    if (strstr(name, "ffn_gate.weight") || strstr(name, "ffn_up.weight") ||
        strstr(name, "ffn_down.weight")) return true;
    return false;
}

static uint32_t select_ggml_type(const char *name, uint32_t quant_type) {
    if (name_is_expert_weight(name)) return GGML_TYPE_Q4_K;
    if (quant_type == GGML_TYPE_F32) return GGML_TYPE_F32;
    if (name_is_f32_forced(name)) return GGML_TYPE_F32;
    if (strcmp(name, "token_embd.weight") == 0) return GGML_TYPE_F16;
    if (name_is_q8_0_forced(name)) return GGML_TYPE_Q8_0;
    return quant_type;
}

/* =========================================================================
 * Main Converter
 * ========================================================================= */

static void write_gguf_header(buf_t *b, const char *arch, int n_tensors, int n_kv) {
    buf_u32(b, GGUF_MAGIC);
    buf_u32(b, 3);     // version
    buf_u64(b, n_tensors);
    buf_u64(b, n_kv);
}

static void write_kv_str(buf_t *b, const char *key, const char *val) {
    buf_str(b, key);
    buf_u32(b, 8);     // GGUF_VALUE_STRING
    buf_str(b, val);
}

static void write_kv_u32(buf_t *b, const char *key, uint32_t val) {
    buf_str(b, key);
    buf_u32(b, 4);     // GGUF_VALUE_UINT32
    buf_u32(b, val);
}

static void write_kv_f32(buf_t *b, const char *key, float val) {
    buf_str(b, key);
    buf_u32(b, 6);     // GGUF_VALUE_FLOAT32
    buf_f32(b, val);
}

static void write_kv_arr_start(buf_t *b, const char *key, uint32_t elem_type, uint64_t n) {
    buf_str(b, key);
    buf_u32(b, 9);     // GGUF_VALUE_ARRAY
    buf_u32(b, elem_type);
    buf_u64(b, n);
}

static int count_tokenizer_kv(const char *dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    size_t json_len;
    char *json = read_file(path, &json_len);
    if (!json) return 0; /* no tokenizer file */

    /* Count tokens in vocab object */
    char *model_vocab = json_extract_value(json, "vocab");
    if (!model_vocab) { free(json); return 0; }

    int count = 0;
    const char *p = model_vocab;
    while (*p) { if (*p == '"') count++; if (*p) p++; }
    count /= 2; /* each entry has 2 quotes: "key": value */
    free(model_vocab);
    free(json);

    return count;
}

/* We need a function to count, and a function to write.
 * The count is used to set n_kv before writing.
 * For simplicity and to avoid parsing tokenizer.json twice, we do:
 * 1. Count to set n_kv
 * 2. Write everything including tokenizer in one pass */

static void write_tokenizer_data(buf_t *hdr, const char *dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    size_t json_len;
    char *json = read_file(path, &json_len);
    if (!json) {
        fprintf(stderr, "warning: cannot read tokenizer.json\n");
        return;
    }

    char *model_vocab = json_extract_value(json, "vocab");
    if (!model_vocab) { free(json); fprintf(stderr, "warning: no vocab in tokenizer.json\n"); return; }

    /* Count tokens */
    int capacity = 0;
    const char *p = model_vocab;
    while (*p) { if (*p == '"') capacity++; if (*p) p++; }
    capacity /= 2; /* each entry has 2 quotes: "key": value */
    if (capacity <= 0) { free(model_vocab); free(json); return; }

    typedef struct { int id; char *text; } tok_entry;
    tok_entry *tokens = calloc(capacity, sizeof(tok_entry));
    int n_tok = 0;

    p = model_vocab + 1;
    for (int i = 0; i < capacity; i++) {
        while (*p && *p != '"') p++;
        if (!*p) break;
        p++;
        const char *ts = p;
        while (*p && *p != '"') { if (*p == '\\') p++; if (*p) p++; }
        size_t tlen = (size_t)(p - ts);
        tokens[n_tok].text = malloc(tlen + 1);
        memcpy(tokens[n_tok].text, ts, tlen);
        tokens[n_tok].text[tlen] = 0;
        /* Unescape */
        char *src = tokens[n_tok].text, *dst = src;
        while (*src) {
            if (*src == '\\' && *(src+1)) {
                src++;
                if (*src == 'n') *dst++ = '\n';
                else if (*src == 't') *dst++ = '\t';
                else if (*src == 'r') *dst++ = '\r';
                else if (*src == '\\') *dst++ = '\\';
                else if (*src == '"') *dst++ = '"';
                else *dst++ = *src;
            } else { *dst++ = *src; }
            src++;
        }
        *dst = 0;
        p++;
        while (*p && isspace(*p)) p++;
        if (*p == ':') p++;
        while (*p && isspace(*p)) p++;
        int id = 0;
        while (*p >= '0' && *p <= '9') { id = id * 10 + (*p - '0'); p++; }
        tokens[n_tok].id = id;
        n_tok++;
        if (*p == ',') p++;
    }
    free(model_vocab);

    /* Sort by id */
    for (int i = 0; i < n_tok; i++)
        for (int j = i+1; j < n_tok; j++)
            if (tokens[j].id < tokens[i].id) {
                tok_entry tmp = tokens[i]; tokens[i] = tokens[j]; tokens[j] = tmp;
            }

    /* Also read added_tokens */
    char *added = json_extract_value(json, "added_tokens");
    if (added) {
        const char *ap = added + 1;
        while (*ap && *ap != ']') {
            while (*ap && *ap != '{') { if (*ap == ']') goto done_added; ap++; }
            if (*ap != '{') break;
            ap++;
            char *id_str = json_extract_value(ap, "id");
            char *content_str = json_extract_value(ap, "content");
            if (id_str && content_str) {
                int aid = atoi(id_str);
                int found = 0;
                for (int i = 0; i < n_tok; i++)
                    if (tokens[i].id == aid) { found = 1; break; }
                if (!found) {
                    n_tok++;
                    tokens = realloc(tokens, n_tok * sizeof(tok_entry));
                    tokens[n_tok-1].id = aid;
                    size_t clen = strlen(content_str);
                    if (clen >= 2 && content_str[0] == '"' && content_str[clen-1] == '"') {
                        content_str[clen-1] = 0;
                        tokens[n_tok-1].text = strdup(content_str + 1);
                        content_str[clen-1] = '"';
                    } else tokens[n_tok-1].text = strdup(content_str);
                    for (int k = n_tok-1; k > 0 && tokens[k].id < tokens[k-1].id; k--) {
                        tok_entry tmp = tokens[k]; tokens[k] = tokens[k-1]; tokens[k-1] = tmp;
                    }
                }
            }
            free(id_str); free(content_str);
            while (*ap && *ap != '}') ap++;
            if (*ap == '}') ap++;
            while (*ap && isspace(*ap)) ap++;
        }
    done_added:
        free(added);
    }
    free(json);

    /* Write tokenizer KV entries */
    write_kv_arr_start(hdr, "tokenizer.ggml.tokens", 8, n_tok);
    for (int i = 0; i < n_tok; i++)
        buf_str(hdr, tokens[i].text);

    write_kv_arr_start(hdr, "tokenizer.ggml.scores", 6, n_tok);
    for (int i = 0; i < n_tok; i++)
        buf_f32(hdr, 0.0f);

    fprintf(stderr, "read_tokenizer: %d tokens\n", n_tok);
    for (int i = 0; i < n_tok; i++) free(tokens[i].text);
    free(tokens);
}

static void write_tensor_info(buf_t *b, const char *name, uint32_t type,
                               int ndim, const uint64_t *dims, uint64_t offset) {
    buf_str(b, name);
    buf_u32(b, ndim);
    for (int d = 0; d < ndim; d++) buf_u64(b, dims[d]);
    buf_u32(b, type);
    buf_u64(b, offset);
}

static int hy3_convert(const char *input_dir, const char *output_path,
                       int quant_type) {
    st_db db;
    if (st_db_open(&db, input_dir) != 0) {
        fprintf(stderr, "error: failed to open safetensors database\n");
        return -1;
    }

    fprintf(stderr, "hy3-convert: %d tensors, %d shards\n", db.n_tensors, db.n_shards);

    /* Build GGUF metadata and tensor directory. We use a two-pass approach:
     * first write header + metadata + tensor info, then write tensor data. */

    buf_t hdr;
    buf_init(&hdr, 65536);

    buf_t tensors_buf;
    buf_init(&tensors_buf, 65536);

    int n_tensors = db.n_tensors;

    /* Count tokenizer entries (+4 for tokens array, scores array, bos, eos) */
    int n_tok = count_tokenizer_kv(input_dir);
    int n_kv = 19 + (n_tok > 0 ? 4 : 0);

    write_gguf_header(&hdr, "hy_v3", n_tensors, n_kv);

    write_kv_str(&hdr, "general.architecture", "hy_v3");
    write_kv_str(&hdr, "general.name", "HYV3");
    write_kv_u32(&hdr, "hy_v3.block_count", 80);
    write_kv_u32(&hdr, "hy_v3.context_length", 262144);
    write_kv_u32(&hdr, "hy_v3.embedding_length", 4096);
    write_kv_u32(&hdr, "hy_v3.feed_forward_length", 13312);
    write_kv_u32(&hdr, "hy_v3.attention.head_count", 64);
    write_kv_u32(&hdr, "hy_v3.attention.head_count_kv", 8);
    write_kv_u32(&hdr, "hy_v3.attention.key_length", 128);
    write_kv_u32(&hdr, "hy_v3.attention.value_length", 128);
    write_kv_u32(&hdr, "hy_v3.rope.freq_base", 11158840);
    write_kv_u32(&hdr, "hy_v3.expert_count", 192);
    write_kv_u32(&hdr, "hy_v3.expert_used_count", 8);
    write_kv_u32(&hdr, "hy_v3.expert_feed_forward_length", 1536);
    write_kv_u32(&hdr, "hy_v3.expert_shared_count", 1);
    write_kv_u32(&hdr, "hy_v3.vocab_size", 120832);
    write_kv_f32(&hdr, "hy_v3.attention.layer_norm_rms_epsilon", 1e-5f);
    write_kv_u32(&hdr, "hy_v3.rope.dimension_count", 64);

    uint64_t alignment = 32;
    write_kv_u32(&hdr, "general.alignment", alignment);

    /* Write tokenizer data */
    if (n_tok > 0) {
        write_tokenizer_data(&hdr, input_dir);
        write_kv_u32(&hdr, "tokenizer.ggml.bos_token_id", 120000);
        write_kv_u32(&hdr, "tokenizer.ggml.eos_token_id", 120025);
    }

    /* Calculate data offsets while building tensor directory */
    uint64_t data_offset = 0;

    /* HF -> GGUF name mapping */
    for (int i = 0; i < db.n_tensors; i++) {
        char gguf_name[256];
        const char *hf_name = db.tensors[i].name;
        int expert_idx = -1;

        /* Map HF tensor names to GGUF names */
        if (strcmp(hf_name, "model.embed_tokens.weight") == 0) {
            strcpy(gguf_name, "token_embd.weight");
        } else if (strcmp(hf_name, "model.norm.weight") == 0) {
            strcpy(gguf_name, "output_norm.weight");
        } else if (strcmp(hf_name, "lm_head.weight") == 0) {
            strcpy(gguf_name, "output.weight");
        } else if (strncmp(hf_name, "model.layers.", 13) == 0) {
            int layer;
            char rest[128];
            if (sscanf(hf_name, "model.layers.%d.%s", &layer, rest) == 2) {
                if (strcmp(rest, "input_layernorm.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.attn_norm.weight", layer);
                } else if (strcmp(rest, "post_attention_layernorm.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_norm.weight", layer);
                } else if (strcmp(rest, "self_attn.q_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.attn_q.weight", layer);
                } else if (strcmp(rest, "self_attn.k_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.attn_k.weight", layer);
                } else if (strcmp(rest, "self_attn.v_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.attn_v.weight", layer);
                } else if (strcmp(rest, "self_attn.o_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.attn_output.weight", layer);
                } else if (strcmp(rest, "self_attn.q_norm.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.attn_q_norm.weight", layer);
                } else if (strcmp(rest, "self_attn.k_norm.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.attn_k_norm.weight", layer);
                } else if (strcmp(rest, "mlp.gate_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_gate.weight", layer);
                } else if (strcmp(rest, "mlp.up_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_up.weight", layer);
                } else if (strcmp(rest, "mlp.down_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_down.weight", layer);
                } else if (strcmp(rest, "mlp.shared_mlp.gate_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_gate_shexp.weight", layer);
                } else if (strcmp(rest, "mlp.shared_mlp.up_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_up_shexp.weight", layer);
                } else if (strcmp(rest, "mlp.shared_mlp.down_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_down_shexp.weight", layer);
                } else if (strcmp(rest, "mlp.router.gate.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_gate_inp.weight", layer);
                } else if (strcmp(rest, "mlp.expert_bias") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_gate_exps_b.bias", layer);
                } else if (strncmp(rest, "mlp.experts.", 12) == 0) {
                    char expert_name[64];
                    if (sscanf(rest, "mlp.experts.%d.%s", &expert_idx, expert_name) == 2) {
                        if (strcmp(expert_name, "gate_proj.weight") == 0) {
                            snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_gate_exps.%d.gate_proj.weight", layer, expert_idx);
                        } else if (strcmp(expert_name, "up_proj.weight") == 0) {
                            snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_up_exps.%d.up_proj.weight", layer, expert_idx);
                        } else if (strcmp(expert_name, "down_proj.weight") == 0) {
                            snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_down_exps.%d.down_proj.weight", layer, expert_idx);
                        } else {
                            strcpy(gguf_name, hf_name);
                        }
                    } else {
                        strcpy(gguf_name, hf_name);
                    }
                } else if (strcmp(rest, "eh_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.eh_proj.weight", layer);
                } else if (strcmp(rest, "enorm.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.enorm.weight", layer);
                } else if (strcmp(rest, "hnorm.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.hnorm.weight", layer);
                } else if (strcmp(rest, "final_layernorm.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.final_norm.weight", layer);
                } else {
                    strcpy(gguf_name, hf_name);
                }
            } else {
                strcpy(gguf_name, hf_name);
            }
        } else {
            strcpy(gguf_name, hf_name);
        }

        st_tensor *t = &db.tensors[i];
        uint64_t dims[4];
        for (int d = 0; d < t->ndim; d++) dims[d] = t->shape[d];

        uint64_t elems = 1;
        for (int d = 0; d < t->ndim; d++) elems *= dims[d];

        const char *name = gguf_name;
        uint32_t ggml_type = select_ggml_type(name, quant_type);
        uint64_t tensor_bytes = type_size(ggml_type, elems);

        write_tensor_info(&tensors_buf, gguf_name, ggml_type, t->ndim, dims, data_offset);
        data_offset += tensor_bytes;
        data_offset = align_up(data_offset, alignment);
    }

    /* Write file */
    FILE *fp = fopen(output_path, "wb");
    if (!fp) {
        fprintf(stderr, "error: cannot write %s: %s\n", output_path, strerror(errno));
        st_db_free(&db);
        return -1;
    }

    /* Pad header to alignment boundary */
    uint64_t hdr_end = hdr.len + tensors_buf.len;
    uint64_t data_start = align_up(hdr_end, alignment);
    size_t pad = (size_t)(data_start - hdr_end);

    fwrite(hdr.ptr, 1, hdr.len, fp);
    fwrite(tensors_buf.ptr, 1, tensors_buf.len, fp);
    if (pad > 0) {
        uint8_t zero = 0;
        for (size_t i = 0; i < pad; i++) fwrite(&zero, 1, 1, fp);
    }

    /* Write tensor data */
    uint64_t write_offset = 0;
    fprintf(stderr, "hy3-convert: writing tensors...\n");
    for (int i = 0; i < db.n_tensors; i++) {
        if (i > 0 && i % 1000 == 0)
            fprintf(stderr, "hy3-convert: tensor %d/%d (%.1f%%)\n", i, db.n_tensors, 100.0 * i / db.n_tensors);
        st_tensor *t = &db.tensors[i];
        uint64_t elems = 1;
        for (int d = 0; d < t->ndim; d++) elems *= t->shape[d];

        char gguf_name[256];
        /* Recompute gguf_name (same as above) */
        if (strcmp(t->name, "model.embed_tokens.weight") == 0)
            strcpy(gguf_name, "token_embd.weight");
        else if (strcmp(t->name, "model.norm.weight") == 0)
            strcpy(gguf_name, "output_norm.weight");
        else if (strcmp(t->name, "lm_head.weight") == 0)
            strcpy(gguf_name, "output.weight");
        else {
            int layer;
            char rest[128];
            if (sscanf(t->name, "model.layers.%d.%s", &layer, rest) == 2) {
                if (strcmp(rest, "mlp.expert_bias") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_gate_exps_b.bias", layer);
                } else if (strcmp(rest, "mlp.router.gate.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_gate_inp.weight", layer);
                } else if (strcmp(rest, "mlp.gate_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_gate.weight", layer);
                } else if (strcmp(rest, "mlp.up_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_up.weight", layer);
                } else if (strcmp(rest, "mlp.down_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_down.weight", layer);
                } else if (strcmp(rest, "mlp.shared_mlp.gate_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_gate_shexp.weight", layer);
                } else if (strcmp(rest, "mlp.shared_mlp.up_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_up_shexp.weight", layer);
                } else if (strcmp(rest, "mlp.shared_mlp.down_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_down_shexp.weight", layer);
                } else if (strcmp(rest, "input_layernorm.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.attn_norm.weight", layer);
                } else if (strcmp(rest, "post_attention_layernorm.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_norm.weight", layer);
                } else if (strcmp(rest, "self_attn.q_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.attn_q.weight", layer);
                } else if (strcmp(rest, "self_attn.k_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.attn_k.weight", layer);
                } else if (strcmp(rest, "self_attn.v_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.attn_v.weight", layer);
                } else if (strcmp(rest, "self_attn.o_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.attn_output.weight", layer);
                } else if (strcmp(rest, "self_attn.q_norm.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.attn_q_norm.weight", layer);
                } else if (strcmp(rest, "self_attn.k_norm.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.attn_k_norm.weight", layer);
                } else if (strcmp(rest, "eh_proj.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.eh_proj.weight", layer);
                } else if (strcmp(rest, "enorm.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.enorm.weight", layer);
                } else if (strcmp(rest, "hnorm.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.hnorm.weight", layer);
                } else if (strcmp(rest, "final_layernorm.weight") == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "blk.%d.final_norm.weight", layer);
                } else if (strncmp(rest, "mlp.experts.", 12) == 0) {
                    int expert_num;
                    char weight_type[64];
                    if (sscanf(rest, "mlp.experts.%d.%s", &expert_num, weight_type) == 2) {
                        if (strcmp(weight_type, "gate_proj.weight") == 0)
                            snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_gate_exps.%d.gate_proj.weight", layer, expert_num);
                        else if (strcmp(weight_type, "up_proj.weight") == 0)
                            snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_up_exps.%d.up_proj.weight", layer, expert_num);
                        else if (strcmp(weight_type, "down_proj.weight") == 0)
                            snprintf(gguf_name, sizeof(gguf_name), "blk.%d.ffn_down_exps.%d.down_proj.weight", layer, expert_num);
                        else
                            snprintf(gguf_name, sizeof(gguf_name), "%s", t->name);
                    } else {
                        snprintf(gguf_name, sizeof(gguf_name), "%s", t->name);
                    }
                } else {
                    snprintf(gguf_name, sizeof(gguf_name), "%s", t->name);
                }
            } else {
                snprintf(gguf_name, sizeof(gguf_name), "%s", t->name);
            }
        }

        uint32_t ggml_type = select_ggml_type(gguf_name, quant_type);
        uint64_t tensor_bytes = type_size(ggml_type, elems);

        float *f32_buf = malloc(elems * sizeof(float));
        if (st_read_tensor(&db, i, f32_buf) != 0) {
            fprintf(stderr, "error: failed to read tensor %s\n", t->name);
            free(f32_buf);
            continue;
        }

        if (strcmp(t->name, "model.embed_tokens.weight") == 0) {
            fprintf(stderr, "embed_tokens.weight: tensor_bytes=%llu ggml_type=%u write_offset=%llu (%.3f GB)\n",
                    (unsigned long long)tensor_bytes, ggml_type, (unsigned long long)write_offset,
                    write_offset / 1e9);
            fprintf(stderr, "  first 8: ");
            for (int k = 0; k < 8; k++) fprintf(stderr, "%f ", f32_buf[k]);
            fprintf(stderr, "\n");
        }

        if (ggml_type == GGML_TYPE_F32) {
            fwrite(f32_buf, sizeof(float), (size_t)elems, fp);
        } else if (ggml_type == GGML_TYPE_F16) {
            uint16_t *hbuf = malloc(elems * sizeof(uint16_t));
            #pragma omp parallel for schedule(static)
            for (int64_t k = 0; k < (int64_t)elems; k++) hbuf[k] = fp32_to_fp16(f32_buf[k]);
            fwrite(hbuf, sizeof(uint16_t), (size_t)elems, fp);
            free(hbuf);
        } else if (ggml_type == GGML_TYPE_Q8_0) {
            uint64_t nb = elems / 32;
            typedef struct { float d; int8_t qs[32]; } q8_block;
            uint8_t *qbuf = malloc(nb * sizeof(q8_block));
            #pragma omp parallel for schedule(static)
            for (uint64_t j = 0; j < nb; j++) {
                float amax = 0.0f;
                for (int k = 0; k < 32; k++) {
                    float a = fabsf(f32_buf[j * 32 + k]);
                    if (a > amax) amax = a;
                }
                float d = amax / 127.0f;
                if (d < 1e-10f) d = 1e-10f;
                float id = 1.0f / d;
                q8_block *blk = &((q8_block *)qbuf)[j];
                blk->d = d;
                for (int k = 0; k < 32; k++) {
                    float v = f32_buf[j * 32 + k] * id;
                    int q = (int)(v + (v >= 0.0f ? 0.5f : -0.5f)); /* round, not truncate */
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    blk->qs[k] = (int8_t)q;
                }
            }
            for (uint64_t j = 0; j < nb; j++) {
                q8_block *blk = &((q8_block *)qbuf)[j];
                fwrite(&blk->d, sizeof(float), 1, fp);
                fwrite(blk->qs, 1, 32, fp);
            }
            free(qbuf);
        } else if (ggml_type == GGML_TYPE_Q4_K) {
            uint64_t nb = elems / QK_K;
            uint8_t *qbuf = malloc(nb * 144);
            quantize_row_q4_K(f32_buf, qbuf, (int)(elems));
            fwrite(qbuf, 1, nb * 144, fp);
            free(qbuf);
        }

        free(f32_buf);

        uint64_t padded = align_up(tensor_bytes, alignment);
        uint64_t cur_pos = data_start + write_offset + tensor_bytes;
        while (cur_pos < data_start + write_offset + padded) {
            uint8_t zero = 0;
            fwrite(&zero, 1, 1, fp);
            cur_pos++;
        }
        write_offset += padded;
    }

    fclose(fp);
    fprintf(stderr, "hy3-convert: wrote %s\n", output_path);

    st_db_free(&db);
    free(hdr.ptr);
    free(tensors_buf.ptr);
    return 0;
}

int main(int argc, char **argv) {
    const char *input_dir = NULL;
    const char *output_path = NULL;
    int quant_type = GGML_TYPE_F32;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) input_dir = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            const char *t = argv[++i];
            if (strcmp(t, "f32") == 0) quant_type = GGML_TYPE_F32;
            else if (strcmp(t, "q8_0") == 0) quant_type = GGML_TYPE_Q8_0;
            else if (strcmp(t, "q4_k") == 0) quant_type = GGML_TYPE_Q4_K;
            else { fprintf(stderr, "unknown type: %s (use f32, q8_0, q4_k)\n", t); return 1; }
        } else if (strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: hy3_convert -i <input_dir> -o <output.gguf> [-t f32|q8_0|q4_k]\n");
            fprintf(stderr, "  f32          everything F32 (reference/debug; huge output)\n");
            fprintf(stderr, "  q8_0, q4_k   mixed precision (both equivalent): routed experts -> Q4_K,\n");
            fprintf(stderr, "               token_embd -> F16, attention q/k/v/o + output.weight +\n");
            fprintf(stderr, "               shared-expert/dense FFN -> Q8_0, norms/router/bias -> F32\n");
            return 0;
        }
    }

    if (!input_dir || !output_path) {
        fprintf(stderr, "Usage: hy3_convert -i <input_dir> -o <output.gguf> [-t f32|q8_0|q4_k]\n");
        return 1;
    }

    return hy3_convert(input_dir, output_path, quant_type);
}
