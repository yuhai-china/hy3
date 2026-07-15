/* hy3_metal.m - Metal backend for macOS / Apple Silicon.
 *
 * Design (deliberately different from hy3_gpu.cu's CUDA backend, because
 * Apple Silicon has unified memory instead of a discrete GPU with its own
 * VRAM budget):
 *
 *   - The GGUF file is already mmap'd by hy3_model_load() (m->gguf.map).
 *     Rather than dequantizing weights on the CPU and cudaMemcpy'ing a copy
 *     into device VRAM (what the CUDA backend must do), we wrap the *same*
 *     mmap'd pages as one or more zero-copy MTLBuffers
 *     (newBufferWithBytesNoCopy, MTLResourceStorageModeShared). The GPU
 *     dereferences the exact same physical memory the CPU mmap'd -- no
 *     duplication, which matters because this model (173GB+) would not fit
 *     in memory twice even on a 192GB Mac Studio.
 *   - Because there's no separate VRAM to run out of, ALL 80 layers run on
 *     Metal; there is no CPU-tail-layers split like --gpu-layers on CUDA.
 *   - A single MTLBuffer has a device-specific maxBufferLength that is
 *     usually smaller than the whole model, so the mmap is wrapped as
 *     several overlapping "views" (mirroring ds4_metal.m's approach): the
 *     overlap between adjacent views is sized to be larger than the
 *     largest single tensor, so every tensor is guaranteed to lie
 *     entirely within at least one view and no kernel ever needs a tensor
 *     split across two buffers.
 *   - Quantized formats (Q8_0, Q4_K) are dequantized *inline* inside the
 *     matmul kernels (see hy3.metal) rather than pre-expanded to F32,
 *     since we no longer have a "dequantize once during upload" step.
 *   - The KV cache and small per-token scratch buffers are ordinary
 *     (copying) MTLBuffers -- they're mutable and tiny compared to the
 *     weights.
 *
 * All math (RoPE convention, KV-cache interleaving, Q4_K bit layout, MoE
 * routing) mirrors the already-validated hy3.c / hy3_gpu.cu implementations
 * exactly; see hy3.metal's header comment for the specifics.
 *
 * This backend has been built and tested on Apple Silicon (macOS, clang +
 * Metal toolchain); see run_metal.sh for the build/run steps.
 */
#ifdef __cplusplus
extern "C" {
#endif
#include "hy3.h"
#ifdef __cplusplus
}
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <time.h>

#ifndef HY3_METAL_SHADER_PATH
#define HY3_METAL_SHADER_PATH "hy3.metal"
#endif

#define HY3_METAL_MAX_VIEWS 64

#define METAL_CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "hy3_metal: %s\n", (msg)); exit(1); } \
} while (0)

typedef struct {
    id<MTLBuffer> buffer;
    uint64_t model_offset; /* byte offset into m->gguf.map that buffer[0] corresponds to */
    uint64_t length;
} hy3_metal_view_t;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;

    id<MTLComputePipelineState> pipe_rms_norm;
    id<MTLComputePipelineState> pipe_rms_norm_offset;
    id<MTLComputePipelineState> pipe_rms_norm_heads;
    id<MTLComputePipelineState> pipe_rms_norm_heads_rope;
    id<MTLComputePipelineState> pipe_silu_mul;
    id<MTLComputePipelineState> pipe_sigmoid;
    id<MTLComputePipelineState> pipe_add;
    id<MTLComputePipelineState> pipe_scale_add;
    id<MTLComputePipelineState> pipe_fill_zero;
    id<MTLComputePipelineState> pipe_embed_f16;
    id<MTLComputePipelineState> pipe_embed_f32;
    id<MTLComputePipelineState> pipe_rope;
    id<MTLComputePipelineState> pipe_attention;
    id<MTLComputePipelineState> pipe_attention_split;
    id<MTLComputePipelineState> pipe_attention_split_q8;
    id<MTLComputePipelineState> pipe_attention_reduce;
    id<MTLComputePipelineState> pipe_matmul_f32;
    id<MTLComputePipelineState> pipe_matmul_f16;
    id<MTLComputePipelineState> pipe_matmul_q8_0;
    id<MTLComputePipelineState> pipe_matmul_q8_0_mm;
    id<MTLComputePipelineState> pipe_matmul_q4_k;
    id<MTLComputePipelineState> pipe_matmul_q4_k_mm;
    id<MTLComputePipelineState> pipe_kv_cache_write;
    id<MTLComputePipelineState> pipe_kv_cache_write_q8;
    id<MTLComputePipelineState> pipe_touch;

    /* Fast MoE path pipelines. */
    id<MTLComputePipelineState> pipe_router_topk;
    id<MTLComputePipelineState> pipe_matmul_q4_k_id;
    id<MTLComputePipelineState> pipe_moe_swiglu_id;
    id<MTLComputePipelineState> pipe_moe_combine_id;

    hy3_metal_view_t views[HY3_METAL_MAX_VIEWS];
    int n_views;
    uint64_t max_tensor_bytes;
    id residency_set;   /* MTLResidencySet (macOS 15+); nil if unavailable/disabled */

    /* Mutable per-token working buffers (all MTLResourceStorageModeShared,
     * i.e. directly readable/writable from the CPU once the command buffer
     * that wrote them has completed -- Apple Silicon unified memory needs
     * no explicit device->host copy). */
    id<MTLBuffer> d_embed;       /* HY3_N_EMBD: residual stream x */
    id<MTLBuffer> d_s;           /* HY3_N_EMBD: rms_norm scratch */
    id<MTLBuffer> d_q;           /* HY3_N_HEAD*HY3_HEAD_DIM */
    id<MTLBuffer> d_k;           /* HY3_N_KV_HEAD*HY3_HEAD_DIM */
    id<MTLBuffer> d_v;           /* HY3_N_KV_HEAD*HY3_HEAD_DIM */
    id<MTLBuffer> d_attn_out;    /* HY3_N_HEAD*HY3_HEAD_DIM */
    id<MTLBuffer> d_attn_partials; /* n_heads * ATTN_SPLITS * (2 + head_dim) floats */
    id<MTLBuffer> d_o_proj;      /* HY3_N_EMBD */
    id<MTLBuffer> d_gate;        /* max(HY3_DENSE_INTERMED, HY3_MOE_INTERMED) */
    id<MTLBuffer> d_up;          /* max(HY3_DENSE_INTERMED, HY3_MOE_INTERMED) */
    id<MTLBuffer> d_mid;         /* max(HY3_DENSE_INTERMED, HY3_MOE_INTERMED): silu(gate)*up */
    id<MTLBuffer> d_ffn_out;     /* HY3_N_EMBD */
    id<MTLBuffer> d_router;      /* HY3_N_EXPERT */
    id<MTLBuffer> d_expert_out;  /* HY3_N_EMBD accumulator across routed experts */
    id<MTLBuffer> d_logits;      /* HY3_N_VOCAB */

    /* Fast MoE path (ds4-style, GPU-resident routing). */
    id<MTLBuffer> d_router_ids;  /* HY3_N_EXPERT_USED int32 expert ids */
    id<MTLBuffer> d_router_wts;  /* HY3_N_EXPERT_USED float combine weights */
    id<MTLBuffer> d_rope_inv_freq; /* HY3_HEAD_DIM/2 precomputed 1/theta^(2d/dim) */
    float rope_attn_factor;        /* YaRN mscale, 1.0 when disabled */
    id<MTLBuffer> d_bias;        /* HY3_N_LAYER * HY3_N_EXPERT float expert bias (all layers, uploaded once) */
    id<MTLBuffer> d_gate_k;      /* HY3_N_EXPERT_USED * HY3_MOE_INTERMED */
    id<MTLBuffer> d_up_k;        /* HY3_N_EXPERT_USED * HY3_MOE_INTERMED */
    id<MTLBuffer> d_mid_k;       /* HY3_N_EXPERT_USED * HY3_MOE_INTERMED */
    id<MTLBuffer> d_down_k;      /* HY3_N_EXPERT_USED * HY3_N_EMBD */

    id<MTLBuffer> d_k_cache;
    id<MTLBuffer> d_v_cache;
    id<MTLBuffer> d_k_cache_q8;
    id<MTLBuffer> d_v_cache_q8;
    id<MTLBuffer> d_k_scales;
    id<MTLBuffer> d_v_scales;
    int ctx_cap_slots; /* capacity of d_k_cache/d_v_cache in interleaved slots */
    int concurrent;    /* 1 = concurrent encoder; helpers skip auto barriers and
                        * the caller inserts explicit barriers at dependency edges */
} hy3_metal_ctx_t;

#define METAL_ATTN_SPLITS 16

/* =========================================================================
 * Zero-copy model wrapping
 * ========================================================================= */

static uint64_t hy3_round_up(uint64_t v, uint64_t align) { return (v + align - 1) & ~(align - 1); }

static double hy3_metal_now(void);

/* Pin every wrapped model view into a single MTLResidencySet so the driver
 * establishes residency for the whole model at load time rather than faulting
 * it in lazily on the first inference token. No-op on macOS < 15 or when
 * HY3_METAL_RESIDENCY=0. */
static void hy3_metal_request_residency(hy3_metal_ctx_t *ctx) {
    const char *env = getenv("HY3_METAL_RESIDENCY");
    if (env && env[0] == '0') {
        fprintf(stderr, "hy3_metal: residency set disabled (HY3_METAL_RESIDENCY=0)\n");
        return;
    }
    if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"hy3_model";
        desc.initialCapacity = ctx->n_views;
        NSError *err = nil;
        id<MTLResidencySet> set = [ctx->device newResidencySetWithDescriptor:desc error:&err];
        if (!set) {
            fprintf(stderr, "hy3_metal: residency set creation failed: %s (falling back to demand paging)\n",
                    err ? err.localizedDescription.UTF8String : "unknown");
            return;
        }
        for (int i = 0; i < ctx->n_views; i++)
            [set addAllocation:ctx->views[i].buffer];
        [set commit];
        double t0 = hy3_metal_now();
        [set requestResidency];
        [ctx->queue addResidencySet:set];
        ctx->residency_set = set;
        fprintf(stderr, "hy3_metal: requested residency for %d view(s) in %.2fs\n",
                ctx->n_views, hy3_metal_now() - t0);
    } else {
        fprintf(stderr, "hy3_metal: MTLResidencySet needs macOS 15+; using demand paging\n");
    }
}

static bool hy3_metal_wrap_model(hy3_metal_ctx_t *ctx, hy3_model *m) {
    uint64_t page = (uint64_t)getpagesize();
    uintptr_t model_addr = (uintptr_t)m->gguf.map;
    if (model_addr % page != 0) {
        fprintf(stderr, "hy3_metal: GGUF mmap base is not page aligned (unexpected)\n");
        return false;
    }

    uint64_t max_tensor_bytes = 0;
    for (uint64_t i = 0; i < m->gguf.n_tensors; i++)
        if (m->gguf.tensors[i].bytes > max_tensor_bytes) max_tensor_bytes = m->gguf.tensors[i].bytes;
    ctx->max_tensor_bytes = max_tensor_bytes;

    uint64_t mapped_size = hy3_round_up(m->gguf.size, page);
    uint64_t max_buffer = (uint64_t)[ctx->device maxBufferLength];
    max_buffer &= ~(page - 1);
    if (max_buffer == 0) {
        fprintf(stderr, "hy3_metal: device reports maxBufferLength=0\n");
        return false;
    }

    /* Overlap must cover not just the largest single tensor but the largest
     * *contiguous expert block* (all 192 experts of one projection), because
     * the fast MoE id-kernel indexes experts as base + id*stride and needs the
     * whole block inside one view. The gate/up experts are interleaved so a
     * gate block spans ~2x its own bytes; use the observed max block span. */
    uint64_t max_expert_span = 0;
    for (int il = HY3_N_LAYER_DENSE; il < HY3_N_LAYER; il++) {
        hy3_layer_weights *l = &m->w.layers[il];
        if (!l->ffn_gate_exps[0].t || !l->ffn_gate_exps[HY3_N_EXPERT-1].t) continue;
        struct {
            const hy3_weight *lo, *hi;
        } blocks[3] = {
            { &l->ffn_gate_exps[0], &l->ffn_gate_exps[HY3_N_EXPERT-1] },
            { &l->ffn_up_exps[0],   &l->ffn_up_exps[HY3_N_EXPERT-1] },
            { &l->ffn_down_exps[0], &l->ffn_down_exps[HY3_N_EXPERT-1] },
        };
        for (int b = 0; b < 3; b++) {
            if (!blocks[b].lo->t || !blocks[b].hi->t) continue;
            uint64_t span = blocks[b].hi->t->abs_offset + blocks[b].hi->t->bytes
                          - blocks[b].lo->t->abs_offset;
            if (span > max_expert_span) max_expert_span = span;
        }
    }
    uint64_t overlap_bytes = max_tensor_bytes > max_expert_span ? max_tensor_bytes : max_expert_span;
    uint64_t overlap = hy3_round_up(overlap_bytes, page) + page;
    if (max_buffer <= overlap) {
        fprintf(stderr, "hy3_metal: maxBufferLength (%.2f GiB) too small for the largest expert block (%.2f GiB)\n",
                (double)max_buffer / 1e9, (double)overlap_bytes / 1e9);
        return false;
    }

    uint64_t step = max_buffer - overlap;
    uint64_t off = 0;
    for (;;) {
        if (ctx->n_views >= HY3_METAL_MAX_VIEWS) {
            fprintf(stderr, "hy3_metal: model needs more than %d views, increase HY3_METAL_MAX_VIEWS\n",
                    HY3_METAL_MAX_VIEWS);
            return false;
        }
        uint64_t view_bytes = mapped_size - off;
        if (view_bytes > max_buffer) view_bytes = max_buffer;

        /* Zero-copy: wrap the file mmap directly as a Metal buffer. The GGUF
         * mmap is prefaulted (read-touched) in hy3_model_load so every page is
         * resident, and iogpu.wired_limit_mb is raised to ~180 GiB so the GPU
         * can wire those pages (otherwise the GPU reads ZERO instead of
         * faulting the page in, silencing the model). Skip the prefault with
         * HY3_NO_PREFAULT only if the pages are guaranteed resident. */
        id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:(void *)(model_addr + off)
                                                              length:(NSUInteger)view_bytes
                                                             options:MTLResourceStorageModeShared
                                                       deallocator:nil];
        if (!buf) {
            fprintf(stderr, "hy3_metal: failed to wrap model view at offset %.2f GiB, size %.2f GiB\n",
                    (double)off / 1e9, (double)view_bytes / 1e9);
            return false;
        }
        ctx->views[ctx->n_views].buffer = buf;
        ctx->views[ctx->n_views].model_offset = off;
        ctx->views[ctx->n_views].length = view_bytes;
        ctx->n_views++;

        if (off + view_bytes >= mapped_size) break;
        off += step;
    }
    fprintf(stderr, "hy3_metal: wrapped %.2f GiB model into %d zero-copy view(s) (largest tensor %.2f MiB)\n",
            (double)m->gguf.size / 1e9, ctx->n_views, (double)max_tensor_bytes / 1e6);

    /* Pin all model views with an MTLResidencySet (macOS 15+). This lets the
     * driver build the GPU page tables / do VM validation and residency
     * accounting for the whole ~174GiB model up front, during load, instead of
     * lazily on the first inference token -- which is what made the first
     * prompt token pay a ~70s cold-start stall. Gated by HY3_METAL_RESIDENCY
     * (default on): set it to 0 to fall back to plain demand paging if a driver
     * ever serves zero for a pinned no-copy view (an old bug this used to hit
     * with a single giant overlapping view; the per-view set below is what ds4
     * uses successfully). */
    hy3_metal_request_residency(ctx);
    return true;
}

static double hy3_metal_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Warm the mmap'd model into GPU-accessible residency by touching one byte
 * per `stride` bytes of every view with a GPU kernel, so the first inference
 * token doesn't pay the cost of faulting the zero-copy weight pages in.
 * (Mirrors ds4_metal.m's kernel_touch_u8_stride approach.) */
static void hy3_metal_warm_views(hy3_metal_ctx_t *ctx) {
    const char *env = getenv("HY3_METAL_WARM");
    if (!(env && env[0] == '1')) return;

    uint32_t stride = 16384; /* one page (16KiB on Apple Silicon) */
    id<MTLBuffer> sink = [ctx->device newBufferWithLength:sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
    if (!sink) return;

    double t0 = hy3_metal_now();
    for (int i = 0; i < ctx->n_views; i++) {
        hy3_metal_view_t *v = &ctx->views[i];
        uint32_t n_steps = (uint32_t)((v->length + stride - 1) / stride);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->pipe_touch];
        [enc setBuffer:v->buffer offset:0 atIndex:0];
        [enc setBuffer:sink offset:0 atIndex:1];
        [enc setBytes:&stride length:sizeof(stride) atIndex:2];
        [enc setBytes:&n_steps length:sizeof(n_steps) atIndex:3];
        NSUInteger tg = 256;
        [enc dispatchThreadgroups:MTLSizeMake((n_steps + tg - 1) / tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    fprintf(stderr, "hy3_metal: warmed %d view(s) into GPU residency in %.2fs\n",
            ctx->n_views, hy3_metal_now() - t0);
}

static bool hy3_metal_find_view(hy3_metal_ctx_t *ctx, uint64_t abs_offset, uint64_t bytes,
                                 id<MTLBuffer> *buf_out, uint64_t *local_off_out) {
    for (int i = 0; i < ctx->n_views; i++) {
        hy3_metal_view_t *v = &ctx->views[i];
        if (abs_offset >= v->model_offset && abs_offset + bytes <= v->model_offset + v->length) {
            *buf_out = v->buffer;
            *local_off_out = abs_offset - v->model_offset;
            return true;
        }
    }
    return false;
}

/* Bind a GGUF-resident weight tensor at argument index `idx`. */
static bool hy3_bind_weight(id<MTLComputeCommandEncoder> enc, hy3_metal_ctx_t *ctx,
                             const hy3_weight *w, int idx) {
    if (!w->data || !w->t) {
        fprintf(stderr, "hy3_metal: attempted to bind a missing weight at index %d\n", idx);
        return false;
    }
    id<MTLBuffer> buf;
    uint64_t off;
    if (!hy3_metal_find_view(ctx, w->t->abs_offset, w->t->bytes, &buf, &off)) {
        fprintf(stderr, "hy3_metal: tensor at offset %llu (%llu bytes) is not covered by any view\n",
                (unsigned long long)w->t->abs_offset, (unsigned long long)w->t->bytes);
        return false;
    }
    [enc setBuffer:buf offset:(NSUInteger)off atIndex:idx];
    return true;
}

/* =========================================================================
 * Pipeline / library setup
 * ========================================================================= */

static id<MTLComputePipelineState> hy3_make_pipeline(id<MTLDevice> dev, id<MTLLibrary> lib, const char *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) {
        fprintf(stderr, "hy3_metal: shader function '%s' not found in library\n", name);
        exit(1);
    }
    id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pipe) {
        fprintf(stderr, "hy3_metal: failed to build pipeline '%s': %s\n",
                name, err ? err.localizedDescription.UTF8String : "unknown error");
        exit(1);
    }
    return pipe;
}

static id<MTLLibrary> hy3_load_library(id<MTLDevice> dev) {
    const char *path = getenv("HY3_METAL_SHADER");
    if (!path || !path[0]) path = HY3_METAL_SHADER_PATH;

    NSString *nsPath = [NSString stringWithUTF8String:path];
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:nsPath encoding:NSUTF8StringEncoding error:&err];
    if (!src) {
        fprintf(stderr, "hy3_metal: could not read shader source '%s': %s\n"
                        "  (set HY3_METAL_SHADER=/path/to/hy3.metal if it's not next to the binary)\n",
                path, err ? err.localizedDescription.UTF8String : "unknown error");
        exit(1);
    }
    MTLCompileOptions *opts = [MTLCompileOptions new];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        fprintf(stderr, "hy3_metal: shader compilation failed:\n%s\n",
                err ? err.localizedDescription.UTF8String : "unknown error");
        exit(1);
    }
    return lib;
}

/* =========================================================================
 * Kernel dispatch helpers (encode-only; caller commits the command buffer)
 * ========================================================================= */

static void m_rms_norm(hy3_metal_ctx_t *ctx, id<MTLComputeCommandEncoder> enc,
                        id<MTLBuffer> out, id<MTLBuffer> x, const hy3_weight *w, uint32_t n) {
    [enc setComputePipelineState:ctx->pipe_rms_norm];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    if (!hy3_bind_weight(enc, ctx, w, 2)) exit(1);
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* In-place per-head RMS norm: buf[off..off+n) is normalized using shared
 * weight w. Used for Q/K norm across HY3_N_HEAD / HY3_N_KV_HEAD heads. */
static void m_rms_norm_offset(hy3_metal_ctx_t *ctx, id<MTLComputeCommandEncoder> enc,
                               id<MTLBuffer> buf, const hy3_weight *w, uint32_t n, uint32_t off) {
    [enc setComputePipelineState:ctx->pipe_rms_norm_offset];
    [enc setBuffer:buf offset:0 atIndex:0];
    if (!hy3_bind_weight(enc, ctx, w, 1)) exit(1);
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc setBytes:&off length:sizeof(off) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* Fused per-head RMS norm: normalizes n_heads heads of head_dim each in a
 * single dispatch (grid.x = n_heads). Replaces the per-head loop. */
static void m_rms_norm_heads(hy3_metal_ctx_t *ctx, id<MTLComputeCommandEncoder> enc,
                             id<MTLBuffer> buf, const hy3_weight *w,
                             uint32_t head_dim, uint32_t n_heads) {
    [enc setComputePipelineState:ctx->pipe_rms_norm_heads];
    [enc setBuffer:buf offset:0 atIndex:0];
    if (!hy3_bind_weight(enc, ctx, w, 1)) exit(1);
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* Fused per-head Q/K RMSNorm + RoPE in a single dispatch (grid = n_heads +
 * n_kv_heads threadgroups). Avoids the separate rms_norm_heads x2 + rope
 * sequence and the barrier between norm and rope. */
static void m_rms_norm_heads_rope(hy3_metal_ctx_t *ctx, id<MTLComputeCommandEncoder> enc,
                                  id<MTLBuffer> q, id<MTLBuffer> k,
                                  const hy3_weight *qw, const hy3_weight *kw,
                                  uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
                                  int pos) {
    [enc setComputePipelineState:ctx->pipe_rms_norm_heads_rope];
    [enc setBuffer:q offset:0 atIndex:0];
    [enc setBuffer:k offset:0 atIndex:1];
    if (!hy3_bind_weight(enc, ctx, qw, 2)) exit(1);
    if (!hy3_bind_weight(enc, ctx, kw, 3)) exit(1);
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:4];
    [enc setBytes:&n_heads length:sizeof(n_heads) atIndex:5];
    [enc setBytes:&n_kv_heads length:sizeof(n_kv_heads) atIndex:6];
    [enc setBytes:&pos length:sizeof(pos) atIndex:7];
    [enc setBuffer:ctx->d_rope_inv_freq offset:0 atIndex:8];
    [enc setBytes:&ctx->rope_attn_factor length:sizeof(float) atIndex:9];
    uint32_t grid = n_heads + n_kv_heads;
    [enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void m_mul_mat(hy3_metal_ctx_t *ctx, id<MTLComputeCommandEncoder> enc,
                       const hy3_weight *w, id<MTLBuffer> dst, id<MTLBuffer> x,
                       uint32_t m_rows, uint32_t n_cols) {
    if (!w->t || !w->data) {
        fprintf(stderr, "hy3_metal: matmul requested on a missing/unloaded weight tensor\n");
        exit(1);
    }
    id<MTLComputePipelineState> pipe = nil;
    bool q4k_mm = false;
    bool q8_mm = false;
    switch (w->t->ggml_type) {
        case 0:  pipe = ctx->pipe_matmul_f32;  break;
        case 1:  pipe = ctx->pipe_matmul_f16;  break;
        case 8:
            /* SIMD-group q8_0 variant requires m_rows % 4 == 0 and
             * n_cols % 32 == 0. Gated by HY3_Q8_MM (default on). */
            if ((m_rows % 4) == 0 && (n_cols % 32) == 0) {
                static int use_mm = -1;
                if (use_mm < 0) {
                    const char *e = getenv("HY3_Q8_MM");
                    use_mm = (!e || e[0] != '0');
                }
                if (use_mm) { pipe = ctx->pipe_matmul_q8_0_mm; q8_mm = true; }
                else        { pipe = ctx->pipe_matmul_q8_0; }
            } else {
                pipe = ctx->pipe_matmul_q8_0;
            }
            break;
        case 12:
            /* SIMD-group q4_k variant requires m_rows % 4 == 0 and
             * n_cols % 256 == 0; every q4_k weight in this model satisfies
             * both. Gated by HY3_Q4K_MM (default on) for easy A/B profiling. */
            if ((m_rows % 4) == 0 && (n_cols % 256) == 0) {
                static int use_mm = -1;
                if (use_mm < 0) {
                    const char *e = getenv("HY3_Q4K_MM");
                    use_mm = (!e || e[0] != '0');
                }
                if (use_mm) { pipe = ctx->pipe_matmul_q4_k_mm; q4k_mm = true; }
                else        { pipe = ctx->pipe_matmul_q4_k; }
            } else {
                pipe = ctx->pipe_matmul_q4_k;
            }
            break;
        default:
            fprintf(stderr, "hy3_metal: unsupported ggml_type %u for matmul\n", w->t->ggml_type);
            exit(1);
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:dst offset:0 atIndex:0];
    if (!hy3_bind_weight(enc, ctx, w, 1)) exit(1);
    [enc setBuffer:x offset:0 atIndex:2];
    [enc setBytes:&n_cols length:sizeof(n_cols) atIndex:3];
    if (q4k_mm || q8_mm) {
        [enc dispatchThreadgroups:MTLSizeMake(m_rows / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake(m_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void m_silu_mul(hy3_metal_ctx_t *ctx, id<MTLComputeCommandEncoder> enc,
                        id<MTLBuffer> out, id<MTLBuffer> gate, id<MTLBuffer> up, uint32_t n) {
    [enc setComputePipelineState:ctx->pipe_silu_mul];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc setBuffer:gate offset:0 atIndex:1];
    [enc setBuffer:up offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    NSUInteger tg = 256;
    [enc dispatchThreadgroups:MTLSizeMake((n + tg - 1) / tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void m_add(hy3_metal_ctx_t *ctx, id<MTLComputeCommandEncoder> enc,
                   id<MTLBuffer> a, id<MTLBuffer> b, uint32_t n) {
    [enc setComputePipelineState:ctx->pipe_add];
    [enc setBuffer:a offset:0 atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    NSUInteger tg = 256;
    [enc dispatchThreadgroups:MTLSizeMake((n + tg - 1) / tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void m_scale_add(hy3_metal_ctx_t *ctx, id<MTLComputeCommandEncoder> enc,
                         id<MTLBuffer> acc, id<MTLBuffer> b, float scale, uint32_t n) {
    [enc setComputePipelineState:ctx->pipe_scale_add];
    [enc setBuffer:acc offset:0 atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    [enc setBytes:&scale length:sizeof(scale) atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    NSUInteger tg = 256;
    [enc dispatchThreadgroups:MTLSizeMake((n + tg - 1) / tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void m_fill_zero(hy3_metal_ctx_t *ctx, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> buf, uint32_t n) {
    [enc setComputePipelineState:ctx->pipe_fill_zero];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBytes:&n length:sizeof(n) atIndex:1];
    NSUInteger tg = 256;
    [enc dispatchThreadgroups:MTLSizeMake((n + tg - 1) / tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void m_rope(hy3_metal_ctx_t *ctx, id<MTLComputeCommandEncoder> enc,
                    id<MTLBuffer> q, id<MTLBuffer> k, int pos,
                    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads) {
    [enc setComputePipelineState:ctx->pipe_rope];
    [enc setBuffer:q offset:0 atIndex:0];
    [enc setBuffer:k offset:0 atIndex:1];
    [enc setBytes:&pos length:sizeof(pos) atIndex:2];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:3];
    [enc setBytes:&n_heads length:sizeof(n_heads) atIndex:4];
    [enc setBytes:&n_kv_heads length:sizeof(n_kv_heads) atIndex:5];
    [enc setBuffer:ctx->d_rope_inv_freq offset:0 atIndex:6];
    [enc setBytes:&ctx->rope_attn_factor length:sizeof(float) atIndex:7];
    uint32_t total = (n_heads + n_kv_heads) * (head_dim / 2);
    NSUInteger tg = 256;
    [enc dispatchThreadgroups:MTLSizeMake((total + tg - 1) / tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void m_kv_cache_write(hy3_metal_ctx_t *ctx, id<MTLComputeCommandEncoder> enc,
                              int dst_slot, uint32_t kv_size) {
    uint32_t slot_u = (uint32_t)dst_slot;
    uint32_t n_kv_heads = HY3_N_KV_HEAD;
    [enc setComputePipelineState:ctx->pipe_kv_cache_write_q8];
    [enc setBuffer:ctx->d_k_cache_q8 offset:0 atIndex:0];
    [enc setBuffer:ctx->d_v_cache_q8 offset:0 atIndex:1];
    [enc setBuffer:ctx->d_k_scales  offset:0 atIndex:2];
    [enc setBuffer:ctx->d_v_scales  offset:0 atIndex:3];
    [enc setBuffer:ctx->d_k offset:0 atIndex:4];
    [enc setBuffer:ctx->d_v offset:0 atIndex:5];
    [enc setBytes:&kv_size length:sizeof(kv_size) atIndex:6];
    [enc setBytes:&slot_u length:sizeof(slot_u) atIndex:7];
    [enc setBytes:&n_kv_heads length:sizeof(n_kv_heads) atIndex:8];
    [enc setThreadgroupMemoryLength:64 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n_kv_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void m_attention(hy3_metal_ctx_t *ctx, id<MTLComputeCommandEncoder> enc,
                         id<MTLBuffer> out, id<MTLBuffer> q,
                         uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                         int kv_len, uint32_t kv_group, int layer_id, int n_layers) {
    uint32_t n_splits = METAL_ATTN_SPLITS;

    /* --- Split pass: n_heads * n_splits threadgroups, head_dim threads each --- */
    [enc setComputePipelineState:ctx->pipe_attention_split_q8];
    [enc setBuffer:ctx->d_attn_partials offset:0 atIndex:0];
    [enc setBuffer:q offset:0 atIndex:1];
    [enc setBuffer:ctx->d_k_cache_q8 offset:0 atIndex:2];
    [enc setBuffer:ctx->d_v_cache_q8 offset:0 atIndex:3];
    [enc setBuffer:ctx->d_k_scales offset:0 atIndex:12];
    [enc setBuffer:ctx->d_v_scales offset:0 atIndex:13];
    [enc setBytes:&n_heads length:sizeof(n_heads) atIndex:4];
    [enc setBytes:&n_kv_heads length:sizeof(n_kv_heads) atIndex:5];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
    [enc setBytes:&kv_len length:sizeof(kv_len) atIndex:7];
    [enc setBytes:&kv_group length:sizeof(kv_group) atIndex:8];
    [enc setBytes:&layer_id length:sizeof(layer_id) atIndex:9];
    [enc setBytes:&n_layers length:sizeof(n_layers) atIndex:10];
    [enc setBytes:&n_splits length:sizeof(n_splits) atIndex:11];
    [enc setThreadgroupMemoryLength:64 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n_heads * n_splits, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];  /* split → reduce dependency */

    /* --- Reduce pass: n_heads threadgroups, head_dim threads each --- */
    [enc setComputePipelineState:ctx->pipe_attention_reduce];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc setBuffer:ctx->d_attn_partials offset:0 atIndex:1];
    [enc setBytes:&n_heads length:sizeof(n_heads) atIndex:2];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:3];
    [enc setBytes:&n_splits length:sizeof(n_splits) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
    if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* =========================================================================
 * Model init
 * ========================================================================= */

static id<MTLBuffer> hy3_alloc(id<MTLDevice> dev, uint64_t n_floats) {
    id<MTLBuffer> b = [dev newBufferWithLength:(NSUInteger)(n_floats * sizeof(float))
                                        options:MTLResourceStorageModeShared];
    METAL_CHECK(b != nil, "buffer allocation failed (out of memory?)");
    return b;
}

/* Allocate n_halves fp16 elements (KV cache is stored as fp16 to halve memory
 * and attention read bandwidth; quantization to fp16 is quality-neutral). */
static id<MTLBuffer> hy3_alloc_half(id<MTLDevice> dev, uint64_t n_halves) {
    id<MTLBuffer> b = [dev newBufferWithLength:(NSUInteger)(n_halves * sizeof(uint16_t))
                                        options:MTLResourceStorageModeShared];
    METAL_CHECK(b != nil, "buffer allocation failed (out of memory?)");
    return b;
}

int hy3_metal_init(hy3_model *m) {
    hy3_metal_ctx_t *ctx = (hy3_metal_ctx_t *)calloc(1, sizeof(hy3_metal_ctx_t));
    if (!ctx) return -1;

    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "hy3_metal: no Metal device found\n");
        free(ctx);
        return -1;
    }
    fprintf(stderr, "hy3_metal: device = %s\n", ctx->device.name.UTF8String);

    /* attention() in hy3.metal caps attended tokens at 8192 and uses one
     * float of threadgroup memory per attended token (32768 bytes total).
     * This has been a safe assumption across Apple7+ GPU families, but
     * check explicitly so a violation is a clear startup error instead of
     * a cryptic dispatch-time Metal validation failure. */
    NSUInteger maxTgMem = ctx->device.maxThreadgroupMemoryLength;
    if (maxTgMem < 8192 * sizeof(float)) {
        fprintf(stderr, "hy3_metal: device maxThreadgroupMemoryLength (%lu bytes) is smaller than "
                        "the attention kernel needs (32768 bytes)\n", (unsigned long)maxTgMem);
        free(ctx);
        return -1;
    }

    ctx->queue = [ctx->device newCommandQueue];
    ctx->library = hy3_load_library(ctx->device);

    ctx->pipe_rms_norm        = hy3_make_pipeline(ctx->device, ctx->library, "rms_norm");
    ctx->pipe_rms_norm_offset = hy3_make_pipeline(ctx->device, ctx->library, "rms_norm_offset");
    ctx->pipe_rms_norm_heads  = hy3_make_pipeline(ctx->device, ctx->library, "rms_norm_heads");
    ctx->pipe_rms_norm_heads_rope = hy3_make_pipeline(ctx->device, ctx->library, "rms_norm_heads_rope");
    ctx->pipe_silu_mul        = hy3_make_pipeline(ctx->device, ctx->library, "silu_mul");
    ctx->pipe_sigmoid         = hy3_make_pipeline(ctx->device, ctx->library, "sigmoid_inplace");
    ctx->pipe_add             = hy3_make_pipeline(ctx->device, ctx->library, "add_inplace");
    ctx->pipe_scale_add       = hy3_make_pipeline(ctx->device, ctx->library, "scale_add_inplace");
    ctx->pipe_fill_zero       = hy3_make_pipeline(ctx->device, ctx->library, "fill_zero");
    ctx->pipe_embed_f16       = hy3_make_pipeline(ctx->device, ctx->library, "embed_lookup_f16");
    ctx->pipe_embed_f32       = hy3_make_pipeline(ctx->device, ctx->library, "embed_lookup_f32");
    ctx->pipe_rope            = hy3_make_pipeline(ctx->device, ctx->library, "rope");
    ctx->pipe_attention       = hy3_make_pipeline(ctx->device, ctx->library, "attention");
    ctx->pipe_attention_split = hy3_make_pipeline(ctx->device, ctx->library, "attention_split");
    ctx->pipe_attention_split_q8 = hy3_make_pipeline(ctx->device, ctx->library, "attention_split_q8");
    ctx->pipe_attention_reduce= hy3_make_pipeline(ctx->device, ctx->library, "attention_reduce");
    ctx->pipe_matmul_f32      = hy3_make_pipeline(ctx->device, ctx->library, "matmul_f32");
    ctx->pipe_matmul_f16      = hy3_make_pipeline(ctx->device, ctx->library, "matmul_f16");
    ctx->pipe_matmul_q8_0     = hy3_make_pipeline(ctx->device, ctx->library, "matmul_q8_0");
    ctx->pipe_matmul_q8_0_mm  = hy3_make_pipeline(ctx->device, ctx->library, "matmul_q8_0_mm");
    ctx->pipe_matmul_q4_k     = hy3_make_pipeline(ctx->device, ctx->library, "matmul_q4_k");
    ctx->pipe_matmul_q4_k_mm  = hy3_make_pipeline(ctx->device, ctx->library, "matmul_q4_k_mm");
    ctx->pipe_kv_cache_write_q8 = hy3_make_pipeline(ctx->device, ctx->library, "kv_cache_write_q8");
    ctx->pipe_touch           = hy3_make_pipeline(ctx->device, ctx->library, "touch_u8_stride");
    ctx->pipe_router_topk     = hy3_make_pipeline(ctx->device, ctx->library, "router_topk");
    ctx->pipe_matmul_q4_k_id  = hy3_make_pipeline(ctx->device, ctx->library, "matmul_q4_k_id");
    ctx->pipe_moe_swiglu_id   = hy3_make_pipeline(ctx->device, ctx->library, "moe_swiglu_id");
    ctx->pipe_moe_combine_id  = hy3_make_pipeline(ctx->device, ctx->library, "moe_combine_id");

    if (!hy3_metal_wrap_model(ctx, m)) {
        free(ctx);
        return -1;
    }

    hy3_metal_warm_views(ctx);

    uint32_t inter = HY3_DENSE_INTERMED > HY3_MOE_INTERMED ? HY3_DENSE_INTERMED : HY3_MOE_INTERMED;
    ctx->d_embed      = hy3_alloc(ctx->device, HY3_N_EMBD);
    ctx->d_s          = hy3_alloc(ctx->device, HY3_N_EMBD);
    ctx->d_q          = hy3_alloc(ctx->device, HY3_N_HEAD * HY3_HEAD_DIM);
    ctx->d_k          = hy3_alloc(ctx->device, HY3_N_KV_HEAD * HY3_HEAD_DIM);
    ctx->d_v          = hy3_alloc(ctx->device, HY3_N_KV_HEAD * HY3_HEAD_DIM);
    ctx->d_attn_out   = hy3_alloc(ctx->device, HY3_N_HEAD * HY3_HEAD_DIM);
    {
        uint32_t n = HY3_N_HEAD * METAL_ATTN_SPLITS * (uint32_t)(2 + HY3_HEAD_DIM);
        ctx->d_attn_partials = hy3_alloc(ctx->device, n);
    }
    ctx->d_o_proj     = hy3_alloc(ctx->device, HY3_N_EMBD);
    ctx->d_gate       = hy3_alloc(ctx->device, inter);
    ctx->d_up         = hy3_alloc(ctx->device, inter);
    ctx->d_mid        = hy3_alloc(ctx->device, inter);
    ctx->d_ffn_out    = hy3_alloc(ctx->device, HY3_N_EMBD);
    ctx->d_router     = hy3_alloc(ctx->device, HY3_N_EXPERT);
    ctx->d_expert_out = hy3_alloc(ctx->device, HY3_N_EMBD);
    ctx->d_logits     = hy3_alloc(ctx->device, HY3_N_VOCAB);

    /* Fast MoE path buffers. */
    ctx->d_router_ids = hy3_alloc(ctx->device, HY3_N_EXPERT_USED);           /* int32 reuses float size */
    ctx->d_router_wts = hy3_alloc(ctx->device, HY3_N_EXPERT_USED);
    ctx->d_bias       = hy3_alloc(ctx->device, (uint64_t)HY3_N_LAYER * HY3_N_EXPERT);
    ctx->d_gate_k     = hy3_alloc(ctx->device, (uint64_t)HY3_N_EXPERT_USED * HY3_MOE_INTERMED);
    ctx->d_up_k       = hy3_alloc(ctx->device, (uint64_t)HY3_N_EXPERT_USED * HY3_MOE_INTERMED);
    ctx->d_mid_k      = hy3_alloc(ctx->device, (uint64_t)HY3_N_EXPERT_USED * HY3_MOE_INTERMED);
    ctx->d_down_k     = hy3_alloc(ctx->device, (uint64_t)HY3_N_EXPERT_USED * HY3_N_EMBD);

    /* RoPE inv_freq[d] (YaRN-aware) + mscale, resolved once in hy3_rope_init()
     * and shared with the CPU/CUDA backends via hy3_rope_get_params(). */
    {
        uint32_t half_dim = HY3_HEAD_DIM / 2;
        float inv[HY3_HEAD_DIM / 2];
        hy3_rope_get_params(m, inv, &ctx->rope_attn_factor);
        ctx->d_rope_inv_freq = [ctx->device newBufferWithBytes:inv
                                                        length:half_dim * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    }

    /* Upload every layer's expert bias once (static). Layers without a bias
     * tensor get zeros. The fast MoE router reads bias at offset il*NE. */
    {
        float *bp = (float *)ctx->d_bias.contents;
        memset(bp, 0, (size_t)HY3_N_LAYER * HY3_N_EXPERT * sizeof(float));
        for (int il = 0; il < HY3_N_LAYER; il++) {
            hy3_layer_weights *l = &m->w.layers[il];
            if (l->has_expert_bias)
                memcpy(bp + (size_t)il * HY3_N_EXPERT, l->expert_bias, HY3_N_EXPERT * sizeof(float));
        }
    }

    /* KV cache: interleaved by layer (slot = token*HY3_N_LAYER + layer),
     * same layout as the CPU/CUDA backends. Default to 8192 tokens of
     * context; grows on demand in hy3_eval_metal (see gpu_ensure_kv_capacity
     * in hy3_gpu.cu for the CUDA analog of the same fix). */
    int default_ctx_tokens = 8192;
    const char *ctx_env = getenv("HY3_METAL_CTX_TOKENS");
    if (ctx_env && ctx_env[0]) {
        int v = atoi(ctx_env);
        if (v > 0) default_ctx_tokens = v;
    }
    uint32_t kv_dim = HY3_N_KV_HEAD * HY3_HEAD_DIM;
    ctx->ctx_cap_slots = default_ctx_tokens * HY3_N_LAYER;
    ctx->d_k_cache = hy3_alloc_half(ctx->device, (uint64_t)ctx->ctx_cap_slots * kv_dim);
    ctx->d_v_cache = hy3_alloc_half(ctx->device, (uint64_t)ctx->ctx_cap_slots * kv_dim);

    /* Q8 KV cache (default on). Per-head absmax → int8 quantization. */
    {
        uint64_t kv_bytes = (uint64_t)ctx->ctx_cap_slots * kv_dim;
        ctx->d_k_cache_q8 = [ctx->device newBufferWithLength:(NSUInteger)kv_bytes options:MTLResourceStorageModeShared];
        ctx->d_v_cache_q8 = [ctx->device newBufferWithLength:(NSUInteger)kv_bytes options:MTLResourceStorageModeShared];
        uint32_t n_scales = (uint32_t)ctx->ctx_cap_slots * HY3_N_KV_HEAD;
        ctx->d_k_scales = hy3_alloc(ctx->device, n_scales);
        ctx->d_v_scales = hy3_alloc(ctx->device, n_scales);
        fprintf(stderr, "hy3_metal: Q8 KV cache enabled (default), %.2f GiB\n",
                2.0 * (double)ctx->ctx_cap_slots * kv_dim * sizeof(uint8_t) / 1e9);
    }
    fprintf(stderr, "hy3_metal: KV cache sized for %d tokens (%d layers x %d slots, %.2f GiB total, fp16)\n",
            default_ctx_tokens, HY3_N_LAYER, ctx->ctx_cap_slots,
            2.0 * ctx->ctx_cap_slots * kv_dim * sizeof(uint16_t) / 1e9);

    m->metal_ctx = ctx;
    fprintf(stderr, "hy3_metal: initialized, all %d layers Metal-resident (zero-copy)\n", HY3_N_LAYER);
    if (getenv("HY3_METAL_SELFTEST")) {
        for (int i = 0; i < HY3_N_EMBD; i++) ((float *)ctx->d_s.contents)[i] = 1.0f;
        hy3_layer_weights *l0 = &m->w.layers[0];
        const float *wc = (const float *)l0->attn_norm.data;
        double cs = 0; for (int i = 0; i < HY3_N_EMBD; i++) cs += wc[i];
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        m_mul_mat(ctx, enc, &l0->attn_norm, ctx->d_logits, ctx->d_s, 1, HY3_N_EMBD);
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        fprintf(stderr, "hy3_metal: SELFTEST L0 dot=%.4f CPU=%.4f\n", ((float *)ctx->d_logits.contents)[0], cs);
    }
    return 0;
}

static void hy3_metal_grow_kv_cache(hy3_metal_ctx_t *ctx, int needed_slots) {
    if (needed_slots <= ctx->ctx_cap_slots) return;
    uint32_t kv_dim = HY3_N_KV_HEAD * HY3_HEAD_DIM;
    int new_cap = needed_slots + 8192 * HY3_N_LAYER; /* headroom: ~8192 more tokens */
    id<MTLBuffer> nk = hy3_alloc_half(ctx->device, (uint64_t)new_cap * kv_dim);
    id<MTLBuffer> nv = hy3_alloc_half(ctx->device, (uint64_t)new_cap * kv_dim);
    memcpy(nk.contents, ctx->d_k_cache.contents, (size_t)ctx->ctx_cap_slots * kv_dim * sizeof(uint16_t));
    memcpy(nv.contents, ctx->d_v_cache.contents, (size_t)ctx->ctx_cap_slots * kv_dim * sizeof(uint16_t));
    ctx->d_k_cache = nk;
    ctx->d_v_cache = nv;
    ctx->ctx_cap_slots = new_cap;
    fprintf(stderr, "hy3_metal: grew KV cache to %d slots (%.2f GiB each, fp16)\n",
            new_cap, (double)new_cap * kv_dim * sizeof(uint16_t) / 1e9);
}

void hy3_metal_free(hy3_model *m) {
    hy3_metal_ctx_t *ctx = (hy3_metal_ctx_t *)m->metal_ctx;
    if (!ctx) return;
    /* ARC (or manual release under MRC, depending on build flags) reclaims
     * the Objective-C objects when the struct is freed and these fields go
     * out of scope; explicitly nil the zero-copy view buffers first so
     * Metal drops its reference to the mmap'd pages before hy3_model_free()
     * calls munmap() on the underlying file. */
    if (ctx->residency_set) {
        if (@available(macOS 15.0, *)) {
            id<MTLResidencySet> set = ctx->residency_set;
            [ctx->queue removeResidencySet:set];
            [set endResidency];
        }
        ctx->residency_set = nil;
    }
    for (int i = 0; i < ctx->n_views; i++) ctx->views[i].buffer = nil;
    ctx->n_views = 0;
    free(ctx);
    m->metal_ctx = NULL;
}

/* =========================================================================
 * Forward pass
 * ========================================================================= */

static void metal_forward_layer(hy3_metal_ctx_t *ctx, hy3_model *m, int il, int pos) {
    hy3_layer_weights *l = &m->w.layers[il];
    bool is_dense = (il < HY3_N_LAYER_DENSE);
    uint32_t q_size = HY3_N_HEAD * HY3_HEAD_DIM;
    uint32_t kv_size = HY3_N_KV_HEAD * HY3_HEAD_DIM;
    int kv_len = m->cache_len; /* slot this token/layer writes to, before increment */

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    m_rms_norm(ctx, enc, ctx->d_s, ctx->d_embed, &l->attn_norm, HY3_N_EMBD);

    m_mul_mat(ctx, enc, &l->attn_q, ctx->d_q, ctx->d_s, q_size, HY3_N_EMBD);
    m_mul_mat(ctx, enc, &l->attn_k, ctx->d_k, ctx->d_s, kv_size, HY3_N_EMBD);
    m_mul_mat(ctx, enc, &l->attn_v, ctx->d_v, ctx->d_s, kv_size, HY3_N_EMBD);

    if (l->attn_q_norm.data) {
        for (int h = 0; h < HY3_N_HEAD; h++)
            m_rms_norm_offset(ctx, enc, ctx->d_q, &l->attn_q_norm, HY3_HEAD_DIM, h * HY3_HEAD_DIM);
    }
    if (l->attn_k_norm.data) {
        for (int h = 0; h < HY3_N_KV_HEAD; h++)
            m_rms_norm_offset(ctx, enc, ctx->d_k, &l->attn_k_norm, HY3_HEAD_DIM, h * HY3_HEAD_DIM);
    }

    m_rope(ctx, enc, ctx->d_q, ctx->d_k, pos, HY3_HEAD_DIM, HY3_N_HEAD, HY3_N_KV_HEAD);

    /* Write this token's K/V into the interleaved cache at slot kv_len,
     * then bump cache_len -- matches forward_layer_dense/forward_layer_moe
     * in hy3.c exactly (one slot per (token,layer), see attention()'s
     * comment there for the interleaving invariant). This must be a GPU
     * kernel (not a CPU memcpy of d_k/d_v.contents) because nothing has
     * been committed yet at this point -- the rope kernel above is only
     * *encoded* so far, so its output isn't actually written until the
     * command buffer runs; a CPU-side copy here would race the GPU and
     * could grab stale/pre-RoPE data. Encoding this as a kernel in the
     * same encoder lets Metal's automatic hazard tracking order it
     * correctly after rope. */
    m_kv_cache_write(ctx, enc, kv_len, kv_size);
    m->cache_len = kv_len + 1;

    m_attention(ctx, enc, ctx->d_attn_out, ctx->d_q,
                HY3_N_HEAD, HY3_N_KV_HEAD, HY3_HEAD_DIM,
                m->cache_len, HY3_N_HEAD / HY3_N_KV_HEAD, il, HY3_N_LAYER);

    m_mul_mat(ctx, enc, &l->attn_output, ctx->d_o_proj, ctx->d_attn_out, HY3_N_EMBD, q_size);
    m_add(ctx, enc, ctx->d_embed, ctx->d_o_proj, HY3_N_EMBD);

    m_rms_norm(ctx, enc, ctx->d_s, ctx->d_embed, &l->ffn_norm, HY3_N_EMBD);

    if (is_dense) {
        m_mul_mat(ctx, enc, &l->ffn_gate, ctx->d_gate, ctx->d_s, HY3_DENSE_INTERMED, HY3_N_EMBD);
        m_mul_mat(ctx, enc, &l->ffn_up, ctx->d_up, ctx->d_s, HY3_DENSE_INTERMED, HY3_N_EMBD);
        m_silu_mul(ctx, enc, ctx->d_mid, ctx->d_gate, ctx->d_up, HY3_DENSE_INTERMED);
        m_mul_mat(ctx, enc, &l->ffn_down, ctx->d_ffn_out, ctx->d_mid, HY3_N_EMBD, HY3_DENSE_INTERMED);
        m_add(ctx, enc, ctx->d_embed, ctx->d_ffn_out, HY3_N_EMBD);

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return;
    }

    /* --- MoE layer --- */
    m_mul_mat(ctx, enc, &l->ffn_gate_inp, ctx->d_router, ctx->d_s, HY3_N_EXPERT, HY3_N_EMBD);
    {
        uint32_t n_expert = HY3_N_EXPERT;
        [enc setComputePipelineState:ctx->pipe_sigmoid];
        [enc setBuffer:ctx->d_router offset:0 atIndex:0];
        [enc setBytes:&n_expert length:sizeof(n_expert) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake((HY3_N_EXPERT + 63) / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    /* Shared expert (always active, independent of routing) is encoded into
     * the same command buffer as the router/sigmoid kernels above; they
     * touch disjoint buffers so ordering between them doesn't matter. Both
     * must finish before the CPU reads d_router below, which the
     * waitUntilCompleted after commit guarantees. */
    m_mul_mat(ctx, enc, &l->ffn_gate_shexp, ctx->d_gate, ctx->d_s, HY3_MOE_INTERMED, HY3_N_EMBD);
    m_mul_mat(ctx, enc, &l->ffn_up_shexp, ctx->d_up, ctx->d_s, HY3_MOE_INTERMED, HY3_N_EMBD);
    m_silu_mul(ctx, enc, ctx->d_mid, ctx->d_gate, ctx->d_up, HY3_MOE_INTERMED);
    m_mul_mat(ctx, enc, &l->ffn_down_shexp, ctx->d_ffn_out, ctx->d_mid, HY3_N_EMBD, HY3_MOE_INTERMED);

    m_fill_zero(ctx, enc, ctx->d_expert_out, HY3_N_EMBD);

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted]; /* need router logits on the CPU before we know which experts to run */

    /* --- Top-k routing on the CPU (192 elements; trivial cost). Mirrors
     * forward_layer_moe in hy3.c: select top-k by (sigmoid + bias), but
     * combine using the *unbiased* sigmoid weights renormalized to sum 1
     * and scaled by router_scaling_factor. --- */
    float *sigmoid_vals = (float *)ctx->d_router.contents;
    float scored[HY3_N_EXPERT];
    memcpy(scored, sigmoid_vals, sizeof(scored));
    if (l->has_expert_bias)
        for (int i = 0; i < HY3_N_EXPERT; i++) scored[i] += l->expert_bias[i];

    int topk_inds[HY3_N_EXPERT_USED];
    float topk_vals[HY3_N_EXPERT_USED];
    int n_used = m->n_expert_used;
    {
        int idx[HY3_N_EXPERT];
        float val[HY3_N_EXPERT];
        for (int i = 0; i < HY3_N_EXPERT; i++) { idx[i] = i; val[i] = scored[i]; }
        for (int i = 0; i < n_used; i++) {
            int best = i;
            for (int j = i + 1; j < HY3_N_EXPERT; j++) if (val[j] > val[best]) best = j;
            float tv = val[i]; val[i] = val[best]; val[best] = tv;
            int ti = idx[i]; idx[i] = idx[best]; idx[best] = ti;
        }
        float sum_w = 0.0f;
        for (int i = 0; i < n_used; i++) {
            topk_inds[i] = idx[i];
            topk_vals[i] = sigmoid_vals[idx[i]];
            sum_w += topk_vals[i];
        }
        float inv_sum = 1.0f / (sum_w + 1e-20f);
        const float router_scaling = 2.826f;
        for (int i = 0; i < n_used; i++) topk_vals[i] = topk_vals[i] * inv_sum * router_scaling;
    }

    id<MTLCommandBuffer> cb2 = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoder];
    for (int e = 0; e < n_used; e++) {
        int ei = topk_inds[e];
        m_mul_mat(ctx, enc2, &l->ffn_gate_exps[ei], ctx->d_gate, ctx->d_s, HY3_MOE_INTERMED, HY3_N_EMBD);
        m_mul_mat(ctx, enc2, &l->ffn_up_exps[ei], ctx->d_up, ctx->d_s, HY3_MOE_INTERMED, HY3_N_EMBD);
        m_silu_mul(ctx, enc2, ctx->d_mid, ctx->d_gate, ctx->d_up, HY3_MOE_INTERMED);
        m_mul_mat(ctx, enc2, &l->ffn_down_exps[ei], ctx->d_gate, ctx->d_mid, HY3_N_EMBD, HY3_MOE_INTERMED);
        m_scale_add(ctx, enc2, ctx->d_expert_out, ctx->d_gate, topk_vals[e], HY3_N_EMBD);
    }
    m_add(ctx, enc2, ctx->d_embed, ctx->d_ffn_out, HY3_N_EMBD);   /* + shared expert down-proj */
    m_add(ctx, enc2, ctx->d_embed, ctx->d_expert_out, HY3_N_EMBD); /* + routed experts */
    [enc2 endEncoding];
    [cb2 commit];
    [cb2 waitUntilCompleted];
}

/* ---------------------------------------------------------------------------
 * FAST PATH: whole-token single-command-buffer forward with GPU-resident MoE
 * routing (no per-layer commit/wait, no CPU round-trip for expert selection).
 * Gated by HY3_FAST (default on). ds4-style batch encoder validated in
 * fast_metal.m (2.4x fewer syncs).
 * ------------------------------------------------------------------------- */

static void metal_encode_attention(hy3_metal_ctx_t *ctx, hy3_model *m, int il, int pos,
                                   id<MTLComputeCommandEncoder> enc) {
    hy3_layer_weights *l = &m->w.layers[il];
    uint32_t q_size = HY3_N_HEAD * HY3_HEAD_DIM;
    uint32_t kv_size = HY3_N_KV_HEAD * HY3_HEAD_DIM;
    int kv_len = m->cache_len;
    #define BAR() do { if (ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; } while(0)

    m_rms_norm(ctx, enc, ctx->d_s, ctx->d_embed, &l->attn_norm, HY3_N_EMBD);
    BAR();  /* d_s ready before q/k/v read it */
    /* q,k,v are independent (write disjoint buffers, read d_s) -> run concurrently */
    m_mul_mat(ctx, enc, &l->attn_q, ctx->d_q, ctx->d_s, q_size, HY3_N_EMBD);
    m_mul_mat(ctx, enc, &l->attn_k, ctx->d_k, ctx->d_s, kv_size, HY3_N_EMBD);
    m_mul_mat(ctx, enc, &l->attn_v, ctx->d_v, ctx->d_s, kv_size, HY3_N_EMBD);
    BAR();  /* q,k,v ready before norm/rope */
    if (l->attn_q_norm.data && l->attn_k_norm.data)
        m_rms_norm_heads_rope(ctx, enc, ctx->d_q, ctx->d_k, &l->attn_q_norm, &l->attn_k_norm,
                              HY3_HEAD_DIM, HY3_N_HEAD, HY3_N_KV_HEAD, pos);
    else {
        if (l->attn_q_norm.data)
            m_rms_norm_heads(ctx, enc, ctx->d_q, &l->attn_q_norm, HY3_HEAD_DIM, HY3_N_HEAD);
        if (l->attn_k_norm.data)
            m_rms_norm_heads(ctx, enc, ctx->d_k, &l->attn_k_norm, HY3_HEAD_DIM, HY3_N_KV_HEAD);
        BAR();  /* q/k norm before rope */
        m_rope(ctx, enc, ctx->d_q, ctx->d_k, pos, HY3_HEAD_DIM, HY3_N_HEAD, HY3_N_KV_HEAD);
    }
    BAR();  /* rope before kv write + attention read q */
    m_kv_cache_write(ctx, enc, kv_len, kv_size);
    m->cache_len = kv_len + 1;
    BAR();  /* kv written before attention reads cache */
    m_attention(ctx, enc, ctx->d_attn_out, ctx->d_q,
                HY3_N_HEAD, HY3_N_KV_HEAD, HY3_HEAD_DIM,
                m->cache_len, HY3_N_HEAD / HY3_N_KV_HEAD, il, HY3_N_LAYER);
    BAR();  /* attn_out before o proj */
    m_mul_mat(ctx, enc, &l->attn_output, ctx->d_o_proj, ctx->d_attn_out, HY3_N_EMBD, q_size);
    BAR();  /* o_proj before residual add */
    m_add(ctx, enc, ctx->d_embed, ctx->d_o_proj, HY3_N_EMBD);
    BAR();  /* embed updated before next stage */
    #undef BAR
}

/* Id-matmul over the contiguous expert weight block: computes all `k` selected
 * experts in one dispatch. `base` is expert 0's tensor; experts are equal-stride. */
static void m_mul_mat_id(hy3_metal_ctx_t *ctx, id<MTLComputeCommandEncoder> enc,
                         const hy3_weight *base, uint64_t expert_stride,
                         id<MTLBuffer> dst, id<MTLBuffer> x,
                         uint32_t m_rows, uint32_t n_cols, uint32_t k,
                         uint32_t x_per_slot) {
    id<MTLBuffer> buf; uint64_t off;
    /* The id-kernel indexes experts as base + id*stride, so the WHOLE expert
     * block (up to the highest selected id) must live in one view. Validate the
     * full span, not just expert 0's tensor, or a block straddling a view
     * boundary would index into invalid pages (NaN/garbage). Use the full
     * 192-expert span to be safe. */
    uint64_t full_span = expert_stride * (uint64_t)(HY3_N_EXPERT - 1) + base->t->bytes;
    if (!hy3_metal_find_view(ctx, base->t->abs_offset, full_span, &buf, &off)) {
        fprintf(stderr, "hy3_metal: expert block (span %.2f GiB) crosses a view boundary\n",
                (double)full_span / 1e9);
        exit(1);
    }
    uint32_t stride32 = (uint32_t)expert_stride;
    [enc setComputePipelineState:ctx->pipe_matmul_q4_k_id];
    [enc setBuffer:dst offset:0 atIndex:0];
    [enc setBuffer:buf offset:(NSUInteger)off atIndex:1];
    [enc setBuffer:x offset:0 atIndex:2];
    [enc setBytes:&n_cols length:4 atIndex:3];
    [enc setBytes:&m_rows length:4 atIndex:4];
    [enc setBytes:&stride32 length:4 atIndex:5];
    [enc setBuffer:ctx->d_router_ids offset:0 atIndex:6];
    [enc setBytes:&x_per_slot length:4 atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(m_rows / 4, k, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void metal_encode_moe(hy3_metal_ctx_t *ctx, hy3_model *m, int il,
                             id<MTLComputeCommandEncoder> enc) {
    hy3_layer_weights *l = &m->w.layers[il];
    uint32_t k = (uint32_t)m->n_expert_used;
    #define BAR() do { if (ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; } while(0)

    m_rms_norm(ctx, enc, ctx->d_s, ctx->d_embed, &l->ffn_norm, HY3_N_EMBD);
    BAR();  /* d_s ready */

    m_mul_mat(ctx, enc, &l->ffn_gate_inp, ctx->d_router, ctx->d_s, HY3_N_EXPERT, HY3_N_EMBD);
    BAR();  /* router logits ready before top-k */
    {
        uint32_t NE = HY3_N_EXPERT, has_bias = l->has_expert_bias ? 1u : 0u;
        float scaling = 2.826f;
        [enc setComputePipelineState:ctx->pipe_router_topk];
        [enc setBuffer:ctx->d_router offset:0 atIndex:0];
        [enc setBuffer:ctx->d_bias offset:(NSUInteger)il * HY3_N_EXPERT * sizeof(float) atIndex:1];
        [enc setBuffer:ctx->d_router_ids offset:0 atIndex:2];
        [enc setBuffer:ctx->d_router_wts offset:0 atIndex:3];
        [enc setBytes:&NE length:4 atIndex:4];
        [enc setBytes:&k length:4 atIndex:5];
        [enc setBytes:&has_bias length:4 atIndex:6];
        [enc setBytes:&scaling length:4 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    BAR();  /* router_ids/wts ready before routed matmuls; also gates shexp */

    uint64_t gate_stride = l->ffn_gate_exps[1].t->abs_offset - l->ffn_gate_exps[0].t->abs_offset;
    uint64_t up_stride   = l->ffn_up_exps[1].t->abs_offset   - l->ffn_up_exps[0].t->abs_offset;
    uint64_t down_stride = l->ffn_down_exps[1].t->abs_offset - l->ffn_down_exps[0].t->abs_offset;

    /* Shared + routed gate/up all read d_s and write disjoint buffers -> concurrent. */
    m_mul_mat(ctx, enc, &l->ffn_gate_shexp, ctx->d_gate, ctx->d_s, HY3_MOE_INTERMED, HY3_N_EMBD);
    m_mul_mat(ctx, enc, &l->ffn_up_shexp, ctx->d_up, ctx->d_s, HY3_MOE_INTERMED, HY3_N_EMBD);
    m_mul_mat_id(ctx, enc, &l->ffn_gate_exps[0], gate_stride, ctx->d_gate_k, ctx->d_s,
                 HY3_MOE_INTERMED, HY3_N_EMBD, k, 0);
    m_mul_mat_id(ctx, enc, &l->ffn_up_exps[0], up_stride, ctx->d_up_k, ctx->d_s,
                 HY3_MOE_INTERMED, HY3_N_EMBD, k, 0);
    BAR();  /* gate/up (shared+routed) ready before swiglu */

    /* Shared silu*up and routed swiglu write disjoint buffers -> concurrent. */
    m_silu_mul(ctx, enc, ctx->d_mid, ctx->d_gate, ctx->d_up, HY3_MOE_INTERMED);
    {
        uint32_t total = k * HY3_MOE_INTERMED, tg = 256;
        [enc setComputePipelineState:ctx->pipe_moe_swiglu_id];
        [enc setBuffer:ctx->d_mid_k offset:0 atIndex:0];
        [enc setBuffer:ctx->d_gate_k offset:0 atIndex:1];
        [enc setBuffer:ctx->d_up_k offset:0 atIndex:2];
        [enc setBytes:&total length:4 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((total + tg - 1) / tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    BAR();  /* mid (shared+routed) ready before down */

    /* Shared down + routed down write disjoint buffers -> concurrent. */
    m_mul_mat(ctx, enc, &l->ffn_down_shexp, ctx->d_ffn_out, ctx->d_mid, HY3_N_EMBD, HY3_MOE_INTERMED);
    m_mul_mat_id(ctx, enc, &l->ffn_down_exps[0], down_stride, ctx->d_down_k, ctx->d_mid_k,
                 HY3_N_EMBD, HY3_MOE_INTERMED, k, 1);
    BAR();  /* down outputs ready before combine into residual */

    m_add(ctx, enc, ctx->d_embed, ctx->d_ffn_out, HY3_N_EMBD);
    BAR();  /* shared added before routed combine (both write d_embed) */
    {
        uint32_t M = HY3_N_EMBD, tg = 256;
        [enc setComputePipelineState:ctx->pipe_moe_combine_id];
        [enc setBuffer:ctx->d_embed offset:0 atIndex:0];
        [enc setBuffer:ctx->d_down_k offset:0 atIndex:1];
        [enc setBuffer:ctx->d_router_wts offset:0 atIndex:2];
        [enc setBytes:&M length:4 atIndex:3];
        [enc setBytes:&k length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((M + tg - 1) / tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        if (!ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    BAR();  /* embed updated before next layer */
    #undef BAR
}

static void metal_encode_dense(hy3_metal_ctx_t *ctx, hy3_model *m, int il,
                               id<MTLComputeCommandEncoder> enc) {
    hy3_layer_weights *l = &m->w.layers[il];
    #define BAR() do { if (ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; } while(0)
    m_rms_norm(ctx, enc, ctx->d_s, ctx->d_embed, &l->ffn_norm, HY3_N_EMBD);
    BAR();
    m_mul_mat(ctx, enc, &l->ffn_gate, ctx->d_gate, ctx->d_s, HY3_DENSE_INTERMED, HY3_N_EMBD);
    m_mul_mat(ctx, enc, &l->ffn_up, ctx->d_up, ctx->d_s, HY3_DENSE_INTERMED, HY3_N_EMBD);
    BAR();
    m_silu_mul(ctx, enc, ctx->d_mid, ctx->d_gate, ctx->d_up, HY3_DENSE_INTERMED);
    BAR();
    m_mul_mat(ctx, enc, &l->ffn_down, ctx->d_ffn_out, ctx->d_mid, HY3_N_EMBD, HY3_DENSE_INTERMED);
    BAR();
    m_add(ctx, enc, ctx->d_embed, ctx->d_ffn_out, HY3_N_EMBD);
    BAR();
    #undef BAR
}

static void metal_forward_model_fast(hy3_metal_ctx_t *ctx, hy3_model *m, int token, int want_logits) {
    static int concurrent = -1;
    if (concurrent < 0) {
        const char *e = getenv("HY3_CONCURRENT");
        concurrent = (!e || e[0] != '0');   /* default on */
    }
    ctx->concurrent = concurrent;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = concurrent
        ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
        : [cb computeCommandEncoder];

    if (m->w.token_embd.t->ggml_type == 1)
        [enc setComputePipelineState:ctx->pipe_embed_f16];
    else
        [enc setComputePipelineState:ctx->pipe_embed_f32];
    [enc setBuffer:ctx->d_embed offset:0 atIndex:0];
    if (!hy3_bind_weight(enc, ctx, &m->w.token_embd, 1)) exit(1);
    uint32_t token_u = (uint32_t)token, dim = HY3_N_EMBD;
    [enc setBytes:&token_u length:sizeof(token_u) atIndex:2];
    [enc setBytes:&dim length:sizeof(dim) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((HY3_N_EMBD + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    for (int il = 0; il < HY3_N_LAYER; il++) {
        int pos = m->cache_len / HY3_N_LAYER;
        metal_encode_attention(ctx, m, il, pos, enc);
        if (il < HY3_N_LAYER_DENSE) {
            metal_encode_dense(ctx, m, il, enc);
        } else {
            metal_encode_moe(ctx, m, il, enc);
        }
    }

    /* Final norm + logits folded into the same command buffer (saves one
     * commit/wait per generated token). */
    if (want_logits) {
        m_rms_norm(ctx, enc, ctx->d_embed, ctx->d_embed, &m->w.output_norm, HY3_N_EMBD);
        if (ctx->concurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        m_mul_mat(ctx, enc, &m->w.output, ctx->d_logits, ctx->d_embed, HY3_N_VOCAB, HY3_N_EMBD);
    }

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    ctx->concurrent = 0;  /* restore for any serial path */
}

static void metal_forward_model(hy3_metal_ctx_t *ctx, hy3_model *m, int token) {
    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (m->w.token_embd.t->ggml_type == 1) {
        [enc setComputePipelineState:ctx->pipe_embed_f16];
    } else {
        [enc setComputePipelineState:ctx->pipe_embed_f32];
    }
    [enc setBuffer:ctx->d_embed offset:0 atIndex:0];
    if (!hy3_bind_weight(enc, ctx, &m->w.token_embd, 1)) exit(1);
    uint32_t token_u = (uint32_t)token, dim = HY3_N_EMBD;
    [enc setBytes:&token_u length:sizeof(token_u) atIndex:2];
    [enc setBytes:&dim length:sizeof(dim) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((HY3_N_EMBD + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (getenv("HY3_METAL_TRACE")) {
        float *e = (float *)ctx->d_embed.contents;
        fprintf(stderr, "TRACE after embed: %.4f %.4f %.4f %.4f\n", e[0], e[1], e[2], e[3]);
    }

    for (int il = 0; il < HY3_N_LAYER; il++) {
        int pos = m->cache_len / HY3_N_LAYER;
        double lt0 = hy3_metal_now();
        metal_forward_layer(ctx, m, il, pos);
        if (getenv("HY3_METAL_TRACE") && (il == 0 || il == 1 || il == 79)) {
            float *e = (float *)ctx->d_embed.contents;
            fprintf(stderr, "TRACE after layer %d: %.4f %.4f %.4f %.4f\n", il, e[0], e[1], e[2], e[3]);
        }
        if (getenv("HY3_METAL_PROFILE")) {
            double dt = hy3_metal_now() - lt0;
            fprintf(stderr, "hy3_metal: layer %d took %.3fms (%s)\n",
                    il, dt * 1000.0, il < HY3_N_LAYER_DENSE ? "dense" : "moe");
        }
    }
}

int hy3_eval_metal(hy3_model *m, const hy3_tokens *tokens, float *logits, int *pos) {
    hy3_metal_ctx_t *ctx = (hy3_metal_ctx_t *)m->metal_ctx;
    if (!ctx) return -1;

    static int use_fast = -1;
    if (use_fast < 0) {
        const char *e = getenv("HY3_FAST");
        use_fast = (!e || e[0] != '0');   /* default on */
    }

    for (int i = 0; i < tokens->len; i++) {
        int token = tokens->v[i];
        hy3_metal_grow_kv_cache(ctx, m->cache_len + HY3_N_LAYER);
        if (use_fast) {
            /* Only the last token needs logits; fold them into its cb. */
            metal_forward_model_fast(ctx, m, token, i == tokens->len - 1);
        } else {
            metal_forward_model(ctx, m, token);
        }
    }

    if (!use_fast) {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        m_rms_norm(ctx, enc, ctx->d_embed, ctx->d_embed, &m->w.output_norm, HY3_N_EMBD);
        // DO NOT REMOVE: in-place norm then matmul read; barrier orders them.
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        m_mul_mat(ctx, enc, &m->w.output, ctx->d_logits, ctx->d_embed, HY3_N_VOCAB, HY3_N_EMBD);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    if (getenv("HY3_METAL_TRACE")) {
        float *e = (float *)ctx->d_embed.contents;
        float *g = (float *)ctx->d_logits.contents;
        fprintf(stderr, "TRACE post-norm embed: %.4f %.4f %.4f | logits: %.4f %.4f %.4f\n",
                e[0], e[1], e[2], g[0], g[1], g[2]);
    }

    memcpy(logits, ctx->d_logits.contents, HY3_N_VOCAB * sizeof(float));
    *pos = m->cache_len;
    return 0;
}
