// fast_metal.m — standalone Metal kernel benchmark/experiment bed.
//
// Purpose: iterate on faster matmul (mat-vec) kernels for hy3 WITHOUT touching
// the production hy3_metal.m / hy3.metal. It synthesizes quantized weights and
// activations, runs each kernel variant, checks numerical agreement against a
// reference, and reports GB/s so we can see which idea actually wins before
// porting anything back.
//
// Focus: the MoE-layer bottleneck is cb1 = attention (q/k/v/o proj) + shared
// expert, all Q8_0. So the headline shapes here are the Q8_0 projections.
//
// Build:  see `make fast_metal`  (Makefile target added alongside this file)
// Run:    ./fast_metal

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double now_s(void) {
    static mach_timebase_info_data_t t;
    if (!t.denom) mach_timebase_info(&t);
    return mach_absolute_time() * (double)t.numer / t.denom / 1e9;
}

// ---------------------------------------------------------------------------
// Kernel source. Q8_0 block = { float d; int8 qs[32]; } = 36 bytes / 32 elems.
// Args match hy3: dst[buffer0], w[buffer1], x[buffer2], n[buffer3] (n=cols).
// Each kernel writes m_rows outputs; grid/threadgroup chosen per variant.
// ---------------------------------------------------------------------------
static NSString *kSrc = @R"METAL(
#include <metal_stdlib>
using namespace metal;

// ---- V0: baseline, one threadgroup(256) per row (current production style) ----
kernel void q8_v0(device float *dst [[buffer(0)]],
                  device const uchar *w [[buffer(1)]],
                  device const float *x [[buffer(2)]],
                  constant uint &n [[buffer(3)]],
                  uint row [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint tgSz [[threads_per_threadgroup]],
                  uint simd_lane [[thread_index_in_simdgroup]],
                  uint simd_grp [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[8];
    uint nb = n / 32;
    device const uchar *wr = w + (size_t)row * nb * 36;
    float sum = 0.0f;
    for (uint j = tid; j < nb; j += tgSz) {
        device const uchar *blk = wr + (size_t)j * 36;
        float d = *(device const float *)blk;
        device const char4 *qs = (device const char4 *)(blk + 4);
        device const float4 *xv = (device const float4 *)(x + j * 32);
        float local = 0.0f;
        for (uint l = 0; l < 8; l++) {
            float4 qf = float4(qs[l]);
            float4 xf = xv[l];
            local += qf.x*xf.x + qf.y*xf.y + qf.z*xf.z + qf.w*xf.w;
        }
        sum += d * local;
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) partial[simd_grp] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (uint i = 0; i < tgSz / 32; i++) t += partial[i];
        dst[row] = t;
    }
}

// ---- V1: current mm variant, 1 SIMD-group(32) computes NR0=4 rows ----
#define V1_NR0 4
kernel void q8_v1(device float *dst [[buffer(0)]],
                  device const uchar *w [[buffer(1)]],
                  device const float *x [[buffer(2)]],
                  constant uint &n [[buffer(3)]],
                  uint tgpig [[threadgroup_position_in_grid]],
                  ushort tiisg [[thread_index_in_simdgroup]])
{
    const short NQ = 8;
    const short ix = tiisg / 4;   // 0..7 block within a stride
    const short il = tiisg % 4;   // 0..3 slice of 8 quants
    const uint nb = n / 32;
    const uint first_row = (uint)tgpig * V1_NR0;
    const uint row_stride = nb * 36;
    float sumf[V1_NR0] = {0.f};
    float yl[8];
    device const float *yb = x + ix * 32 + il * NQ;
    for (uint ib = ix; ib < nb; ib += 8) {
        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];
        device const uchar *blk0 = w + first_row * row_stride + (size_t)ib * 36;
        for (short row = 0; row < V1_NR0; row++) {
            device const uchar *blk = blk0 + (size_t)row * row_stride;
            float d = *(device const float *)blk;
            device const char *qs = (device const char *)(blk + 4) + il * NQ;
            float sq = 0.f;
            for (short i = 0; i < NQ; ++i) sq += (float)qs[i] * yl[i];
            sumf[row] += sq * d;
        }
        yb += 8 * 32;
    }
    for (short row = 0; row < V1_NR0; row++) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0) dst[first_row + row] = tot;
    }
}

// ---- V2: like V1 but NR0=8 rows per SIMD-group (more weight reuse of x) ----
#define V2_NR0 8
kernel void q8_v2(device float *dst [[buffer(0)]],
                  device const uchar *w [[buffer(1)]],
                  device const float *x [[buffer(2)]],
                  constant uint &n [[buffer(3)]],
                  uint tgpig [[threadgroup_position_in_grid]],
                  ushort tiisg [[thread_index_in_simdgroup]])
{
    const short NQ = 8;
    const short ix = tiisg / 4;
    const short il = tiisg % 4;
    const uint nb = n / 32;
    const uint first_row = (uint)tgpig * V2_NR0;
    const uint row_stride = nb * 36;
    float sumf[V2_NR0] = {0.f};
    float yl[8];
    device const float *yb = x + ix * 32 + il * NQ;
    for (uint ib = ix; ib < nb; ib += 8) {
        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];
        device const uchar *blk0 = w + first_row * row_stride + (size_t)ib * 36;
        for (short row = 0; row < V2_NR0; row++) {
            device const uchar *blk = blk0 + (size_t)row * row_stride;
            float d = *(device const float *)blk;
            device const char *qs = (device const char *)(blk + 4) + il * NQ;
            float sq = 0.f;
            for (short i = 0; i < NQ; ++i) sq += (float)qs[i] * yl[i];
            sumf[row] += sq * d;
        }
        yb += 8 * 32;
    }
    for (short row = 0; row < V2_NR0; row++) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0) dst[first_row + row] = tot;
    }
}

// ---- V3: multi-SIMD-group threadgroup. TG has NSG simdgroups, each does NR0
//         rows, so one threadgroup covers NSG*NR0 rows. Same math as V1 but
//         amortizes threadgroup dispatch over more rows. ----
#define V3_NR0 4
#define V3_NSG 8
kernel void q8_v3(device float *dst [[buffer(0)]],
                  device const uchar *w [[buffer(1)]],
                  device const float *x [[buffer(2)]],
                  constant uint &n [[buffer(3)]],
                  uint tgpig [[threadgroup_position_in_grid]],
                  ushort tiisg [[thread_index_in_simdgroup]],
                  ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NQ = 8;
    const short ix = tiisg / 4;
    const short il = tiisg % 4;
    const uint nb = n / 32;
    const uint first_row = ((uint)tgpig * V3_NSG + sgitg) * V3_NR0;
    const uint row_stride = nb * 36;
    float sumf[V3_NR0] = {0.f};
    float yl[8];
    device const float *yb = x + ix * 32 + il * NQ;
    for (uint ib = ix; ib < nb; ib += 8) {
        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];
        device const uchar *blk0 = w + first_row * row_stride + (size_t)ib * 36;
        for (short row = 0; row < V3_NR0; row++) {
            device const uchar *blk = blk0 + (size_t)row * row_stride;
            float d = *(device const float *)blk;
            device const char *qs = (device const char *)(blk + 4) + il * NQ;
            float sq = 0.f;
            for (short i = 0; i < NQ; ++i) sq += (float)qs[i] * yl[i];
            sumf[row] += sq * d;
        }
        yb += 8 * 32;
    }
    for (short row = 0; row < V3_NR0; row++) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0) dst[first_row + row] = tot;
    }
}

// ---- GPU top-k router: pick top-K experts by (sigmoid(logit)+bias), then
//      emit their ids and the renormalized*scaled combine weights, entirely on
//      the GPU so the CPU never sees router logits mid-token. One threadgroup,
//      NE=192 experts, K<=8. Simple selection (K passes over NE). ----
kernel void router_topk(device const float *logits [[buffer(0)]],  // NE
                        device const float *bias   [[buffer(1)]],  // NE (or null)
                        device int         *out_ids [[buffer(2)]], // K
                        device float       *out_wts [[buffer(3)]], // K
                        constant uint      &NE      [[buffer(4)]],
                        constant uint      &K       [[buffer(5)]],
                        constant uint      &has_bias[[buffer(6)]],
                        constant float     &scaling [[buffer(7)]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint tgSz [[threads_per_threadgroup]])
{
    threadgroup float sig[256];   // sigmoid(logit)
    threadgroup float scored[256];// sigmoid+bias
    threadgroup int   chosen[8];
    threadgroup float chosen_sig[8];
    for (uint i = tid; i < NE; i += tgSz) {
        float s = 1.0f / (1.0f + exp(-logits[i]));
        sig[i] = s;
        scored[i] = s + (has_bias ? bias[i] : 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // K sequential argmax passes, single thread (K,NE small)
    if (tid == 0) {
        float sum_w = 0.0f;
        for (uint k = 0; k < K; k++) {
            int best = -1; float bv = -1e30f;
            for (uint i = 0; i < NE; i++) {
                float v = scored[i];
                if (v > bv) { bv = v; best = (int)i; }
            }
            chosen[k] = best;
            chosen_sig[k] = sig[best];
            sum_w += sig[best];
            scored[best] = -1e30f; // remove
        }
        float inv = 1.0f / (sum_w + 1e-20f);
        for (uint k = 0; k < K; k++) {
            out_ids[k] = chosen[k];
            out_wts[k] = chosen_sig[k] * inv * scaling;
        }
    }
}

// ---- mul_mv_id: one dispatch computes all K routed experts. tgpig.z selects
//      which expert slot; the real expert id comes from ids[slot], and its
//      weight base is experts_base + id*expert_stride_bytes. Same Q8_0 v1 math.
//      dst layout: [slot][row]  (K * M floats). ----
kernel void q8_mul_mv_id(device float *dst [[buffer(0)]],          // K*M
                         device const uchar *experts [[buffer(1)]],// all experts contiguous
                         device const float *x [[buffer(2)]],      // N
                         constant uint &n [[buffer(3)]],
                         constant uint &M [[buffer(4)]],
                         constant uint &expert_stride [[buffer(5)]], // bytes per expert
                         device const int *ids [[buffer(6)]],       // K
                         uint3 tgpig [[threadgroup_position_in_grid]],
                         ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint slot = tgpig.y;
    const int eid = ids[slot];
    device const uchar *w = experts + (size_t)eid * expert_stride;
    const short NQ = 8;
    const short ix = tiisg / 4;
    const short il = tiisg % 4;
    const uint nb = n / 32;
    const uint first_row = (uint)tgpig.x * 4;
    const uint row_stride = nb * 36;
    float sumf[4] = {0.f};
    float yl[8];
    device const float *yb = x + ix * 32 + il * NQ;
    for (uint ib = ix; ib < nb; ib += 8) {
        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];
        device const uchar *blk0 = w + first_row * row_stride + (size_t)ib * 36;
        for (short row = 0; row < 4; row++) {
            device const uchar *blk = blk0 + (size_t)row * row_stride;
            float d = *(device const float *)blk;
            device const char *qs = (device const char *)(blk + 4) + il * NQ;
            float sq = 0.f;
            for (short i = 0; i < NQ; ++i) sq += (float)qs[i] * yl[i];
            sumf[row] += sq * d;
        }
        yb += 8 * 32;
    }
    for (short row = 0; row < 4; row++) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0) dst[(size_t)slot * M + first_row + row] = tot;
    }
}
)METAL";

// ---------------------------------------------------------------------------
typedef struct { const char *name; int nr0; int nsg; } Variant;

static id<MTLComputePipelineState> make_pipe(id<MTLDevice> dev, id<MTLLibrary> lib, const char *fn) {
    NSError *e = nil;
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
    if (!f) { fprintf(stderr, "missing %s\n", fn); exit(1); }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&e];
    if (!p) { fprintf(stderr, "pipe %s: %s\n", fn, e.localizedDescription.UTF8String); exit(1); }
    return p;
}

int main(void) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError *e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:[MTLCompileOptions new] error:&e];
    if (!lib) { fprintf(stderr, "compile: %s\n", e.localizedDescription.UTF8String); return 1; }

    id<MTLComputePipelineState> p0 = make_pipe(dev, lib, "q8_v0");
    id<MTLComputePipelineState> p1 = make_pipe(dev, lib, "q8_v1");
    id<MTLComputePipelineState> p2 = make_pipe(dev, lib, "q8_v2");
    id<MTLComputePipelineState> p3 = make_pipe(dev, lib, "q8_v3");

    // Headline shapes from a MoE layer's cb1 (all Q8_0). {M rows, N cols}.
    struct { const char *tag; int M; int N; } shapes[] = {
        {"attn_q  8192x4096", 8192, 4096},
        {"attn_o  4096x8192", 4096, 8192},
        {"shexp_gate 1536x4096", 1536, 4096},
        {"shexp_down 4096x1536", 4096, 1536},
    };

    for (int s = 0; s < 4; s++) {
        int M = shapes[s].M, N = shapes[s].N;
        int nb = N / 32, bb = 36;
        size_t wB = (size_t)M * nb * bb;
        double gib = wB / 1073741824.0;
        id<MTLBuffer> bW = [dev newBufferWithLength:wB options:MTLResourceStorageModeShared];
        id<MTLBuffer> bX = [dev newBufferWithLength:N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bR = [dev newBufferWithLength:M * sizeof(float) options:MTLResourceStorageModeShared]; // reference
        id<MTLBuffer> bO = [dev newBufferWithLength:M * sizeof(float) options:MTLResourceStorageModeShared];

        // deterministic pseudo-random weights + activations
        unsigned char *w = (unsigned char *)bW.contents;
        srand(1234 + s);
        for (int r = 0; r < M; r++) for (int b = 0; b < nb; b++) {
            unsigned char *blk = w + ((size_t)r * nb + b) * bb;
            *(float *)blk = 0.02f + ((r + b) % 7) * 0.005f;
            signed char *qs = (signed char *)(blk + 4);
            for (int l = 0; l < 32; l++) qs[l] = (signed char)((rand() % 255) - 127);
        }
        float *x = (float *)bX.contents;
        for (int i = 0; i < N; i++) x[i] = ((rand() % 2000) - 1000) / 1000.0f;
        uint32_t n = N;

        // reference via V0
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:p0];
            [enc setBuffer:bR offset:0 atIndex:0];[enc setBuffer:bW offset:0 atIndex:1];
            [enc setBuffer:bX offset:0 atIndex:2];[enc setBytes:&n length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(M,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding];[cb commit];[cb waitUntilCompleted];
        }

        printf("\n=== %s  (%.3f GiB) ===\n", shapes[s].tag, gib);

        // Bench one variant. To mirror the real forward pass (many dispatches
        // encoded into ONE command buffer, a single commit/wait at the end),
        // we encode R dispatches into a single encoder and time the whole
        // buffer. That isolates GPU compute throughput from per-command-buffer
        // commit/wait latency. R is large so the fixed cost amortizes.
        #define BENCH(PIPE, GRID, TG, NAME) do { \
            int R=200; \
            /* warmup */ \
            { id<MTLCommandBuffer> cb=[q commandBuffer]; \
              id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder]; \
              [enc setComputePipelineState:PIPE];[enc setBuffer:bO offset:0 atIndex:0]; \
              [enc setBuffer:bW offset:0 atIndex:1];[enc setBuffer:bX offset:0 atIndex:2]; \
              [enc setBytes:&n length:4 atIndex:3]; \
              [enc dispatchThreadgroups:GRID threadsPerThreadgroup:TG]; \
              [enc endEncoding];[cb commit];[cb waitUntilCompleted]; } \
            double t0=now_s(); \
            id<MTLCommandBuffer> cb=[q commandBuffer]; \
            id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder]; \
            [enc setComputePipelineState:PIPE];[enc setBuffer:bW offset:0 atIndex:1]; \
            [enc setBuffer:bX offset:0 atIndex:2];[enc setBytes:&n length:4 atIndex:3]; \
            for (int r=0;r<R;r++){ [enc setBuffer:bO offset:0 atIndex:0]; \
                [enc dispatchThreadgroups:GRID threadsPerThreadgroup:TG]; } \
            [enc endEncoding];[cb commit];[cb waitUntilCompleted]; \
            double dt=(now_s()-t0)/R; \
            float *o=(float*)bO.contents,*ref=(float*)bR.contents; \
            float mx=0,mxabs=0; int badrow=-1; float bo=0,br=0; \
            for(int i=0;i<M;i++){float d=fabsf(o[i]-ref[i]); \
                float rl=d/(fabsf(ref[i])+1e-3f); \
                if(d>mxabs)mxabs=d; \
                if(rl>mx){mx=rl;badrow=i;bo=o[i];br=ref[i];}} \
            printf("  %-8s %.4f ms  %6.1f GB/s  absErr=%.1e\n", \
                   NAME, dt*1e3, gib*1.0737/dt, mxabs); \
        } while(0)

        BENCH(p0, MTLSizeMake(M,1,1),        MTLSizeMake(256,1,1), "v0");
        BENCH(p1, MTLSizeMake(M/4,1,1),      MTLSizeMake(32,1,1),  "v1(nr4)");
        BENCH(p2, MTLSizeMake(M/8,1,1),      MTLSizeMake(32,1,1),  "v2(nr8)");
        BENCH(p3, MTLSizeMake(M/(4*8),1,1),  MTLSizeMake(32*8,1,1),"v3(8sg)");
        #undef BENCH
    }

    // -----------------------------------------------------------------------
    // CORRECTNESS: GPU router top-k and mul_mv_id (the two new ds4-style pieces
    // that let the whole token stay on the GPU).
    // -----------------------------------------------------------------------
    {
        id<MTLComputePipelineState> pr = make_pipe(dev, lib, "router_topk");
        id<MTLComputePipelineState> pid = make_pipe(dev, lib, "q8_mul_mv_id");
        printf("\n=== CORRECTNESS: router_topk + mul_mv_id ===\n");

        // router: NE=192, K=4, with bias
        const uint NE = 192, K = 4; const float scaling = 2.826f;
        id<MTLBuffer> lg = [dev newBufferWithLength:NE*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bi = [dev newBufferWithLength:NE*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> oid= [dev newBufferWithLength:K*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> owt= [dev newBufferWithLength:K*4 options:MTLResourceStorageModeShared];
        float *lgp=(float*)lg.contents,*bip=(float*)bi.contents;
        srand(99);
        for (uint i=0;i<NE;i++){ lgp[i]=((rand()%2000)-1000)/500.0f; bip[i]=((rand()%200)-100)/1000.0f; }
        uint hb=1;
        { id<MTLCommandBuffer> cb=[q commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:pr];
          [e setBuffer:lg offset:0 atIndex:0];[e setBuffer:bi offset:0 atIndex:1];
          [e setBuffer:oid offset:0 atIndex:2];[e setBuffer:owt offset:0 atIndex:3];
          [e setBytes:&NE length:4 atIndex:4];[e setBytes:&K length:4 atIndex:5];
          [e setBytes:&hb length:4 atIndex:6];[e setBytes:&scaling length:4 atIndex:7];
          [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
          [e endEncoding];[cb commit];[cb waitUntilCompleted]; }
        // CPU reference top-k
        int cpu_id[8]; float cpu_wt[8];
        { float sc[256],sg[256]; for(uint i=0;i<NE;i++){float s=1.f/(1.f+expf(-lgp[i]));sg[i]=s;sc[i]=s+bip[i];}
          float sum=0; for(uint k=0;k<K;k++){int b=-1;float bv=-1e30f;for(uint i=0;i<NE;i++)if(sc[i]>bv){bv=sc[i];b=i;}
            cpu_id[k]=b;sum+=sg[b];sc[b]=-1e30f;}
          float inv=1.f/(sum+1e-20f); for(uint k=0;k<K;k++)cpu_wt[k]=sg[cpu_id[k]]*inv*scaling; }
        int *gid=(int*)oid.contents; float *gwt=(float*)owt.contents;
        int ok=1; for(uint k=0;k<K;k++){ if(gid[k]!=cpu_id[k]||fabsf(gwt[k]-cpu_wt[k])>1e-5f) ok=0; }
        printf("  router_topk: %s  gpu_ids=[%d %d %d %d] cpu_ids=[%d %d %d %d]\n",
               ok?"PASS":"FAIL", gid[0],gid[1],gid[2],gid[3], cpu_id[0],cpu_id[1],cpu_id[2],cpu_id[3]);

        // mul_mv_id: 4 experts of shape 1536x4096 Q8_0, contiguous
        const uint EM=1536, EN=4096; const uint enb=EN/32; const uint estride=EM*enb*36;
        const uint NEXP=8; // pool of 8 experts, we select 4
        id<MTLBuffer> EW=[dev newBufferWithLength:(size_t)NEXP*estride options:MTLResourceStorageModeShared];
        id<MTLBuffer> EX=[dev newBufferWithLength:EN*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> EO=[dev newBufferWithLength:(size_t)K*EM*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> ER=[dev newBufferWithLength:EM*4 options:MTLResourceStorageModeShared];
        unsigned char *ew=(unsigned char*)EW.contents;
        srand(7);
        for(size_t b=0;b<(size_t)NEXP*estride;b++) ew[b]=rand()&0xFF;
        for(uint ex=0;ex<NEXP;ex++)for(uint r=0;r<EM;r++)for(uint bb=0;bb<enb;bb++){
            unsigned char*blk=ew+(size_t)ex*estride+((size_t)r*enb+bb)*36; *(float*)blk=0.03f+((r+bb)%5)*0.004f; }
        float *exx=(float*)EX.contents; for(uint i=0;i<EN;i++)exx[i]=((rand()%2000)-1000)/1000.0f;
        int sel[4]={5,2,7,0}; memcpy(oid.contents,sel,sizeof(sel));
        uint en=EN;
        { id<MTLCommandBuffer> cb=[q commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:pid];
          [e setBuffer:EO offset:0 atIndex:0];[e setBuffer:EW offset:0 atIndex:1];
          [e setBuffer:EX offset:0 atIndex:2];[e setBytes:&en length:4 atIndex:3];
          [e setBytes:&EM length:4 atIndex:4];[e setBytes:&estride length:4 atIndex:5];
          [e setBuffer:oid offset:0 atIndex:6];
          [e dispatchThreadgroups:MTLSizeMake(EM/4,K,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
          [e endEncoding];[cb commit];[cb waitUntilCompleted]; }
        // CPU ref for slot 0 (expert sel[0])
        float *eo=(float*)EO.contents; int mism=0; float mxe=0;
        for(uint slot=0;slot<K;slot++){
            unsigned char*w=ew+(size_t)sel[slot]*estride;
            for(uint r=0;r<EM;r++){ float acc=0;
                for(uint bb=0;bb<enb;bb++){ unsigned char*blk=w+((size_t)r*enb+bb)*36; float d=*(float*)blk;
                    signed char*qs=(signed char*)(blk+4); float loc=0; for(int l=0;l<32;l++)loc+=(float)qs[l]*exx[bb*32+l]; acc+=d*loc; }
                float g=eo[(size_t)slot*EM+r]; float e2=fabsf(g-acc); if(e2>mxe)mxe=e2; if(e2>1e-2f)mism++; }
        }
        printf("  mul_mv_id  : %s  (K=%d experts in 1 dispatch, maxAbsErr=%.1e, mism=%d)\n",
               mism==0?"PASS":"FAIL", K, mxe, mism);
    }

    // -----------------------------------------------------------------------
    // FULL-TOKEN SIMULATION.
    //
    // Reproduce one decode token's worth of MoE-layer matmuls (the dominant
    // path) and compare command-buffer strategies:
    //   A) per-layer: 2 command buffers/layer + waitUntilCompleted each
    //      (mirrors current hy3: cb1 attn+shexp, CPU topk, cb2 routed).
    //   B) one command buffer for the whole token, single wait at the end
    //      (mirrors ds4's batch encoder — only possible if expert selection
    //      stays on the GPU so the CPU never needs router logits mid-token).
    //
    // Each "layer" here encodes the real Q8_0/Q4_K matmul shapes:
    //   attn: q 8192x4096, k 1024x4096, v 1024x4096, o 4096x8192   (Q8_0)
    //   shexp: gate 1536x4096, up 1536x4096, down 4096x1536        (Q8_0)
    //   routed(x4): gate 1536x4096, up 1536x4096, down 4096x1536   (Q4_K)
    // We reuse the Q8_0 v1 pipeline for every matmul (shape-accurate timing;
    // Q4_K is similar cost per byte), which is enough to compare A vs B.
    // -----------------------------------------------------------------------
    {
        const int NLAYER = 79;      // MoE layers
        const int NROUTED = 4;      // experts per token (top_k=4)
        // biggest weight we touch: attn_o 4096x8192 Q8_0
        int maxN = 8192, maxM = 8192;
        int nb = maxN / 32;
        size_t wB = (size_t)maxM * nb * 36;
        id<MTLBuffer> W = [dev newBufferWithLength:wB options:MTLResourceStorageModeShared];
        id<MTLBuffer> X = [dev newBufferWithLength:maxN * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> O = [dev newBufferWithLength:maxM * sizeof(float) options:MTLResourceStorageModeShared];
        memset(W.contents, 1, wB);
        for (int i = 0; i < maxN; i++) ((float*)X.contents)[i] = 0.01f;

        // (M rows, N cols) of every matmul in one MoE layer
        struct { int M, N; } mm[] = {
            {8192,4096},{1024,4096},{1024,4096},{4096,8192}, // attn q k v o
            {1536,4096},{1536,4096},{4096,1536},             // shexp gate up down
        };
        int n_attn_shexp = 7;

        #define ENCODE_MM(ENC, M_, N_) do { \
            uint32_t nn=(N_); [ENC setComputePipelineState:p1]; \
            [ENC setBuffer:O offset:0 atIndex:0];[ENC setBuffer:W offset:0 atIndex:1]; \
            [ENC setBuffer:X offset:0 atIndex:2];[ENC setBytes:&nn length:4 atIndex:3]; \
            [ENC dispatchThreadgroups:MTLSizeMake((M_)/4,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; \
        } while(0)

        // ---- Strategy A: 2 cb/layer + wait each ----
        for (int warm=0; warm<2; warm++) {
            for (int L=0; L<NLAYER; L++) {
                id<MTLCommandBuffer> cb=[q commandBuffer];
                id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
                for (int i=0;i<n_attn_shexp;i++) ENCODE_MM(e, mm[i].M, mm[i].N);
                [e endEncoding];[cb commit];[cb waitUntilCompleted];
                id<MTLCommandBuffer> cb2=[q commandBuffer];
                id<MTLComputeCommandEncoder> e2=[cb2 computeCommandEncoder];
                for (int r=0;r<NROUTED;r++){ ENCODE_MM(e2,1536,4096);ENCODE_MM(e2,1536,4096);ENCODE_MM(e2,4096,1536);}
                [e2 endEncoding];[cb2 commit];[cb2 waitUntilCompleted];
            }
        }
        double tA0=now_s();
        for (int L=0; L<NLAYER; L++) {
            id<MTLCommandBuffer> cb=[q commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            for (int i=0;i<n_attn_shexp;i++) ENCODE_MM(e, mm[i].M, mm[i].N);
            [e endEncoding];[cb commit];[cb waitUntilCompleted];
            id<MTLCommandBuffer> cb2=[q commandBuffer];
            id<MTLComputeCommandEncoder> e2=[cb2 computeCommandEncoder];
            for (int r=0;r<NROUTED;r++){ ENCODE_MM(e2,1536,4096);ENCODE_MM(e2,1536,4096);ENCODE_MM(e2,4096,1536);}
            [e2 endEncoding];[cb2 commit];[cb2 waitUntilCompleted];
        }
        double tA=now_s()-tA0;

        // ---- Strategy B: one cb for the whole token, single wait ----
        for (int warm=0; warm<2; warm++) {
            id<MTLCommandBuffer> cb=[q commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            for (int L=0; L<NLAYER; L++) {
                for (int i=0;i<n_attn_shexp;i++) ENCODE_MM(e, mm[i].M, mm[i].N);
                for (int r=0;r<NROUTED;r++){ ENCODE_MM(e,1536,4096);ENCODE_MM(e,1536,4096);ENCODE_MM(e,4096,1536);}
            }
            [e endEncoding];[cb commit];[cb waitUntilCompleted];
        }
        double tB0=now_s();
        {
            id<MTLCommandBuffer> cb=[q commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            for (int L=0; L<NLAYER; L++) {
                for (int i=0;i<n_attn_shexp;i++) ENCODE_MM(e, mm[i].M, mm[i].N);
                for (int r=0;r<NROUTED;r++){ ENCODE_MM(e,1536,4096);ENCODE_MM(e,1536,4096);ENCODE_MM(e,4096,1536);}
            }
            [e endEncoding];[cb commit];[cb waitUntilCompleted];
        }
        double tB=now_s()-tB0;

        // ---- Strategy C: concurrent encoder (no auto hazard tracking) ----
        // Independent matmuls in a layer (gate/up, the K routed experts) can
        // overlap. We only barrier between dependent stages. Here we simply run
        // every matmul concurrently within a layer, barrier between layers.
        for (int warm=0; warm<2; warm++) {
            id<MTLCommandBuffer> cb=[q commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            for (int L=0; L<NLAYER; L++) {
                for (int i=0;i<n_attn_shexp;i++) ENCODE_MM(e, mm[i].M, mm[i].N);
                for (int r=0;r<NROUTED;r++){ ENCODE_MM(e,1536,4096);ENCODE_MM(e,1536,4096);ENCODE_MM(e,4096,1536);}
                [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [e endEncoding];[cb commit];[cb waitUntilCompleted];
        }
        double tC0=now_s();
        {
            id<MTLCommandBuffer> cb=[q commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            for (int L=0; L<NLAYER; L++) {
                for (int i=0;i<n_attn_shexp;i++) ENCODE_MM(e, mm[i].M, mm[i].N);
                for (int r=0;r<NROUTED;r++){ ENCODE_MM(e,1536,4096);ENCODE_MM(e,1536,4096);ENCODE_MM(e,4096,1536);}
                [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [e endEncoding];[cb commit];[cb waitUntilCompleted];
        }
        double tC=now_s()-tC0;
        #undef ENCODE_MM

        printf("\n=== FULL TOKEN (79 MoE layers, top_k=4) ===\n");
        printf("  A) 2 cb/layer + wait each : %.2f ms/token  (%.1f tok/s)\n", tA*1e3, 1.0/tA);
        printf("  B) 1 cb/token serial      : %.2f ms/token  (%.1f tok/s)\n", tB*1e3, 1.0/tB);
        printf("  C) 1 cb/token concurrent  : %.2f ms/token  (%.1f tok/s)\n", tC*1e3, 1.0/tC);
        printf("  speedup B/A: %.2fx   C/B: %.2fx\n", tA/tB, tB/tC);
    }
    return 0;
}}
