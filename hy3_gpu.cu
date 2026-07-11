#ifdef __cplusplus
extern "C" {
#endif
#include "hy3.h"
#ifdef __cplusplus
}
#endif
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <float.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define CUDA_CHECK(call) do { cudaError_t e=call; if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error: %s (%s:%d)\n",cudaGetErrorString(e),__FILE__,__LINE__); exit(1);} } while(0)
#define CUBLAS_CHECK(call) do { cublasStatus_t s=call; if(s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS error %d (%s:%d)\n",(int)s,__FILE__,__LINE__); exit(1);} } while(0)

#define BLOCK_DIM 256
#define CUDA_QK_K 256
#define ATTN_SPLITS_MAX 128 /* upper bound for dynamic split-KV; actual splits adapt to ntok */

/* INT8 KV cache: per-head quant, FP16 scale + 128×int8 per head.
 * Slot = 8×(2+128)=1040 bytes  (vs 2048 for FP16) — ~49 % less memory & bandwidth. */
#define KV_INT8_QOFF   ((HY3_N_KV_HEAD)*2)              /* 16 bytes of FP16 scales */
#define KV_INT8_STRIDE (KV_INT8_QOFF + (HY3_N_KV_HEAD)*(HY3_HEAD_DIM))  /* 1040 */

typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[CUDA_QK_K/2]; } cuda_block_q4_K;

__device__ static float dev_f16_to_f32(uint16_t v){ return __half2float(*(const __half*)&v); }
__device__ static void dev_q4_K_get_scale_min(uint32_t j,const uint8_t*s,uint8_t*d,uint8_t*m){
    if(j<4u){*d=s[j]&63u;*m=s[j+4u]&63u;}
    else{*d=(s[j+4u]&0x0fu)|((s[j-4u]>>6u)<<4u);*m=(s[j+4u]>>4u)|((s[j]>>6u)<<4u);}
}
/* Coalesced warp-cooperative Q4_K row dot: the 32 lanes read each block's
 * 128-byte qs as 32 contiguous uint32 (one coalesced transaction), lane L
 * owning byte range [L*4, L*4+3] = pair p=L/8, elements e0=(L%8)*4..+3.
 * Each lane applies its sub-block scale/min to its 4 low + 4 high nibble
 * contributions; the caller warp-reduces. This fixes the ~56x-off-bandwidth
 * uncoalesced access that dominated decode. Returns this lane's partial. */
__device__ static float warp_row_dot_q4k(const cuda_block_q4_K*wr,const float*xs,uint32_t nb,uint32_t lane){
    uint32_t p=lane>>3, e0=(lane&7u)*4u;
    float acc=0.f;
    for(uint32_t b=0;b<nb;b++){
        const cuda_block_q4_K*blk=wr+b;
        float xd=dev_f16_to_f32(blk->d),xmin=dev_f16_to_f32(blk->dmin);
        uint8_t sc0,m0,sc1,m1;
        dev_q4_K_get_scale_min(2u*p,blk->scales,&sc0,&m0);
        dev_q4_K_get_scale_min(2u*p+1u,blk->scales,&sc1,&m1);
        uint32_t w=((const uint32_t*)blk->qs)[lane];
        const float*xb=xs+(size_t)b*CUDA_QK_K;
        const float*xlo=xb+2u*p*32u+e0,*xhi=xb+(2u*p+1u)*32u+e0;
        float dlo=0.f,dhi=0.f,slo=0.f,shi=0.f;
        #pragma unroll
        for(int bi=0;bi<4;bi++){
            uint32_t byte=(w>>(bi*8))&0xFFu; float xl=xlo[bi],xh=xhi[bi];
            dlo+=(float)(byte&0xFu)*xl; slo+=xl;
            dhi+=(float)(byte>>4)*xh;   shi+=xh;
        }
        acc+=(float)sc0*xd*dlo-(float)m0*xmin*slo+(float)sc1*xd*dhi-(float)m1*xmin*shi;
    }
    return acc;
}

/* ===== GPU-resident MoE (ds4/hy3.metal) ===== */
__global__ void router_topk_kernel(const float*logits,const float*bias,int*out_ids,float*out_wts,
    uint32_t NE,uint32_t K,uint32_t has_bias,float scaling){
    __shared__ float sig[256],scored[256]; uint32_t tid=threadIdx.x;
    for(uint32_t i=tid;i<NE;i+=blockDim.x){ float s=1.f/(1.f+expf(-logits[i])); sig[i]=s; scored[i]=s+(has_bias?bias[i]:0.f); }
    __syncthreads();
    if(tid==0){
        float sw=0.f;
        for(uint32_t k=0;k<K;k++){
            int best=0; float bv=-1e30f;
            for(uint32_t i=0;i<NE;i++){ if(scored[i]>bv){bv=scored[i];best=(int)i;} }
            out_ids[k]=best; out_wts[k]=sig[best]; sw+=sig[best]; scored[best]=-1e30f;
        }
        float inv=1.f/(sw+1e-20f);
        for(uint32_t k=0;k<K;k++) out_wts[k]=out_wts[k]*inv*scaling;
    }
}
/* Warp-per-row Q4_K expert matmul: one warp (32 lanes) computes one output
 * row via full-warp reduction; grid.x = ceil(M/WPB), grid.y = slot. Gives
 * ~M/8 * nu blocks (e.g. 1536 for gate) vs the old 24 -- full SM occupancy
 * hides the Q4_K dequant + weight-read latency that was ~85% of decode time. */
#define MOE_WPB 8   /* warps per block (8 warps = 256 threads; safe & fast) */
__global__ static void moe_matmul_q4k_id_kernel(float*dst,const uint8_t*const*ptrs,
    const float*x,const int*ids,uint32_t n,uint32_t M,uint32_t xps){
    __shared__ float sxs[HY3_N_EMBD];
    uint32_t slot=blockIdx.y; int eid=ids[slot];
    const uint8_t*w=ptrs[eid]; const float*xg=xps?(x+(uint64_t)slot*n):x;
    for(uint32_t i=threadIdx.x;i<n;i+=blockDim.x) sxs[i]=xg[i];
    __syncthreads();
    uint32_t warp=threadIdx.x>>5,lane=threadIdx.x&31u,nb=n/CUDA_QK_K;
    uint32_t row=blockIdx.x*MOE_WPB+warp; if(row>=M) return;
    const cuda_block_q4_K*wr=(const cuda_block_q4_K*)(w+(uint64_t)row*(uint64_t)nb*sizeof(cuda_block_q4_K));
    float acc=warp_row_dot_q4k(wr,sxs,nb,lane);
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffffu,acc,o);
    if(lane==0) dst[(uint64_t)slot*M+row]=acc;
}
__global__ static void moe_matmul_q4k_id_gate_up_kernel(float*go,float*uo,
    const uint8_t*const*gp,const uint8_t*const*up,const float*x,const int*ids,
    uint32_t n,uint32_t M,uint32_t xps){
    __shared__ float sxs[HY3_N_EMBD];
    uint32_t slot=blockIdx.y; int eid=ids[slot];
    const uint8_t*gw=gp[eid],*uw=up[eid]; const float*xg=xps?(x+(uint64_t)slot*n):x;
    for(uint32_t i=threadIdx.x;i<n;i+=blockDim.x) sxs[i]=xg[i];
    __syncthreads();
    uint32_t warp=threadIdx.x>>5,lane=threadIdx.x&31u,nb=n/CUDA_QK_K;
    uint32_t row=blockIdx.x*MOE_WPB+warp; if(row>=M) return;
    uint64_t rb=(uint64_t)nb*sizeof(cuda_block_q4_K);
    const cuda_block_q4_K*gr=(const cuda_block_q4_K*)(gw+(uint64_t)row*rb);
    const cuda_block_q4_K*ur=(const cuda_block_q4_K*)(uw+(uint64_t)row*rb);
    float g=warp_row_dot_q4k(gr,sxs,nb,lane),u=warp_row_dot_q4k(ur,sxs,nb,lane);
    for(int o=16;o>0;o>>=1){ g+=__shfl_down_sync(0xffffffffu,g,o); u+=__shfl_down_sync(0xffffffffu,u,o); }
    if(lane==0){ go[(uint64_t)slot*M+row]=g; uo[(uint64_t)slot*M+row]=u; }
}
__global__ void moe_combine_id_kernel(float*emb,const float*dk,const float*wt,uint32_t M,uint32_t K){
    uint32_t r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=M) return;
    float acc=0.f; for(uint32_t k=0;k<K;k++) acc+=wt[k]*dk[(uint64_t)k*M+r]; emb[r]+=acc;
}

/* ===== standard + fused kernels ===== */
__global__ void rms_norm_kernel(float*out,const float*x,const float*w,int n){
    extern __shared__ float sd[]; int tid=threadIdx.x; float sum=0.f;
    for(int i=tid;i<n;i+=blockDim.x) sum+=x[i]*x[i]; sd[tid]=sum; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(tid<s) sd[tid]+=sd[tid+s]; __syncthreads(); }
    float r=rsqrtf(sd[0]/(float)n+1e-5f);
    for(int i=tid;i<n;i+=blockDim.x) out[i]=x[i]*r*w[i];
}
__global__ void silu_mul_kernel(float*out,const float*g,const float*u,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float x=g[i]; out[i]=(x/(1.f+expf(-x)))*u[i]; }
}
/* fused Q/K norm + rope: one block per head (Q heads then K heads) */
__global__ void qk_norm_rope_fused_kernel(float*q,float*k,const float*qw,const float*kw,
    int hdim,int nh,int nkh,const int*d_pos){
    int pos=*d_pos;
    int head=blockIdx.x,iskv=(head>=nh),h=iskv?(head-nh):head;
    float*buf=iskv?k:q,*x=buf+(size_t)h*hdim; const float*w=iskv?kw:qw;
    extern __shared__ float sd[]; int tid=threadIdx.x; float sum=0.f;
    for(int i=tid;i<hdim;i+=blockDim.x) sum+=x[i]*x[i]; sd[tid]=sum; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(tid<s) sd[tid]+=sd[tid+s]; __syncthreads(); }
    float r=rsqrtf(sd[0]/(float)hdim+1e-5f); __syncthreads();
    for(int i=tid;i<hdim;i+=blockDim.x) x[i]=x[i]*r*w[i]; __syncthreads();
    int half=hdim/2;
    if(tid<half){ float fr=(float)pos/powf(11158840.f,(float)(2*tid)/(float)hdim);
        float c=cosf(fr),s=sinf(fr),v0=x[tid],v1=x[tid+half];
        x[tid]=v0*c-v1*s; x[tid+half]=v1*c+v0*s; }
}
/* fused norm + router GEMV */
__global__ void norm_router_gemv_fused_kernel(float*rout,const float*x,const float*nw,const float*rw,int ne,int nex){
    int row=blockIdx.x; if(row>=nex) return;
    extern __shared__ float sd[]; int tid=threadIdx.x; float sum=0.f;
    for(int i=tid;i<ne;i+=blockDim.x) sum+=x[i]*x[i]; sd[tid]=sum; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(tid<s) sd[tid]+=sd[tid+s]; __syncthreads(); }
    float r=rsqrtf(sd[0]/(float)ne+1e-5f); __syncthreads();
    const float*wr=rw+(size_t)row*ne; float acc=0.f;
    for(int i=tid;i<ne;i+=blockDim.x) acc+=x[i]*r*nw[i]*wr[i];
    sd[tid]=acc; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(tid<s) sd[tid]+=sd[tid+s]; __syncthreads(); }
    if(tid==0) rout[row]=sd[0];
}
/* split-KV online-softmax attention, INT8 KV cache with per-head FP16 scales.
 * Each head subdivides the KV sequence into nsplits chunks; one warp per
 * chunk computes partials (local max / sum / per-lane accumulator).  A reduce
 * kernel merges all partials for each head.  This parallelises the O(context)
 * attention loop that dominated long-context decode. */
__global__ void attention_split_kv_kernel(
    const float*q,const uint8_t*kc,const uint8_t*vc,
    int nh,int nkh,int hdim,const int*d_pos,
    int kvg,int lid,int nl,int nsplits,float*partials)
{
    int tid=blockIdx.x,h=tid/nsplits; if(h>=nh) return;
    int kvh=h/kvg,ntok=(*d_pos)+1; if(ntok<1) return;
    int chunk=(ntok+nsplits-1)/nsplits,s=tid%nsplits;
    int rs=s*chunk,re=(rs+chunk<ntok)?(rs+chunk):ntok;
    const float*qh=q+(size_t)h*hdim;
    float scale=rsqrtf((float)hdim); int lane=threadIdx.x&31u;
    float4 qv=((const float4*)qh)[lane];
    float ms=-INFINITY,ss=0.f; float4 ov=make_float4(0,0,0,0);
    __shared__ float4 kvs[8*32*2],osh[32];
    /* empty split → zero partials (reduce skips pss<=0) */
    if(rs>=re){ osh[lane]=make_float4(0,0,0,0); __syncthreads();
        if(lane==0){ float*p=partials+(size_t)tid*130; p[0]=-INFINITY;p[1]=0.f;
            for(int i=0;i<128;i++) p[2+i]=0.f; } return; }
    for(int r0=rs;r0<re;r0+=8){
        int nr=(re-r0<8)?(re-r0):8;
        for(int off=threadIdx.x;off<nr*32;off+=blockDim.x){
            int r=off/32,c=off&31,t=r0+r;
            size_t base=(size_t)(t*nl+lid)*KV_INT8_STRIDE;
            float ks=__half2float(__ldg((const half*)(kc+base)+kvh));
            float vs=__half2float(__ldg((const half*)(vc+base)+kvh));
            const int8_t*kq=(const int8_t*)(kc+base+KV_INT8_QOFF)+(size_t)kvh*hdim;
            const int8_t*vq=(const int8_t*)(vc+base+KV_INT8_QOFF)+(size_t)kvh*hdim;
            int c4=c*4; float4 tm;
            uint32_t kp=__ldg((const uint32_t*)(kq+c4)),vp=__ldg((const uint32_t*)(vq+c4));
            tm.x=(float)(int8_t)(kp&0xFF)*ks; tm.y=(float)(int8_t)((kp>>8)&0xFF)*ks;
            tm.z=(float)(int8_t)((kp>>16)&0xFF)*ks; tm.w=(float)(int8_t)((kp>>24)&0xFF)*ks; kvs[off]=tm;
            tm.x=(float)(int8_t)(vp&0xFF)*vs; tm.y=(float)(int8_t)((vp>>8)&0xFF)*vs;
            tm.z=(float)(int8_t)((vp>>16)&0xFF)*vs; tm.w=(float)(int8_t)((vp>>24)&0xFF)*vs; kvs[nr*32+off]=tm;
        }
        __syncthreads();
        for(int r=0;r<nr;r++){
            float4 k4=kvs[r*32+lane],v4=kvs[nr*32+r*32+lane];
            float sc=qv.x*k4.x+qv.y*k4.y+qv.z*k4.z+qv.w*k4.w;
            for(int o=16;o>0;o>>=1) sc+=__shfl_down_sync(0xffffffffu,sc,o);
            sc=__shfl_sync(0xffffffffu,sc,0)*scale;
            float nm=fmaxf(ms,sc),os=__expf(ms-nm),rs=__expf(sc-nm);
            ss=ss*os+rs; ov.x=ov.x*os+v4.x*rs; ov.y=ov.y*os+v4.y*rs; ov.z=ov.z*os+v4.z*rs; ov.w=ov.w*os+v4.w*rs; ms=nm;
        }
        __syncthreads();
    }
    osh[lane]=ov; __syncthreads();
    if(lane==0){ float*p=partials+(size_t)tid*130; p[0]=ms;p[1]=ss; for(int i=0;i<32;i++){ float4 v=osh[i]; p[2+i*4+0]=v.x;p[2+i*4+1]=v.y;p[2+i*4+2]=v.z;p[2+i*4+3]=v.w; } }
}
/* Reduce per-head split partials with the same online-softmax merge rule,
 * processing only this thread's lane index across all partials. */
__global__ void attention_reduce_kernel(float*out,const float*partials,int nh,int hdim,int nsplits){
    int h=blockIdx.x,lane=threadIdx.x; if(h>=nh) return;
    float ms=-INFINITY,ss=0.f; float4 ov=make_float4(0,0,0,0);
    for(int s=0;s<nsplits;s++){
        const float*p=partials+(size_t)(h*nsplits+s)*130; float pms=p[0],pss=p[1];
        if(pss<=0.f) continue;
        float nm=fmaxf(ms,pms),os=__expf(ms-nm),rs=__expf(pms-nm);
        ss=ss*os+pss*rs;
        const float*pv=p+2+lane*4;
        ov.x=ov.x*os+pv[0]*rs; ov.y=ov.y*os+pv[1]*rs; ov.z=ov.z*os+pv[2]*rs; ov.w=ov.w*os+pv[3]*rs;
        ms=nm;
    }
    float inv=ss>0.f?1.f/ss:0.f; ov.x*=inv;ov.y*=inv;ov.z*=inv;ov.w*=inv;
    float*oh=out+(size_t)h*hdim; ((float4*)oh)[lane]=ov;
}
/* INT8 KV quantise: per-head absmax scale → FP16 scale + 128×int8.
 * One block per head, warp-reduce finds max-abs. */
__global__ void kv_quantize_int8_kernel(uint8_t*kc,uint8_t*vc,const float*k,const float*v,
    const int*d_pos,int il,int nl){
    int head=blockIdx.x; if(head>=HY3_N_KV_HEAD)return;
    const float*kh=k+(size_t)head*HY3_HEAD_DIM,*vh=v+(size_t)head*HY3_HEAD_DIM;
    float mk=0.f,mv=0.f;
    for(int i=threadIdx.x;i<HY3_HEAD_DIM;i+=blockDim.x){
        mk=fmaxf(mk,fabsf(kh[i])); mv=fmaxf(mv,fabsf(vh[i]));
    }
    for(int o=16;o>0;o>>=1){mk=fmaxf(mk,__shfl_down_sync(0xffffffffu,mk,o));mv=fmaxf(mv,__shfl_down_sync(0xffffffffu,mv,o));}
    mk=__shfl_sync(0xffffffffu,mk,0); mv=__shfl_sync(0xffffffffu,mv,0);
    float sk=mk/127.f+1e-8f,sv=mv/127.f+1e-8f,isk=1.f/sk,isv=1.f/sv;
    size_t base=(size_t)((*d_pos)*nl+il)*KV_INT8_STRIDE;
    if(threadIdx.x==0){((half*)(kc+base))[head]=__float2half(sk);((half*)(vc+base))[head]=__float2half(sv);}
    int8_t*kq=(int8_t*)(kc+base+KV_INT8_QOFF)+(size_t)head*HY3_HEAD_DIM;
    int8_t*vq=(int8_t*)(vc+base+KV_INT8_QOFF)+(size_t)head*HY3_HEAD_DIM;
    for(int i=threadIdx.x;i<HY3_HEAD_DIM;i+=blockDim.x){
        float fk=kh[i]*isk,fv=vh[i]*isv;
        kq[i]=(int8_t)roundf(fmaxf(-127.f,fminf(127.f,fk)));
        vq[i]=(int8_t)roundf(fmaxf(-127.f,fminf(127.f,fv)));
    }
}
__global__ void embed_lookup_kernel(float*out,const float*table,int token,int dim){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<dim) out[i]=table[(size_t)token*dim+i];
}
/* Fused residual add + RMS norm: x += a; out = rmsnorm(x)*w. One block.
 * Removes a separate add kernel + its inter-kernel gap on the critical path. */
__global__ void add_rmsnorm_kernel(float*out,float*x,const float*a,const float*w,int n){
    extern __shared__ float sd[]; int tid=threadIdx.x; float sum=0.f;
    for(int i=tid;i<n;i+=blockDim.x){ float v=x[i]+a[i]; x[i]=v; sum+=v*v; }
    sd[tid]=sum; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(tid<s) sd[tid]+=sd[tid+s]; __syncthreads(); }
    float r=rsqrtf(sd[0]/(float)n+1e-5f);
    for(int i=tid;i<n;i+=blockDim.x) out[i]=x[i]*r*w[i];
}
__global__ void add_kernel(float*out,const float*a,const float*b,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=a[i]+b[i];
}

/* ===== Context ===== */
typedef struct { uint8_t*data; int bytes; } q4k_buf_t;
typedef struct {
    cublasHandle_t cublas; cudaStream_t stream;
    cudaStream_t stream2;            /* routed-expert matmuls, overlapped w/ shared GEMV */
    cudaEvent_t ev_fork,ev_join;     /* fork/join for multi-stream graph capture */
    float *d_token_embd,*d_output_norm; half *d_output;   /* d_output: FP16 GEMV */
    half  *d_layer_attn_qkv[81],*d_layer_attn_output[81]; /* FP16 GEMV weights */
    float *d_layer_attn_q_norm[81],*d_layer_attn_k_norm[81],*d_layer_attn_norm[81],*d_layer_ffn_norm[81];
    float *d_layer_ffn_gate_inp[81];   /* router: FP32 for stable top-k argmax */
    half  *d_layer_ffn_down_shexp[81],*d_layer_ffn_gateup_shexp[81];
    float *d_layer_eh_proj[81],*d_layer_enorm[81],*d_layer_hnorm[81],*d_layer_final_norm[81];
    float *d_layer_expert_bias[81];
    half  *d_layer_dense_ffn_gate[81],*d_layer_dense_ffn_up[81],*d_layer_dense_ffn_down[81];
    q4k_buf_t d_q4k_gate_exps[81][192],d_q4k_up_exps[81][192],d_q4k_down_exps[81][192];
    uint8_t **d_gate_ptrs[81],**d_up_ptrs[81],**d_down_ptrs[81];
    uint8_t *d_k_cache,*d_v_cache; int ctx_cap;
    float *d_embed,*d_scratch,*d_scratch2,*d_logits,*d_attn_partials;
    int *d_router_ids; float *d_router_wts,*d_moe_gate_k,*d_moe_up_k,*d_moe_mid_k,*d_moe_down_k;
    half *d_xf16;  /* F16 activation scratch for cublasGemmEx (max input dim) */
    int *d_pos; int *h_pos;      /* device + pinned-host token position for graph replay */
    cudaGraphExec_t graph_exec; int graph_ready; int graph_warmed; int attn_splits;
    uint8_t *graph_kc,*graph_vc;    /* KV cache ptrs baked into graph; realloc -> recapture */
} gpu_ctx_t;

/* ===== upload ===== */
static inline float fp16_to_float(uint16_t h){
    uint32_t s=(uint32_t)(h>>15),e=(uint32_t)((h>>10)&0x1f),m=(uint32_t)(h&0x3ff),f;
    if(e==0)f=(s<<31)|((0x7f-15)<<23)|(m<<13);
    else if(e==31)f=(s<<31)|0x7f800000|(m<<13);
    else f=(s<<31)|((e+0x70)<<23)|(m<<13);
    float r;memcpy(&r,&f,4);return r;
}
static float *upload_f32(const uint8_t*d,uint64_t n){float*b;CUDA_CHECK(cudaMalloc(&b,n*sizeof(float)));CUDA_CHECK(cudaMemcpy(b,d,n*sizeof(float),cudaMemcpyHostToDevice));return b;}
static float *upload_q8_0(const uint8_t*s,uint64_t n){static const int Q=32;float*b;CUDA_CHECK(cudaMalloc(&b,n*sizeof(float)));float*h=(float*)malloc(n*sizeof(float));uint64_t nb=n/Q;for(uint64_t j=0;j<nb;j++){float d;memcpy(&d,s+j*36,4);const int8_t*qs=(const int8_t*)(s+j*36+4);for(int k=0;k<Q;k++)h[j*Q+k]=(float)qs[k]*d;}CUDA_CHECK(cudaMemcpy(b,h,n*sizeof(float),cudaMemcpyHostToDevice));free(h);return b;}
static float *upload_f16(const uint8_t*d,uint64_t n){float*b;CUDA_CHECK(cudaMalloc(&b,n*sizeof(float)));float*h=(float*)malloc(n*sizeof(float));const uint16_t*s=(const uint16_t*)d;for(uint64_t i=0;i<n;i++)h[i]=fp16_to_float(s[i]);CUDA_CHECK(cudaMemcpy(b,h,n*sizeof(float),cudaMemcpyHostToDevice));free(h);return b;}
static void upload_q4k_compressed(q4k_buf_t*o,const uint8_t*d,uint64_t n){uint64_t nb=n/CUDA_QK_K;o->bytes=(int)(nb*sizeof(cuda_block_q4_K));CUDA_CHECK(cudaMalloc(&o->data,o->bytes));CUDA_CHECK(cudaMemcpy(o->data,d,o->bytes,cudaMemcpyHostToDevice));}
static void hq4ksm(int j,const uint8_t*q,uint8_t*sc,uint8_t*m){if(j<4){*sc=q[j]&63;*m=q[j+4]&63;}else{*sc=(q[j+4]&0xF)|((q[j-4]>>6)<<4);*m=(q[j+4]>>4)|((q[j]>>6)<<4);}}
static float *upload_q4k_dense(const uint8_t*s,uint64_t n){static const int Q=256;float*b;CUDA_CHECK(cudaMalloc(&b,n*sizeof(float)));float*h=(float*)malloc(n*sizeof(float));uint64_t nb=n/Q;for(uint64_t i=0;i<nb;i++){const uint8_t*blk=s+i*144;uint16_t d16,dm16;memcpy(&d16,blk,2);memcpy(&dm16,blk+2,2);const uint8_t*sc=blk+4,*q=blk+16;float d=fp16_to_float(d16),dm=fp16_to_float(dm16);float*y=h+i*Q;int is=0;for(int j=0;j<Q;j+=64){uint8_t s1,m1,s2,m2;hq4ksm(is+0,sc,&s1,&m1);hq4ksm(is+1,sc,&s2,&m2);float d1=d*(float)s1,dm1=dm*(float)m1,d2=d*(float)s2,dm2=dm*(float)m2;for(int l=0;l<32;l++)y[l]=d1*(float)(q[l]&0xF)-dm1;for(int l=0;l<32;l++)y[32+l]=d2*(float)(q[l]>>4)-dm2;y+=64;q+=32;is+=2;}}CUDA_CHECK(cudaMemcpy(b,h,n*sizeof(float),cudaMemcpyHostToDevice));free(h);return b;}
static float *upload_weight_dense(const hy3_weight*w){
    if(!w||!w->data||!w->t)return NULL; const hy3_tensor_info*t=w->t; uint64_t n=t->elements; if(n==0)return NULL;
    switch(t->ggml_type){case 0:return upload_f32(w->data,n);case 1:return upload_f16(w->data,n);case 8:return upload_q8_0(w->data,n);case 12:return upload_q4k_dense(w->data,n);default:{float*b;CUDA_CHECK(cudaMalloc(&b,n*sizeof(float)));CUDA_CHECK(cudaMemset(b,0,n*sizeof(float)));return b;}}
}
/* Convert an F32 device buffer to F16 (frees the F32), for cuBLAS FP16
 * tensor-core GEMV (cublasGemmEx). Halves weight bandwidth (GEMVs are
 * memory-bound at batch-1) and roughly doubles throughput on Blackwell. */
__global__ void f32_to_f16_conv_kernel(half*dst,const float*src,uint64_t n){
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=__float2half(src[i]);
}
static half *convert_to_half_free(float *f32, uint64_t n){
    if(!f32||n==0)return NULL;
    half*h; CUDA_CHECK(cudaMalloc(&h,n*sizeof(half)));
    f32_to_f16_conv_kernel<<<(int)((n+255u)/256u),256>>>(h,f32,n);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(f32); return h;
}
static half *upload_weight_half(const hy3_weight*w){
    float *f32=upload_weight_dense(w); if(!f32)return NULL;
    return convert_to_half_free(f32,w->t->elements);
}

#ifdef __cplusplus
extern "C" {
#endif

int hy3_gpu_init(hy3_model *m,int n_gpu_layers){
    gpu_ctx_t*ctx=(gpu_ctx_t*)calloc(1,sizeof(gpu_ctx_t)); if(!ctx)return -1;
    CUBLAS_CHECK(cublasCreate(&ctx->cublas)); CUDA_CHECK(cudaStreamCreate(&ctx->stream));
    CUDA_CHECK(cudaStreamCreate(&ctx->stream2));
    CUDA_CHECK(cudaEventCreateWithFlags(&ctx->ev_fork,cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ctx->ev_join,cudaEventDisableTiming));
    CUBLAS_CHECK(cublasSetStream(ctx->cublas,ctx->stream));
    CUBLAS_CHECK(cublasSetMathMode(ctx->cublas,CUBLAS_TF32_TENSOR_OP_MATH));
    ctx->d_token_embd=upload_weight_dense(&m->w.token_embd);
    ctx->d_output_norm=upload_weight_dense(&m->w.output_norm);
    ctx->d_output=upload_weight_half(&m->w.output);
    if(n_gpu_layers<=0)n_gpu_layers=81; if(n_gpu_layers>81)n_gpu_layers=81; m->gpu_layers=n_gpu_layers;
    for(int il=0;il<n_gpu_layers;il++){
        hy3_layer_weights*l=&m->w.layers[il];
        ctx->d_layer_attn_norm[il]=upload_weight_dense(&l->attn_norm);
        { float*q=upload_weight_dense(&l->attn_q),*k=upload_weight_dense(&l->attn_k),*v=upload_weight_dense(&l->attn_v);
          size_t qn=(size_t)HY3_N_HEAD*HY3_HEAD_DIM*HY3_N_EMBD,kn=(size_t)HY3_N_KV_HEAD*HY3_HEAD_DIM*HY3_N_EMBD;
          float *qkv; CUDA_CHECK(cudaMalloc(&qkv,(qn+2*kn)*sizeof(float)));
          CUDA_CHECK(cudaMemcpy(qkv,q,qn*sizeof(float),cudaMemcpyDeviceToDevice));
          CUDA_CHECK(cudaMemcpy(qkv+qn,k,kn*sizeof(float),cudaMemcpyDeviceToDevice));
          CUDA_CHECK(cudaMemcpy(qkv+qn+kn,v,kn*sizeof(float),cudaMemcpyDeviceToDevice));
          cudaFree(q);cudaFree(k);cudaFree(v);
          ctx->d_layer_attn_qkv[il]=convert_to_half_free(qkv,qn+2*kn); }
        ctx->d_layer_attn_output[il]=upload_weight_half(&l->attn_output);
        ctx->d_layer_attn_q_norm[il]=upload_weight_dense(&l->attn_q_norm);
        ctx->d_layer_attn_k_norm[il]=upload_weight_dense(&l->attn_k_norm);
        ctx->d_layer_ffn_norm[il]=upload_weight_dense(&l->ffn_norm);
        ctx->d_layer_eh_proj[il]=upload_weight_dense(&l->eh_proj);
        ctx->d_layer_enorm[il]=upload_weight_dense(&l->enorm);
        ctx->d_layer_hnorm[il]=upload_weight_dense(&l->hnorm);
        ctx->d_layer_final_norm[il]=upload_weight_dense(&l->final_norm);
        if(il<HY3_N_LAYER_DENSE){
            ctx->d_layer_dense_ffn_gate[il]=upload_weight_half(&l->ffn_gate);
            ctx->d_layer_dense_ffn_up[il]=upload_weight_half(&l->ffn_up);
            ctx->d_layer_dense_ffn_down[il]=upload_weight_half(&l->ffn_down);
        } else {
            ctx->d_layer_ffn_gate_inp[il]=upload_weight_dense(&l->ffn_gate_inp);
            ctx->d_layer_ffn_down_shexp[il]=upload_weight_half(&l->ffn_down_shexp);
            { float*g=upload_weight_dense(&l->ffn_gate_shexp),*u=upload_weight_dense(&l->ffn_up_shexp);
              size_t gm=(size_t)HY3_MOE_INTERMED*HY3_N_EMBD;
              float *gu; CUDA_CHECK(cudaMalloc(&gu,2*gm*sizeof(float)));
              CUDA_CHECK(cudaMemcpy(gu,g,gm*sizeof(float),cudaMemcpyDeviceToDevice));
              CUDA_CHECK(cudaMemcpy(gu+gm,u,gm*sizeof(float),cudaMemcpyDeviceToDevice));
              cudaFree(g);cudaFree(u);
              ctx->d_layer_ffn_gateup_shexp[il]=convert_to_half_free(gu,2*gm); }
            if(l->has_expert_bias){CUDA_CHECK(cudaMalloc(&ctx->d_layer_expert_bias[il],HY3_N_EXPERT*sizeof(float)));CUDA_CHECK(cudaMemcpy(ctx->d_layer_expert_bias[il],l->expert_bias,HY3_N_EXPERT*sizeof(float),cudaMemcpyHostToDevice));}
            for(int e=0;e<HY3_N_EXPERT;e++){upload_q4k_compressed(&ctx->d_q4k_gate_exps[il][e],l->ffn_gate_exps[e].data,l->ffn_gate_exps[e].t->elements);upload_q4k_compressed(&ctx->d_q4k_up_exps[il][e],l->ffn_up_exps[e].data,l->ffn_up_exps[e].t->elements);upload_q4k_compressed(&ctx->d_q4k_down_exps[il][e],l->ffn_down_exps[e].data,l->ffn_down_exps[e].t->elements);}
            { uint8_t*hg[192],*hu[192],*hd[192];
              for(int e=0;e<HY3_N_EXPERT;e++){hg[e]=ctx->d_q4k_gate_exps[il][e].data;hu[e]=ctx->d_q4k_up_exps[il][e].data;hd[e]=ctx->d_q4k_down_exps[il][e].data;}
              CUDA_CHECK(cudaMalloc(&ctx->d_gate_ptrs[il],HY3_N_EXPERT*sizeof(uint8_t*)));CUDA_CHECK(cudaMemcpy(ctx->d_gate_ptrs[il],hg,HY3_N_EXPERT*sizeof(uint8_t*),cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMalloc(&ctx->d_up_ptrs[il],HY3_N_EXPERT*sizeof(uint8_t*)));CUDA_CHECK(cudaMemcpy(ctx->d_up_ptrs[il],hu,HY3_N_EXPERT*sizeof(uint8_t*),cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMalloc(&ctx->d_down_ptrs[il],HY3_N_EXPERT*sizeof(uint8_t*)));CUDA_CHECK(cudaMemcpy(ctx->d_down_ptrs[il],hd,HY3_N_EXPERT*sizeof(uint8_t*),cudaMemcpyHostToDevice));}
        }
    }
    int maxct=8192; size_t islots=(size_t)maxct*HY3_N_LAYER;
    CUDA_CHECK(cudaMalloc(&ctx->d_k_cache,islots*KV_INT8_STRIDE));
    CUDA_CHECK(cudaMalloc(&ctx->d_v_cache,islots*KV_INT8_STRIDE));
    CUDA_CHECK(cudaMalloc(&ctx->d_embed,HY3_N_EMBD*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_scratch,(HY3_DENSE_INTERMED*2+HY3_N_EMBD*4+HY3_HEAD_DIM*256)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_scratch2,(HY3_N_EXPERT*4+HY3_MOE_INTERMED*8)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_logits,HY3_N_VOCAB*sizeof(float))); ctx->ctx_cap=(int)islots;
    CUDA_CHECK(cudaMalloc(&ctx->d_attn_partials,(size_t)HY3_N_HEAD*ATTN_SPLITS_MAX*130*sizeof(float)));
    ctx->attn_splits=16; /* initial, recaptured to optimal on first graph */
    CUDA_CHECK(cudaMalloc(&ctx->d_router_ids,HY3_N_EXPERT_USED*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_router_wts,HY3_N_EXPERT_USED*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_moe_gate_k,(size_t)HY3_N_EXPERT_USED*HY3_MOE_INTERMED*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_moe_up_k,(size_t)HY3_N_EXPERT_USED*HY3_MOE_INTERMED*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_moe_mid_k,(size_t)HY3_N_EXPERT_USED*HY3_MOE_INTERMED*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_moe_down_k,(size_t)HY3_N_EXPERT_USED*HY3_N_EMBD*sizeof(float)));
    { int mx=HY3_DENSE_INTERMED>HY3_N_EMBD?HY3_DENSE_INTERMED:HY3_N_EMBD; CUDA_CHECK(cudaMalloc(&ctx->d_xf16,(size_t)mx*sizeof(half))); }
    CUDA_CHECK(cudaMalloc(&ctx->d_pos,sizeof(int)));
    CUDA_CHECK(cudaHostAlloc((void**)&ctx->h_pos,sizeof(int),cudaHostAllocDefault));
    m->gpu_ctx=ctx; fprintf(stderr,"hy3_gpu: initialized (%d layers, INT8 KV, online attn)\n",n_gpu_layers);
    return 0;
}

void hy3_gpu_free(hy3_model *m){
    gpu_ctx_t*ctx=(gpu_ctx_t*)m->gpu_ctx; if(!ctx)return;
    #define GF(p) do{if(p){cudaFree(p);p=NULL;}}while(0)
    GF(ctx->d_token_embd);GF(ctx->d_output_norm);GF(ctx->d_output);
    for(int il=0;il<81;il++){GF(ctx->d_layer_attn_qkv[il]);GF(ctx->d_layer_attn_output[il]);GF(ctx->d_layer_attn_q_norm[il]);GF(ctx->d_layer_attn_k_norm[il]);GF(ctx->d_layer_attn_norm[il]);GF(ctx->d_layer_ffn_norm[il]);GF(ctx->d_layer_ffn_gate_inp[il]);GF(ctx->d_layer_ffn_down_shexp[il]);GF(ctx->d_layer_ffn_gateup_shexp[il]);GF(ctx->d_layer_eh_proj[il]);GF(ctx->d_layer_enorm[il]);GF(ctx->d_layer_hnorm[il]);GF(ctx->d_layer_final_norm[il]);GF(ctx->d_layer_expert_bias[il]);GF(ctx->d_gate_ptrs[il]);GF(ctx->d_up_ptrs[il]);GF(ctx->d_down_ptrs[il]);GF(ctx->d_layer_dense_ffn_gate[il]);GF(ctx->d_layer_dense_ffn_up[il]);GF(ctx->d_layer_dense_ffn_down[il]);for(int e=0;e<HY3_N_EXPERT;e++){GF(ctx->d_q4k_gate_exps[il][e].data);GF(ctx->d_q4k_up_exps[il][e].data);GF(ctx->d_q4k_down_exps[il][e].data);}}
    GF(ctx->d_k_cache);GF(ctx->d_v_cache);GF(ctx->d_embed);GF(ctx->d_scratch);GF(ctx->d_scratch2);GF(ctx->d_attn_partials);GF(ctx->d_logits);GF(ctx->d_router_ids);GF(ctx->d_router_wts);GF(ctx->d_moe_gate_k);GF(ctx->d_moe_up_k);GF(ctx->d_moe_mid_k);GF(ctx->d_moe_down_k);GF(ctx->d_xf16);GF(ctx->d_pos);if(ctx->h_pos){cudaFreeHost(ctx->h_pos);ctx->h_pos=NULL;}if(ctx->graph_ready)cudaGraphExecDestroy(ctx->graph_exec);
    if(ctx->ev_fork)cudaEventDestroy(ctx->ev_fork); if(ctx->ev_join)cudaEventDestroy(ctx->ev_join);
    if(ctx->stream2)cudaStreamDestroy(ctx->stream2);
    if(ctx->stream)cudaStreamDestroy(ctx->stream); if(ctx->cublas)cublasDestroy(ctx->cublas);
    free(ctx); m->gpu_ctx=NULL;
}

/* FP16 GEMV via cuBLAS tensor cores: convert activation to F16, then
 * W(F16) x(F16) -> dst(F32), F32 accumulate. ~2x plain TF32 SGEMM for
 * these skinny batch-1 GEMVs (halves weight bandwidth; they're mem-bound). */
__global__ void f2h_vec_kernel(half*d,const float*s,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=__float2half(s[i]);}
static void gpu_mul_mat(gpu_ctx_t*ctx,const float*x,float*dst,const half*w,int m,int n){
    f2h_vec_kernel<<<(n+255)/256,256,0,ctx->stream>>>(ctx->d_xf16,x,n);
    float a=1.f,b=0.f;
    cublasGemmEx(ctx->cublas,CUBLAS_OP_T,CUBLAS_OP_N,m,1,n,&a,
                 w,CUDA_R_16F,n, ctx->d_xf16,CUDA_R_16F,n, &b, dst,CUDA_R_32F,m,
                 CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
}
/* Full-FP32 GEMV (no FP16 rounding of weights or activations). Used for the
 * MoE router projection, whose top-k argmax over near-equal expert scores is
 * the model's most precision-sensitive step: FP16 rounding there flips routing
 * decisions and compounds across depth. dst[j] = dot(row j of w, x). */
static void gpu_mul_mat_f32(gpu_ctx_t*ctx,const float*x,float*dst,const float*w,int m,int n){
    float a=1.f,b=0.f;
    cublasSgemv(ctx->cublas,CUBLAS_OP_T,n,m,&a,w,n,x,1,&b,dst,1);
}
static void gpu_rms_norm(gpu_ctx_t*ctx,float*o,const float*x,const float*w,int n){
    rms_norm_kernel<<<1,BLOCK_DIM,BLOCK_DIM*sizeof(float),ctx->stream>>>(o,x,w,n);
}
static void gpu_ensure_kv_capacity(gpu_ctx_t*ctx,int ns){
    if(ns<=ctx->ctx_cap)return; size_t nc=(size_t)ns+(size_t)8192*HY3_N_LAYER;
    uint8_t*nk,*nv; CUDA_CHECK(cudaMalloc(&nk,nc*KV_INT8_STRIDE));CUDA_CHECK(cudaMalloc(&nv,nc*KV_INT8_STRIDE));
    if(ctx->d_k_cache){CUDA_CHECK(cudaMemcpy(nk,ctx->d_k_cache,(size_t)ctx->ctx_cap*KV_INT8_STRIDE,cudaMemcpyDeviceToDevice));CUDA_CHECK(cudaMemcpy(nv,ctx->d_v_cache,(size_t)ctx->ctx_cap*KV_INT8_STRIDE,cudaMemcpyDeviceToDevice));}
    cudaFree(ctx->d_k_cache);cudaFree(ctx->d_v_cache);ctx->d_k_cache=nk;ctx->d_v_cache=nv;ctx->ctx_cap=(int)nc;
}

/* Encode one GPU-resident layer using ctx->d_pos for all position-dependent
 * kernels (rope, KV append, attention). No host<->device mirror, no capacity
 * management -- both handled by the caller outside any capture region. This
 * is what the CUDA graph captures, and what the full-GPU warmup runs eagerly. */
static void encode_layer_gpu(gpu_ctx_t*ctx,hy3_model*m,int il){
    int kvd=HY3_N_KV_HEAD*HY3_HEAD_DIM,qs=HY3_N_HEAD*HY3_HEAD_DIM;
    float*x=ctx->d_embed,*s=ctx->d_scratch,*s2=ctx->d_scratch+HY3_N_EMBD*2,*ao=ctx->d_scratch+HY3_N_EMBD*4;
    gpu_rms_norm(ctx,s,x,ctx->d_layer_attn_norm[il],HY3_N_EMBD);
    float*qg=s2,*kg=s2+qs,*vg=s2+qs+kvd;
    gpu_mul_mat(ctx,s,s2,ctx->d_layer_attn_qkv[il],qs+2*kvd,HY3_N_EMBD);
    qk_norm_rope_fused_kernel<<<HY3_N_HEAD+HY3_N_KV_HEAD,BLOCK_DIM,BLOCK_DIM*sizeof(float),ctx->stream>>>(qg,kg,ctx->d_layer_attn_q_norm[il],ctx->d_layer_attn_k_norm[il],HY3_HEAD_DIM,HY3_N_HEAD,HY3_N_KV_HEAD,ctx->d_pos);
    kv_quantize_int8_kernel<<<HY3_N_KV_HEAD,32,0,ctx->stream>>>(ctx->d_k_cache,ctx->d_v_cache,kg,vg,ctx->d_pos,il,HY3_N_LAYER);
    attention_split_kv_kernel<<<HY3_N_HEAD*ctx->attn_splits,32,0,ctx->stream>>>(qg,ctx->d_k_cache,ctx->d_v_cache,HY3_N_HEAD,HY3_N_KV_HEAD,HY3_HEAD_DIM,ctx->d_pos,HY3_N_HEAD/HY3_N_KV_HEAD,il,HY3_N_LAYER,ctx->attn_splits,ctx->d_attn_partials);
    attention_reduce_kernel<<<HY3_N_HEAD,32,0,ctx->stream>>>(ao,ctx->d_attn_partials,HY3_N_HEAD,HY3_HEAD_DIM,ctx->attn_splits);
    gpu_mul_mat(ctx,ao,s,ctx->d_layer_attn_output[il],HY3_N_EMBD,qs);
    /* fused: x += o-proj (s); s = rmsnorm(x)*ffn_norm  (one kernel, no gap) */
    add_rmsnorm_kernel<<<1,BLOCK_DIM,BLOCK_DIM*sizeof(float),ctx->stream>>>(s,x,s,ctx->d_layer_ffn_norm[il],HY3_N_EMBD);
    if(il<HY3_N_LAYER_DENSE){
        float*g=s2,*u=s2+HY3_DENSE_INTERMED;
        gpu_mul_mat(ctx,s,g,ctx->d_layer_dense_ffn_gate[il],HY3_DENSE_INTERMED,HY3_N_EMBD);
        gpu_mul_mat(ctx,s,u,ctx->d_layer_dense_ffn_up[il],HY3_DENSE_INTERMED,HY3_N_EMBD);
        silu_mul_kernel<<<(HY3_DENSE_INTERMED+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,ctx->stream>>>(s2,g,u,HY3_DENSE_INTERMED);
        gpu_mul_mat(ctx,s2,s,ctx->d_layer_dense_ffn_down[il],HY3_N_EMBD,HY3_DENSE_INTERMED);
        add_kernel<<<(HY3_N_EMBD+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,ctx->stream>>>(x,x,s,HY3_N_EMBD);
    } else {
        int nu=m->n_expert_used;
        gpu_mul_mat_f32(ctx,s,ctx->d_scratch2,ctx->d_layer_ffn_gate_inp[il],HY3_N_EXPERT,HY3_N_EMBD);
        router_topk_kernel<<<1,256,0,ctx->stream>>>(ctx->d_scratch2,ctx->d_layer_expert_bias[il],ctx->d_router_ids,ctx->d_router_wts,HY3_N_EXPERT,(uint32_t)nu,m->w.layers[il].has_expert_bias?1u:0u,2.826f);
        /* Fork: the routed-expert Q4_K matmuls (which dominate the layer) run on
         * stream2 while the small shared-expert FP16 GEMVs run on the main
         * stream. Both read s read-only; shared-down writes ao (not s) to avoid a
         * WAR hazard with the routed gate_up read of s. Joined before combine. */
        CUDA_CHECK(cudaEventRecord(ctx->ev_fork,ctx->stream));
        CUDA_CHECK(cudaStreamWaitEvent(ctx->stream2,ctx->ev_fork,0));
        int gg=(HY3_MOE_INTERMED+MOE_WPB-1)/MOE_WPB;
        moe_matmul_q4k_id_gate_up_kernel<<<dim3(gg,nu),MOE_WPB*32,0,ctx->stream2>>>(ctx->d_moe_gate_k,ctx->d_moe_up_k,(const uint8_t*const*)ctx->d_gate_ptrs[il],(const uint8_t*const*)ctx->d_up_ptrs[il],s,ctx->d_router_ids,HY3_N_EMBD,HY3_MOE_INTERMED,0);
        silu_mul_kernel<<<(nu*HY3_MOE_INTERMED+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,ctx->stream2>>>(ctx->d_moe_mid_k,ctx->d_moe_gate_k,ctx->d_moe_up_k,nu*HY3_MOE_INTERMED);
        int gd=(HY3_N_EMBD+MOE_WPB-1)/MOE_WPB;
        moe_matmul_q4k_id_kernel<<<dim3(gd,nu),MOE_WPB*32,0,ctx->stream2>>>(ctx->d_moe_down_k,(const uint8_t*const*)ctx->d_down_ptrs[il],ctx->d_moe_mid_k,ctx->d_router_ids,HY3_MOE_INTERMED,HY3_N_EMBD,1);
        CUDA_CHECK(cudaEventRecord(ctx->ev_join,ctx->stream2));
        float*sg=(float*)ctx->d_scratch2+HY3_N_EXPERT*2,*su=sg+HY3_MOE_INTERMED;
        gpu_mul_mat(ctx,s,sg,ctx->d_layer_ffn_gateup_shexp[il],2*HY3_MOE_INTERMED,HY3_N_EMBD);
        silu_mul_kernel<<<(HY3_MOE_INTERMED+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,ctx->stream>>>(s2,sg,su,HY3_MOE_INTERMED);
        gpu_mul_mat(ctx,s2,ao,ctx->d_layer_ffn_down_shexp[il],HY3_N_EMBD,HY3_MOE_INTERMED);
        add_kernel<<<(HY3_N_EMBD+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,ctx->stream>>>(x,x,ao,HY3_N_EMBD);
        CUDA_CHECK(cudaStreamWaitEvent(ctx->stream,ctx->ev_join,0));
        moe_combine_id_kernel<<<(HY3_N_EMBD+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,ctx->stream>>>(x,ctx->d_moe_down_k,ctx->d_router_wts,HY3_N_EMBD,(uint32_t)nu);
    }
}

/* Pick a split count that keeps per-block work ≈32 tokens for best latency. */
static int gpu_optimal_splits(int ntok){
    int s=ntok/32; if(s<16)s=16; if(s>ATTN_SPLITS_MAX)s=ATTN_SPLITS_MAX; return s;
}

/* Capture (once) and replay the full 80-layer forward as a CUDA graph.
 * Collapses ~1700 host-issued kernel/cuBLAS launches per token into one
 * graph launch, removing the launch-latency that dominates batch-1 decode. */
static void gpu_decode_graph(gpu_ctx_t*ctx,hy3_model*m){
    int ntok=*ctx->h_pos+1, opt=gpu_optimal_splits(ntok);
    if(opt!=ctx->attn_splits){
        if(ctx->graph_ready){cudaGraphExecDestroy(ctx->graph_exec);ctx->graph_ready=0;}
        ctx->attn_splits=opt;
    }
    if(!ctx->graph_ready||ctx->graph_kc!=ctx->d_k_cache||ctx->graph_vc!=ctx->d_v_cache){
        if(ctx->graph_ready){cudaGraphExecDestroy(ctx->graph_exec);ctx->graph_ready=0;}
        cudaGraph_t g;
        CUDA_CHECK(cudaStreamBeginCapture(ctx->stream,cudaStreamCaptureModeThreadLocal));
        for(int il=0;il<HY3_N_LAYER;il++) encode_layer_gpu(ctx,m,il);
        CUDA_CHECK(cudaStreamEndCapture(ctx->stream,&g));
        CUDA_CHECK(cudaGraphInstantiate(&ctx->graph_exec,g,0));
        cudaGraphDestroy(g);
        ctx->graph_ready=1; ctx->graph_kc=ctx->d_k_cache; ctx->graph_vc=ctx->d_v_cache;
        if(getenv("HY3_TIMING")) fprintf(stderr,"hy3_gpu: captured decode graph (splits=%d)\n",ctx->attn_splits);
    }
    if(getenv("HY3_GRAPH_BENCH")){
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        for(int w=0;w<3;w++) cudaGraphLaunch(ctx->graph_exec,ctx->stream);
        cudaStreamSynchronize(ctx->stream);
        cudaEventRecord(a,ctx->stream);
        for(int r=0;r<50;r++) cudaGraphLaunch(ctx->graph_exec,ctx->stream);
        cudaEventRecord(b,ctx->stream); cudaStreamSynchronize(ctx->stream);
        float ms=0; cudaEventElapsedTime(&ms,a,b);
        fprintf(stderr,"hy3_gpu: graph replay %.3f ms/token (%.1f tok/s pure GPU)\n",ms/50.0,1000.0/(ms/50.0));
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    CUDA_CHECK(cudaGraphLaunch(ctx->graph_exec,ctx->stream));
}

static double now_sec_cpu(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (double)ts.tv_sec+(double)ts.tv_nsec*1e-9; }
int hy3_eval_gpu(hy3_model *m,const hy3_tokens *tokens,float *logits,int *pos){
    gpu_ctx_t*ctx=(gpu_ctx_t*)m->gpu_ctx; if(!ctx)return -1;
    int kvs=HY3_N_KV_HEAD*HY3_HEAD_DIM,ng=m->gpu_layers;
    if(ng<=0)ng=HY3_N_LAYER; if(ng>HY3_N_LAYER)ng=HY3_N_LAYER;
    int fg=(ng>=HY3_N_LAYER),kvd=HY3_N_KV_HEAD*HY3_HEAD_DIM;
    /* ---- per-phase timing (HY3_TIMING only) ---- */
    static double tp_graph=0,tp_final=0,tp_dtoh=0; static long tp_n=0;
    double tpg0=0,tpg1=0,tpf0=0,tpf1=0,tpd0=0;
    int do_tp = getenv("HY3_TIMING")!=NULL;
    for(int i=0;i<tokens->len;i++){
        int token=tokens->v[i],cb=m->cache_len,tp=cb/HY3_N_LAYER;
        int grid=(HY3_N_EMBD+BLOCK_DIM-1)/BLOCK_DIM;
        embed_lookup_kernel<<<grid,BLOCK_DIM,0,ctx->stream>>>(ctx->d_embed,ctx->d_token_embd,token,HY3_N_EMBD);
        /* device-side token position for graph replay */
        *ctx->h_pos=tp;
        CUDA_CHECK(cudaMemcpyAsync(ctx->d_pos,ctx->h_pos,sizeof(int),cudaMemcpyHostToDevice,ctx->stream));
        if(do_tp){ tpg0=now_sec_cpu(); tpg1=tpg0; }

        if(fg){
            /* Full-GPU: ensure KV capacity (host-side, may realloc), then
             * warmup one token eagerly (so cuBLAS allocs its workspace before
             * capture) and replay a CUDA graph thereafter. */
            gpu_ensure_kv_capacity(ctx,cb+HY3_N_LAYER);
            if(!ctx->graph_warmed){ for(int il=0;il<HY3_N_LAYER;il++) encode_layer_gpu(ctx,m,il); ctx->graph_warmed=1; }
            else gpu_decode_graph(ctx,m);
            if(do_tp){ cudaStreamSynchronize(ctx->stream); tpg1=now_sec_cpu(); }
            m->cache_len=cb+HY3_N_LAYER;
            continue;
        }

        /* CPU-side KV cache (only used for partial-GPU / CPU-fallback path) */
        size_t nd=(size_t)(cb+HY3_N_LAYER)*kvs;
        if(nd>(size_t)m->ctx_size*kvs){size_t nc=cb+HY3_N_LAYER+1024;m->cache_k=(float*)realloc(m->cache_k,nc*kvs*sizeof(float));m->cache_v=(float*)realloc(m->cache_v,nc*kvs*sizeof(float));m->ctx_size=(int)nc;}
        if(!m->cache_k){m->ctx_size=4096;m->cache_k=(float*)calloc((size_t)m->ctx_size*kvs,sizeof(float));m->cache_v=(float*)calloc((size_t)m->ctx_size*kvs,sizeof(float));}

        /* Partial-GPU: eager, with per-layer KV mirror for the CPU tail. */
        for(int il=0;il<ng;il++){
            if(!ctx->d_layer_attn_qkv[il])break;
            int kvl=cb+il,qs=HY3_N_HEAD*HY3_HEAD_DIM;
            float*x=ctx->d_embed,*s=ctx->d_scratch,*s2=ctx->d_scratch+HY3_N_EMBD*2,*ao=ctx->d_scratch+HY3_N_EMBD*4;
            gpu_rms_norm(ctx,s,x,ctx->d_layer_attn_norm[il],HY3_N_EMBD);
            float*qg=s2,*kg=s2+qs,*vg=s2+qs+kvd;
            gpu_mul_mat(ctx,s,s2,ctx->d_layer_attn_qkv[il],qs+2*kvd,HY3_N_EMBD);
            qk_norm_rope_fused_kernel<<<HY3_N_HEAD+HY3_N_KV_HEAD,BLOCK_DIM,BLOCK_DIM*sizeof(float),ctx->stream>>>(qg,kg,ctx->d_layer_attn_q_norm[il],ctx->d_layer_attn_k_norm[il],HY3_HEAD_DIM,HY3_N_HEAD,HY3_N_KV_HEAD,ctx->d_pos);
            gpu_ensure_kv_capacity(ctx,kvl+1);
            kv_quantize_int8_kernel<<<HY3_N_KV_HEAD,32,0,ctx->stream>>>(ctx->d_k_cache,ctx->d_v_cache,kg,vg,ctx->d_pos,il,HY3_N_LAYER);
            {float*hk=m->cache_k+(size_t)kvl*kvd,*hv=m->cache_v+(size_t)kvl*kvd;CUDA_CHECK(cudaMemcpyAsync(hk,kg,kvd*sizeof(float),cudaMemcpyDeviceToHost,ctx->stream));CUDA_CHECK(cudaMemcpyAsync(hv,vg,kvd*sizeof(float),cudaMemcpyDeviceToHost,ctx->stream));}
            int ns=gpu_optimal_splits(tp+1);
            attention_split_kv_kernel<<<HY3_N_HEAD*ns,32,0,ctx->stream>>>(qg,ctx->d_k_cache,ctx->d_v_cache,HY3_N_HEAD,HY3_N_KV_HEAD,HY3_HEAD_DIM,ctx->d_pos,HY3_N_HEAD/HY3_N_KV_HEAD,il,HY3_N_LAYER,ns,ctx->d_attn_partials);
            attention_reduce_kernel<<<HY3_N_HEAD,32,0,ctx->stream>>>(ao,ctx->d_attn_partials,HY3_N_HEAD,HY3_HEAD_DIM,ns);
            gpu_mul_mat(ctx,ao,s,ctx->d_layer_attn_output[il],HY3_N_EMBD,qs);
            add_kernel<<<(HY3_N_EMBD+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,ctx->stream>>>(x,x,s,HY3_N_EMBD);
            if(il<HY3_N_LAYER_DENSE){
                gpu_rms_norm(ctx,s,x,ctx->d_layer_ffn_norm[il],HY3_N_EMBD);
                float*g=s2,*u=s2+HY3_DENSE_INTERMED;
                gpu_mul_mat(ctx,s,g,ctx->d_layer_dense_ffn_gate[il],HY3_DENSE_INTERMED,HY3_N_EMBD);
                gpu_mul_mat(ctx,s,u,ctx->d_layer_dense_ffn_up[il],HY3_DENSE_INTERMED,HY3_N_EMBD);
                silu_mul_kernel<<<(HY3_DENSE_INTERMED+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,ctx->stream>>>(s2,g,u,HY3_DENSE_INTERMED);
                gpu_mul_mat(ctx,s2,s,ctx->d_layer_dense_ffn_down[il],HY3_N_EMBD,HY3_DENSE_INTERMED);
                add_kernel<<<(HY3_N_EMBD+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,ctx->stream>>>(x,x,s,HY3_N_EMBD);
            } else {
                int nu=m->n_expert_used;
                gpu_rms_norm(ctx,s,x,ctx->d_layer_ffn_norm[il],HY3_N_EMBD);
                gpu_mul_mat_f32(ctx,s,ctx->d_scratch2,ctx->d_layer_ffn_gate_inp[il],HY3_N_EXPERT,HY3_N_EMBD);
                router_topk_kernel<<<1,256,0,ctx->stream>>>(ctx->d_scratch2,ctx->d_layer_expert_bias[il],ctx->d_router_ids,ctx->d_router_wts,HY3_N_EXPERT,(uint32_t)nu,m->w.layers[il].has_expert_bias?1u:0u,2.826f);
                float*sg=(float*)ctx->d_scratch2+HY3_N_EXPERT*2,*su=sg+HY3_MOE_INTERMED;
                gpu_mul_mat(ctx,s,sg,ctx->d_layer_ffn_gateup_shexp[il],2*HY3_MOE_INTERMED,HY3_N_EMBD);
                silu_mul_kernel<<<(HY3_MOE_INTERMED+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,ctx->stream>>>(s2,sg,su,HY3_MOE_INTERMED);
                int gg=(HY3_MOE_INTERMED+MOE_WPB-1)/MOE_WPB;
                moe_matmul_q4k_id_gate_up_kernel<<<dim3(gg,nu),MOE_WPB*32,0,ctx->stream>>>(ctx->d_moe_gate_k,ctx->d_moe_up_k,(const uint8_t*const*)ctx->d_gate_ptrs[il],(const uint8_t*const*)ctx->d_up_ptrs[il],s,ctx->d_router_ids,HY3_N_EMBD,HY3_MOE_INTERMED,0);
                silu_mul_kernel<<<(nu*HY3_MOE_INTERMED+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,ctx->stream>>>(ctx->d_moe_mid_k,ctx->d_moe_gate_k,ctx->d_moe_up_k,nu*HY3_MOE_INTERMED);
                int gd=(HY3_N_EMBD+MOE_WPB-1)/MOE_WPB;
                moe_matmul_q4k_id_kernel<<<dim3(gd,nu),MOE_WPB*32,0,ctx->stream>>>(ctx->d_moe_down_k,(const uint8_t*const*)ctx->d_down_ptrs[il],ctx->d_moe_mid_k,ctx->d_router_ids,HY3_MOE_INTERMED,HY3_N_EMBD,1);
                gpu_mul_mat(ctx,s2,s,ctx->d_layer_ffn_down_shexp[il],HY3_N_EMBD,HY3_MOE_INTERMED);
                add_kernel<<<(HY3_N_EMBD+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,ctx->stream>>>(x,x,s,HY3_N_EMBD);
                moe_combine_id_kernel<<<(HY3_N_EMBD+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM,0,ctx->stream>>>(x,ctx->d_moe_down_k,ctx->d_router_wts,HY3_N_EMBD,(uint32_t)nu);
            }
        }
        m->cache_len=cb+ng;
        CUDA_CHECK(cudaMemcpyAsync(m->embed,ctx->d_embed,HY3_N_EMBD*sizeof(float),cudaMemcpyDeviceToHost,ctx->stream));CUDA_CHECK(cudaStreamSynchronize(ctx->stream));for(int il=ng;il<HY3_N_LAYER;il++){int p=cb/HY3_N_LAYER;if(il<HY3_N_LAYER_DENSE)forward_layer_dense(m,il,p);else forward_layer_moe(m,il,p);}CUDA_CHECK(cudaMemcpyAsync(ctx->d_embed,m->embed,HY3_N_EMBD*sizeof(float),cudaMemcpyHostToDevice,ctx->stream));
        m->cache_len=cb+HY3_N_LAYER;
    }
    if(do_tp) tpf0=now_sec_cpu();
    if(ctx->d_output_norm)gpu_rms_norm(ctx,ctx->d_embed,ctx->d_embed,ctx->d_output_norm,HY3_N_EMBD);
    gpu_mul_mat(ctx,ctx->d_embed,ctx->d_logits,ctx->d_output,HY3_N_VOCAB,HY3_N_EMBD);
    if(do_tp){ cudaStreamSynchronize(ctx->stream); tpf1=now_sec_cpu(); tpd0=now_sec_cpu(); }
    CUDA_CHECK(cudaMemcpyAsync(logits,ctx->d_logits,HY3_N_VOCAB*sizeof(float),cudaMemcpyDeviceToHost,ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream)); *pos=m->cache_len;
    if(do_tp){ double tpd1=now_sec_cpu(); tp_graph+=tpg1-tpg0; tp_final+=tpf1-tpf0; tp_dtoh+=tpd1-tpd0; tp_n++;
        if(tp_n%64==0) fprintf(stderr,"hy3_gpu: phases | graph %.2f final(matmul+norm) %.2f dtoh(logits) %.2f ms (avg %ld tok)\n",
            1000*tp_graph/tp_n,1000*tp_final/tp_n,1000*tp_dtoh/tp_n,tp_n); }
    return 0;
}
#ifdef __cplusplus
}
#endif
