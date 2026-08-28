// Microbench: serial q8 flash vs tiled FA2 q8 (qwen2.5-0.5b shape)
// Shape: n_heads=14, n_kv_heads=2, head_dim=128
// Compares time_current (k_flash_gqa_q8_0) vs time_fa2 (tiled FA2 grouped GQA + split-K)
// Build: nvcc -O3 -arch=native --resource-usage -o build/micro_fa2_q8 tools/micro_fa2_q8.cu -L$HOME/mmcuda/lib -lcudart
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

#ifndef BLOCK_Q8_0_DEFINED
#define BLOCK_Q8_0_DEFINED
struct BlockQ8_0 {
    half d;
    int8_t qs[32];
};
#endif

__device__ __forceinline__ float warp_sum(float v){
#pragma unroll
    for(int off=16; off>0; off/=2) v+= __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// ---------------- current kernel copy (serial per-warp) ----------------
__global__ void k_flash_gqa_q8_0(
    const float     *__restrict__ q,
    const BlockQ8_0 *__restrict__ Kc_q8,
    const BlockQ8_0 *__restrict__ Vc_q8,
    float           *__restrict__ out,
    const int       *__restrict__ d_pos,
    int n_heads, int n_kv_heads, int head_dim, int max_ctx,
    float scale, int window) {
    const int pos = *d_pos;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int lane = threadIdx.x;
    const int elems = head_dim / 32;
    const int kvh = h / (n_heads / n_kv_heads);
    const int blocks_per_head = head_dim / 32;
    const int blocks_per_slot = n_kv_heads * blocks_per_head;
    const float *qh = q + (long)h * head_dim + lane * elems;
    int t0 = 0;
    if (window > 0 && pos >= window) t0 = pos - window + 1;
    float qreg[16];
#pragma unroll
    for (int i = 0; i < 16; i++) qreg[i] = (i < elems) ? qh[i] : 0.0f;
    float m_prev = -1e30f, l_prev = 0.0f;
    float oreg[16] = {0.0f};
    const int block_in_head = (lane * elems) / 32;
    const int elem_sub_idx = (lane * elems) % 32;
    const int block_idx = kvh * blocks_per_head + block_in_head;
    const int block_byte_off = block_idx * 34;
    const int wsc = block_byte_off >> 2;
    const int sh_d = block_byte_off & 2;
    const int a0 = (block_byte_off + 2) >> 2;
    const int sh_qs = (block_byte_off + 2) & 2;
    const int k_elem_word = elem_sub_idx >> 2;
    const long stride = (long)blocks_per_slot * 34;
    const char *k_ptr = (const char *)Kc_q8 + (long)t0 * stride;
    const char *v_ptr = (const char *)Vc_q8 + (long)t0 * stride;
    for (int t = t0; t <= pos; t++) {
        const uint32_t *k_slot_u32 = (const uint32_t *)k_ptr;
        const uint32_t *v_slot_u32 = (const uint32_t *)v_ptr;
        k_ptr += stride; v_ptr += stride;
        const uint32_t dw_k = k_slot_u32[wsc];
        const unsigned short d16_k = (unsigned short)(sh_d ? (dw_k >> 16) : (dw_k & 0xFFFFu));
        const float dk = __half2float(__ushort_as_half(d16_k));
        const uint32_t lo_k = k_slot_u32[a0 + k_elem_word];
        const uint32_t vv_k = sh_qs ? __byte_perm(lo_k, k_slot_u32[a0 + k_elem_word + 1], 0x5432) : lo_k;
        float score = 0.0f;
        if (elems == 4) {
            float k0 = (float)((int8_t)(vv_k      ));
            float k1 = (float)((int8_t)(vv_k >>  8));
            float k2 = (float)((int8_t)(vv_k >> 16));
            float k3 = (float)((int8_t)(vv_k >> 24));
            score = (qreg[0]*k0 + qreg[1]*k1 + qreg[2]*k2 + qreg[3]*k3) * dk;
        } else if (elems == 2) {
            int shift = (elem_sub_idx & 2) ? 16 : 0;
            float k0 = (float)((int8_t)(vv_k >> shift));
            float k1 = (float)((int8_t)(vv_k >> (shift + 8)));
            score = (qreg[0]*k0 + qreg[1]*k1) * dk;
        } else {
#pragma unroll
            for (int i = 0; i < elems; i++) {
                int shift = ((elem_sub_idx + i) & 3) * 8;
                float k_val = (float)((int8_t)(vv_k >> shift));
                score += qreg[i] * (k_val * dk);
            }
        }
        score = warp_sum(score);
        score = __shfl_sync(0xffffffff, score, 0) * scale;
        float m_curr = fmaxf(m_prev, score);
        float p = expf(score - m_curr);
        float alpha = expf(m_prev - m_curr);
        float l_curr = l_prev * alpha + p;
        const uint32_t dw_v = v_slot_u32[wsc];
        const unsigned short d16_v = (unsigned short)(sh_d ? (dw_v >> 16) : (dw_v & 0xFFFFu));
        const float dv = __half2float(__ushort_as_half(d16_v));
        const uint32_t lo_v = v_slot_u32[a0 + k_elem_word];
        const uint32_t vv_v = sh_qs ? __byte_perm(lo_v, v_slot_u32[a0 + k_elem_word + 1], 0x5432) : lo_v;
        const float pdv = p * dv;
        if (elems == 4) {
            float v0 = (float)((int8_t)(vv_v      ));
            float v1 = (float)((int8_t)(vv_v >>  8));
            float v2 = (float)((int8_t)(vv_v >> 16));
            float v3 = (float)((int8_t)(vv_v >> 24));
            oreg[0] = oreg[0] * alpha + pdv * v0;
            oreg[1] = oreg[1] * alpha + pdv * v1;
            oreg[2] = oreg[2] * alpha + pdv * v2;
            oreg[3] = oreg[3] * alpha + pdv * v3;
        } else if (elems == 2) {
            int shift = (elem_sub_idx & 2) ? 16 : 0;
            float v0 = (float)((int8_t)(vv_v >> shift));
            float v1 = (float)((int8_t)(vv_v >> (shift + 8)));
            oreg[0] = oreg[0] * alpha + pdv * v0;
            oreg[1] = oreg[1] * alpha + pdv * v1;
        } else {
#pragma unroll
            for (int i = 0; i < elems; i++) {
                int shift = ((elem_sub_idx + i) & 3) * 8;
                float v_val = (float)((int8_t)(vv_v >> shift));
                oreg[i] = oreg[i] * alpha + pdv * v_val;
            }
        }
        m_prev = m_curr; l_prev = l_curr;
    }
    float *outh = out + (long)h * head_dim + lane * elems;
#pragma unroll
    for (int i = 0; i < 16; i++) if (i < elems) outh[i] = oreg[i] / l_prev;
}

// ---------------- FP32 reference kernel (for correctness) ----------------
__global__ void k_flash_gqa_fp32(const float *__restrict__ q, const float *__restrict__ Kc, const float *__restrict__ Vc, float *__restrict__ out, const int *__restrict__ d_pos, int n_heads, int n_kv_heads, int head_dim, int max_ctx, float scale, int window){
    const int pos=*d_pos; int h=blockIdx.x; if(h>=n_heads) return;
    int lane=threadIdx.x; int kvh=h/(n_heads/n_kv_heads); int elems=head_dim/32;
    const float *qh=q+(long)h*head_dim+lane*elems;
    int t0=0; if(window>0 && pos>=window) t0=pos-window+1;
    float qreg[16];
#pragma unroll
for(int i=0;i<16;i++) qreg[i]=(i<elems)?qh[i]:0;
    float m_prev=-1e30f,l_prev=0; float oreg[16]={0};
    for(int t=t0; t<=pos; t++){
        long off=((long)t*n_kv_heads+kvh)*head_dim+lane*elems;
        const float*kp=Kc+off; const float*vp=Vc+off;
        float score=0;
#pragma unroll
for(int i=0;i<16;i++) if(i<elems) score+=qreg[i]*kp[i];
        score=warp_sum(score); score=__shfl_sync(0xffffffff,score,0)*scale;
        float m_new=fmaxf(m_prev,score); float ex=expf(score-m_new); float alpha=expf(m_prev-m_new);
        l_prev=l_prev*alpha+ex;
#pragma unroll
for(int i=0;i<16;i++) if(i<elems) oreg[i]=oreg[i]*alpha+ex*vp[i];
        m_prev=m_new;
    }
    float *oh=out+(long)h*head_dim+lane*elems;
#pragma unroll
for(int i=0;i<16;i++) if(i<elems) oh[i]=oreg[i]/(l_prev+1e-8f);
}

// scatter q8
__global__ void k_kv_scatter_q8_0(const float *__restrict__ kst, const float *__restrict__ vst, BlockQ8_0 *__restrict__ Kc, BlockQ8_0 *__restrict__ Vc, const int *__restrict__ d_pos, int n_kv_heads, int head_dim, int max_ctx){
    int block_idx=blockIdx.x*blockDim.x+threadIdx.x; int num_blocks_per_slot=(n_kv_heads*head_dim)/32; if(block_idx>=num_blocks_per_slot) return;
    int slot=(*d_pos)%max_ctx; int src_offset=block_idx*32;
    float k_vals[32],v_vals[32]; float max_k=0,max_v=0;
    for(int i=0;i<32;i++){k_vals[i]=kst[src_offset+i]; v_vals[i]=vst[src_offset+i]; max_k=fmaxf(max_k,fabsf(k_vals[i])); max_v=fmaxf(max_v,fabsf(v_vals[i]));}
    float scale_k=(max_k>0)?(max_k/127.0f):1.0f; float inv_k=(max_k>0)?(127.0f/max_k):0; float scale_v=(max_v>0)?(max_v/127.0f):1.0f; float inv_v=(max_v>0)?(127.0f/max_v):0;
    BlockQ8_0 *kd=Kc+(long)slot*num_blocks_per_slot+block_idx; BlockQ8_0 *vd=Vc+(long)slot*num_blocks_per_slot+block_idx;
    kd->d=__float2half(scale_k); vd->d=__float2half(scale_v);
    for(int i=0;i<32;i++){kd->qs[i]=(int8_t)__float2int_rn(k_vals[i]*inv_k); vd->qs[i]=(int8_t)__float2int_rn(v_vals[i]*inv_v);}
}
__global__ void k_kv_scatter_fp32(const float *__restrict__ kst, const float *__restrict__ vst, float *__restrict__ Kc, float *__restrict__ Vc, const int *__restrict__ d_pos, int n_kv_heads, int head_dim, int max_ctx){
    int i=threadIdx.x+blockIdx.x*blockDim.x; int kvdim=n_kv_heads*head_dim; if(i>=kvdim) return; int slot=(*d_pos)%max_ctx; Kc[(long)slot*kvdim+i]=kst[i]; Vc[(long)slot*kvdim+i]=vst[i];
}

// ---------------- New FA2 tiled Q8 kernel ----------------
// Design: BC=32 KV block, BR concept is 1 query (decode). GQA grouping: 7 heads per KV head share smem tile.
// Grid: dim3(S, n_kv_heads) where S = ceil(N/BC). Each block = 7 warps (224 threads) = one GQA group for one BC slice.
// Each block cooperatively loads its BC slice's Q8 blocks into smem using vectorized loads (128-bit where possible via BlockQ8_0 struct),
// then each warp does online softmax loop over its slice while sharing smem K/V (broadcast). Partial (m,l,acc) written to global.
// Second kernel merges S partials per head via block-wise online softmax.
#define FA2_BC 32
#define FA2_BC_MAX 32

__global__ void k_fa2_q8_split(
    const float *__restrict__ q,
    const BlockQ8_0 *__restrict__ Kc_q8,
    const BlockQ8_0 *__restrict__ Vc_q8,
    float *__restrict__ p_acc, // [S][n_heads][head_dim]
    float *__restrict__ p_m,   // [S][n_heads]
    float *__restrict__ p_l,   // [S][n_heads]
    const int *__restrict__ d_pos,
    int n_heads, int n_kv_heads, int head_dim,
    float scale, int window, int S)
{
    const int pos = *d_pos;
    int t0=0; if(window>0 && pos>=window) t0=pos-window+1;
    int total = pos - t0 + 1;
    int chunk = (total + S -1)/ S; // tokens per split
    const int s = blockIdx.x; // 0..S-1
    const int kv = blockIdx.y; // 0..n_kv_heads-1
    if(s >= S || kv >= n_kv_heads) return;
    const int G = n_heads / n_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    // blockDim = G*32 (e.g., 224)
    if(warp >= G) return;

    const int head = kv * G + warp; // global head id for this warp

    // token range for this split
    const int begin = t0 + s * chunk;
    const int end = min(pos+1, t0 + (s+1)*chunk);
    int active = end - begin;
    if(active < 0) active = 0;

    const int blocks_per_head = head_dim / 32; // 4 for 128
    // shared tiles for this BC slice (max 32*4=128 blocks)
    __shared__ BlockQ8_0 sK[FA2_BC_MAX * 4];
    __shared__ BlockQ8_0 sV[FA2_BC_MAX * 4];

    // cooperative vectorized load of Q8 blocks for this slice
    // Use 128-bit vectorization concept: each BlockQ8_0 is 34 bytes; we load as struct via __ldg (compiler will vectorize qs via 4x int32)
    int totalBlocks = active * blocks_per_head;
    // vectorized loop: stride = blockDim.x; each thread loads one BlockQ8_0 at a time
    for(int i = tid; i < totalBlocks; i += blockDim.x){
        int tok = i / blocks_per_head;
        int b = i % blocks_per_head;
        long g_idx = ((long)(begin + tok) * n_kv_heads + kv) * blocks_per_head + b;
        // __ldg vectorized load hint
        sK[i] = Kc_q8[g_idx];
        sV[i] = Vc_q8[g_idx];
    }
    __syncthreads();

    // q registers for this head/warp (each lane holds 4 elems)
    // head_dim=128 => elems=4
    const int elems = head_dim / 32;
    const float *qh = q + (long)head * head_dim + lane * elems;
    float qreg[4];
#pragma unroll
    for(int i=0;i<4;i++) qreg[i] = (i < elems) ? qh[i] : 0.0f;

    float m_prev = -1e30f, l_prev = 0.0f;
    float oreg[4] = {0,0,0,0};

    if(active==0){
        // write empty partial
        if(lane==0){ p_m[(size_t)s * n_heads + head] = -INFINITY; p_l[(size_t)s * n_heads + head] = 0.0f; }
        float *myacc = p_acc + ((size_t)s * n_heads + head) * head_dim + lane * elems;
        for(int i=0;i<4;i++) if(i<elems) myacc[i]=0;
        return;
    }

    // online softmax loop over active tokens; K/V fetched from smem (broadcast)
    for(int ti=0; ti<active; ++ti){
        int blk_base = ti * blocks_per_head;
        // lane's block: for head_dim 128, lane 0..31, elems 4 => block = lane>>3, offset = (lane&7)*4
        int block_in_head = lane >> 3; // 0..3
        int elem_off = (lane & 7) * 4;
        const BlockQ8_0 &bk = sK[blk_base + block_in_head];
        const BlockQ8_0 &bv = sV[blk_base + block_in_head];
        float dk = __half2float(bk.d);
        float dv = __half2float(bv.d);
        // dequant K on fly via shared broadcast (vectorized 4 int8)
        // Use 32-bit word load for qs chunk: int32 = *(int32*)&qs[elem_off]
        // But keep scalar for clarity (compiler will vectorize)
        float k0 = (float)bk.qs[elem_off];
        float k1 = (float)bk.qs[elem_off+1];
        float k2 = (float)bk.qs[elem_off+2];
        float k3 = (float)bk.qs[elem_off+3];
        float score = (qreg[0]*k0 + qreg[1]*k1 + qreg[2]*k2 + qreg[3]*k3) * dk;
        score = warp_sum(score);
        score = __shfl_sync(0xffffffff, score, 0) * scale;

        float m_curr = fmaxf(m_prev, score);
        float p = expf(score - m_curr);
        float alpha = expf(m_prev - m_curr);
        float l_curr = l_prev * alpha + p;

        // rescale + add V
        float v0 = (float)bv.qs[elem_off];
        float v1 = (float)bv.qs[elem_off+1];
        float v2 = (float)bv.qs[elem_off+2];
        float v3 = (float)bv.qs[elem_off+3];
        oreg[0] = oreg[0] * alpha + p * dv * v0;
        oreg[1] = oreg[1] * alpha + p * dv * v1;
        oreg[2] = oreg[2] * alpha + p * dv * v2;
        oreg[3] = oreg[3] * alpha + p * dv * v3;

        m_prev = m_curr; l_prev = l_curr;
    }

    // write partials
    if(lane==0){
        p_m[(size_t)s * n_heads + head] = m_prev;
        p_l[(size_t)s * n_heads + head] = l_prev;
    }
    float *myacc2 = p_acc + ((size_t)s * n_heads + head) * head_dim + lane * elems;
    for(int i=0;i<4;i++) if(i<elems) myacc2[i]=oreg[i];
}

__global__ void k_fa2_combine(
    const float *__restrict__ p_acc,
    const float *__restrict__ p_m,
    const float *__restrict__ p_l,
    float *__restrict__ out,
    int n_heads, int head_dim, int S)
{
    int h = blockIdx.x;
    int lane = threadIdx.x;
    int elems = head_dim / 32;
    if(h>= n_heads) return;
    float m = -INFINITY, l=0;
    float oreg[4]={0,0,0,0};
    for(int s=0;s<S;s++){
        size_t idx = (size_t)s * n_heads + h;
        float ls = p_l[idx];
        if(!(ls>0)) continue;
        float ms = p_m[idx];
        float m_new = fmaxf(m, ms);
        float alpha = expf(m - m_new);
        float beta = expf(ms - m_new);
        const float *acc = p_acc + idx * head_dim + lane * elems;
        l = l * alpha + ls * beta;
        for(int i=0;i<4;i++) if(i<elems) oreg[i]= oreg[i]*alpha + acc[i]*beta;
        m = m_new;
    }
    float inv = 1.0f / (l + 1e-8f);
    float *oh = out + (long)h * head_dim + lane * elems;
    for(int i=0;i<4;i++) if(i<elems) oh[i]=oreg[i]*inv;
}

// ---------------- bench ----------------
static float frand(){ return (float)rand()/(float)RAND_MAX*2.0f -1.0f; }

int main(){
    const int n_heads=14, n_kv_heads=2, head_dim=128;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const int window=0;
    const int max_ctx=8192;
    std::vector<int> Ns = {512,1024,2048,4096};
    printf("FA2 Q8 microbench: n_heads=%d n_kv_heads=%d head_dim=%d BC=%d scale=%.5f\n", n_heads,n_kv_heads,head_dim,FA2_BC,scale);
    printf("N | time_current(ms) | time_fa2(ms) | speedup | max_abs_err_q8_vs_fp32 | max_abs_err_fa2_vs_current | NaN/Inf\n");

    size_t q_bytes=(size_t)n_heads*head_dim*sizeof(float);
    size_t out_bytes=q_bytes;
    size_t kvdim=(size_t)n_kv_heads*head_dim;
    size_t fp32_cache_bytes=(size_t)max_ctx * kvdim * sizeof(float);
    int blocks_per_head=head_dim/32;
    int blocks_per_slot=n_kv_heads*blocks_per_head;
    size_t q8_cache_bytes=(size_t)max_ctx * blocks_per_slot * sizeof(BlockQ8_0);

    float *h_q=(float*)malloc(q_bytes);
    float *h_out_cur=(float*)malloc(out_bytes);
    float *h_out_fa2=(float*)malloc(out_bytes);
    float *h_out_ref=(float*)malloc(out_bytes);
    float *h_kst=(float*)malloc(kvdim*sizeof(float));
    float *h_vst=(float*)malloc(kvdim*sizeof(float));
    for(size_t i=0;i<(size_t)n_heads*head_dim;i++) h_q[i]=frand()*0.5f;

    float *d_q,*d_out_cur,*d_out_fa2,*d_out_ref;
    float *d_Kc_fp32,*d_Vc_fp32;
    BlockQ8_0 *d_Kc_q8,*d_Vc_q8;
    int *d_pos;
    float *d_kst,*d_vst;
    cudaMalloc(&d_q,q_bytes); cudaMalloc(&d_out_cur,out_bytes); cudaMalloc(&d_out_fa2,out_bytes); cudaMalloc(&d_out_ref,out_bytes);
    cudaMalloc(&d_Kc_fp32,fp32_cache_bytes); cudaMalloc(&d_Vc_fp32,fp32_cache_bytes);
    cudaMalloc(&d_Kc_q8,q8_cache_bytes); cudaMalloc(&d_Vc_q8,q8_cache_bytes);
    cudaMalloc(&d_pos,sizeof(int)); cudaMalloc(&d_kst,kvdim*sizeof(float)); cudaMalloc(&d_vst,kvdim*sizeof(float));
    cudaMemcpy(d_q,h_q,q_bytes,cudaMemcpyHostToDevice);
    cudaMemset(d_Kc_fp32,0,fp32_cache_bytes); cudaMemset(d_Vc_fp32,0,fp32_cache_bytes);
    cudaMemset(d_Kc_q8,0,q8_cache_bytes); cudaMemset(d_Vc_q8,0,q8_cache_bytes);

    // workspace for FA2 split
    int Smax = (4096 + FA2_BC -1)/FA2_BC; // 128
    size_t pacc_bytes=(size_t)Smax * n_heads * head_dim * sizeof(float);
    size_t pm_bytes=(size_t)Smax * n_heads * sizeof(float);
    float *d_pacc,*d_pm,*d_pl;
    cudaMalloc(&d_pacc,pacc_bytes); cudaMalloc(&d_pm,pm_bytes); cudaMalloc(&d_pl,pm_bytes);

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

    for(int N: Ns){
        // fill KV caches for 0..N-1
        cudaMemset(d_Kc_fp32,0,fp32_cache_bytes); cudaMemset(d_Vc_fp32,0,fp32_cache_bytes);
        cudaMemset(d_Kc_q8,0,q8_cache_bytes); cudaMemset(d_Vc_q8,0,q8_cache_bytes);
        for(int t=0;t<N;t++){
            for(size_t i=0;i<kvdim;i++){ h_kst[i]=frand()*0.8f; h_vst[i]=frand()*0.8f; }
            cudaMemcpy(d_kst,h_kst,kvdim*sizeof(float),cudaMemcpyHostToDevice);
            cudaMemcpy(d_vst,h_vst,kvdim*sizeof(float),cudaMemcpyHostToDevice);
            cudaMemcpy(d_pos,&t,sizeof(int),cudaMemcpyHostToDevice);
            int nb=(n_kv_heads*head_dim)/32;
            k_kv_scatter_fp32<<<(kvdim+255)/256,256>>>(d_kst,d_vst,d_Kc_fp32,d_Vc_fp32,d_pos,n_kv_heads,head_dim,max_ctx);
            k_kv_scatter_q8_0<<<(nb+255)/256,256>>>(d_kst,d_vst,d_Kc_q8,d_Vc_q8,d_pos,n_kv_heads,head_dim,max_ctx);
        }
        cudaDeviceSynchronize();
        int cur_pos=N-1; cudaMemcpy(d_pos,&cur_pos,sizeof(int),cudaMemcpyHostToDevice);
        // correctness single run
        k_flash_gqa_q8_0<<<n_heads,32>>>(d_q,d_Kc_q8,d_Vc_q8,d_out_cur,d_pos,n_heads,n_kv_heads,head_dim,max_ctx,scale,window);
        k_flash_gqa_fp32<<<n_heads,32>>>(d_q,d_Kc_fp32,d_Vc_fp32,d_out_ref,d_pos,n_heads,n_kv_heads,head_dim,max_ctx,scale,window);
        // FA2
        int S=(N + FA2_BC -1)/FA2_BC;
        if(S<1) S=1;
        dim3 grid_fa2(S, n_kv_heads);
        int G=n_heads/n_kv_heads;
        int threads_fa2=G*32;
        k_fa2_q8_split<<<grid_fa2, threads_fa2>>>(d_q,d_Kc_q8,d_Vc_q8,d_pacc,d_pm,d_pl,d_pos,n_heads,n_kv_heads,head_dim,scale,window,S);
        k_fa2_combine<<<n_heads,32>>>(d_pacc,d_pm,d_pl,d_out_fa2,n_heads,head_dim,S);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out_cur,d_out_cur,out_bytes,cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_fa2,d_out_fa2,out_bytes,cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_ref,d_out_ref,out_bytes,cudaMemcpyDeviceToHost);
        float max_err_cur=0, max_err_fa2=0, max_err_fa2_cur=0;
        int naninf=0;
        for(size_t i=0;i<(size_t)n_heads*head_dim;i++){
            if(!isfinite(h_out_cur[i])||!isfinite(h_out_fa2[i])) naninf++;
            float e1=fabsf(h_out_cur[i]-h_out_ref[i]);
            float e2=fabsf(h_out_fa2[i]-h_out_ref[i]);
            float e3=fabsf(h_out_fa2[i]-h_out_cur[i]);
            max_err_cur=std::max(max_err_cur,e1);
            max_err_fa2=std::max(max_err_fa2,e2);
            max_err_fa2_cur=std::max(max_err_fa2_cur,e3);
        }
        // benchmark current
        const int warmup=20, iters=200;
        for(int i=0;i<warmup;i++) k_flash_gqa_q8_0<<<n_heads,32>>>(d_q,d_Kc_q8,d_Vc_q8,d_out_cur,d_pos,n_heads,n_kv_heads,head_dim,max_ctx,scale,window);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for(int i=0;i<iters;i++) k_flash_gqa_q8_0<<<n_heads,32>>>(d_q,d_Kc_q8,d_Vc_q8,d_out_cur,d_pos,n_heads,n_kv_heads,head_dim,max_ctx,scale,window);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms_cur=0; cudaEventElapsedTime(&ms_cur,start,stop); ms_cur/=iters;
        // bench fa2
        for(int i=0;i<warmup;i++){ k_fa2_q8_split<<<grid_fa2, threads_fa2>>>(d_q,d_Kc_q8,d_Vc_q8,d_pacc,d_pm,d_pl,d_pos,n_heads,n_kv_heads,head_dim,scale,window,S); k_fa2_combine<<<n_heads,32>>>(d_pacc,d_pm,d_pl,d_out_fa2,n_heads,head_dim,S); }
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for(int i=0;i<iters;i++){ k_fa2_q8_split<<<grid_fa2, threads_fa2>>>(d_q,d_Kc_q8,d_Vc_q8,d_pacc,d_pm,d_pl,d_pos,n_heads,n_kv_heads,head_dim,scale,window,S); k_fa2_combine<<<n_heads,32>>>(d_pacc,d_pm,d_pl,d_out_fa2,n_heads,head_dim,S); }
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms_fa2=0; cudaEventElapsedTime(&ms_fa2,start,stop); ms_fa2/=iters;
        float speedup = ms_cur / ms_fa2;
        printf("%4d | % .4f | % .4f | % .2fx | cur %.3e fa2 %.3e diff %.3e | %d %s\n", N, ms_cur, ms_fa2, speedup, max_err_cur, max_err_fa2, max_err_fa2_cur, naninf, (max_err_fa2<1e-2 && naninf==0?"PASS":"FAIL"));
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_q); cudaFree(d_out_cur); cudaFree(d_out_fa2); cudaFree(d_out_ref);
    cudaFree(d_Kc_fp32); cudaFree(d_Vc_fp32); cudaFree(d_Kc_q8); cudaFree(d_Vc_q8);
    cudaFree(d_pos); cudaFree(d_kst); cudaFree(d_vst);
    cudaFree(d_pacc); cudaFree(d_pm); cudaFree(d_pl);
    free(h_q); free(h_out_cur); free(h_out_fa2); free(h_out_ref); free(h_kst); free(h_vst);
    return 0;
}
