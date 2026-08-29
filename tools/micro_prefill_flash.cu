// Microbench: batched FlashAttention Q8_0 PREFILL (n queries x ctx KV) vs serial
//
// Compares time_serial (k_flash_gqa_q8_0 called n times, each at a different
// ctx_len = e_pos + i + 1, matching the engine's prefill loop today) against
// time_batched (single launch of k_prefill_flash_q8_0).
//
// Design:
//   Grid  : dim3(num_q_tiles, n_kv_heads) where num_q_tiles = ceil(n / BR)
//   Block : G*32 threads (G = n_heads / n_kv_heads); one warp per query head
//           in the GQA group. Each warp processes its assigned query rows in
//           sequence (warp w -> rows w, w+G, w+2G, ...). For each row, 32
//           lanes cooperatively compute the full head-dim dot product and
//           each lane holds 4 elems of acc (lane m -> head-dim slice
//           [m*4, m*4+4)). Output is written 4 elems per lane, naturally
//           covering the full head-dim with 32 lanes.
//   Smem  : sK[BC*HD], sV[BC*HD] (Q8_0 dequantized to FP16 on load),
//           sQ[BR*HD] (FP16, one-time load of all query rows).
//   Loop  : S = ceil(ctx/BC) inside each block; same KV tile read by every
//           q-tile (L2 reuse across blocks on grid-y).
//   Mask  : causal per-row; query i attends to [0, e_pos + i] ONLY.
//
// Build:
//   cd ~/Storage/repos/nnfromscratch
//   export PATH=$HOME/mmcuda/bin:$PATH
//   nvcc -O3 -arch=native --resource-usage -Iinclude -Isrc \
//       -o build/micro_prefill_flash tools/micro_prefill_flash.cu \
//       -L$HOME/mmcuda/lib -lcudart
//   export LD_LIBRARY_PATH=$HOME/mmcuda/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
//   ./build/micro_prefill_flash
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
    half d;          // 2 bytes FP16 scale
    int8_t qs[32];   // 32 bytes signed int8
};
#endif

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off /= 2) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// ---------------- serial baseline (copy of k_flash_gqa_q8_0) ----------------
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
        }
        m_prev = m_curr; l_prev = l_curr;
    }
    float *outh = out + (long)h * head_dim + lane * elems;
#pragma unroll
    for (int i = 0; i < 16; i++) if (i < elems) outh[i] = oreg[i] / l_prev;
}

// ---------------- NEW batched PREFILL flash kernel ----------------
//
// Grid  : dim3(num_q_tiles, n_kv_heads)
// Block : G*32 threads (G = n_heads / n_kv_heads). With n_heads=14,n_kv_heads=2
//         => G=7 => blockDim=224.
//
// Each block:
//   - Loads BR query rows (all GQA heads) for q-tile blockIdx.x into sQ[BR,HD]
//     (FP16, one-time). Each warp loads its own head rows: warp w loads
//     head = kv*G + w, rows 0..BR-1 of the q-tile.
//   - Iterates S = ceil(ctx/BC) KV tiles; for each s:
//       1) cooperative load + dequant of BC KV rows for this kv-head into
//          sK[BC,HD] and sV[BC,HD] (FP16).
//       2) per warp, per query row: causal-masked online softmax update over
//          the BC-tile using sK/sV (broadcast). Each warp handles its assigned
//          rows in sequence: rows w, w+G, w+2G, ...
//   - Writes Att[query, head, :] = acc / l. Each lane writes 4 elems (its
//     head-dim slice) for the row it just processed.
//
// Causal mask: query i attends to KV [0, e_pos + i] ONLY.
#define BR_PF 8
#define BC_PF 64

__global__ void k_prefill_flash_q8_0(
    const float     *__restrict__ Q,        // [n, n_heads, head_dim]
    const BlockQ8_0 *__restrict__ Kc,       // [ctx_max, n_kv_heads*blocks_per_head]
    const BlockQ8_0 *__restrict__ Vc,       // [ctx_max, n_kv_heads*blocks_per_head]
    float           *__restrict__ Att,      // [n, n_heads, head_dim]
    int n, int ctx, int e_pos,
    int n_heads, int n_kv_heads, int head_dim,
    float scale, int window)
{
    const int G = n_heads / n_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv = blockIdx.y;
    const int q_tile = blockIdx.x;
    const int head = kv * G + warp;
    if (warp >= G) return;

    const int elems = head_dim / 32;             // 4 for HD=128
    const int blocks_per_head = head_dim / 32;   // 4 for HD=128

    // Smem layout: BC_PF KV tokens
    // sK_d, sV_d: [BC_PF, blocks_per_head] FP16 scales
    // sK_q, sV_q: [BC_PF, head_dim] int8 quantized
    extern __shared__ char raw_smem[];
    half   *sK_d = (half*)raw_smem;
    half   *sV_d = sK_d + BC_PF * blocks_per_head;
    int8_t *sK_q = (int8_t*)(sV_d + BC_PF * blocks_per_head);
    int8_t *sV_q = sK_q + BC_PF * head_dim;

    // Each warp is dedicated to one query head (`head`).
    // The 32 lanes in the warp cooperate on each query row:
    // lane m holds elements [m*4 .. m*4+3] of head_dim=128.
    // In registers, each lane holds Q, m, l, acc for all BR_PF rows of this q-tile.
    float qreg[BR_PF][4];
    float m_state[BR_PF];
    float l_state[BR_PF];
    float acc[BR_PF][4];
    int   max_kv[BR_PF];
    int   row_min_t[BR_PF];
    bool  active[BR_PF];

#pragma unroll
    for (int r = 0; r < BR_PF; r++) {
        const int qrow = q_tile * BR_PF + r;
        active[r] = (qrow < n);
        max_kv[r] = active[r] ? (e_pos + qrow) : -1;
        row_min_t[r] = (window > 0 && active[r] && (e_pos + qrow + 1 > window))
                       ? (e_pos + qrow + 1 - window) : 0;

        m_state[r] = -1e30f;
        l_state[r] = 0.0f;
        acc[r][0] = 0.0f; acc[r][1] = 0.0f; acc[r][2] = 0.0f; acc[r][3] = 0.0f;

        if (active[r]) {
            const float *qr = Q + (long)qrow * (n_heads * head_dim) + (long)head * head_dim + lane * elems;
            qreg[r][0] = qr[0]; qreg[r][1] = qr[1]; qreg[r][2] = qr[2]; qreg[r][3] = qr[3];
        } else {
            qreg[r][0] = 0.0f; qreg[r][1] = 0.0f; qreg[r][2] = 0.0f; qreg[r][3] = 0.0f;
        }
    }

    const int S = (ctx + BC_PF - 1) / BC_PF;
    const int block_in_head = lane >> 3;         // lane / 8
    const int elem_off = (lane & 7) * elems;     // (lane % 8) * 4

    for (int s = 0; s < S; s++) {
        const int s_start = s * BC_PF;
        const int s_end_excl = (s_start + BC_PF < ctx) ? (s_start + BC_PF) : ctx;
        const int bc_active = s_end_excl - s_start;
        if (bc_active <= 0) continue;

        // Cooperative load of BC_PF KV blocks into smem
        const int total_blocks = bc_active * blocks_per_head;
        for (int i = tid; i < total_blocks; i += blockDim.x) {
            const int tok = i / blocks_per_head;
            const int b   = i % blocks_per_head;
            const long g_idx = ((long)(s_start + tok) * n_kv_heads + kv) * blocks_per_head + b;
            const BlockQ8_0 bk = Kc[g_idx];
            const BlockQ8_0 bv = Vc[g_idx];
            sK_d[tok * blocks_per_head + b] = bk.d;
            sV_d[tok * blocks_per_head + b] = bv.d;
            const int row_off = tok * head_dim + b * 32;
#pragma unroll
            for (int j = 0; j < 32; j++) {
                sK_q[row_off + j] = bk.qs[j];
                sV_q[row_off + j] = bv.qs[j];
            }
        }
        __syncthreads();

        for (int t = s_start; t < s_end_excl; t++) {
            const int t_in_tile = t - s_start;
            const int k_q_off = t_in_tile * head_dim + block_in_head * 32 + elem_off;
            const int v_q_off = k_q_off;
            const float dk = __half2float(sK_d[t_in_tile * blocks_per_head + block_in_head]);
            const float dv = __half2float(sV_d[t_in_tile * blocks_per_head + block_in_head]);

            const float k0 = (float)sK_q[k_q_off + 0];
            const float k1 = (float)sK_q[k_q_off + 1];
            const float k2 = (float)sK_q[k_q_off + 2];
            const float k3 = (float)sK_q[k_q_off + 3];

            const float v0 = (float)sV_q[v_q_off + 0];
            const float v1 = (float)sV_q[v_q_off + 1];
            const float v2 = (float)sV_q[v_q_off + 2];
            const float v3 = (float)sV_q[v_q_off + 3];

#pragma unroll
            for (int r = 0; r < BR_PF; r++) {
                if (!active[r] || t < row_min_t[r] || t > max_kv[r]) continue;

                // Dot product slice for lane m
                float dot_partial = (qreg[r][0]*k0 + qreg[r][1]*k1 + qreg[r][2]*k2 + qreg[r][3]*k3) * dk;
                float score = warp_sum(dot_partial);
                score = __shfl_sync(0xffffffff, score, 0) * scale;

                float m_curr = fmaxf(m_state[r], score);
                float p = expf(score - m_curr);
                float alpha = expf(m_state[r] - m_curr);
                l_state[r] = l_state[r] * alpha + p;

                float pdv = p * dv;
                acc[r][0] = acc[r][0] * alpha + pdv * v0;
                acc[r][1] = acc[r][1] * alpha + pdv * v1;
                acc[r][2] = acc[r][2] * alpha + pdv * v2;
                acc[r][3] = acc[r][3] * alpha + pdv * v3;

                m_state[r] = m_curr;
            }
        }
        __syncthreads();
    }

    // Write output: 32 lanes cooperatively write each row's 128 elements
#pragma unroll
    for (int r = 0; r < BR_PF; r++) {
        if (active[r]) {
            const int qrow = q_tile * BR_PF + r;
            float inv_l = 1.0f / l_state[r];
            float *out_row = Att + (long)qrow * (n_heads * head_dim) + (long)head * head_dim + lane * elems;
            out_row[0] = acc[r][0] * inv_l;
            out_row[1] = acc[r][1] * inv_l;
            out_row[2] = acc[r][2] * inv_l;
            out_row[3] = acc[r][3] * inv_l;
        }
    }
}

__global__ void k_kv_scatter_q8_0(const float *kst, const float *vst,
                                  BlockQ8_0 *Kc, BlockQ8_0 *Vc,
                                  const int *d_pos,
                                  int n_kv_heads, int head_dim, int max_ctx) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks_per_slot = (n_kv_heads * head_dim) / 32;
    if (block_idx >= num_blocks_per_slot) return;
    int slot = (*d_pos) % max_ctx;
    int src_offset = block_idx * 32;
    float k_vals[32], v_vals[32]; float max_k = 0, max_v = 0;
    for (int i = 0; i < 32; i++) {
        k_vals[i] = kst[src_offset + i];
        v_vals[i] = vst[src_offset + i];
        max_k = fmaxf(max_k, fabsf(k_vals[i]));
        max_v = fmaxf(max_v, fabsf(v_vals[i]));
    }
    float scale_k = (max_k > 0) ? (max_k / 127.0f) : 1.0f;
    float inv_k  = (max_k > 0) ? (127.0f / max_k) : 0.0f;
    float scale_v = (max_v > 0) ? (max_v / 127.0f) : 1.0f;
    float inv_v  = (max_v > 0) ? (127.0f / max_v) : 0.0f;
    BlockQ8_0 *kd = Kc + (long)slot * num_blocks_per_slot + block_idx;
    BlockQ8_0 *vd = Vc + (long)slot * num_blocks_per_slot + block_idx;
    kd->d = __float2half(scale_k);
    vd->d = __float2half(scale_v);
    for (int i = 0; i < 32; i++) {
        kd->qs[i] = (int8_t)__float2int_rn(k_vals[i] * inv_k);
        vd->qs[i] = (int8_t)__float2int_rn(v_vals[i] * inv_v);
    }
}

static float frand(){ return (float)rand()/(float)RAND_MAX*2.0f - 1.0f; }

// ---------------- bench ----------------
int main() {
    const int n_heads = 14, n_kv_heads = 2, head_dim = 128;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const int window = 0;
    const int max_ctx = 8192;
    const int G = n_heads / n_kv_heads;            // 7
    const int blocks_per_head = head_dim / 32;     // 4
    const int blocks_per_slot = n_kv_heads * blocks_per_head;

    std::vector<int> Ns = {32, 64, 128, 256, 512, 1024, 2048};
    std::vector<int> Cs = {32, 64, 128, 256, 512, 1024, 2048};

    printf("PREFILL FA2 Q8 microbench: n_heads=%d n_kv_heads=%d head_dim=%d G=%d BR=%d BC=%d scale=%.5f\n",
           n_heads, n_kv_heads, head_dim, G, BR_PF, BC_PF, scale);
    printf("row format: n | ctx | time_serial(ms) | time_batched(ms) | speedup | max_abs_err | NaN/Inf | PASS?\n");

    size_t qrow_bytes   = (size_t)n_heads * head_dim * sizeof(float);
    size_t att_bytes    = qrow_bytes;
    size_t q_buf_bytes  = (size_t)2048 * n_heads * head_dim * sizeof(float);
    size_t att_buf_bytes= (size_t)2048 * n_heads * head_dim * sizeof(float);
    size_t kvdim        = (size_t)n_kv_heads * head_dim;
    size_t q8_cache_bytes = (size_t)max_ctx * blocks_per_slot * sizeof(BlockQ8_0);
    size_t kv_buf_bytes = kvdim * sizeof(float);

    float *h_qrow   = (float*)malloc(qrow_bytes);
    float *h_att_ser= (float*)malloc(att_bytes);
    float *h_att_bat= (float*)malloc(att_bytes);
    float *h_kst    = (float*)malloc(kv_buf_bytes);
    float *h_vst    = (float*)malloc(kv_buf_bytes);
    for (size_t i = 0; i < qrow_bytes / sizeof(float); i++) h_qrow[i] = frand() * 0.5f;

    float *d_qrow, *d_att_ser, *d_att_bat;
    BlockQ8_0 *d_Kc_q8, *d_Vc_q8;
    int *d_pos;
    float *d_kst, *d_vst;
    float *d_qbuf, *d_attbuf;
    cudaMalloc(&d_qrow,  qrow_bytes);
    cudaMalloc(&d_att_ser, att_buf_bytes);
    cudaMalloc(&d_att_bat, att_buf_bytes);
    cudaMalloc(&d_Kc_q8,  q8_cache_bytes);
    cudaMalloc(&d_Vc_q8,  q8_cache_bytes);
    cudaMalloc(&d_pos,    sizeof(int));
    cudaMalloc(&d_kst,    kv_buf_bytes);
    cudaMalloc(&d_vst,    kv_buf_bytes);
    cudaMalloc(&d_qbuf,   q_buf_bytes);
    cudaMalloc(&d_attbuf, att_buf_bytes);
    cudaMemcpy(d_qrow, h_qrow, qrow_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_Kc_q8, 0, q8_cache_bytes);
    cudaMemset(d_Vc_q8, 0, q8_cache_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int warmup = 20;
    const int iters  = 50;

    for (int n : Ns) { for (int ctx : Cs) { if (ctx != n) continue; if (ctx > max_ctx) continue;
        cudaMemset(d_Kc_q8, 0, q8_cache_bytes);
        cudaMemset(d_Vc_q8, 0, q8_cache_bytes);
        for (int t = 0; t < ctx; t++) {
                for (size_t i = 0; i < kvdim; i++) { h_kst[i] = frand() * 0.8f; h_vst[i] = frand() * 0.8f; }
                cudaMemcpy(d_kst, h_kst, kv_buf_bytes, cudaMemcpyHostToDevice);
                cudaMemcpy(d_vst, h_vst, kv_buf_bytes, cudaMemcpyHostToDevice);
                cudaMemcpy(d_pos, &t, sizeof(int), cudaMemcpyHostToDevice);
                int nb = (n_kv_heads * head_dim) / 32;
                k_kv_scatter_q8_0<<<(nb + 255) / 256, 256>>>(
                    d_kst, d_vst, d_Kc_q8, d_Vc_q8, d_pos, n_kv_heads, head_dim, max_ctx);
            }
            cudaDeviceSynchronize();
            if (n == 32) {
                int nb_dbg = (n_kv_heads * head_dim) / 32;
                BlockQ8_0 *h_Kc = (BlockQ8_0*)malloc((size_t)nb_dbg * sizeof(BlockQ8_0));
                cudaMemcpy(h_Kc, d_Kc_q8 + (long)0 * nb_dbg, (size_t)nb_dbg * sizeof(BlockQ8_0), cudaMemcpyDeviceToHost);
                printf("    [sanity t=0] Kc[0].d=%.5f qs[0..7]=", __half2float(h_Kc[0].d));
                for (int j = 0; j < 8; j++) printf("%d ", (int)h_Kc[0].qs[j]);
                printf("qs[28..31]=");
                for (int j = 28; j < 32; j++) printf("%d ", (int)h_Kc[0].qs[j]);
                printf("\n");
                cudaMemcpy(h_Kc, d_Vc_q8 + (long)0 * nb_dbg, (size_t)nb_dbg * sizeof(BlockQ8_0), cudaMemcpyDeviceToHost);
                printf("    [sanity t=0] Vc[0].d=%.5f qs[0..7]=", __half2float(h_Kc[0].d));
                for (int j = 0; j < 8; j++) printf("%d ", (int)h_Kc[0].qs[j]);
                printf("qs[28..31]=");
                for (int j = 28; j < 32; j++) printf("%d ", (int)h_Kc[0].qs[j]);
                printf("Vc[1].d=%.5f qs[0..7]=", __half2float(h_Kc[1].d));
                for (int j = 0; j < 8; j++) printf("%d ", (int)h_Kc[1].qs[j]);
                printf("\n");
                free(h_Kc);
                fflush(stdout);
            }

            float *h_q = (float*)malloc((size_t)n * n_heads * head_dim * sizeof(float));
            for (size_t i = 0; i < (size_t)n * n_heads * head_dim; i++) h_q[i] = frand() * 0.5f;
            cudaMemcpy(d_qbuf, h_q, (size_t)n * n_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_attbuf, 0, (size_t)n * n_heads * head_dim * sizeof(float));

            const int e_pos = 0;

            // Serial baseline: k_flash_gqa_q8_0 called n times.
            for (int it = 0; it < warmup; it++) {
                for (int i = 0; i < n; i++) {
                    int pos_i = e_pos + i;  // batched's mask: query i attends to [0, e_pos+i]
                    cudaMemcpy(d_pos, &pos_i, sizeof(int), cudaMemcpyHostToDevice);
                    k_flash_gqa_q8_0<<<n_heads, 32>>>(
                        d_qbuf + (long)i * n_heads * head_dim,
                        d_Kc_q8, d_Vc_q8,
                        d_attbuf + (long)i * n_heads * head_dim,
                        d_pos, n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
                }
            }
             cudaDeviceSynchronize();
            if (n == 32) {
                float h_dbg[16];
                cudaMemcpy(h_dbg, d_attbuf, 16 * sizeof(float), cudaMemcpyDeviceToHost);
                printf("    [post-warmup] d_attbuf[0..15] = ");
                for (int i = 0; i < 16; i++) printf("%.4f ", h_dbg[i]);
                printf("\n");
                fflush(stdout);
            }
            cudaEventRecord(start);
            for (int it = 0; it < iters; it++) {
                for (int i = 0; i < n; i++) {
                    int pos_i = e_pos + i;  // batched's mask: query i attends to [0, e_pos+i]
                    cudaMemcpy(d_pos, &pos_i, sizeof(int), cudaMemcpyHostToDevice);
                    k_flash_gqa_q8_0<<<n_heads, 32>>>(
                        d_qbuf + (long)i * n_heads * head_dim,
                        d_Kc_q8, d_Vc_q8,
                        d_attbuf + (long)i * n_heads * head_dim,
                        d_pos, n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
                }
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms_serial = 0;
            cudaEventElapsedTime(&ms_serial, start, stop);
            ms_serial /= iters;

            cudaMemcpy(d_att_ser, d_attbuf, (size_t)n * n_heads * head_dim * sizeof(float),
                       cudaMemcpyDeviceToDevice);

            // Batched prefill flash
            int num_q_tiles = (n + BR_PF - 1) / BR_PF;
            dim3 grid(num_q_tiles, n_kv_heads);
            int threads = G * 32;            // 224
            size_t smem_bytes = 2 * (size_t)BC_PF * blocks_per_head * sizeof(half)  // sK_d, sV_d
                              + 2 * (size_t)BC_PF * head_dim * sizeof(int8_t);     // sK_q, sV_q
            cudaFuncSetAttribute(k_prefill_flash_q8_0,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_bytes);

            for (int it = 0; it < warmup; it++) {
                k_prefill_flash_q8_0<<<grid, threads, smem_bytes>>>(
                    d_qbuf, d_Kc_q8, d_Vc_q8, d_attbuf,
                    n, ctx, e_pos, n_heads, n_kv_heads, head_dim, scale, window);
            }
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            for (int it = 0; it < iters; it++) {
                k_prefill_flash_q8_0<<<grid, threads, smem_bytes>>>(
                    d_qbuf, d_Kc_q8, d_Vc_q8, d_attbuf,
                    n, ctx, e_pos, n_heads, n_kv_heads, head_dim, scale, window);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms_batched = 0;
            cudaEventElapsedTime(&ms_batched, start, stop);
            ms_batched /= iters;

            float *h_att_ser_d = (float*)malloc((size_t)n * n_heads * head_dim * sizeof(float));
            float *h_att_bat_d = (float*)malloc((size_t)n * n_heads * head_dim * sizeof(float));
            cudaMemcpy(h_att_ser_d, d_att_ser, (size_t)n * n_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_att_bat_d, d_attbuf, (size_t)n * n_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
            if (n == 32) {
                // Debug: show first few values
                for (int q = 0; q < std::min(n, 4); q++) {
                    for (int h = 0; h < std::min(n_heads, 4); h++) {
                        printf("  [q=%d h=%d] ser=[", q, h);
                        for (int e = 0; e < 4; e++) {
                            int idx = q*n_heads*head_dim + h*head_dim + 0*4 + e;
                            printf("%.3f ", h_att_ser_d[idx]);
                        }
                        printf("] bat=[");
                        for (int e = 0; e < 4; e++) {
                            int idx = q*n_heads*head_dim + h*head_dim + 0*4 + e;
                            printf("%.3f ", h_att_bat_d[idx]);
                        }
                        printf("]\n");
                    }
                }
                fflush(stdout);
            }
            float max_err = 0.0f;
            int naninf = 0;
            for (int q = 0; q < n; q++) {
                for (int h = 0; h < n_heads; h++) {
                    float qh_max_err = 0;
                    int worst_d = -1;
                    for (int d = 0; d < head_dim; d++) {
                        size_t idx = (size_t)q * n_heads * head_dim + (size_t)h * head_dim + d;
                        if (!isfinite(h_att_ser_d[idx]) || !isfinite(h_att_bat_d[idx])) naninf++;
                        float diff = fabsf(h_att_bat_d[idx] - h_att_ser_d[idx]);
                        if (diff > qh_max_err) { qh_max_err = diff; worst_d = d; }
                        if (diff > max_err) max_err = diff;
                    }
                    if (qh_max_err > 0.01f && n == 32 && q < 4) {
                        size_t idx = (size_t)q * n_heads * head_dim + (size_t)h * head_dim + worst_d;
                        printf("  [ERR n=%d q=%d h=%d worst_d=%d (ser=%.4f bat=%.4f diff=%.4f)]\n",
                               n, q, h, worst_d, h_att_ser_d[idx], h_att_bat_d[idx], qh_max_err);
                    }
                }
            }
            float speedup = ms_serial / ms_batched;
            const char *pass = (max_err < 1e-2f && naninf == 0) ? "PASS" : "FAIL";
            printf("%4d | %4d | %9.4f | %9.4f | %6.2fx | %.3e | %d | %s\n",
                   n, ctx, ms_serial, ms_batched, speedup, max_err, naninf, pass);
            free(h_q);
            free(h_att_ser_d);
            free(h_att_bat_d);
            fflush(stdout);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_qrow); cudaFree(d_att_ser); cudaFree(d_att_bat);
    cudaFree(d_Kc_q8); cudaFree(d_Vc_q8);
    cudaFree(d_pos); cudaFree(d_kst); cudaFree(d_vst);
    cudaFree(d_qbuf); cudaFree(d_attbuf);
    free(h_qrow); free(h_att_ser); free(h_att_bat);
    free(h_kst); free(h_vst);
    return 0;
}
