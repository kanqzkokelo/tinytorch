// Microbench: split-K FlashAttention-2 Q4_0 decode kernel vs serial references & FP32
// Model shape (Qwen2.5-0.5B): n_heads=14, n_kv_heads=2, head_dim=128
// Build:
//   nvcc -O3 -arch=native -Iinclude -Isrc -o build/micro_fa2_decode_q4 tools/micro_fa2_decode_q4.cu -L$HOME/mmcuda/lib -lcudart
// Run:
//   ./build/micro_fa2_decode_q4

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

#ifndef BLOCK_Q4_0_DEFINED
#define BLOCK_Q4_0_DEFINED
struct BlockQ4_0 {
    half d;          // 2 bytes FP16 scale
    uint8_t qs[16];  // 16 bytes = 32 nibbles
};
#endif

static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be 18 bytes");

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

// ---------------- 1. Reference Serial FP32 Kernel ----------------
__global__ void k_flash_gqa_fp32(
    const float *__restrict__ q,
    const float *__restrict__ Kc,
    const float *__restrict__ Vc,
    float       *__restrict__ out,
    const int   *__restrict__ d_pos,
    int n_heads, int n_kv_heads, int head_dim, int max_ctx,
    float scale, int window)
{
    const int pos = *d_pos;
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int lane = threadIdx.x;
    const int elems = head_dim / 32; // 4
    const int kvh = h / (n_heads / n_kv_heads);

    const float *qh = q + (long)h * head_dim + lane * elems;
    int t0 = 0;
    if (window > 0 && pos >= window) t0 = pos - window + 1;

    float qreg[4];
#pragma unroll
    for (int i = 0; i < 4; i++) qreg[i] = qh[i];

    float m_prev = -1e30f, l_prev = 0.0f;
    float oreg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int t = t0; t <= pos; t++) {
        long off = ((long)t * n_kv_heads + kvh) * head_dim + lane * elems;
        const float *kp = Kc + off;
        const float *vp = Vc + off;

        float score = qreg[0] * kp[0] + qreg[1] * kp[1] + qreg[2] * kp[2] + qreg[3] * kp[3];
        score = warp_sum(score);
        score = __shfl_sync(0xffffffff, score, 0) * scale;

        float m_curr = fmaxf(m_prev, score);
        float p = expf(score - m_curr);
        float alpha = expf(m_prev - m_curr);
        l_prev = l_prev * alpha + p;

        oreg[0] = oreg[0] * alpha + p * vp[0];
        oreg[1] = oreg[1] * alpha + p * vp[1];
        oreg[2] = oreg[2] * alpha + p * vp[2];
        oreg[3] = oreg[3] * alpha + p * vp[3];

        m_prev = m_curr;
    }

    float *outh = out + (long)h * head_dim + lane * elems;
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    outh[0] = oreg[0] * inv_l;
    outh[1] = oreg[1] * inv_l;
    outh[2] = oreg[2] * inv_l;
    outh[3] = oreg[3] * inv_l;
}

// ---------------- 2. Reference Serial Q4_0 Kernel ----------------
__global__ void k_flash_gqa_q4_0(
    const float     *__restrict__ q,
    const BlockQ4_0 *__restrict__ Kc_q4,
    const BlockQ4_0 *__restrict__ Vc_q4,
    float           *__restrict__ out,
    const int       *__restrict__ d_pos,
    int n_heads, int n_kv_heads, int head_dim, int max_ctx,
    float scale, int window)
{
    const int pos = *d_pos;
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int lane = threadIdx.x;
    const int elems = head_dim / 32; // 4
    const int kvh = h / (n_heads / n_kv_heads);
    const int blocks_per_head = head_dim / 32; // 4
    const int blocks_per_slot = n_kv_heads * blocks_per_head; // 8

    const float *qh = q + (long)h * head_dim + lane * elems;
    int t0 = 0;
    if (window > 0 && pos >= window) t0 = pos - window + 1;

    float qreg[4];
#pragma unroll
    for (int i = 0; i < 4; i++) qreg[i] = qh[i];

    float m_prev = -1e30f, l_prev = 0.0f;
    float oreg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int block_in_head = lane >> 3; // 0..3
    const int sub = lane & 7;            // 0..7
    const bool is_high = (sub >= 4);
    const int byte_offset = (sub & 3) * 4;
    const uint32_t shift = is_high ? 4 : 0;

    const int global_block_idx = kvh * blocks_per_head + block_in_head;

    for (int t = t0; t <= pos; t++) {
        long slot_idx = (long)t * blocks_per_slot + global_block_idx;
        const BlockQ4_0 *bk = &Kc_q4[slot_idx];
        const BlockQ4_0 *bv = &Vc_q4[slot_idx];

        float dk = __half2float(bk->d);
        float dv = __half2float(bv->d);

        // Read 4 bytes from qs
        const uint8_t *k_qs = &bk->qs[byte_offset];
        const uint8_t *v_qs = &bv->qs[byte_offset];

        uint32_t k_u32 = (uint32_t)k_qs[0] | ((uint32_t)k_qs[1] << 8) | ((uint32_t)k_qs[2] << 16) | ((uint32_t)k_qs[3] << 24);
        uint32_t v_u32 = (uint32_t)v_qs[0] | ((uint32_t)v_qs[1] << 8) | ((uint32_t)v_qs[2] << 16) | ((uint32_t)v_qs[3] << 24);

        uint32_t k_shifted = k_u32 >> shift;
        float k0 = (float)((int)((k_shifted      ) & 0x0F) - 8);
        float k1 = (float)((int)((k_shifted >>  8) & 0x0F) - 8);
        float k2 = (float)((int)((k_shifted >> 16) & 0x0F) - 8);
        float k3 = (float)((int)((k_shifted >> 24) & 0x0F) - 8);

        uint32_t v_shifted = v_u32 >> shift;
        float v0 = (float)((int)((v_shifted      ) & 0x0F) - 8);
        float v1 = (float)((int)((v_shifted >>  8) & 0x0F) - 8);
        float v2 = (float)((int)((v_shifted >> 16) & 0x0F) - 8);
        float v3 = (float)((int)((v_shifted >> 24) & 0x0F) - 8);

        float dot_partial = (qreg[0] * k0 + qreg[1] * k1 + qreg[2] * k2 + qreg[3] * k3) * dk;
        float score = warp_sum(dot_partial);
        score = __shfl_sync(0xffffffff, score, 0) * scale;

        float m_curr = fmaxf(m_prev, score);
        float p = expf(score - m_curr);
        float alpha = expf(m_prev - m_curr);
        l_prev = l_prev * alpha + p;

        float pdv = p * dv;
        oreg[0] = oreg[0] * alpha + pdv * v0;
        oreg[1] = oreg[1] * alpha + pdv * v1;
        oreg[2] = oreg[2] * alpha + pdv * v2;
        oreg[3] = oreg[3] * alpha + pdv * v3;

        m_prev = m_curr;
    }

    float *outh = out + (long)h * head_dim + lane * elems;
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    outh[0] = oreg[0] * inv_l;
    outh[1] = oreg[1] * inv_l;
    outh[2] = oreg[2] * inv_l;
    outh[3] = oreg[3] * inv_l;
}

// ---------------- 3. KV Scatter Kernel: k_kv_scatter_q4_0 ----------------
// Quantizes incoming FP32 K/V staging vectors into BlockQ4_0 on the fly
__global__ void k_kv_scatter_q4_0(
    const float *__restrict__ kst,
    const float *__restrict__ vst,
    BlockQ4_0   *__restrict__ Kc_q4,
    BlockQ4_0   *__restrict__ Vc_q4,
    const int   *__restrict__ d_pos,
    int n_kv_heads, int head_dim, int max_ctx)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blocks_per_slot = (n_kv_heads * head_dim) / 32; // e.g. 2 * 128 / 32 = 8
    if (block_idx >= blocks_per_slot) return;

    int slot = (*d_pos) % max_ctx;
    int src_offset = block_idx * 32;

    float k_vals[32], v_vals[32];
    float max_k = 0.0f, max_v = 0.0f;

#pragma unroll
    for (int i = 0; i < 32; i++) {
        k_vals[i] = kst[src_offset + i];
        v_vals[i] = vst[src_offset + i];
        max_k = fmaxf(max_k, fabsf(k_vals[i]));
        max_v = fmaxf(max_v, fabsf(v_vals[i]));
    }

    // Scale mapping: max_val mapped to 7 (range [-8, 7] -> [0, 15] nibble)
    float scale_k = (max_k > 0.0f) ? (max_k / 7.0f) : 1.0f;
    float inv_k   = (max_k > 0.0f) ? (7.0f / max_k) : 0.0f;

    float scale_v = (max_v > 0.0f) ? (max_v / 7.0f) : 1.0f;
    float inv_v   = (max_v > 0.0f) ? (7.0f / max_v) : 0.0f;

    BlockQ4_0 *kd = Kc_q4 + (long)slot * blocks_per_slot + block_idx;
    BlockQ4_0 *vd = Vc_q4 + (long)slot * blocks_per_slot + block_idx;

    kd->d = __float2half(scale_k);
    vd->d = __float2half(scale_v);

#pragma unroll
    for (int j = 0; j < 16; j++) {
        int q0_k = __float2int_rn(k_vals[j] * inv_k) + 8;
        int q1_k = __float2int_rn(k_vals[j + 16] * inv_k) + 8;
        q0_k = max(0, min(15, q0_k));
        q1_k = max(0, min(15, q1_k));
        kd->qs[j] = (uint8_t)((q0_k & 0x0F) | ((q1_k & 0x0F) << 4));

        int q0_v = __float2int_rn(v_vals[j] * inv_v) + 8;
        int q1_v = __float2int_rn(v_vals[j + 16] * inv_v) + 8;
        q0_v = max(0, min(15, q0_v));
        q1_v = max(0, min(15, q1_v));
        vd->qs[j] = (uint8_t)((q0_v & 0x0F) | ((q1_v & 0x0F) << 4));
    }
}

// FP32 Scatter Kernel for reference benchmark
__global__ void k_kv_scatter_fp32(
    const float *__restrict__ kst,
    const float *__restrict__ vst,
    float       *__restrict__ Kc,
    float       *__restrict__ Vc,
    const int   *__restrict__ d_pos,
    int n_kv_heads, int head_dim, int max_ctx)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int kvdim = n_kv_heads * head_dim;
    if (i >= kvdim) return;

    int slot = (*d_pos) % max_ctx;
    Kc[(long)slot * kvdim + i] = kst[i];
    Vc[(long)slot * kvdim + i] = vst[i];
}

// ---------------- 4. Optimized Split-K FlashAttention-2 Q4_0 Kernel ----------------
#define FA2_DECODE_BC 64

__global__ void k_fa2_q4_split(
    const float     *__restrict__ q,
    const BlockQ4_0 *__restrict__ Kc_q4,
    const BlockQ4_0 *__restrict__ Vc_q4,
    float           *__restrict__ p_acc,
    float           *__restrict__ p_m,
    float           *__restrict__ p_l,
    const int       *__restrict__ d_pos,
    int n_heads, int n_kv_heads, int head_dim,
    float scale, int window, int S)
{
    const int pos = *d_pos;
    const int s = blockIdx.x;
    const int kv = blockIdx.y;
    if (s >= S || kv >= n_kv_heads) return;

    const int G = n_heads / n_kv_heads; // 7
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (warp >= G) return;

    const int head = kv * G + warp;

    // Load Q vector directly into registers for this head & lane (16-byte aligned float4)
    const int elems = head_dim / 32; // 4
    const float4 q_vec = *(const float4 *)(q + (long)head * head_dim + lane * 4);
    float q0 = q_vec.x;
    float q1 = q_vec.y;
    float q2 = q_vec.z;
    float q3 = q_vec.w;

    // Sliding window & slice bounds
    int t_lo = (window > 0 && pos >= window) ? (pos - window + 1) : 0;
    int nslots = pos - t_lo + 1;
    int chunk = (nslots + S - 1) / S;
    int begin = t_lo + s * chunk;
    int end = min(pos + 1, t_lo + (s + 1) * chunk);

    // Inactive slice early exit
    if (begin >= end) {
        if (lane == 0) {
            p_m[(size_t)s * n_heads + head] = -1e30f;
            p_l[(size_t)s * n_heads + head] = 0.0f;
        }
        float *myacc = p_acc + ((size_t)s * n_heads + head) * head_dim + lane * elems;
        myacc[0] = 0.0f;
        myacc[1] = 0.0f;
        myacc[2] = 0.0f;
        myacc[3] = 0.0f;
        return;
    }

    const int blocks_per_head = head_dim / 32; // 4

    // Shared memory layout:
    // BC=64 tokens
    // sK_d, sV_d: [BC, 4] FP16 scales (512 bytes each)
    // sK_q, sV_q: [BC, 64] uint8_t quantized nibble bytes (4096 bytes each)
    // Total smem = 512 * 2 + 4096 * 2 = 9,216 bytes
    __shared__ half    sK_d[FA2_DECODE_BC * 4];
    __shared__ half    sV_d[FA2_DECODE_BC * 4];
    __shared__ uint8_t sK_q[FA2_DECODE_BC * 64];
    __shared__ uint8_t sV_q[FA2_DECODE_BC * 64];

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int block_in_head = lane >> 3; // 0..3
    const int sub = lane & 7;            // 0..7
    const bool is_high = (sub >= 4);
    const int byte_offset = (sub & 3) * 4; // 0, 4, 8, 12 in block's 16 bytes
    const uint32_t shift = is_high ? 4 : 0;

    // Iterate over tokens in [begin, end) in tiles of BC=64
    for (int t_tile = begin; t_tile < end; t_tile += FA2_DECODE_BC) {
        int t_tile_end = min(end, t_tile + FA2_DECODE_BC);
        int bc_active = t_tile_end - t_tile;

        // Cooperative load of bc_active KV blocks into smem
        int total_blocks = bc_active * blocks_per_head;
        for (int i = tid; i < total_blocks; i += blockDim.x) {
            int tok = i >> 2;
            int b   = i & 3;
            long g_idx = ((long)(t_tile + tok) * n_kv_heads + kv) * 4 + b;
            const BlockQ4_0 bk = Kc_q4[g_idx];
            const BlockQ4_0 bv = Vc_q4[g_idx];
            sK_d[tok * 4 + b] = bk.d;
            sV_d[tok * 4 + b] = bv.d;

            int row_off = tok * 64 + b * 16;
            // 16 bytes of qs copied safely
#pragma unroll
            for (int j = 0; j < 8; j++) {
                ((uint16_t *)&sK_q[row_off])[j] = ((const uint16_t *)&bk.qs[0])[j];
                ((uint16_t *)&sV_q[row_off])[j] = ((const uint16_t *)&bv.qs[0])[j];
            }
        }
        __syncthreads();

        // Process tokens in smem tile
        for (int t_idx = 0; t_idx < bc_active; t_idx++) {
            const float dk = __half2float(sK_d[t_idx * 4 + block_in_head]);
            const float dv = __half2float(sV_d[t_idx * 4 + block_in_head]);

            // 32-bit conflict-free smem read (4 packed bytes per lane)
            const uint32_t k_u32 = *(const uint32_t *)&sK_q[t_idx * 64 + block_in_head * 16 + byte_offset];
            const uint32_t v_u32 = *(const uint32_t *)&sV_q[t_idx * 64 + block_in_head * 16 + byte_offset];

            uint32_t k_shifted = k_u32 >> shift;
            const float k0 = (float)((int)((k_shifted      ) & 0x0F) - 8);
            const float k1 = (float)((int)((k_shifted >>  8) & 0x0F) - 8);
            const float k2 = (float)((int)((k_shifted >> 16) & 0x0F) - 8);
            const float k3 = (float)((int)((k_shifted >> 24) & 0x0F) - 8);

            uint32_t v_shifted = v_u32 >> shift;
            const float v0 = (float)((int)((v_shifted      ) & 0x0F) - 8);
            const float v1 = (float)((int)((v_shifted >>  8) & 0x0F) - 8);
            const float v2 = (float)((int)((v_shifted >> 16) & 0x0F) - 8);
            const float v3 = (float)((int)((v_shifted >> 24) & 0x0F) - 8);

            float dot_partial = (q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3) * dk;
            float score = warp_sum(dot_partial);
            score = __shfl_sync(0xffffffff, score, 0) * scale;

            float m_curr = fmaxf(m_prev, score);
            float p = expf(score - m_curr);
            float alpha = expf(m_prev - m_curr);
            l_prev = l_prev * alpha + p;

            float pdv = p * dv;
            acc[0] = acc[0] * alpha + pdv * v0;
            acc[1] = acc[1] * alpha + pdv * v1;
            acc[2] = acc[2] * alpha + pdv * v2;
            acc[3] = acc[3] * alpha + pdv * v3;

            m_prev = m_curr;
        }
        __syncthreads();
    }

    // Write partials
    if (lane == 0) {
        p_m[(size_t)s * n_heads + head] = m_prev;
        p_l[(size_t)s * n_heads + head] = l_prev;
    }
    float *myacc = p_acc + ((size_t)s * n_heads + head) * head_dim + lane * elems;
    myacc[0] = acc[0];
    myacc[1] = acc[1];
    myacc[2] = acc[2];
    myacc[3] = acc[3];
}

// ---------------- 5. Combine Kernel: k_fa2_combine ----------------
__global__ void k_fa2_combine(
    const float *__restrict__ p_acc,
    const float *__restrict__ p_m,
    const float *__restrict__ p_l,
    float       *__restrict__ out,
    int n_heads, int head_dim, int S)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int lane = threadIdx.x;

    float m_global = -1e30f, l_global = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = 0; s < S; s++) {
        float m_s = p_m[(size_t)s * n_heads + h];
        float l_s = p_l[(size_t)s * n_heads + h];
        if (!isfinite(m_s) || !isfinite(l_s) || l_s <= 0.0f || m_s <= -1e20f) continue;
        float m_new = fmaxf(m_global, m_s);
        float alpha_prev = expf(m_global - m_new);
        float alpha_s    = expf(m_s - m_new);
        l_global = l_global * alpha_prev + l_s * alpha_s;
        const float4 p_vec = *(const float4 *)(p_acc + ((size_t)s * n_heads + h) * head_dim + lane * 4);
        acc[0] = acc[0] * alpha_prev + p_vec.x * alpha_s;
        acc[1] = acc[1] * alpha_prev + p_vec.y * alpha_s;
        acc[2] = acc[2] * alpha_prev + p_vec.z * alpha_s;
        acc[3] = acc[3] * alpha_prev + p_vec.w * alpha_s;
        m_global = m_new;
    }

    float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
    *(float4 *)(out + (long)h * head_dim + lane * 4) = make_float4(
        acc[0] * inv_l,
        acc[1] * inv_l,
        acc[2] * inv_l,
        acc[3] * inv_l
    );
}

// ---------------- Benchmark & Verification Harness ----------------
static float frand() {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

int main() {
    const int n_heads = 14;
    const int n_kv_heads = 2;
    const int head_dim = 128;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const int window = 0;
    const int max_ctx = 8192;
    const std::vector<int> Ns = {32, 128, 512, 1024, 2048, 4096, 8192};

    printf("=== Decode FlashAttention-2 Q4_0 Split-K Microbenchmark ===\n");
    printf("Config: n_heads=%d, n_kv_heads=%d, head_dim=%d, BC=%d, scale=%.6f\n\n",
           n_heads, n_kv_heads, head_dim, FA2_DECODE_BC, scale);

    const size_t q_bytes = (size_t)n_heads * head_dim * sizeof(float);
    const size_t out_bytes = q_bytes;
    const int blocks_per_head = head_dim / 32; // 4
    const int blocks_per_slot = n_kv_heads * blocks_per_head; // 8
    const size_t q4_cache_bytes = (size_t)max_ctx * blocks_per_slot * sizeof(BlockQ4_0);
    const size_t fp32_cache_bytes = (size_t)max_ctx * n_kv_heads * head_dim * sizeof(float);

    // Host buffers
    float *h_q = (float *)malloc(q_bytes);
    float *h_out_fp32 = (float *)malloc(out_bytes);
    float *h_out_serial_q4 = (float *)malloc(out_bytes);
    float *h_out_fa2_q4 = (float *)malloc(out_bytes);

    srand(42);
    for (size_t i = 0; i < (size_t)n_heads * head_dim; i++) {
        h_q[i] = frand() * 0.5f;
    }

    // Device buffers
    float *d_q, *d_out_fp32, *d_out_serial_q4, *d_out_fa2_q4;
    float *d_Kc_fp32, *d_Vc_fp32;
    BlockQ4_0 *d_Kc_q4, *d_Vc_q4;
    float *d_kst, *d_vst;
    int *d_pos;

    cudaMalloc(&d_q, q_bytes);
    cudaMalloc(&d_out_fp32, out_bytes);
    cudaMalloc(&d_out_serial_q4, out_bytes);
    cudaMalloc(&d_out_fa2_q4, out_bytes);
    cudaMalloc(&d_Kc_fp32, fp32_cache_bytes);
    cudaMalloc(&d_Vc_fp32, fp32_cache_bytes);
    cudaMalloc(&d_Kc_q4, q4_cache_bytes);
    cudaMalloc(&d_Vc_q4, q4_cache_bytes);
    cudaMalloc(&d_kst, (size_t)n_kv_heads * head_dim * sizeof(float));
    cudaMalloc(&d_vst, (size_t)n_kv_heads * head_dim * sizeof(float));
    cudaMalloc(&d_pos, sizeof(int));

    cudaMemcpy(d_q, h_q, q_bytes, cudaMemcpyHostToDevice);

    // Split-K workspace (S_max = 64)
    const int S_max = 64;
    const size_t pacc_bytes = (size_t)S_max * n_heads * head_dim * sizeof(float);
    const size_t pm_bytes = (size_t)S_max * n_heads * sizeof(float);
    float *d_pacc, *d_pm, *d_pl;
    cudaMalloc(&d_pacc, pacc_bytes);
    cudaMalloc(&d_pm, pm_bytes);
    cudaMalloc(&d_pl, pm_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Populate KV cache across max_ctx using k_kv_scatter_q4_0 and k_kv_scatter_fp32
    printf("Populating synthetic KV cache (%d tokens) via k_kv_scatter_q4_0...\n", max_ctx);
    std::vector<float> h_kst(n_kv_heads * head_dim);
    std::vector<float> h_vst(n_kv_heads * head_dim);
    for (int t = 0; t < max_ctx; t++) {
        for (int i = 0; i < n_kv_heads * head_dim; i++) {
            h_kst[i] = frand() * 0.4f;
            h_vst[i] = frand() * 0.4f;
        }
        cudaMemcpy(d_kst, h_kst.data(), h_kst.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vst, h_vst.data(), h_vst.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos, &t, sizeof(int), cudaMemcpyHostToDevice);

        k_kv_scatter_fp32<<<((n_kv_heads * head_dim) + 255) / 256, 256>>>(
            d_kst, d_vst, d_Kc_fp32, d_Vc_fp32, d_pos, n_kv_heads, head_dim, max_ctx);
        k_kv_scatter_q4_0<<<(blocks_per_slot + 31) / 32, 32>>>(
            d_kst, d_vst, d_Kc_q4, d_Vc_q4, d_pos, n_kv_heads, head_dim, max_ctx);
    }
    cudaDeviceSynchronize();
    printf("KV cache population complete.\n\n");

    printf("%-6s | %-13s | %-13s | %-13s | %-10s | %-11s | %-11s | %-8s | %-6s\n",
           "N", "FP32 Ref(ms)", "Q4 Serial(ms)", "FA2 Q4(ms)", "Speedup", "Err vs Q4", "Err vs FP32", "NaN/Inf", "Status");
    printf("-------+---------------+---------------+---------------+------------+-------------+-------------+----------+--------\n");

    bool all_passed = true;

    for (int N : Ns) {
        int pos_val = N - 1;
        cudaMemcpy(d_pos, &pos_val, sizeof(int), cudaMemcpyHostToDevice);

        // Dynamic split-K S: S in [1, S_max]
        int S = std::max(1, std::min(S_max, (pos_val + 1 + FA2_DECODE_BC - 1) / FA2_DECODE_BC));
        dim3 grid_split(S, n_kv_heads);
        const int G = n_heads / n_kv_heads;
        const int threads_split = G * 32; // 224

        // Correctness verification runs
        k_flash_gqa_fp32<<<n_heads, 32>>>(
            d_q, d_Kc_fp32, d_Vc_fp32, d_out_fp32, d_pos,
            n_heads, n_kv_heads, head_dim, max_ctx, scale, window);

        k_flash_gqa_q4_0<<<n_heads, 32>>>(
            d_q, d_Kc_q4, d_Vc_q4, d_out_serial_q4, d_pos,
            n_heads, n_kv_heads, head_dim, max_ctx, scale, window);

        k_fa2_q4_split<<<grid_split, threads_split>>>(
            d_q, d_Kc_q4, d_Vc_q4, d_pacc, d_pm, d_pl,
            d_pos, n_heads, n_kv_heads, head_dim, scale, window, S);
        k_fa2_combine<<<n_heads, 32>>>(
            d_pacc, d_pm, d_pl, d_out_fa2_q4, n_heads, head_dim, S);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA ERROR at N=%d: %s\n", N, cudaGetErrorString(err));
            all_passed = false;
        }

        cudaMemcpy(h_out_fp32, d_out_fp32, out_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_serial_q4, d_out_serial_q4, out_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_fa2_q4, d_out_fa2_q4, out_bytes, cudaMemcpyDeviceToHost);

        float max_abs_err_q4 = 0.0f;
        float max_abs_err_fp32 = 0.0f;
        int naninf_count = 0;

        for (size_t i = 0; i < (size_t)n_heads * head_dim; i++) {
            if (!std::isfinite(h_out_fa2_q4[i]) || !std::isfinite(h_out_serial_q4[i]) || !std::isfinite(h_out_fp32[i])) {
                naninf_count++;
            }
            float err_q4 = std::fabs(h_out_fa2_q4[i] - h_out_serial_q4[i]);
            if (err_q4 > max_abs_err_q4) max_abs_err_q4 = err_q4;

            float err_fp32 = std::fabs(h_out_fa2_q4[i] - h_out_fp32[i]);
            if (err_fp32 > max_abs_err_fp32) max_abs_err_fp32 = err_fp32;
        }

        // Benchmark timing
        const int warmup = 20;
        const int iters = 100;

        // 1. FP32 Ref Timing
        for (int i = 0; i < warmup; i++) {
            k_flash_gqa_fp32<<<n_heads, 32>>>(
                d_q, d_Kc_fp32, d_Vc_fp32, d_out_fp32, d_pos,
                n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) {
            k_flash_gqa_fp32<<<n_heads, 32>>>(
                d_q, d_Kc_fp32, d_Vc_fp32, d_out_fp32, d_pos,
                n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_fp32 = 0.0f;
        cudaEventElapsedTime(&ms_fp32, start, stop);
        ms_fp32 /= iters;

        // 2. Q4_0 Serial Ref Timing
        for (int i = 0; i < warmup; i++) {
            k_flash_gqa_q4_0<<<n_heads, 32>>>(
                d_q, d_Kc_q4, d_Vc_q4, d_out_serial_q4, d_pos,
                n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) {
            k_flash_gqa_q4_0<<<n_heads, 32>>>(
                d_q, d_Kc_q4, d_Vc_q4, d_out_serial_q4, d_pos,
                n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_serial_q4 = 0.0f;
        cudaEventElapsedTime(&ms_serial_q4, start, stop);
        ms_serial_q4 /= iters;

        // 3. FA2 Q4_0 Split-K Timing
        for (int i = 0; i < warmup; i++) {
            k_fa2_q4_split<<<grid_split, threads_split>>>(
                d_q, d_Kc_q4, d_Vc_q4, d_pacc, d_pm, d_pl,
                d_pos, n_heads, n_kv_heads, head_dim, scale, window, S);
            k_fa2_combine<<<n_heads, 32>>>(
                d_pacc, d_pm, d_pl, d_out_fa2_q4, n_heads, head_dim, S);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) {
            k_fa2_q4_split<<<grid_split, threads_split>>>(
                d_q, d_Kc_q4, d_Vc_q4, d_pacc, d_pm, d_pl,
                d_pos, n_heads, n_kv_heads, head_dim, scale, window, S);
            k_fa2_combine<<<n_heads, 32>>>(
                d_pacc, d_pm, d_pl, d_out_fa2_q4, n_heads, head_dim, S);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_fa2_q4 = 0.0f;
        cudaEventElapsedTime(&ms_fa2_q4, start, stop);
        ms_fa2_q4 /= iters;

        float speedup_vs_serial = ms_serial_q4 / ms_fa2_q4;
        bool pass = (max_abs_err_q4 < 1e-4f) && (naninf_count == 0) && (max_abs_err_fp32 < 0.05f);
        if (!pass) all_passed = false;

        printf("%-6d | %-13.5f | %-13.5f | %-13.5f | %-9.2fx | %-11.3e | %-11.3e | %-8d | %-6s\n",
               N, ms_fp32, ms_serial_q4, ms_fa2_q4, speedup_vs_serial, max_abs_err_q4, max_abs_err_fp32, naninf_count, pass ? "PASS" : "FAIL");
    }

    printf("-------------------------------------------------------------------------------------------------------------------\n");
    printf("Overall Result: %s\n", all_passed ? "ALL PASSED" : "FAILED");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_q);
    cudaFree(d_out_fp32);
    cudaFree(d_out_serial_q4);
    cudaFree(d_out_fa2_q4);
    cudaFree(d_Kc_fp32);
    cudaFree(d_Vc_fp32);
    cudaFree(d_Kc_q4);
    cudaFree(d_Vc_q4);
    cudaFree(d_kst);
    cudaFree(d_vst);
    cudaFree(d_pos);
    cudaFree(d_pacc);
    cudaFree(d_pm);
    cudaFree(d_pl);
    free(h_q);
    free(h_out_fp32);
    free(h_out_serial_q4);
    free(h_out_fa2_q4);

    return all_passed ? 0 : 1;
}
