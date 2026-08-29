// Microbench: split-K FlashAttention-2 Q8_0 decode kernel vs serial reference
// Model shape (Qwen2.5-0.5B): n_heads=14, n_kv_heads=2, head_dim=128
// Build:
//   nvcc -O3 -arch=native --resource-usage -Iinclude -Isrc -o build/micro_fa2_decode_q8 tools/micro_fa2_decode_q8.cu -L$HOME/mmcuda/lib -lcudart
// Run:
//   ./build/micro_fa2_decode_q8

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
    for (int off = 16; off > 0; off /= 2) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

// ---------------- Reference Serial Kernel (k_flash_gqa_q8_0) ----------------
__global__ void k_flash_gqa_q8_0(
    const float     *__restrict__ q,
    const BlockQ8_0 *__restrict__ Kc_q8,
    const BlockQ8_0 *__restrict__ Vc_q8,
    float           *__restrict__ out,
    const int       *__restrict__ d_pos,
    int n_heads, int n_kv_heads, int head_dim, int max_ctx,
    float scale, int window)
{
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
    for (int i = 0; i < 16; i++) {
        qreg[i] = (i < elems) ? qh[i] : 0.0f;
    }

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
        k_ptr += stride;
        v_ptr += stride;

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

        m_prev = m_curr;
        l_prev = l_curr;
    }

    float *outh = out + (long)h * head_dim + lane * elems;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        if (i < elems) outh[i] = oreg[i] / l_prev;
    }
}

// ---------------- 1. Split Kernel: k_fa2_q8_split ----------------
#define FA2_DECODE_BC 64

__global__ void k_fa2_q8_split(
    const float     *__restrict__ q,
    const BlockQ8_0 *__restrict__ Kc_q8,
    const BlockQ8_0 *__restrict__ Vc_q8,
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

    const int G = n_heads / n_kv_heads;
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

    // Shared memory: BC=64 KV tokens
    // sK_d, sV_d: [BC, blocks_per_head] FP16 scales (512 bytes each)
    // sK_q, sV_q: [BC, head_dim] int8 quantized (8192 bytes each)
    // Total = 17,408 bytes
    __shared__ half   sK_d[FA2_DECODE_BC * 4];
    __shared__ half   sV_d[FA2_DECODE_BC * 4];
    __shared__ int8_t sK_q[FA2_DECODE_BC * 128];
    __shared__ int8_t sV_q[FA2_DECODE_BC * 128];

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int block_in_head = lane >> 3;     // 0..3

    // Iterate over tokens in [begin, end) in tiles of BC
    for (int t_tile = begin; t_tile < end; t_tile += FA2_DECODE_BC) {
        int t_tile_end = min(end, t_tile + FA2_DECODE_BC);
        int bc_active = t_tile_end - t_tile;

        // Cooperative load of bc_active KV blocks into smem
        int total_blocks = bc_active * blocks_per_head;
        for (int i = tid; i < total_blocks; i += blockDim.x) {
            int tok = i >> 2;
            int b   = i & 3;
            long g_idx = ((long)(t_tile + tok) * n_kv_heads + kv) * 4 + b;
            const BlockQ8_0 bk = Kc_q8[g_idx];
            const BlockQ8_0 bv = Vc_q8[g_idx];
            sK_d[tok * 4 + b] = bk.d;
            sV_d[tok * 4 + b] = bv.d;
            int row_off = tok * 128 + b * 32;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                ((uint32_t *)&sK_q[row_off])[j] = ((const uint32_t *)&bk.qs[0])[j];
                ((uint32_t *)&sV_q[row_off])[j] = ((const uint32_t *)&bv.qs[0])[j];
            }
        }
        __syncthreads();
        // Process tokens in smem tile
        for (int t_idx = 0; t_idx < bc_active; t_idx++) {
            const float dk = __half2float(sK_d[t_idx * 4 + block_in_head]);
            const float dv = __half2float(sV_d[t_idx * 4 + block_in_head]);

            // 32-bit conflict-free smem read (4 int8s per lane)
            const uint32_t k_u32 = *(const uint32_t *)&sK_q[t_idx * 128 + lane * 4];
            const uint32_t v_u32 = *(const uint32_t *)&sV_q[t_idx * 128 + lane * 4];

            const float k0 = (float)((int8_t)(k_u32      ));
            const float k1 = (float)((int8_t)(k_u32 >>  8));
            const float k2 = (float)((int8_t)(k_u32 >> 16));
            const float k3 = (float)((int8_t)(k_u32 >> 24));

            const float v0 = (float)((int8_t)(v_u32      ));
            const float v1 = (float)((int8_t)(v_u32 >>  8));
            const float v2 = (float)((int8_t)(v_u32 >> 16));
            const float v3 = (float)((int8_t)(v_u32 >> 24));

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

// ---------------- 2. Combine Kernel: k_fa2_combine ----------------
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
        float m_s = p_m[s * n_heads + h];
        float l_s = p_l[s * n_heads + h];
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

// ---------------- KV Cache Fast Initializer ----------------
__global__ void k_init_kv_q8(BlockQ8_0 *Kc, BlockQ8_0 *Vc, int N, int n_kv_heads, int head_dim, int max_ctx) {
    int tok = blockIdx.y;
    if (tok >= N) return;
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks_per_slot = (n_kv_heads * head_dim) / 32;
    if (block_idx >= num_blocks_per_slot) return;
    int slot = tok % max_ctx;
    BlockQ8_0 *kd = Kc + (long)slot * num_blocks_per_slot + block_idx;
    BlockQ8_0 *vd = Vc + (long)slot * num_blocks_per_slot + block_idx;
    kd->d = __float2half(0.04f + (float)((tok + block_idx * 3) % 11) * 0.005f);
    vd->d = __float2half(0.04f + (float)((tok * 5 + block_idx) % 11) * 0.005f);
    for (int i = 0; i < 32; i++) {
        kd->qs[i] = (int8_t)(((tok * 19 + block_idx * 37 + i * 11) % 251) - 125);
        vd->qs[i] = (int8_t)(((tok * 29 + block_idx * 17 + i * 13) % 251) - 125);
    }
}

// ---------------- Benchmarking Main ----------------
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

    printf("=== Decode FlashAttention-2 Q8_0 Split-K Microbenchmark ===\n");
    printf("Config: n_heads=%d, n_kv_heads=%d, head_dim=%d, BC=%d, scale=%.6f\n\n",
           n_heads, n_kv_heads, head_dim, FA2_DECODE_BC, scale);

    const size_t q_bytes = (size_t)n_heads * head_dim * sizeof(float);
    const size_t out_bytes = q_bytes;
    const int blocks_per_head = head_dim / 32;
    const int blocks_per_slot = n_kv_heads * blocks_per_head;
    const size_t q8_cache_bytes = (size_t)max_ctx * blocks_per_slot * sizeof(BlockQ8_0);

    // Host buffers
    float *h_q = (float *)malloc(q_bytes);
    float *h_out_serial = (float *)malloc(out_bytes);
    float *h_out_fa2 = (float *)malloc(out_bytes);
    srand(42);
    for (size_t i = 0; i < (size_t)n_heads * head_dim; i++) {
        h_q[i] = frand() * 0.5f;
    }

    // Device buffers
    float *d_q, *d_out_serial, *d_out_fa2;
    BlockQ8_0 *d_Kc_q8, *d_Vc_q8;
    int *d_pos;
    cudaMalloc(&d_q, q_bytes);
    cudaMalloc(&d_out_serial, out_bytes);
    cudaMalloc(&d_out_fa2, out_bytes);
    cudaMalloc(&d_Kc_q8, q8_cache_bytes);
    cudaMalloc(&d_Vc_q8, q8_cache_bytes);
    cudaMalloc(&d_pos, sizeof(int));
    cudaMemcpy(d_q, h_q, q_bytes, cudaMemcpyHostToDevice);

    // Split-K workspace (S_max = 32)
    const int S_max = 32;
    const size_t pacc_bytes = (size_t)S_max * n_heads * head_dim * sizeof(float);
    const size_t pm_bytes = (size_t)S_max * n_heads * sizeof(float);
    float *d_pacc, *d_pm, *d_pl;
    cudaMalloc(&d_pacc, pacc_bytes);
    cudaMalloc(&d_pm, pm_bytes);
    cudaMalloc(&d_pl, pm_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("%-6s | %-16s | %-16s | %-10s | %-13s | %-8s | %-6s\n",
           "N", "Serial (ms/lay)", "FA2 (ms/lay)", "Speedup", "Max Abs Err", "NaN/Inf", "Status");
    printf("-------+------------------+------------------+------------+---------------+----------+--------\n");

    bool all_passed = true;

    for (int N : Ns) {
        int pos_val = N - 1;
        cudaMemcpy(d_pos, &pos_val, sizeof(int), cudaMemcpyHostToDevice);

        // Initialize KV cache up to N
        dim3 grid_init((blocks_per_slot + 31) / 32, N);
        k_init_kv_q8<<<grid_init, 32>>>(d_Kc_q8, d_Vc_q8, N, n_kv_heads, head_dim, max_ctx);
        cudaDeviceSynchronize();

        // Calculate S
        int S = std::max(1, std::min(S_max, (pos_val + 1 + 31) / 32));
        dim3 grid_split(S, n_kv_heads);
        const int G = n_heads / n_kv_heads;
        const int threads_split = G * 32; // 224

        // Correctness runs
        k_flash_gqa_q8_0<<<n_heads, 32>>>(
            d_q, d_Kc_q8, d_Vc_q8, d_out_serial, d_pos,
            n_heads, n_kv_heads, head_dim, max_ctx, scale, window);

        k_fa2_q8_split<<<grid_split, threads_split>>>(
            d_q, d_Kc_q8, d_Vc_q8, d_pacc, d_pm, d_pl,
            d_pos, n_heads, n_kv_heads, head_dim, scale, window, S);
        k_fa2_combine<<<n_heads, 32>>>(
            d_pacc, d_pm, d_pl, d_out_fa2, n_heads, head_dim, S);
        cudaError_t err1 = cudaGetLastError();
        cudaDeviceSynchronize();
        cudaError_t err2 = cudaGetLastError();
        if (err1 != cudaSuccess || err2 != cudaSuccess) {
            printf("CUDA ERROR at N=%d: %s / %s\n", N, cudaGetErrorString(err1), cudaGetErrorString(err2));
        }
        cudaMemcpy(h_out_serial, d_out_serial, out_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_fa2, d_out_fa2, out_bytes, cudaMemcpyDeviceToHost);

        float max_abs_err = 0.0f;
        int naninf_count = 0;
        for (size_t i = 0; i < (size_t)n_heads * head_dim; i++) {
            if (!std::isfinite(h_out_serial[i]) || !std::isfinite(h_out_fa2[i])) {
                naninf_count++;
            }
            float err = std::fabs(h_out_fa2[i] - h_out_serial[i]);
            if (err > max_abs_err) {
                max_abs_err = err;
            }
        }

        // Benchmark Serial
        const int warmup = 20;
        const int iters = 100;

        for (int i = 0; i < warmup; i++) {
            k_flash_gqa_q8_0<<<n_heads, 32>>>(
                d_q, d_Kc_q8, d_Vc_q8, d_out_serial, d_pos,
                n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) {
            k_flash_gqa_q8_0<<<n_heads, 32>>>(
                d_q, d_Kc_q8, d_Vc_q8, d_out_serial, d_pos,
                n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_serial = 0.0f;
        cudaEventElapsedTime(&ms_serial, start, stop);
        ms_serial /= iters;

        // Benchmark FA2 Split-K
        for (int i = 0; i < warmup; i++) {
            k_fa2_q8_split<<<grid_split, threads_split>>>(
                d_q, d_Kc_q8, d_Vc_q8, d_pacc, d_pm, d_pl,
                d_pos, n_heads, n_kv_heads, head_dim, scale, window, S);
            k_fa2_combine<<<n_heads, 32>>>(
                d_pacc, d_pm, d_pl, d_out_fa2, n_heads, head_dim, S);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) {
            k_fa2_q8_split<<<grid_split, threads_split>>>(
                d_q, d_Kc_q8, d_Vc_q8, d_pacc, d_pm, d_pl,
                d_pos, n_heads, n_kv_heads, head_dim, scale, window, S);
            k_fa2_combine<<<n_heads, 32>>>(
                d_pacc, d_pm, d_pl, d_out_fa2, n_heads, head_dim, S);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_fa2 = 0.0f;
        cudaEventElapsedTime(&ms_fa2, start, stop);
        ms_fa2 /= iters;

        float speedup = ms_serial / ms_fa2;
        bool pass = (max_abs_err < 1e-4f) && (naninf_count == 0);
        if (!pass) all_passed = false;

        printf("%-6d | %-16.5f | %-16.5f | %-9.2fx | %-13.3e | %-8d | %-6s\n",
               N, ms_serial, ms_fa2, speedup, max_abs_err, naninf_count, pass ? "PASS" : "FAIL");
    }
    printf("---------------------------------------------------------------------------------------\n");
    printf("Overall Result: %s\n", all_passed ? "ALL PASSED" : "FAILED");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_q);
    cudaFree(d_out_serial);
    cudaFree(d_out_fa2);
    cudaFree(d_Kc_q8);
    cudaFree(d_Vc_q8);
    cudaFree(d_pos);
    cudaFree(d_pacc);
    cudaFree(d_pm);
    cudaFree(d_pl);
    free(h_q);
    free(h_out_serial);
    free(h_out_fa2);

    return all_passed ? 0 : 1;
}
