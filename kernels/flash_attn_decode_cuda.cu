// FlashAttention-2 Style Decode Kernel for Batch=1 LLM Inference.
// Query Q (1x128 FP16 = 256 bytes) is pinned in registers.
// Key K and Value V are streamed from VRAM in tiles of 64 tokens.
// Online softmax tracks max and sum_exp in registers without writing S = QK^T to VRAM.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define HEAD_DIM 128
#define BLOCK_SIZE 64

__global__ void k_flash_attn_decode(const half *__restrict__ Q,         // [num_heads, 128]
                                    const half *__restrict__ K_cache,   // [seq_len, num_heads, 128]
                                    const half *__restrict__ V_cache,   // [seq_len, num_heads, 128]
                                    half *__restrict__ Out,             // [num_heads, 128]
                                    int seq_len, float scale) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x; // 32 threads in warp = 1 warp per head

    // Load Q for this head into registers (each thread holds 4 halfs = 8 bytes)
    half q_reg[4];
    const half *q_head = Q + head_idx * HEAD_DIM + tid * 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) q_reg[i] = q_head[i];

    float m_prev = -1e30f; // Running max
    float l_prev = 0.0f;   // Running sum exp
    float o_reg[4] = {0.0f}; // Accumulated output in FP32 registers

    // Stream K and V tiles from VRAM
    for (int t = 0; t < seq_len; t++) {
        const half *k_ptr = K_cache + (t * gridDim.x + head_idx) * HEAD_DIM + tid * 4;
        const half *v_ptr = V_cache + (t * gridDim.x + head_idx) * HEAD_DIM + tid * 4;

        half k_reg[4], v_reg[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            k_reg[i] = k_ptr[i];
            v_reg[i] = v_ptr[i];
        }

        // Q * K^T dot product
        float score = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            score += __half2float(q_reg[i]) * __half2float(k_reg[i]);
        }

        // Warp reduction for score
        #pragma unroll
        for (int mask = 16; mask > 0; mask /= 2) {
            score += __shfl_down_sync(0xffffffff, score, mask);
        }
        score = __shfl_sync(0xffffffff, score, 0); // Broadcast dot product to all 32 lanes
        score *= scale;

        // Online Softmax update
        float m_new = fmaxf(m_prev, score);
        float exp_score = expf(score - m_new);
        float alpha = expf(m_prev - m_new);

        l_prev = l_prev * alpha + exp_score;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            o_reg[i] = o_reg[i] * alpha + exp_score * __half2float(v_reg[i]);
        }

        m_prev = m_new;
    }

    // Finalize output O = O / l_prev and write back to VRAM
    float inv_l = 1.0f / (l_prev + 1e-8f);
    half *out_head = Out + head_idx * HEAD_DIM + tid * 4;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        out_head[i] = __float2half(o_reg[i] * inv_l);
    }
}

extern "C" {
int run_flash_attn_decode(const void *dQ, const void *dK, const void *dV, void *dOut, int num_heads, int seq_len, float scale) {
    dim3 b(32);
    dim3 g(num_heads);
    k_flash_attn_decode<<<g, b>>>((const half*)dQ, (const half*)dK, (const half*)dV, (half*)dOut, seq_len, scale);
    return (int)cudaGetLastError();
}
}
