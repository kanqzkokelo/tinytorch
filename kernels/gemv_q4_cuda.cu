// Fused INT4 (q4_0) Quantized GEMV CUDA Kernel for high-speed LLM token generation.
// Computes y[M] = W_q4[MxK] @ x[K] where W_q4 is stored as 32-value q4_0 blocks.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    half d;            // FP16 scale factor
    uint8_t qs[16];    // 32 4-bit nibbles (16 bytes)
} BlockQ4_0;

// Warp reduction helper
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fused q4_0 GEMV CUDA Kernel (1 warp = 32 threads computes 1 output row y[m])
__global__ void k_gemv_q4_0(const BlockQ4_0 *__restrict__ W,
                            const float *__restrict__ x,
                            float *__restrict__ y,
                            int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const int lane = threadIdx.x; // 0..31
    const int num_blocks_per_row = K / 32;
    const BlockQ4_0 *row_W = W + (long)row * num_blocks_per_row;

    float sum = 0.0f;

    // Process blocks in stride of 32
    for (int b = lane; b < num_blocks_per_row; b += 32) {
        BlockQ4_0 blk = row_W[b];
        float d = __half2float(*(const __half*)&blk.d);
        const float *x_blk = x + b * 32;

        // Unpack 16 bytes (32 nibbles) in registers
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t byte = blk.qs[i];
            int q0 = (byte & 0x0F) - 8;
            int q1 = (byte >> 4) - 8;

            float w0 = (float)q0 * d;
            float w1 = (float)q1 * d;

            sum += w0 * x_blk[i * 2 + 0] + w1 * x_blk[i * 2 + 1];
        }
    }

    // Warp-level sum reduction
    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        y[row] = sum;
    }
}

__global__ void k_gemv_q4_0_fast(const BlockQ4_0 *__restrict__ W,
                                 const float *__restrict__ x,
                                 float *__restrict__ y,
                                 int M, int K) {
    const int row = blockIdx.x * 16 + threadIdx.y;
    if (row >= M) return;

    const int lane = threadIdx.x;
    const int num_blocks_per_row = K / 32;
    const BlockQ4_0 *row_W = W + (long)row * num_blocks_per_row;

    float sum = 0.0f;

    for (int b = lane; b < num_blocks_per_row; b += 32) {
        BlockQ4_0 blk = row_W[b];
        float d = __half2float(*(const __half*)&blk.d);
        const float *x_blk = x + b * 32;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t byte = blk.qs[i];
            int q0 = (byte & 0x0F) - 8;
            int q1 = (byte >> 4) - 8;
            sum += ((float)q0 * d) * x_blk[i] + ((float)q1 * d) * x_blk[i + 16];
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

__global__ void k_fused_swiglu_q4_0(const BlockQ4_0 *__restrict__ W_gate,
                                    const BlockQ4_0 *__restrict__ W_up,
                                    const float *__restrict__ x,
                                    float *__restrict__ out,
                                    int M, int K) {
    const int row = blockIdx.x * 16 + threadIdx.y;
    if (row >= M) return;

    const int lane = threadIdx.x;
    const int num_blocks_per_row = K / 32;

    const BlockQ4_0 *gate_row = W_gate + (long)row * num_blocks_per_row;
    const BlockQ4_0 *up_row   = W_up   + (long)row * num_blocks_per_row;

    float sum_gate = 0.0f, sum_up = 0.0f;

    for (int b = lane; b < num_blocks_per_row; b += 32) {
        BlockQ4_0 bg = gate_row[b];
        BlockQ4_0 bu = up_row[b];
        float dg = __half2float(*(const __half*)&bg.d);
        float du = __half2float(*(const __half*)&bu.d);
        const float *x_blk = x + b * 32;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t byte_g = bg.qs[i];
            uint8_t byte_u = bu.qs[i];
            int qg0 = (byte_g & 0x0F) - 8;
            int qg1 = (byte_g >> 4) - 8;
            int qu0 = (byte_u & 0x0F) - 8;
            int qu1 = (byte_u >> 4) - 8;

            float x0 = x_blk[i];
            float x1 = x_blk[i + 16];

            sum_gate += ((float)qg0 * dg) * x0 + ((float)qg1 * dg) * x1;
            sum_up   += ((float)qu0 * du) * x0 + ((float)qu1 * du) * x1;
        }
    }

    sum_gate = warp_reduce_sum(sum_gate);
    sum_up   = warp_reduce_sum(sum_up);

    if (lane == 0) {
        float silu_gate = sum_gate >= 0.0f ? 
            sum_gate / (1.0f + expf(-sum_gate)) : 
            (sum_gate * expf(sum_gate)) / (1.0f + expf(sum_gate));
        out[row] = silu_gate * sum_up;
    }
}

__global__ void k_flash_attn_decode(const float *__restrict__ Q,
                                    const float *__restrict__ K_cache,
                                    const float *__restrict__ V_cache,
                                    float *__restrict__ Out,
                                    int seq_len, float scale) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const float *q_head = Q + head_idx * 64 + tid * 2;

    float q_reg[2] = {q_head[0], q_head[1]};
    float m_prev = -1e30f, l_prev = 0.0f;
    float o_reg[2] = {0.0f, 0.0f};

    for (int t = 0; t < seq_len; t++) {
        const float *k_ptr = K_cache + (t * gridDim.x + head_idx) * 64 + tid * 2;
        const float *v_ptr = V_cache + (t * gridDim.x + head_idx) * 64 + tid * 2;

        float k_reg[2] = {k_ptr[0], k_ptr[1]};
        float v_reg[2] = {v_ptr[0], v_ptr[1]};

        float score = q_reg[0] * k_reg[0] + q_reg[1] * k_reg[1];
        #pragma unroll
        for (int mask = 16; mask > 0; mask /= 2) {
            score += __shfl_down_sync(0xffffffff, score, mask);
        }
        score = __shfl_sync(0xffffffff, score, 0) * scale;

        if (t == 0) {
            m_prev = score;
            l_prev = 1.0f;
            o_reg[0] = v_reg[0];
            o_reg[1] = v_reg[1];
        } else {
            float m_new = fmaxf(m_prev, score);
            float exp_score = expf(score - m_new);
            float alpha = expf(m_prev - m_new);

            l_prev = l_prev * alpha + exp_score;
            o_reg[0] = o_reg[0] * alpha + exp_score * v_reg[0];
            o_reg[1] = o_reg[1] * alpha + exp_score * v_reg[1];
            m_prev = m_new;
        }
    }

    float inv_l = 1.0f / (l_prev + 1e-8f);
    float *out_head = Out + head_idx * 64 + tid * 2;
    out_head[0] = o_reg[0] * inv_l;
    out_head[1] = o_reg[1] * inv_l;
}

extern "C" {
int run_gemv_q4_0(const void *dW, const float *dx, float *dy, int M, int K) {
    // 32 threads per warp (x-dim), 4 warps per block (y-dim)
    dim3 b(32, 4);
    dim3 g((M + 3) / 4);
    k_gemv_q4_0<<<g, b>>>((const BlockQ4_0*)dW, dx, dy, M, K);
    return (int)cudaGetLastError();
}

__global__ void k_update_kv_cache(const float *__restrict__ x, float *__restrict__ K_cache, float *__restrict__ V_cache, int dim) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < dim) {
        K_cache[tid] = x[tid];
        V_cache[tid] = x[tid];
    }
}

int init_transformer_graph(const void **dW_q, const void **dW_k, const void **dW_v, const void **dW_attn, const void **dW_gate, const void **dW_up, const void **dW_down,
                           float *dx, float *dh1, float *dh2, float *dK_cache, float *dV_cache,
                           int n_layers, int max_steps, cudaGraphExec_t *graphExec) {
    (void)max_steps;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    dim3 b_gemv(32, 16);
    dim3 g_attn((896 + 15) / 16);
    dim3 g_kv((128 + 15) / 16);
    dim3 g_mlp((4864 + 15) / 16);

    for (int l = 0; l < n_layers; l++) {
        const BlockQ4_0 *w_q    = (const BlockQ4_0*)dW_q[l];
        const BlockQ4_0 *w_k    = (const BlockQ4_0*)dW_k[l];
        const BlockQ4_0 *w_v    = (const BlockQ4_0*)dW_v[l];
        const BlockQ4_0 *w_attn = (const BlockQ4_0*)dW_attn[l];
        const BlockQ4_0 *w_gate = (const BlockQ4_0*)dW_gate[l];
        const BlockQ4_0 *w_up   = (const BlockQ4_0*)dW_up[l];
        const BlockQ4_0 *w_down = (const BlockQ4_0*)dW_down[l];

        k_gemv_q4_0_fast<<<g_attn, b_gemv, 0, stream>>>(w_q, dx, dh1, 896, 896);
        k_gemv_q4_0_fast<<<g_kv, b_gemv, 0, stream>>>(w_k, dx, dK_cache, 128, 896);
        k_gemv_q4_0_fast<<<g_kv, b_gemv, 0, stream>>>(w_v, dx, dV_cache, 128, 896);
        k_flash_attn_decode<<<14, 32, 0, stream>>>(dh1, dK_cache, dV_cache, dh2, 1, 0.125f);
        k_gemv_q4_0_fast<<<g_attn, b_gemv, 0, stream>>>(w_attn, dh2, dx, 896, 896);
        k_fused_swiglu_q4_0<<<g_mlp, b_gemv, 0, stream>>>(w_gate, w_up, dx, dh1, 4864, 896);
        k_gemv_q4_0_fast<<<g_attn, b_gemv, 0, stream>>>(w_down, dh1, dx, 896, 4864);
    }

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(graphExec, graph, NULL, NULL, 0);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return 0;
}

int launch_transformer_step(cudaGraphExec_t graphExec) {
    cudaGraphLaunch(graphExec, 0);
    return 0;
}

__global__ void k_rmsnorm(float *__restrict__ x, int dim) {
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = x[i];
        sum_sq += val * val;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf((sum_sq / (float)dim) + 1e-6f);
    }
    __syncthreads();

    float rms = s_rms;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        x[i] *= rms;
    }
}

__global__ void k_embed_token_q4_0(const BlockQ4_0 *__restrict__ embd, int token_id, float *__restrict__ dx, int dim) {
    int tid = threadIdx.x;
    int num_blocks = dim / 32;
    if (tid >= num_blocks) return;

    BlockQ4_0 blk = embd[(long)token_id * num_blocks + tid];
    float d = __half2float(*(const __half*)&blk.d);
    float *dx_blk = dx + tid * 32;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t byte = blk.qs[i];
        int q0 = (byte & 0x0F) - 8;
        int q1 = (byte >> 4) - 8;
        dx_blk[i]      = (float)q0 * d;
        dx_blk[i + 16] = (float)q1 * d;
    }
}

int embed_token_gpu(const void *dW_embd, int token_id, float *dx, int dim) {
    k_embed_token_q4_0<<<1, dim / 32>>>((const BlockQ4_0*)dW_embd, token_id, dx, dim);
    k_rmsnorm<<<1, 32>>>(dx, dim);
    return (int)cudaGetLastError();
}

__global__ void k_argmax(const float *__restrict__ logits, int N, int *__restrict__ out_idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int best_i = 0;
        float max_val = logits[0];
        for (int i = 1; i < N; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                best_i = i;
            }
        }
        *out_idx = best_i;
    }
}

__global__ void k_compute_logits_q4_0(const BlockQ4_0 *__restrict__ embd,
                                      const float *__restrict__ x,
                                      float *__restrict__ logits,
                                      int vocab_size, int dim) {
    const int v = blockIdx.x * 16 + threadIdx.y;
    if (v >= vocab_size) return;

    const int lane = threadIdx.x;
    const int num_blocks_per_col = dim / 32;
    const BlockQ4_0 *col_W = embd + (long)v * num_blocks_per_col;

    float sum = 0.0f;
    for (int b = lane; b < num_blocks_per_col; b += 32) {
        BlockQ4_0 blk = col_W[b];
        float d = __half2float(*(const __half*)&blk.d);
        const float *x_blk = x + b * 32;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t byte = blk.qs[i];
            int q0 = (byte & 0x0F) - 8;
            int q1 = (byte >> 4) - 8;
            sum += ((float)q0 * d) * x_blk[i] + ((float)q1 * d) * x_blk[i + 16];
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) logits[v] = sum;
}

__global__ void k_rmsnorm_gamma(float *__restrict__ x, const float *__restrict__ gamma, int dim) {
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = x[i];
        sum_sq += val * val;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf((sum_sq / (float)dim) + 1e-6f);
    }
    __syncthreads();

    float rms = s_rms;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float g = gamma ? gamma[i] : 1.0f;
        x[i] = (x[i] * rms) * g;
    }
}

int sample_next_token_id(const void *dW_head, const float *dx, const float *d_gamma, float *d_logits, int *d_out_id, int vocab_size, int dim) {
    k_rmsnorm_gamma<<<1, 32>>>((float*)dx, d_gamma, dim);
    dim3 b_gemv(32, 16);
    dim3 g((vocab_size + 15) / 16);
    k_compute_logits_q4_0<<<g, b_gemv>>>((const BlockQ4_0*)dW_head, dx, d_logits, vocab_size, dim);
    cudaDeviceSynchronize();

    k_argmax<<<1, 1>>>(d_logits, vocab_size, d_out_id);
    cudaDeviceSynchronize();

    int host_id = 0;
    cudaMemcpy(&host_id, d_out_id, sizeof(int), cudaMemcpyDeviceToHost);

    return host_id;
}
}

