#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

struct BlockQ8_0 {
    half d;          // 2 bytes FP16 scale
    int8_t qs[32];   // 32 bytes signed int8
};

__device__ __forceinline__ static float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

// Scatter kernel: FP32 reference
__global__ void k_kv_scatter(const float *__restrict__ kst,
                             const float *__restrict__ vst,
                             float *__restrict__ Kc,
                             float *__restrict__ Vc,
                             const int *__restrict__ d_pos,
                             int n_kv_heads, int head_dim, int max_ctx) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int kvdim = n_kv_heads * head_dim;
    if (i >= kvdim) return;
    const int slot = (*d_pos) % max_ctx;
    Kc[(long)slot * kvdim + i] = kst[i];
    Vc[(long)slot * kvdim + i] = vst[i];
}

// Scatter kernel: quantizes FP32 K/V staging vectors into Q8_0 cache slot (*d_pos)
__global__ void k_kv_scatter_q8_0(
    const float *__restrict__ kst,
    const float *__restrict__ vst,
    BlockQ8_0   *__restrict__ Kc,
    BlockQ8_0   *__restrict__ Vc,
    const int   *__restrict__ d_pos,
    int n_kv_heads, int head_dim, int max_ctx) {
    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_blocks_per_slot = (n_kv_heads * head_dim) / 32;
    if (block_idx >= num_blocks_per_slot) return;
    const int slot = (*d_pos) % max_ctx;
    const int src_offset = block_idx * 32;

    float k_vals[32], v_vals[32];
    float max_k = 0.0f, max_v = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        k_vals[i] = kst[src_offset + i];
        v_vals[i] = vst[src_offset + i];
        max_k = fmaxf(max_k, fabsf(k_vals[i]));
        max_v = fmaxf(max_v, fabsf(v_vals[i]));
    }

    const float scale_k = (max_k > 0.0f) ? (max_k / 127.0f) : 1.0f;
    const float inv_k   = (max_k > 0.0f) ? (127.0f / max_k) : 0.0f;
    const float scale_v = (max_v > 0.0f) ? (max_v / 127.0f) : 1.0f;
    const float inv_v   = (max_v > 0.0f) ? (127.0f / max_v) : 0.0f;

    BlockQ8_0 *k_dest = Kc + (long)slot * num_blocks_per_slot + block_idx;
    BlockQ8_0 *v_dest = Vc + (long)slot * num_blocks_per_slot + block_idx;

    k_dest->d = __float2half(scale_k);
    v_dest->d = __float2half(scale_v);

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        k_dest->qs[i] = (int8_t)__float2int_rn(k_vals[i] * inv_k);
        v_dest->qs[i] = (int8_t)__float2int_rn(v_vals[i] * inv_v);
    }
}

// Flash GQA kernel: standard FP32
__global__ void k_flash_gqa(const float *__restrict__ q,
                            const float *__restrict__ Kc,
                            const float *__restrict__ Vc,
                            float *__restrict__ out,
                            const int *__restrict__ d_pos,
                            int n_heads, int n_kv_heads, int head_dim,
                            int max_ctx, float scale, int window) {
    const int pos = *d_pos;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int lane = threadIdx.x;
    const int kvh = h / (n_heads / n_kv_heads);
    const int elems = head_dim / 32;
    const float *qh = q + (long)h * head_dim + lane * elems;

    int t0 = 0;
    if (window > 0 && pos >= window) t0 = pos - window + 1;

    float qreg[16];
#pragma unroll
    for (int i = 0; i < 16; i++) qreg[i] = (i < elems) ? qh[i] : 0.0f;

    float m_prev = -1e30f, l_prev = 0.0f;
    float oreg[16] = {0.0f};

    for (int t = t0; t <= pos; t++) {
        const long off = ((long)t * n_kv_heads + kvh) * head_dim + lane * elems;
        const float *kp = Kc + off;
        const float *vp = Vc + off;
        float score = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (i < elems) score += qreg[i] * kp[i];
        score = warp_sum(score);
        score = __shfl_sync(0xffffffff, score, 0) * scale;

        const float m_new = fmaxf(m_prev, score);
        const float ex = expf(score - m_new);
        const float alpha = expf(m_prev - m_new);
        l_prev = l_prev * alpha + ex;
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (i < elems) oreg[i] = oreg[i] * alpha + ex * vp[i];
        m_prev = m_new;
    }

    float *outh = out + (long)h * head_dim + lane * elems;
#pragma unroll
    for (int i = 0; i < 16; i++)
        if (i < elems) outh[i] = oreg[i] / l_prev;
}

// Split-K FP32
__global__ void k_flash_gqa_splitk(const float *__restrict__ q,
                                   const float *__restrict__ Kc,
                                   const float *__restrict__ Vc,
                                   float *__restrict__ p_acc,
                                   float *__restrict__ p_m,
                                   float *__restrict__ p_l,
                                   const int *__restrict__ d_pos,
                                   int n_heads, int n_kv_heads, int head_dim,
                                   float scale, int window, int S) {
    const int pos = *d_pos;
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int lane = threadIdx.x;
    const int kvh = h / (n_heads / n_kv_heads);
    const int elems = head_dim / 32;

    int t_lo = 0;
    if (window > 0 && pos >= window) t_lo = pos - window + 1;
    const int nslots = pos - t_lo + 1;
    const int chunk = (nslots + S - 1) / S;
    const int begin = t_lo + s * chunk;
    const int end = min(pos + 1, t_lo + (s + 1) * chunk);

    float *myacc = p_acc + ((size_t)s * n_heads + h) * head_dim + lane * elems;
    float *mym = p_m + (size_t)s * n_heads + h;
    float *myl = p_l + (size_t)s * n_heads + h;

    if (begin >= end) {
        if (lane == 0) { *mym = -1e30f; *myl = 0.0f; }
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (i < elems) myacc[i] = 0.0f;
        return;
    }

    const float *qh = q + (long)h * head_dim + lane * elems;
    float qreg[16];
#pragma unroll
    for (int i = 0; i < 16; i++) qreg[i] = (i < elems) ? qh[i] : 0.0f;

    float m_prev = -1e30f, l_prev = 0.0f;
    float oreg[16] = {0.0f};

    for (int t = begin; t < end; t++) {
        const long off = ((long)t * n_kv_heads + kvh) * head_dim + lane * elems;
        const float *kp = Kc + off;
        const float *vp = Vc + off;
        float score = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (i < elems) score += qreg[i] * kp[i];
        score = warp_sum(score);
        score = __shfl_sync(0xffffffff, score, 0) * scale;

        const float m_new = fmaxf(m_prev, score);
        const float ex = expf(score - m_new);
        const float alpha = expf(m_prev - m_new);
        l_prev = l_prev * alpha + ex;
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (i < elems) oreg[i] = oreg[i] * alpha + ex * vp[i];
        m_prev = m_new;
    }

    if (lane == 0) { *mym = m_prev; *myl = l_prev; }
#pragma unroll
    for (int i = 0; i < 16; i++)
        if (i < elems) myacc[i] = oreg[i];
}

// Flash GQA kernel reading Q8_0 KV cache, dequantizing on-the-fly in registers
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

    const int lane = threadIdx.x; // 0..31
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

        // Load scale dk
        const uint32_t dw_k = k_slot_u32[wsc];
        const unsigned short d16_k = (unsigned short)(sh_d ? (dw_k >> 16) : (dw_k & 0xFFFFu));
        const float dk = __half2float(__ushort_as_half(d16_k));

        // Load int8 values for K
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

        // Load scale dv
        const uint32_t dw_v = v_slot_u32[wsc];
        const unsigned short d16_v = (unsigned short)(sh_d ? (dw_v >> 16) : (dw_v & 0xFFFFu));
        const float dv = __half2float(__ushort_as_half(d16_v));

        // Load int8 values for V
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

// Split-K Q8_0
__global__ void k_flash_gqa_q8_0_splitk(
    const float     *__restrict__ q,
    const BlockQ8_0 *__restrict__ Kc_q8,
    const BlockQ8_0 *__restrict__ Vc_q8,
    float           *__restrict__ p_acc,
    float           *__restrict__ p_m,
    float           *__restrict__ p_l,
    const int       *__restrict__ d_pos,
    int n_heads, int n_kv_heads, int head_dim,
    float scale, int window, int S) {
    const int pos = *d_pos;
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int lane = threadIdx.x;
    const int kvh = h / (n_heads / n_kv_heads);
    const int elems = head_dim / 32;

    int t_lo = 0;
    if (window > 0 && pos >= window) t_lo = pos - window + 1;
    const int nslots = pos - t_lo + 1;
    const int chunk = (nslots + S - 1) / S;
    const int begin = t_lo + s * chunk;
    const int end = min(pos + 1, t_lo + (s + 1) * chunk);

    float *myacc = p_acc + ((size_t)s * n_heads + h) * head_dim + lane * elems;
    float *mym = p_m + (size_t)s * n_heads + h;
    float *myl = p_l + (size_t)s * n_heads + h;

    if (begin >= end) {
        if (lane == 0) { *mym = -1e30f; *myl = 0.0f; }
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (i < elems) myacc[i] = 0.0f;
        return;
    }

    const float *qh = q + (long)h * head_dim + lane * elems;
    float qreg[16];
#pragma unroll
    for (int i = 0; i < 16; i++) qreg[i] = (i < elems) ? qh[i] : 0.0f;

    float m_prev = -1e30f, l_prev = 0.0f;
    float oreg[16] = {0.0f};

    const int blocks_per_head = head_dim / 32;
    const int blocks_per_slot = n_kv_heads * blocks_per_head;
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
    const char *k_ptr = (const char *)Kc_q8 + (long)begin * stride;
    const char *v_ptr = (const char *)Vc_q8 + (long)begin * stride;

    for (int t = begin; t < end; t++) {
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

    if (lane == 0) { *mym = m_prev; *myl = l_prev; }
#pragma unroll
    for (int i = 0; i < 16; i++)
        if (i < elems) myacc[i] = oreg[i];
}

__global__ void k_flash_gqa_combine(const float *__restrict__ p_acc,
                                   const float *__restrict__ p_m,
                                   const float *__restrict__ p_l,
                                   float *__restrict__ out,
                                   int n_heads, int head_dim, int S) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    const int elems = head_dim / 32;

    float m = -1e30f, l = 0.0f;
    float oreg[16] = {0.0f};

    for (int s = 0; s < S; s++) {
        const size_t idx = (size_t)s * n_heads + h;
        const float ls = p_l[idx];
        if (!(ls > 0.0f)) continue;
        const float ms = p_m[idx];
        const float m_new = fmaxf(m, ms);
        const float alpha = expf(m - m_new);
        const float beta = expf(ms - m_new);
        const float *acc = p_acc + idx * head_dim + lane * elems;
        l = l * alpha + ls * beta;
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (i < elems) oreg[i] = oreg[i] * alpha + acc[i] * beta;
        m = m_new;
    }

    const float inv_l = 1.0f / (l + 1e-8f);
    float *oh = out + (long)h * head_dim + lane * elems;
#pragma unroll
    for (int i = 0; i < 16; i++)
        if (i < elems) oh[i] = oreg[i] * inv_l;
}

static float frand() {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

void bench_shape(int n_heads, int n_kv_heads, int head_dim, const std::vector<int> &ctx_lengths) {
    const int max_ctx = 16384;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const int window = 0;

    printf("\n=================================================================================\n");
    printf("Benchmark: n_heads=%d, n_kv_heads=%d, head_dim=%d (scale=%.4f)\n",
           n_heads, n_kv_heads, head_dim, scale);
    printf("=================================================================================\n");
    printf("%-8s | %-15s | %-15s | %-10s | %-15s\n",
           "Context", "FP32 Time (ms)", "Q8_0 Time (ms)", "Speedup", "Max Abs Error");
    printf("---------------------------------------------------------------------------------\n");

    const size_t q_size = (size_t)n_heads * head_dim * sizeof(float);
    const size_t out_size = q_size;
    const size_t kv_dim = (size_t)n_kv_heads * head_dim;

    float *h_q = (float *)malloc(q_size);
    float *h_out_fp32 = (float *)malloc(out_size);
    float *h_out_q8   = (float *)malloc(out_size);
    float *h_kst = (float *)malloc(kv_dim * sizeof(float));
    float *h_vst = (float *)malloc(kv_dim * sizeof(float));

    for (size_t i = 0; i < n_heads * head_dim; i++) {
        h_q[i] = frand();
    }

    float *d_q, *d_out_fp32, *d_out_q8, *d_kst, *d_vst;
    int *d_pos;
    cudaMalloc(&d_q, q_size);
    cudaMalloc(&d_out_fp32, out_size);
    cudaMalloc(&d_out_q8, out_size);
    cudaMalloc(&d_kst, kv_dim * sizeof(float));
    cudaMalloc(&d_vst, kv_dim * sizeof(float));
    cudaMalloc(&d_pos, sizeof(int));

    cudaMemcpy(d_q, h_q, q_size, cudaMemcpyHostToDevice);

    const size_t fp32_cache_size = (size_t)max_ctx * kv_dim * sizeof(float);
    const size_t blocks_per_slot = kv_dim / 32;
    const size_t q8_cache_size   = (size_t)max_ctx * blocks_per_slot * sizeof(BlockQ8_0);

    float *d_Kc_fp32, *d_Vc_fp32;
    BlockQ8_0 *d_Kc_q8, *d_Vc_q8;
    cudaMalloc(&d_Kc_fp32, fp32_cache_size);
    cudaMalloc(&d_Vc_fp32, fp32_cache_size);
    cudaMalloc(&d_Kc_q8, q8_cache_size);
    cudaMalloc(&d_Vc_q8, q8_cache_size);

    // Workspace for Split-K
    const int max_S = 64;
    const size_t split_pacc_size = (size_t)max_S * n_heads * head_dim * sizeof(float);
    const size_t split_pm_size   = (size_t)max_S * n_heads * sizeof(float);
    float *d_pacc_fp32, *d_pm_fp32, *d_pl_fp32;
    float *d_pacc_q8,   *d_pm_q8,   *d_pl_q8;
    cudaMalloc(&d_pacc_fp32, split_pacc_size);
    cudaMalloc(&d_pm_fp32,   split_pm_size);
    cudaMalloc(&d_pl_fp32,   split_pm_size);
    cudaMalloc(&d_pacc_q8,   split_pacc_size);
    cudaMalloc(&d_pm_q8,     split_pm_size);
    cudaMalloc(&d_pl_q8,     split_pm_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int N : ctx_lengths) {
        if (N > max_ctx) continue;

        // Populate cache slots 0..N-1
        for (int t = 0; t < N; t++) {
            for (size_t i = 0; i < kv_dim; i++) {
                h_kst[i] = frand();
                h_vst[i] = frand();
            }
            cudaMemcpy(d_kst, h_kst, kv_dim * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_vst, h_vst, kv_dim * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_pos, &t, sizeof(int), cudaMemcpyHostToDevice);

            k_kv_scatter<<<(kv_dim + 255) / 256, 256>>>(
                d_kst, d_vst, d_Kc_fp32, d_Vc_fp32, d_pos, n_kv_heads, head_dim, max_ctx);

            k_kv_scatter_q8_0<<<(blocks_per_slot + 255) / 256, 256>>>(
                d_kst, d_vst, d_Kc_q8, d_Vc_q8, d_pos, n_kv_heads, head_dim, max_ctx);
        }
        cudaDeviceSynchronize();

        // Run correctness check for pos = N - 1
        const int cur_pos = N - 1;
        cudaMemcpy(d_pos, &cur_pos, sizeof(int), cudaMemcpyHostToDevice);

        int S = 1;
        if (N >= 512) S = std::min(max_S, std::max(2, N / 128));

        if (S > 1) {
            dim3 grid_split(n_heads, S);
            k_flash_gqa_splitk<<<grid_split, 32>>>(
                d_q, d_Kc_fp32, d_Vc_fp32, d_pacc_fp32, d_pm_fp32, d_pl_fp32,
                d_pos, n_heads, n_kv_heads, head_dim, scale, window, S);
            k_flash_gqa_combine<<<n_heads, 32>>>(
                d_pacc_fp32, d_pm_fp32, d_pl_fp32, d_out_fp32, n_heads, head_dim, S);

            k_flash_gqa_q8_0_splitk<<<grid_split, 32>>>(
                d_q, d_Kc_q8, d_Vc_q8, d_pacc_q8, d_pm_q8, d_pl_q8,
                d_pos, n_heads, n_kv_heads, head_dim, scale, window, S);
            k_flash_gqa_combine<<<n_heads, 32>>>(
                d_pacc_q8, d_pm_q8, d_pl_q8, d_out_q8, n_heads, head_dim, S);
        } else {
            k_flash_gqa<<<n_heads, 32>>>(
                d_q, d_Kc_fp32, d_Vc_fp32, d_out_fp32, d_pos,
                n_heads, n_kv_heads, head_dim, max_ctx, scale, window);

            k_flash_gqa_q8_0<<<n_heads, 32>>>(
                d_q, d_Kc_q8, d_Vc_q8, d_out_q8, d_pos,
                n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
        }

        cudaMemcpy(h_out_fp32, d_out_fp32, out_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_q8, d_out_q8, out_size, cudaMemcpyDeviceToHost);

        float max_abs_err = 0.0f;
        for (size_t i = 0; i < (size_t)n_heads * head_dim; i++) {
            float err = fabsf(h_out_fp32[i] - h_out_q8[i]);
            if (err > max_abs_err) max_abs_err = err;
        }

        // Benchmark FP32 kernel
        const int warmup = 20;
        const int iters = 200;
        if (S > 1) {
            dim3 grid_split(n_heads, S);
            for (int i = 0; i < warmup; i++) {
                k_flash_gqa_splitk<<<grid_split, 32>>>(
                    d_q, d_Kc_fp32, d_Vc_fp32, d_pacc_fp32, d_pm_fp32, d_pl_fp32,
                    d_pos, n_heads, n_kv_heads, head_dim, scale, window, S);
                k_flash_gqa_combine<<<n_heads, 32>>>(
                    d_pacc_fp32, d_pm_fp32, d_pl_fp32, d_out_fp32, n_heads, head_dim, S);
            }
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            for (int i = 0; i < iters; i++) {
                k_flash_gqa_splitk<<<grid_split, 32>>>(
                    d_q, d_Kc_fp32, d_Vc_fp32, d_pacc_fp32, d_pm_fp32, d_pl_fp32,
                    d_pos, n_heads, n_kv_heads, head_dim, scale, window, S);
                k_flash_gqa_combine<<<n_heads, 32>>>(
                    d_pacc_fp32, d_pm_fp32, d_pl_fp32, d_out_fp32, n_heads, head_dim, S);
            }
            cudaEventRecord(stop);
        } else {
            for (int i = 0; i < warmup; i++) {
                k_flash_gqa<<<n_heads, 32>>>(
                    d_q, d_Kc_fp32, d_Vc_fp32, d_out_fp32, d_pos,
                    n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
            }
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            for (int i = 0; i < iters; i++) {
                k_flash_gqa<<<n_heads, 32>>>(
                    d_q, d_Kc_fp32, d_Vc_fp32, d_out_fp32, d_pos,
                    n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
            }
            cudaEventRecord(stop);
        }
        cudaEventSynchronize(stop);
        float ms_fp32 = 0.0f;
        cudaEventElapsedTime(&ms_fp32, start, stop);
        ms_fp32 /= iters;

        // Benchmark Q8_0 kernel
        if (S > 1) {
            dim3 grid_split(n_heads, S);
            for (int i = 0; i < warmup; i++) {
                k_flash_gqa_q8_0_splitk<<<grid_split, 32>>>(
                    d_q, d_Kc_q8, d_Vc_q8, d_pacc_q8, d_pm_q8, d_pl_q8,
                    d_pos, n_heads, n_kv_heads, head_dim, scale, window, S);
                k_flash_gqa_combine<<<n_heads, 32>>>(
                    d_pacc_q8, d_pm_q8, d_pl_q8, d_out_q8, n_heads, head_dim, S);
            }
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            for (int i = 0; i < iters; i++) {
                k_flash_gqa_q8_0_splitk<<<grid_split, 32>>>(
                    d_q, d_Kc_q8, d_Vc_q8, d_pacc_q8, d_pm_q8, d_pl_q8,
                    d_pos, n_heads, n_kv_heads, head_dim, scale, window, S);
                k_flash_gqa_combine<<<n_heads, 32>>>(
                    d_pacc_q8, d_pm_q8, d_pl_q8, d_out_q8, n_heads, head_dim, S);
            }
            cudaEventRecord(stop);
        } else {
            for (int i = 0; i < warmup; i++) {
                k_flash_gqa_q8_0<<<n_heads, 32>>>(
                    d_q, d_Kc_q8, d_Vc_q8, d_out_q8, d_pos,
                    n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
            }
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            for (int i = 0; i < iters; i++) {
                k_flash_gqa_q8_0<<<n_heads, 32>>>(
                    d_q, d_Kc_q8, d_Vc_q8, d_out_q8, d_pos,
                    n_heads, n_kv_heads, head_dim, max_ctx, scale, window);
            }
            cudaEventRecord(stop);
        }
        cudaEventSynchronize(stop);
        float ms_q8 = 0.0f;
        cudaEventElapsedTime(&ms_q8, start, stop);
        ms_q8 /= iters;

        float speedup = ms_fp32 / ms_q8;

        printf("%-8d | %-15.4f | %-15.4f | %-10.2fx | %-15.4e\n",
               N, ms_fp32, ms_q8, speedup, max_abs_err);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_q); cudaFree(d_out_fp32); cudaFree(d_out_q8);
    cudaFree(d_kst); cudaFree(d_vst); cudaFree(d_pos);
    cudaFree(d_Kc_fp32); cudaFree(d_Vc_fp32);
    cudaFree(d_Kc_q8); cudaFree(d_Vc_q8);
    cudaFree(d_pacc_fp32); cudaFree(d_pm_fp32); cudaFree(d_pl_fp32);
    cudaFree(d_pacc_q8);   cudaFree(d_pm_q8);   cudaFree(d_pl_q8);
    free(h_q); free(h_out_fp32); free(h_out_q8);
    free(h_kst); free(h_vst);
}

int main() {
    printf("Q8_0 KV Cache Attention Microbenchmark (with Split-K for N>=512)\n");
    std::vector<int> ctx_lengths = {128, 512, 1024, 2048, 4096, 8192};

    // Shape 1: Qwen2-0.5B (n_heads=14, n_kv_heads=2, head_dim=64)
    bench_shape(14, 2, 64, ctx_lengths);

    // Shape 2: Qwen2-1.5B / Gemma2 style (n_heads=16, n_kv_heads=2, head_dim=128)
    bench_shape(16, 2, 128, ctx_lengths);

    return 0;
}
