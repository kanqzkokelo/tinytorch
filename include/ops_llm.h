#ifndef OPS_LLM_H
#define OPS_LLM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
void tt_rmsnorm(float *out, const float *x, const float *weight, int size, float eps);

// RoPE: Rotary Position Embeddings (in-place on q or k)
void tt_rope(float *q_or_k, int pos, int head_dim, int num_heads, float freq_base);

// SwiGLU: out = (x1 * weight_gate * sigmoid(x1 * weight_gate)) * (x1 * weight_up)
void tt_swiglu(float *out, const float *gate, const float *up, int size);

// Softmax inline
void tt_softmax_inline(float *x, int size);

#ifdef __cplusplus
}
#endif

#endif // OPS_LLM_H
