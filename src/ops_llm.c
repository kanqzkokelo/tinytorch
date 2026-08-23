#include "ops_llm.h"
#include <math.h>
#include <stdlib.h>

void tt_rmsnorm(float *out, const float *x, const float *weight, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss /= (float)size;
    float scale = 1.0f / sqrtf(ss + eps);
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * scale * weight[i];
    }
}

void tt_rope(float *vec, int pos, int head_dim, int num_heads, float freq_base) {
    for (int h = 0; h < num_heads; h++) {
        float *head = vec + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
            float val = (float)pos * freq;
            float cos_v = cosf(val);
            float sin_v = sinf(val);

            float v0 = head[i];
            float v1 = head[i + head_dim / 2];

            head[i] = v0 * cos_v - v1 * sin_v;
            head[i + head_dim / 2] = v0 * sin_v + v1 * cos_v;
        }
    }
}

void tt_swiglu(float *out, const float *gate, const float *up, int size) {
    for (int i = 0; i < size; i++) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

void tt_softmax_inline(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}
