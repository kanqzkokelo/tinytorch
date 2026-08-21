#include "tensor.h"
#include <math.h>
#include <string.h>

/* map multi-index -> element offset (elements) */
static inline long tt_offset(const Tensor *t, const int *idx) {
    long off = 0;
    for (int i = 0; i < t->ndim; i++) off += (long)idx[i] * t->strides[i];
    return off;
}

static inline int same_shape(const Tensor *a, const Tensor *b) {
    if (a->ndim != b->ndim) return 0;
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i]) return 0;
    return 1;
}

/* allocate output tensor with same shape as a */
static Tensor *new_like(const Tensor *a) {
    long shp[8];
    for (int i = 0; i < a->ndim; i++) shp[i] = a->shape[i];
    return tt_new(shp, a->ndim);
}

static Tensor *elementwise(const Tensor *a, const Tensor *b,
                           float (*f)(float, float)) {
    if (!a || !b || !same_shape(a, b)) return NULL;
    Tensor *out = new_like(a);
    if (!out) return NULL;
    int idx[8];
    memset(idx, 0, sizeof(idx));
    for (long n = 0; n < out->numel; n++) {
        out->data[n] = f(a->data[tt_offset(a, idx)],
                         b->data[tt_offset(b, idx)]);
        for (int d = out->ndim - 1; d >= 0; d--) {
            if (++idx[d] < out->shape[d]) break;
            idx[d] = 0;
        }
    }
    return out;
}

static float op_add(float x, float y) { return x + y; }
static float op_mul(float x, float y) { return x * y; }

Tensor *tt_add(const Tensor *a, const Tensor *b) {
    return elementwise(a, b, op_add);
}

Tensor *tt_mul(const Tensor *a, const Tensor *b) {
    return elementwise(a, b, op_mul);
}

static Tensor *unary(const Tensor *a, float (*f)(float)) {
    if (!a) return NULL;
    Tensor *out = new_like(a);
    if (!out) return NULL;
    int idx[8];
    memset(idx, 0, sizeof(idx));
    for (long n = 0; n < out->numel; n++) {
        out->data[n] = f(a->data[tt_offset(a, idx)]);
        for (int d = out->ndim - 1; d >= 0; d--) {
            if (++idx[d] < out->shape[d]) break;
            idx[d] = 0;
        }
    }
    return out;
}

static Tensor *scalar_op(const Tensor *a, float s, int is_add) {
    if (!a) return NULL;
    Tensor *out = new_like(a);
    if (!out) return NULL;
    for (long i = 0; i < out->numel; i++)
        out->data[i] = is_add ? a->data[i] + s : a->data[i] * s;
    return out;
}

Tensor *tt_addscalar(const Tensor *a, float s) { return scalar_op(a, s, 1); }

Tensor *tt_mulscalar(const Tensor *a, float s) { return scalar_op(a, s, 0); }

Tensor *tt_matmul(const Tensor *a, const Tensor *b) {
    if (!a || !b || a->ndim != 2 || b->ndim != 2) return NULL;
    const int M = a->shape[0], K = a->shape[1], N = b->shape[1];
    if (K != b->shape[0]) return NULL;
    const long shp[2] = {M, N};
    Tensor *out = tt_new(shp, 2);
    if (!out) return NULL;
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            float av = a->data[(long)m * a->strides[0] + (long)k * a->strides[1]];
            if (av == 0.0f) continue;
            const float *brow = b->data + (long)k * b->strides[0];
            float *orow = out->data + (long)m * N;
            for (int n = 0; n < N; n++)
                orow[n] += av * brow[(long)n * b->strides[1]];
        }
    }
    return out;
}

static float op_relu(float x) { return x > 0.0f ? x : 0.0f; }

Tensor *tt_relu(const Tensor *a) { return unary(a, op_relu); }

Tensor *tt_softmax(const Tensor *a) {
    if (!a || a->ndim < 1) return NULL;
    Tensor *out = new_like(a);
    if (!out) return NULL;
    const int N = a->shape[a->ndim - 1];       /* last axis */
    const long outer = a->numel / N;
    const int s_in = a->strides[a->ndim - 1];
    /* rows are index vectors over all but last dim */
    int idx[8];
    memset(idx, 0, sizeof(idx));
    for (long r = 0; r < outer; r++) {
        long base = tt_offset(a, idx);
        float mx = -INFINITY;
        for (int j = 0; j < N; j++) {
            float v = a->data[base + (long)j * s_in];
            if (v > mx) mx = v;
        }
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            float e = expf(a->data[base + (long)j * s_in] - mx);
            out->data[(long)r * N + j] = e;
            sum += e;
        }
        for (int j = 0; j < N; j++)
            out->data[(long)r * N + j] /= sum;
        for (int d = a->ndim - 2; d >= 0; d--) {
            if (++idx[d] < a->shape[d]) break;
            idx[d] = 0;
        }
    }
    return out;
}
