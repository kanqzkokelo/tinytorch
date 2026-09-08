#include "tensor.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* Sane upper bound on elements per tensor (~1T float32 = 4TB).
   Rejects insane shapes before calloc can hang/OOM the host. */
#define TT_MAX_NUMEL (1L << 40)

static Tensor *tt_alloc(const long *shape, int ndim) {
    if (ndim <= 0 || ndim > 8) return NULL;
    long numel = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) return NULL;
        if (numel > LONG_MAX / shape[i]) return NULL;  /* overflow guard */
        numel *= shape[i];
    }
    if (numel > TT_MAX_NUMEL) return NULL;  /* sane cap, pre-alloc */
    Tensor *t = (Tensor *)calloc(1, sizeof(Tensor));
    if (!t) return NULL;
    t->data = (float *)calloc((size_t)numel, sizeof(float));
    t->shape = (long *)malloc(sizeof(long) * (size_t)ndim);
    t->strides = (long *)malloc(sizeof(long) * (size_t)ndim);
    if (!t->data || !t->shape || !t->strides) {
        free(t->data); free(t->shape); free(t->strides); free(t);
        return NULL;
    }
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    /* row-major contiguous strides */
    t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
    t->ndim = ndim;
    t->numel = numel;
    t->refcount = 1;
    return t;
}

Tensor *tt_new(const long *shape, int ndim) { return tt_alloc(shape, ndim); }

Tensor *tt_fromdata(const float *data, const long *shape, int ndim) {
    Tensor *t = tt_alloc(shape, ndim);
    if (!t) return NULL;
    memcpy(t->data, data, sizeof(float) * (size_t)t->numel);
    return t;
}

Tensor *tt_retain(Tensor *t) {
    if (!t) return NULL;
    t->refcount++;
    return t;
}

void tt_release(Tensor *t) {
    if (!t) return;
    if (--t->refcount > 0) return;
    free(t->data);
    free(t->shape);
    free(t->strides);
    free(t);
}

float *tt_data(const Tensor *t) { return t ? t->data : NULL; }
long   tt_numel(const Tensor *t) { return t ? t->numel : 0; }
int    tt_ndim(const Tensor *t) { return t ? t->ndim : 0; }

void tt_shape(const Tensor *t, long *out) {
    if (!t || !out) return;
    for (int i = 0; i < t->ndim; i++) out[i] = t->shape[i];
}

void tt_strides(const Tensor *t, long *out) {
    if (!t || !out) return;
    for (int i = 0; i < t->ndim; i++) out[i] = t->strides[i];
}
