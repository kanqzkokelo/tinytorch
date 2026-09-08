#ifndef TINYTORCH_H
#define TINYTORCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Tensor {
    float *data;     /* row-major payload */
    long  *shape;    /* ndim entries */
    long  *strides;  /* in elements, per dim; informational: tensors are always contiguous row-major */
    int    ndim;
    long   numel;
    int    refcount;
} Tensor;

/* lifecycle */
Tensor *tt_new(const long *shape, int ndim);              /* zero-filled */
Tensor *tt_fromdata(const float *data, const long *shape, int ndim); /* copies */
Tensor *tt_retain(Tensor *t);
void    tt_release(Tensor *t);

/* accessors */
float  *tt_data(const Tensor *t);
long    tt_numel(const Tensor *t);
int     tt_ndim(const Tensor *t);
void    tt_shape(const Tensor *t, long *out);
void    tt_strides(const Tensor *t, long *out);

/* ops (allocate fresh output; NULL on shape/arg error) */
Tensor *tt_add(const Tensor *a, const Tensor *b);
Tensor *tt_mul(const Tensor *a, const Tensor *b);
Tensor *tt_addscalar(const Tensor *a, float s);
Tensor *tt_mulscalar(const Tensor *a, float s);
Tensor *tt_matmul(const Tensor *a, const Tensor *b);  /* 2D @ 2D naive */
Tensor *tt_matmul_fast(const Tensor *a, const Tensor *b);   /* AVX2+FMA, 1 thread */
Tensor *tt_matmul_omp(const Tensor *a, const Tensor *b, int nthreads);
Tensor *tt_relu(const Tensor *a);
Tensor *tt_softmax(const Tensor *a);                  /* along last axis */

/* spatial & shape ops */
Tensor *tt_reshape(const Tensor *a, const long *new_shape, int new_ndim); /* copy, not a view */
Tensor *tt_conv2d(const Tensor *a, const Tensor *w, const Tensor *b,
                  int stride_h, int stride_w, int pad_h, int pad_w);
Tensor *tt_maxpool2d(const Tensor *a, int pool_h, int pool_w,
                     int stride_h, int stride_w);
Tensor *tt_avgpool2d(const Tensor *a, int pool_h, int pool_w,
                     int stride_h, int stride_w);

#ifdef __cplusplus
}
#endif
#endif
