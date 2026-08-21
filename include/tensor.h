#ifndef TINYTORCH_H
#define TINYTORCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Tensor {
    float *data;     /* row-major payload */
    int   *shape;    /* ndim entries */
    int   *strides;  /* in elements, per dim */
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
void    tt_strides(const Tensor *t, int *out);

/* ops (allocate fresh output; NULL on shape/arg error) */
Tensor *tt_add(const Tensor *a, const Tensor *b);
Tensor *tt_mul(const Tensor *a, const Tensor *b);
Tensor *tt_addscalar(const Tensor *a, float s);
Tensor *tt_mulscalar(const Tensor *a, float s);
Tensor *tt_matmul(const Tensor *a, const Tensor *b);  /* 2D @ 2D naive */
Tensor *tt_relu(const Tensor *a);
Tensor *tt_softmax(const Tensor *a);                  /* along last axis */

#ifdef __cplusplus
}
#endif
#endif
