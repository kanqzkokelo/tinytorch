#ifndef TINYTORCH_AUTOGRAD_H
#define TINYTORCH_AUTOGRAD_H

#include "tensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AGNode AGNode;

/* graph construction: ops mirror the M0 tensor API but record the DAG */
AGNode *ag_leaf(Tensor *t, int requires_grad);   /* retains t */
AGNode *ag_add(AGNode *a, AGNode *b);
AGNode *ag_mul(AGNode *a, AGNode *b);
AGNode *ag_addscalar(AGNode *a, float s);
AGNode *ag_mulscalar(AGNode *a, float s);
AGNode *ag_matmul(AGNode *a, AGNode *b);         /* 2D @ 2D */
AGNode *ag_relu(AGNode *a);
AGNode *ag_softmax(AGNode *a);                   /* last axis */
AGNode *ag_padones(AGNode *a);                   /* append ones column */

/* spatial & shape autograd nodes */
AGNode *ag_reshape(AGNode *a, const long *new_shape, int new_ndim);
AGNode *ag_conv2d(AGNode *a, AGNode *w, AGNode *b,
                  int stride_h, int stride_w, int pad_h, int pad_w);
AGNode *ag_maxpool2d(AGNode *a, int pool_h, int pool_w,
                     int stride_h, int stride_w);
AGNode *ag_avgpool2d(AGNode *a, int pool_h, int pool_w,
                     int stride_h, int stride_w);

/* accessors */
Tensor *ag_value(const AGNode *n);
Tensor *ag_grad(const AGNode *n);                /* NULL until backward */
int     ag_requires_grad(const AGNode *n);

/* reverse-mode AD.
 * ag_backward(out): seed out->grad with 1.0 (scalar losses).
 * ag_backward_from(out, seed, seed_numel): copy seed (seed_numel floats)
 * into out->grad. seed_numel must exactly equal out->val->numel;
 * on mismatch the call returns without touching grad (no memcpy,
 * no propagate). Callers must surface this as an error. */
void    ag_backward(AGNode *out);
void    ag_backward_from(AGNode *out, const float *seed, size_t seed_numel);

void    ag_release(AGNode *n);                   /* refcounted node */
AGNode *ag_retain(AGNode *n);

#ifdef __cplusplus
}
#endif
#endif
