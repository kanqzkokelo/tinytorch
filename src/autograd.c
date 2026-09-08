#include "autograd.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    OP_LEAF,
    OP_ADD,
    OP_MUL,
    OP_ADDSCALAR,
    OP_MULSCALAR,
    OP_MATMUL,
    OP_RELU,
    OP_SOFTMAX,
    OP_PADONES,
    OP_RESHAPE,
    OP_CONV2D,
    OP_MAXPOOL2D,
    OP_AVGPOOL2D
} OpKind;

struct AGNode {
    Tensor *val;
    Tensor *grad;
    AGNode *parents[3];
    int     nparents;
    OpKind  kind;
    float   scalar;
    int     params[4];
    int     requires_grad;
    int     mark;      /* topo DFS state */
    int     refcount;
};

/* ---------- helpers ---------- */

static Tensor *new_like(const Tensor *a) {
    long shp[8];
    for (int i = 0; i < a->ndim; i++) shp[i] = a->shape[i];
    return tt_new(shp, a->ndim);
}

static Tensor *tensor_copy(const Tensor *a) {
    Tensor *out = new_like(a);
    if (!out) return NULL;
    memcpy(out->data, a->data, sizeof(float) * (size_t)a->numel);
    return out;
}

/* out += a (elementwise, same shape) */
static void accum_inplace(Tensor *out, const Tensor *a) {
    for (long i = 0; i < out->numel; i++) out->data[i] += a->data[i];
}

static void accum(AGNode *p, const Tensor *g) {
    if (!p || !p->requires_grad || !g) return;
    if (!p->grad) {
        p->grad = tensor_copy(g);
        return;
    }
    accum_inplace(p->grad, g);
}

/* r = x @ y^T  (2D); x:(M,K), y:(N,K) -> r:(M,N) */
static Tensor *matmul_nt(const Tensor *x, const Tensor *y) {
    const int M = x->shape[0], K = x->shape[1], N = y->shape[0];
    if (y->shape[1] != K) return NULL;
    const long shp[2] = {M, N};
    Tensor *r = tt_new(shp, 2);
    if (!r) return NULL;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            const float *xr = x->data + (long)m * K;
            const float *yr = y->data + (long)n * K;
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += xr[k] * yr[k];
            r->data[(long)m * N + n] = s;
        }
    return r;
}

/* r = x^T @ y  (2D) */
static Tensor *matmul_tn(const Tensor *x, const Tensor *y) {
    const int M = x->shape[1], K = x->shape[0], N = y->shape[1];
    const long shp[2] = {M, N};
    Tensor *r = tt_new(shp, 2);
    if (!r) return NULL;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            float xv = x->data[(long)k * M + m];
            const float *yrow = y->data + (long)k * N;
            float *rrow = r->data + (long)m * N;
            for (int n = 0; n < N; n++) rrow[n] += xv * yrow[n];
        }
    return r;
}

/* ---------- node construction ---------- */

static AGNode *node_new(Tensor *val, int requires_grad) {
    AGNode *n = (AGNode *)calloc(1, sizeof(AGNode));
    if (!n) { tt_release(val); return NULL; }
    n->val = val;
    n->requires_grad = requires_grad;
    n->refcount = 1;
    return n;
}

static AGNode *node_binary(AGNode *a, AGNode *b, Tensor *val, OpKind k) {
    if (!val) return NULL;
    AGNode *n = node_new(val, a->requires_grad || b->requires_grad);
    if (!n) return NULL;
    n->parents[0] = ag_retain(a);
    n->parents[1] = ag_retain(b);
    n->nparents = 2;
    n->kind = k;
    return n;
}

AGNode *ag_leaf(Tensor *t, int requires_grad) {
    if (!t) return NULL;
    AGNode *n = node_new(tt_retain(t), requires_grad);
    if (n) n->kind = OP_LEAF;
    return n;
}

AGNode *ag_add(AGNode *a, AGNode *b) {
    if (!a || !b) return NULL;
    return node_binary(a, b, tt_add(a->val, b->val), OP_ADD);
}

AGNode *ag_mul(AGNode *a, AGNode *b) {
    if (!a || !b) return NULL;
    return node_binary(a, b, tt_mul(a->val, b->val), OP_MUL);
}

AGNode *ag_addscalar(AGNode *a, float s) {
    if (!a) return NULL;
    Tensor *v = tt_addscalar(a->val, s);
    AGNode *n = node_new(v, a->requires_grad);
    if (!n) return NULL;
    n->parents[0] = ag_retain(a);
    n->nparents = 1;
    n->kind = OP_ADDSCALAR;
    n->scalar = s;
    return n;
}

AGNode *ag_mulscalar(AGNode *a, float s) {
    if (!a) return NULL;
    Tensor *v = tt_mulscalar(a->val, s);
    AGNode *n = node_new(v, a->requires_grad);
    if (!n) return NULL;
    n->parents[0] = ag_retain(a);
    n->nparents = 1;
    n->kind = OP_MULSCALAR;
    n->scalar = s;
    return n;
}

AGNode *ag_matmul(AGNode *a, AGNode *b) {
    if (!a || !b) return NULL;
    return node_binary(a, b, tt_matmul_omp(a->val, b->val, 12), OP_MATMUL);
}

AGNode *ag_relu(AGNode *a) {
    if (!a) return NULL;
    Tensor *v = tt_relu(a->val);
    AGNode *n = node_new(v, a->requires_grad);
    if (!n) return NULL;
    n->parents[0] = ag_retain(a);
    n->nparents = 1;
    n->kind = OP_RELU;
    return n;
}

AGNode *ag_softmax(AGNode *a) {
    if (!a) return NULL;
    Tensor *v = tt_softmax(a->val);
    AGNode *n = node_new(v, a->requires_grad);
    if (!n) return NULL;
    n->parents[0] = ag_retain(a);
    n->nparents = 1;
    n->kind = OP_SOFTMAX;
    return n;
}

AGNode *ag_padones(AGNode *a) {
    if (!a || a->val->ndim != 2) return NULL;
    const int M = a->val->shape[0], K = a->val->shape[1];
    const long shp[2] = {M, K + 1};
    Tensor *v = tt_new(shp, 2);
    if (!v) return NULL;
    for (int m = 0; m < M; m++) {
        memcpy(v->data + (long)m * (K + 1), a->val->data + (long)m * K,
               sizeof(float) * (size_t)K);
        v->data[(long)m * (K + 1) + K] = 1.0f;
    }
    AGNode *n = node_new(v, a->requires_grad);
    if (!n) return NULL;
    n->parents[0] = ag_retain(a);
    n->nparents = 1;
    n->kind = OP_PADONES;
    return n;
}

AGNode *ag_reshape(AGNode *a, const long *new_shape, int new_ndim) {
    if (!a) return NULL;
    Tensor *v = tt_reshape(a->val, new_shape, new_ndim);
    if (!v) return NULL;
    AGNode *n = node_new(v, a->requires_grad);
    if (!n) return NULL;
    n->parents[0] = ag_retain(a);
    n->nparents = 1;
    n->kind = OP_RESHAPE;
    return n;
}

AGNode *ag_conv2d(AGNode *a, AGNode *w, AGNode *b,
                  int stride_h, int stride_w, int pad_h, int pad_w) {
    if (!a || !w) return NULL;
    Tensor *v = tt_conv2d(a->val, w->val, b ? b->val : NULL, stride_h, stride_w, pad_h, pad_w);
    if (!v) return NULL;
    int req = a->requires_grad || w->requires_grad || (b && b->requires_grad);
    AGNode *n = node_new(v, req);
    if (!n) return NULL;
    n->parents[0] = ag_retain(a);
    n->parents[1] = ag_retain(w);
    n->nparents = 2;
    if (b) {
        n->parents[2] = ag_retain(b);
        n->nparents = 3;
    }
    n->kind = OP_CONV2D;
    n->params[0] = stride_h; n->params[1] = stride_w;
    n->params[2] = pad_h;    n->params[3] = pad_w;
    return n;
}

AGNode *ag_maxpool2d(AGNode *a, int pool_h, int pool_w,
                     int stride_h, int stride_w) {
    if (!a) return NULL;
    Tensor *v = tt_maxpool2d(a->val, pool_h, pool_w, stride_h, stride_w);
    if (!v) return NULL;
    AGNode *n = node_new(v, a->requires_grad);
    if (!n) return NULL;
    n->parents[0] = ag_retain(a);
    n->nparents = 1;
    n->kind = OP_MAXPOOL2D;
    n->params[0] = pool_h;   n->params[1] = pool_w;
    n->params[2] = stride_h; n->params[3] = stride_w;
    return n;
}

AGNode *ag_avgpool2d(AGNode *a, int pool_h, int pool_w,
                     int stride_h, int stride_w) {
    if (!a) return NULL;
    Tensor *v = tt_avgpool2d(a->val, pool_h, pool_w, stride_h, stride_w);
    if (!v) return NULL;
    AGNode *n = node_new(v, a->requires_grad);
    if (!n) return NULL;
    n->parents[0] = ag_retain(a);
    n->nparents = 1;
    n->kind = OP_AVGPOOL2D;
    n->params[0] = pool_h;   n->params[1] = pool_w;
    n->params[2] = stride_h; n->params[3] = stride_w;
    return n;
}

/* ---------- accessors ---------- */

Tensor *ag_value(const AGNode *n) { return n ? n->val : NULL; }
Tensor *ag_grad(const AGNode *n)  { return n ? n->grad : NULL; }
int     ag_requires_grad(const AGNode *n) { return n ? n->requires_grad : 0; }

/* ---------- backward rules ---------- */

static void backward_add(AGNode *n) {
    accum(n->parents[0], n->grad);
    accum(n->parents[1], n->grad);
}

static void backward_mul(AGNode *n) {
    Tensor *g = n->grad;
    Tensor *t1 = tt_mul(g, n->parents[1]->val);   /* d/da = g ⊙ b */
    accum(n->parents[0], t1);
    tt_release(t1);
    Tensor *t2 = tt_mul(g, n->parents[0]->val);   /* d/db = g ⊙ a */
    accum(n->parents[1], t2);
    tt_release(t2);
}

static void backward_addscalar(AGNode *n) { accum(n->parents[0], n->grad); }

static void backward_mulscalar(AGNode *n) {
    Tensor *t = tt_mulscalar(n->grad, n->scalar);
    accum(n->parents[0], t);
    tt_release(t);
}

static void backward_matmul(AGNode *n) {
    const AGNode *a = n->parents[0], *b = n->parents[1];
    Tensor *ga = matmul_nt(n->grad, b->val);      /* g @ b^T */
    accum((AGNode *)a, ga);
    tt_release(ga);
    Tensor *gb = matmul_tn(a->val, n->grad);      /* a^T @ g */
    accum((AGNode *)b, gb);
    tt_release(gb);
}

static void backward_relu(AGNode *n) {
    const Tensor *x = n->parents[0]->val;
    Tensor *g = n->grad;
    Tensor *t = new_like(x);
    if (!t) return;
    for (long i = 0; i < t->numel; i++)
        t->data[i] = x->data[i] > 0.0f ? g->data[i] : 0.0f;
    accum(n->parents[0], t);
    tt_release(t);
}

static void backward_softmax(AGNode *n) {
    const Tensor *s = n->val;                     /* softmax output */
    const Tensor *g = n->grad;
    const int N = s->shape[s->ndim - 1];
    const long rows = s->numel / N;
    Tensor *t = new_like(s);
    if (!t) return;
    for (long r = 0; r < rows; r++) {
        const float *sr = s->data + r * N;
        const float *gr = g->data + r * N;
        float dot = 0.0f;
        for (int j = 0; j < N; j++) dot += gr[j] * sr[j];
        float *tr = t->data + r * N;
        for (int j = 0; j < N; j++) tr[j] = sr[j] * (gr[j] - dot);
    }
    accum(n->parents[0], t);
    tt_release(t);
}

static void backward_padones(AGNode *n) {
    /* parent grad += g[:, :-1] */
    const Tensor *g = n->grad;
    const int M = g->shape[0], K = n->parents[0]->val->shape[1];
    Tensor *t = new_like(n->parents[0]->val);
    if (!t) return;
    for (int m = 0; m < M; m++)
        memcpy(t->data + (long)m * K, g->data + (long)m * (K + 1),
               sizeof(float) * (size_t)K);
    accum(n->parents[0], t);
    tt_release(t);
}

static void backward_reshape(AGNode *n) {
    if (!n->parents[0]->requires_grad) return;
    long orig_shape[8];
    for (int i = 0; i < n->parents[0]->val->ndim; i++)
        orig_shape[i] = n->parents[0]->val->shape[i];
    Tensor *g_reshaped = tt_reshape(n->grad, orig_shape, n->parents[0]->val->ndim);
    accum(n->parents[0], g_reshaped);
    tt_release(g_reshaped);
}

static void batched_im2col_local(const float *data_im, int N, int C, int H, int W_in,
                                 int HH, int WW, int pad_h, int pad_w,
                                 int stride_h, int stride_w, float *data_col) {
    if (HH <= 0 || WW <= 0 || stride_h <= 0 || stride_w <= 0 || pad_h < 0 || pad_w < 0) return;
    int Hout = (H + 2 * pad_h - HH) / stride_h + 1;
    int Wout = (W_in + 2 * pad_w - WW) / stride_w + 1;
    int channels_col = C * HH * WW;
    int N_spatial = Hout * Wout;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int c = 0; c < channels_col; c++) {
        for (int n = 0; n < N; n++) {
            int w_offset = c % WW;
            int h_offset = (c / WW) % HH;
            int c_im = c / (HH * WW);
            const float *im_n = data_im + (long)n * C * H * W_in + (long)c_im * H * W_in;
            float *col_c_n = data_col + (long)c * (N * N_spatial) + (long)n * N_spatial;

            for (int ho = 0; ho < Hout; ho++) {
                int im_row = ho * stride_h - pad_h + h_offset;
                for (int wo = 0; wo < Wout; wo++) {
                    int im_col = wo * stride_w - pad_w + w_offset;
                    int out_idx = ho * Wout + wo;
                    if (im_row >= 0 && im_row < H && im_col >= 0 && im_col < W_in)
                        col_c_n[out_idx] = im_n[(long)im_row * W_in + im_col];
                    else
                        col_c_n[out_idx] = 0.0f;
                }
            }
        }
    }
}

static void batched_col2im_local(const float *data_col, int N, int C, int H, int W_in,
                                 int HH, int WW, int pad_h, int pad_w,
                                 int stride_h, int stride_w, float *data_im) {
    if (HH <= 0 || WW <= 0 || stride_h <= 0 || stride_w <= 0 || pad_h < 0 || pad_w < 0) return;
    memset(data_im, 0, sizeof(float) * (size_t)N * C * H * W_in);
    int Hout = (H + 2 * pad_h - HH) / stride_h + 1;
    int Wout = (W_in + 2 * pad_w - WW) / stride_w + 1;
    int channels_col = C * HH * WW;
    int N_spatial = Hout * Wout;

    for (int c = 0; c < channels_col; c++) {
        int w_offset = c % WW;
        int h_offset = (c / WW) % HH;
        int c_im = c / (HH * WW);
        for (int n = 0; n < N; n++) {
            float *im_n_c = data_im + (long)n * C * H * W_in + (long)c_im * H * W_in;
            const float *col_c_n = data_col + (long)c * (N * N_spatial) + (long)n * N_spatial;
            for (int ho = 0; ho < Hout; ho++) {
                int im_row = ho * stride_h - pad_h + h_offset;
                for (int wo = 0; wo < Wout; wo++) {
                    int im_col = wo * stride_w - pad_w + w_offset;
                    if (im_row >= 0 && im_row < H && im_col >= 0 && im_col < W_in) {
                        im_n_c[(long)im_row * W_in + im_col] += col_c_n[ho * Wout + wo];
                    }
                }
            }
        }
    }
}

/* C(MxN) = A(MxK) * B(NxK)^T with B kept row-major: the KxN transpose
 * is fused into the inner-product read, so no data_col_t buffer is
 * needed. K-loop stays sequential per output to match GEMM order. */
static void sgemm_A_Bt_local(int M, int N, long K,
                             const float *A, const float *B, float *C) {
    if (M <= 0 || N <= 0 || K <= 0 || !A || !B || !C) return;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            const float *a = A + (long)i * K;
            const float *b = B + (long)j * K;
            float sum = 0.0f;
            for (long k = 0; k < K; k++) sum += a[k] * b[k];
            C[(long)i * N + j] = sum;
        }
    }
}

static void backward_conv2d(AGNode *n) {
    AGNode *a = n->parents[0], *w = n->parents[1];
    AGNode *b = n->nparents > 2 ? n->parents[2] : NULL;
    int sh = n->params[0], sw = n->params[1], ph = n->params[2], pw = n->params[3];
    Tensor *g = n->grad;

    int N = a->val->shape[0], C = a->val->shape[1], H = a->val->shape[2], W_in = a->val->shape[3];
    int F = w->val->shape[0], HH = w->val->shape[2], WW = w->val->shape[3];
    if (HH <= 0 || WW <= 0 || sh <= 0 || sw <= 0 || ph < 0 || pw < 0) return;
    int Hout = g->shape[2], Wout = g->shape[3];

    if (b && b->requires_grad) {
        long b_shp[1] = {F};
        Tensor *gb = tt_new(b_shp, 1);
        if (gb) {
            for (int f = 0; f < F; f++) {
                float sum = 0.0f;
                for (int n_idx = 0; n_idx < N; n_idx++)
                    for (int ho = 0; ho < Hout; ho++)
                        for (int wo = 0; wo < Wout; wo++)
                            sum += g->data[((long)n_idx * F + f) * Hout * Wout + (long)ho * Wout + wo];
                gb->data[f] = sum;
            }
            accum(b, gb);
            tt_release(gb);
        }
    }

    int K_col = C * HH * WW;
    long N_col = (long)N * Hout * Wout;

    if (w->requires_grad) {
        long w_shp[4] = {F, C, HH, WW};
        Tensor *gw = tt_new(w_shp, 4);
        float *data_col = (float *)malloc(sizeof(float) * (size_t)K_col * N_col);
        float *g_perm = NULL;
        if (gw && data_col)
            g_perm = (float *)malloc(sizeof(float) * (size_t)F * N_col);
        if (gw && data_col && g_perm) {
            batched_im2col_local(a->val->data, N, C, H, W_in, HH, WW, ph, pw, sh, sw, data_col);

            long N_spatial = Hout * Wout;
            for (int f = 0; f < F; f++) {
                for (int n_idx = 0; n_idx < N; n_idx++) {
                    const float *src = g->data + ((long)n_idx * F + f) * N_spatial;
                    float *dst = g_perm + (long)f * N_col + (long)n_idx * N_spatial;
                    memcpy(dst, src, sizeof(float) * N_spatial);
                }
            }
            /* fused: gw = g_perm * data_col^T, no KxN transpose buffer,
             * no fromdata wrapper copies, GEMM writes gw->data directly */
            sgemm_A_Bt_local(F, K_col, N_col, g_perm, data_col, gw->data);
            accum(w, gw);
        }
        free(data_col);
        free(g_perm);
        if (gw) tt_release(gw);
    }

    if (a->requires_grad) {
        long a_shp[4] = {N, C, H, W_in};
        Tensor *ga = tt_new(a_shp, 4);
        float *w_mat_t = (float *)malloc(sizeof(float) * (size_t)F * K_col);
        float *g_perm = (float *)malloc(sizeof(float) * (size_t)F * N_col);

        if (ga && w_mat_t && g_perm) {
            for (int f = 0; f < F; f++)
                for (int k = 0; k < K_col; k++)
                    w_mat_t[k * F + f] = w->val->data[f * K_col + k];

            long N_spatial = Hout * Wout;
            for (int f = 0; f < F; f++) {
                for (int n_idx = 0; n_idx < N; n_idx++) {
                    const float *src = g->data + ((long)n_idx * F + f) * N_spatial;
                    float *dst = g_perm + (long)f * N_col + (long)n_idx * N_spatial;
                    memcpy(dst, src, sizeof(float) * N_spatial);
                }
            }

            long shape_w_t[2] = {K_col, F};
            long shape_g_mat[2] = {F, N_col};
            Tensor *w_t = tt_fromdata(w_mat_t, shape_w_t, 2);
            Tensor *g_mat = tt_fromdata(g_perm, shape_g_mat, 2);
            Tensor *d_col = tt_matmul_omp(w_t, g_mat, 12);

            if (d_col) {
                batched_col2im_local(d_col->data, N, C, H, W_in, HH, WW, ph, pw, sh, sw, ga->data);
                tt_release(d_col);
            }
            tt_release(w_t);
            tt_release(g_mat);
            accum(a, ga);
            tt_release(ga);
        }
        free(w_mat_t);
        free(g_perm);
    }
}

static void backward_maxpool2d(AGNode *n) {
    AGNode *a = n->parents[0];
    if (!a->requires_grad) return;
    int ph = n->params[0], pw = n->params[1], sh = n->params[2], sw = n->params[3];
    Tensor *g = n->grad;
    int N = a->val->shape[0], C = a->val->shape[1], H = a->val->shape[2], W = a->val->shape[3];
    int Hout = g->shape[2], Wout = g->shape[3];
    if (ph <= 0 || pw <= 0 || sh <= 0 || sw <= 0) return;

    long a_shp[4] = {N, C, H, W};
    Tensor *ga = tt_new(a_shp, 4);
    if (!ga) return;

    for (int n_idx = 0; n_idx < N; n_idx++) {
        for (int c = 0; c < C; c++) {
            for (int ho = 0; ho < Hout; ho++) {
                int h_start = ho * sh;
                for (int wo = 0; wo < Wout; wo++) {
                    int w_start = wo * sw;
                    float g_val = g->data[((long)n_idx * C + c) * Hout * Wout + (long)ho * Wout + wo];
                    float max_val = -INFINITY;
                    int max_h = h_start, max_w = w_start;
                    for (int kh = 0; kh < ph; kh++) {
                        for (int kw = 0; kw < pw; kw++) {
                            float v = a->val->data[((long)n_idx * C + c) * H * W + (long)(h_start + kh) * W + (w_start + kw)];
                            if (v > max_val) {
                                max_val = v;
                                max_h = h_start + kh;
                                max_w = w_start + kw;
                            }
                        }
                    }
                    ga->data[((long)n_idx * C + c) * H * W + (long)max_h * W + max_w] += g_val;
                }
            }
        }
    }
    accum(a, ga);
    tt_release(ga);
}

static void backward_avgpool2d(AGNode *n) {
    AGNode *a = n->parents[0];
    if (!a->requires_grad) return;
    int ph = n->params[0], pw = n->params[1], sh = n->params[2], sw = n->params[3];
    Tensor *g = n->grad;
    int N = a->val->shape[0], C = a->val->shape[1], H = a->val->shape[2], W = a->val->shape[3];
    int Hout = g->shape[2], Wout = g->shape[3];
    if (ph <= 0 || pw <= 0 || sh <= 0 || sw <= 0) return;
    float norm = 1.0f / (float)(ph * pw);

    long a_shp[4] = {N, C, H, W};
    Tensor *ga = tt_new(a_shp, 4);
    if (!ga) return;

    for (int n_idx = 0; n_idx < N; n_idx++) {
        for (int c = 0; c < C; c++) {
            for (int ho = 0; ho < Hout; ho++) {
                int h_start = ho * sh;
                for (int wo = 0; wo < Wout; wo++) {
                    int w_start = wo * sw;
                    float g_val = g->data[((long)n_idx * C + c) * Hout * Wout + (long)ho * Wout + wo] * norm;
                    for (int kh = 0; kh < ph; kh++) {
                        for (int kw = 0; kw < pw; kw++) {
                            ga->data[((long)n_idx * C + c) * H * W + (long)(h_start + kh) * W + (w_start + kw)] += g_val;
                        }
                    }
                }
            }
        }
    }
    accum(a, ga);
    tt_release(ga);
}

/* ---------- topological sort + propagation ---------- */

typedef struct {
    AGNode **items;
    size_t len, cap;
} NodeList;

static int nl_push(NodeList *l, AGNode *n) {
    if (l->len == l->cap) {
        size_t nc = l->cap ? l->cap * 2 : 64;
        AGNode **ni = (AGNode **)realloc(l->items, nc * sizeof(AGNode *));
        if (!ni) return 0;
        l->items = ni;
        l->cap = nc;
    }
    l->items[l->len++] = n;
    return 1;
}

/* iterative postorder DFS; parents pushed after children */
static int topo_sort(AGNode *root, NodeList *order) {
    typedef struct { AGNode *n; int next; } Frame;
    size_t cap = 64, top = 0;
    Frame *stack = (Frame *)malloc(cap * sizeof(Frame));
    if (!stack) return 0;
    stack[top++] = (Frame){root, 0};
    root->mark = 1;
    while (top > 0) {
        Frame *f = &stack[top - 1];
        if (f->next < f->n->nparents) {
            AGNode *c = f->n->parents[f->next++];
            if (c->mark == 0) {
                if (top == cap) {
                    cap *= 2;
                    Frame *ns = (Frame *)realloc(stack, cap * sizeof(Frame));
                    if (!ns) { free(stack); return 0; }
                    stack = ns;
                    f = &stack[top - 1];  /* realloc may move */
                }
                c->mark = 1;
                stack[top++] = (Frame){c, 0};
            }
        } else {
            f->n->mark = 2;
            if (!nl_push(order, f->n)) { free(stack); return 0; }
            top--;
        }
    }
    free(stack);
    return 1;
}

static void propagate(NodeList *order) {
    for (size_t i = order->len; i-- > 0;) {
        AGNode *n = order->items[i];
        if (!n->grad || n->nparents == 0) continue;
        switch (n->kind) {
            case OP_ADD:       backward_add(n); break;
            case OP_MUL:       backward_mul(n); break;
            case OP_ADDSCALAR: backward_addscalar(n); break;
            case OP_MULSCALAR: backward_mulscalar(n); break;
            case OP_MATMUL:    backward_matmul(n); break;
            case OP_RELU:      backward_relu(n); break;
            case OP_SOFTMAX:   backward_softmax(n); break;
            case OP_PADONES:   backward_padones(n); break;
            case OP_RESHAPE:   backward_reshape(n); break;
            case OP_CONV2D:    backward_conv2d(n); break;
            case OP_MAXPOOL2D: backward_maxpool2d(n); break;
            case OP_AVGPOOL2D: backward_avgpool2d(n); break;
            default: break;
        }
    }
}

void ag_backward_from(AGNode *out, const float *seed, size_t seed_numel) {
    if (!out || !out->requires_grad) return;
    if (seed && (!out->val || seed_numel != (size_t)out->val->numel)) return;
    NodeList order = {0};
    if (!topo_sort(out, &order)) {
        free(order.items);
        return;
    }
    if (!out->grad)
        out->grad = new_like(out->val);
    if (out->grad) {
        if (seed)
            memcpy(out->grad->data, seed,
                   sizeof(float) * (size_t)out->val->numel);
        else
            for (long i = 0; i < out->grad->numel; i++)
                out->grad->data[i] = 1.0f;
        propagate(&order);
    }
    for (size_t i = 0; i < order.len; i++) order.items[i]->mark = 0;
    free(order.items);
}

void ag_backward(AGNode *out) { ag_backward_from(out, NULL, 0); }

/* ---------- lifecycle ---------- */

AGNode *ag_retain(AGNode *n) {
    if (!n) return NULL;
    n->refcount++;
    return n;
}

void ag_release(AGNode *n) {
    if (!n) return;
    if (--n->refcount > 0) return;
    tt_release(n->val);
    tt_release(n->grad);
    for (int i = 0; i < n->nparents; i++)
        ag_release(n->parents[i]);
    free(n);
}
