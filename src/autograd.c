#include "autograd.h"
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
    OP_PADONES
} OpKind;

struct AGNode {
    Tensor *val;
    Tensor *grad;
    AGNode *parents[2];
    int     nparents;
    OpKind  kind;
    float   scalar;
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
    for (int k = 0; k < K; k++)
        for (int m = 0; m < M; m++) {
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
    return node_binary(a, b, tt_matmul(a->val, b->val), OP_MATMUL);
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
            default: break;
        }
    }
}

void ag_backward_from(AGNode *out, const float *seed) {
    if (!out || !out->requires_grad) return;
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

void ag_backward(AGNode *out) { ag_backward_from(out, NULL); }

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
