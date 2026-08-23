/* spatial & shape ops for M4 with batched im2col + single GEMM acceleration */
#include "tensor.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

Tensor *tt_reshape(const Tensor *a, const long *new_shape, int new_ndim) {
    if (!a || !new_shape || new_ndim <= 0 || new_ndim > 8) return NULL;
    long numel = 1;
    for (int i = 0; i < new_ndim; i++) {
        if (new_shape[i] <= 0) return NULL;
        numel *= new_shape[i];
    }
    if (numel != a->numel) return NULL;
    Tensor *out = tt_new(new_shape, new_ndim);
    if (!out) return NULL;
    memcpy(out->data, a->data, sizeof(float) * (size_t)numel);
    return out;
}

/* Batched im2col: fills data_col of shape (C * HH * WW, N * Hout * Wout) */
static void batched_im2col(const float *data_im, int N, int C, int H, int W_in,
                           int HH, int WW, int pad_h, int pad_w,
                           int stride_h, int stride_w, float *data_col) {
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

Tensor *tt_conv2d(const Tensor *a, const Tensor *w, const Tensor *b,
                  int stride_h, int stride_w, int pad_h, int pad_w) {
    if (!a || !w || a->ndim != 4 || w->ndim != 4) return NULL;
    int N = a->shape[0], C = a->shape[1], H = a->shape[2], W_in = a->shape[3];
    int F = w->shape[0], C_w = w->shape[1], HH = w->shape[2], WW = w->shape[3];
    if (C != C_w) return NULL;
    if (b && (b->ndim != 1 || b->shape[0] != F)) return NULL;

    int Hout = (H + 2 * pad_h - HH) / stride_h + 1;
    int Wout = (W_in + 2 * pad_w - WW) / stride_w + 1;
    if (Hout <= 0 || Wout <= 0) return NULL;

    long shp[4] = {N, F, Hout, Wout};
    Tensor *out = tt_new(shp, 4);
    if (!out) return NULL;

    int K_col = C * HH * WW;
    long N_col = (long)N * Hout * Wout;
    float *data_col = (float *)malloc(sizeof(float) * (size_t)K_col * N_col);
    if (!data_col) {
        tt_release(out);
        return NULL;
    }

    batched_im2col(a->data, N, C, H, W_in, HH, WW, pad_h, pad_w, stride_h, stride_w, data_col);

    long shape_w_mat[2] = {F, K_col};
    long shape_col_mat[2] = {K_col, N_col};
    Tensor *w_mat = tt_fromdata(w->data, shape_w_mat, 2);
    Tensor *col_mat = tt_fromdata(data_col, shape_col_mat, 2);

    Tensor *out_mat = tt_matmul_omp(w_mat, col_mat, 12);

    if (out_mat) {
        long N_spatial = Hout * Wout;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int n = 0; n < N; n++) {
            for (int f = 0; f < F; f++) {
                float bias_val = b ? b->data[f] : 0.0f;
                const float *src = out_mat->data + (long)f * N_col + (long)n * N_spatial;
                float *dst = out->data + ((long)n * F + f) * N_spatial;
                for (long i = 0; i < N_spatial; i++) {
                    dst[i] = src[i] + bias_val;
                }
            }
        }
        tt_release(out_mat);
    }

    tt_release(w_mat);
    tt_release(col_mat);
    free(data_col);
    return out;
}

Tensor *tt_maxpool2d(const Tensor *a, int pool_h, int pool_w,
                     int stride_h, int stride_w) {
    if (!a || a->ndim != 4) return NULL;
    int N = a->shape[0], C = a->shape[1], H = a->shape[2], W = a->shape[3];
    int Hout = (H - pool_h) / stride_h + 1;
    int Wout = (W - pool_w) / stride_w + 1;
    if (Hout <= 0 || Wout <= 0) return NULL;

    long shp[4] = {N, C, Hout, Wout};
    Tensor *out = tt_new(shp, 4);
    if (!out) return NULL;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ho = 0; ho < Hout; ho++) {
                int h_start = ho * stride_h;
                for (int wo = 0; wo < Wout; wo++) {
                    int w_start = wo * stride_w;
                    float max_val = -1e30f;
                    for (int kh = 0; kh < pool_h; kh++) {
                        for (int kw = 0; kw < pool_w; kw++) {
                            float v = a->data[((long)n * C + c) * H * W + (long)(h_start + kh) * W + (w_start + kw)];
                            if (v > max_val) max_val = v;
                        }
                    }
                    out->data[((long)n * C + c) * Hout * Wout + (long)ho * Wout + wo] = max_val;
                }
            }
        }
    }
    return out;
}

Tensor *tt_avgpool2d(const Tensor *a, int pool_h, int pool_w,
                     int stride_h, int stride_w) {
    if (!a || a->ndim != 4) return NULL;
    int N = a->shape[0], C = a->shape[1], H = a->shape[2], W = a->shape[3];
    int Hout = (H - pool_h) / stride_h + 1;
    int Wout = (W - pool_w) / stride_w + 1;
    if (Hout <= 0 || Wout <= 0) return NULL;

    long shp[4] = {N, C, Hout, Wout};
    Tensor *out = tt_new(shp, 4);
    if (!out) return NULL;
    float norm = 1.0f / (float)(pool_h * pool_w);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ho = 0; ho < Hout; ho++) {
                int h_start = ho * stride_h;
                for (int wo = 0; wo < Wout; wo++) {
                    int w_start = wo * stride_w;
                    float sum = 0.0f;
                    for (int kh = 0; kh < pool_h; kh++) {
                        for (int kw = 0; kw < pool_w; kw++) {
                            sum += a->data[((long)n * C + c) * H * W + (long)(h_start + kh) * W + (w_start + kw)];
                        }
                    }
                    out->data[((long)n * C + c) * Hout * Wout + (long)ho * Wout + wo] = sum * norm;
                }
            }
        }
    }
    return out;
}
