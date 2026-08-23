/* Accelerated backward_conv2d using im2col / col2im + tt_matmul_omp */
#include "autograd.h"
#include "tensor.h"
#include <stdlib.h>
#include <string.h>

static void col2im(const float *data_col, int channels, int height, int width,
                   int kernel_h, int kernel_w, int pad_h, int pad_w,
                   int stride_h, int stride_w, float *data_im) {
    memset(data_im, 0, sizeof(float) * (size_t)channels * height * width);
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;

    for (int c = 0; c < channels_col; c++) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / (kernel_h * kernel_w);
        for (int h = 0; h < height_col; h++) {
            int im_row = h * stride_h - pad_h + h_offset;
            for (int w = 0; w < width_col; w++) {
                int im_col = w * stride_w - pad_w + w_offset;
                if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
                    int col_index = (c * height_col + h) * width_col + w;
                    data_im[(c_im * height + im_row) * width + im_col] += data_col[col_index];
                }
            }
        }
    }
}

/* Transpose 2D tensor: A(M, N) -> At(N, M) */
static Tensor *tensor_transpose_2d(const Tensor *a) {
    if (!a || a->ndim != 2) return NULL;
    long M = a->shape[0], N = a->shape[1];
    long shp[2] = {N, M};
    Tensor *out = tt_new(shp, 2);
    if (!out) return NULL;
    for (long i = 0; i < M; i++)
        for (long j = 0; j < N; j++)
            out->data[j * M + i] = a->data[i * N + j];
    return out;
}
