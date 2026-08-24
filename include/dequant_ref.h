#ifndef DEQUANT_REF_H
#define DEQUANT_REF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Reference CPU dequantization for every Tier-1 GGML quant type.
 *
 * Math semantics copied from llama.cpp ggml/src/ggml-quants.c
 * (dequantize_row_q4_0/q4_1/q5_0/q5_1/q8_0/q4_K/q5_K/q6_K).
 * Format/layout definitions come from ggml/src/ggml-common.h.
 * Credit: GGML / llama.cpp authors (https://github.com/ggml-org/llama.cpp).
 */

/* Type codes mirror GGML_TYPE_* from oracle/llama.cpp/ggml/include/ggml.h */
enum {
    TTQ_F32  = 0,
    TTQ_F16  = 1,
    TTQ_Q4_0 = 2,
    TTQ_Q4_1 = 3,   /* 20 B per 32 values: fp16 d, fp16 m, 16 nibble bytes */
    TTQ_Q5_0 = 6,   /* 22 B per 32 values: fp16 d, 4 high-bit bytes, 16 nibbles */
    TTQ_Q5_1 = 7,   /* 24 B per 32 values: like q5_0 + fp16 m */
    TTQ_Q8_0 = 8,   /* 34 B per 32 values: fp16 d, 32 int8 */
    TTQ_Q4_K = 12,  /* 144 B per 256 values (Q4_K_S files share this layout) */
    TTQ_Q5_K = 13,  /* 176 B per 256 values (Q5_K_S likewise) */
    TTQ_Q6_K = 14,  /* 210 B per 256 values */
};

/*
 * Load named tensor from a GGUF file and dequantize it row-major into `out`
 * (float32). `type_code` must be one of the TTQ_* values above and must match
 * the tensor's on-disk type. Returns the number of floats written, or a
 * negative value on error (-1 open/load, -2 tensor not found, -3 type
 * mismatch, -4 out_cap too small, -5 unsupported type).
 */
long ttq_roundtrip(const char *gguf_path, const char *tensor_name,
                   int type_code, float *out, long out_cap);

/*
 * Dequantize `numel` values of raw block data (`type_code` = TTQ_*) into
 * float32 `out` (out must hold numel floats). Reusable entry point for the
 * golden GPU GEMV tests: same math as ttq_roundtrip without GGUF I/O.
 * Returns numel, or -5 on unsupported type / bad alignment.
 */
long ttq_dequant(const void *data, int type_code, long numel, float *out);

#ifdef __cplusplus
}
#endif

#endif /* DEQUANT_REF_H */
