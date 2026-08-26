// IQ3_XXS dequant + GEMV prototype (E4B-fits-in-4GB enabler feasibility).
//
// ============================================================================
// LAYOUT (mirrors oracle/llama.cpp/ggml/src/ggml-{common.h,quants.c})
// ============================================================================
//
// GGML type code:    GGML_TYPE_IQ3_XXS = 18   (oracle/.../ggml/include/ggml.h:408)
// Block size:        QK_K = 256 elements per super-block
// Bytes per block:   sizeof(ggml_half) + 3*(QK_K/8) = 2 + 96 = 98 bytes
// Effective rate:    3.0625 bpw  (256*3 / 8 = 96 quants, +2 byte fp16 d)
//
// struct block_iq3_xxs {                 // ggml-common.h
//     ggml_half d;            // @0   fp16 super-block scale
//     uint8_t   qs[3*32];     // @2   packed 3-bit quants (96 bytes)
//                             //        Each 32-element sub-block is 32*3/8 = 12 bytes,
//                             //        so qs holds 8 sub-blocks (256 elts total).
// };
//
// qs[0..11]   = 3-bit codes for elements  0..31
// qs[12..23]  = 3-bit codes for elements 32..63
// ...         = 3-bit codes for elements 224..255
//
// The "scales_and_signs" packed fields the dequant routine uses are NOT
// stored separately; they are EMBEDDED in the high bits of qs[0..95].
// The layout (from dequantize_row_iq3_xxs, ggml-quants.c:2575):
//
//   for ib32 in 0..7:        # 8 sub-blocks of 32 elts
//     aux32 = u32 at qs[64 + 4*ib32]   # the "scales_and_signs" word
//     db    = d * (0.5 + (aux32>>28)) * 0.5     # 4-bit sub-scale
//     for l in 0..3:                       # 4 lane groups of 8
//       signs = ksigns_iq2xs[(aux32 >> 7*l) & 127]   # 7-bit sign pattern
//       idx1  = qs[ib32*8 + 2*l + 0]                 # 3-bit grid idx (j+0..3)
//       idx2  = qs[ib32*8 + 2*l + 1]                 # 3-bit grid idx (j+4..7)
//       for j in 0..3:
//         y[j+0] = db * iq3xxs_grid[idx1][j] * sign
//         y[j+4] = db * iq3xxs_grid[idx2][j] * sign
//
// iq3xxs_grid is uint32_t[256] (ggml-common.h:1017); each entry packs 4 bytes
// whose nibble values (0x0/0x1/0x2/0x3 mapped to 0,1,2,3 with a small
// additive bias) are the 3-bit quantized levels.
//
// ============================================================================
// TABLES (inlined; identical content to ggml-common.h, MIT-licensed upstream)
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define IQ3_BLK        98       /* 2 (d) + 96 (qs) bytes per 256-element block */
#define QK             256
#define NB8            (QK/32)  /* 8 sub-blocks of 32 inside one super-block */

static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

// ====================== HOST dequant ======================
// Mirrors ggml-quants.c:dequantize_row_iq3_xxs. Returns y[k] as fp32.
static void host_dequant_iq3_xxs(const uint8_t *W, float *y, int64_t k) {
    assert(k % QK == 0);
    const int64_t nb = k / QK;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = __half2float(*(const __half *)(W + i*IQ3_BLK));
        const uint8_t *qs = W + i*IQ3_BLK + 2;             /* 96 bytes */
        const uint8_t *sc  = qs + QK/4;                    /* @64..95 */
        float *yp = y + i*QK;
        for (int ib32 = 0; ib32 < NB8; ++ib32) {
            uint32_t aux;
            memcpy(&aux, sc + 4*ib32, sizeof(aux));
            const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
            const uint8_t *qb = qs + 8*ib32;                /* 12 bytes... */
            /* (only first 8 used for the 4 lane groups; 4 trailing bytes
               are padding-like but per the spec qs is 12 bytes for 32
               elements of 3-bit packing: 32*3/8 = 12). */
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux >> 7*l) & 127];
                const uint8_t *g1 = (const uint8_t *)(iq3xxs_grid + qb[2*l + 0]);
                const uint8_t *g2 = (const uint8_t *)(iq3xxs_grid + qb[2*l + 1]);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    yp[j+0] = db * (float)g1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    yp[j+4] = db * (float)g2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                yp += 8;
            }
        }
    }
}

// Random-but-deterministic IQ3_XXS buffer fabricator.
// We need *plausible* scales (fp16) and *plausible* qs bytes — this
// is not a real quantizer, just enough structure to make the dequant
// path exercise all branches (nonzero d, varied aux, varied grid idx).
static uint32_t seed_state = 0x12345;
static uint32_t lcg(void) {
    seed_state = seed_state * 1664525u + 1013904223u;
    return seed_state;
}
static void fabricate_iq3_xxs(uint8_t *W, int64_t numel) {
    const int64_t nb = numel / QK;
    for (int64_t i = 0; i < nb; ++i) {
        /* fp16 d in [0.01, 0.5] */
        float d_f = 0.01f + (lcg() & 0xff) / 510.0f;
        __half dh = __float2half(d_f);
        memcpy(W + i*IQ3_BLK, &dh, 2);
        uint8_t *qs = W + i*IQ3_BLK + 2;
        for (int j = 0; j < 96; ++j) qs[j] = (uint8_t)(lcg() & 0xff);
    }
}

// ====================== DEVICE constant mirrors ======================
__device__ __constant__ uint8_t d_kmask_iq2xs[8]    = {1,2,4,8,16,32,64,128};
__device__ __constant__ uint8_t d_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
__device__ __constant__ uint32_t d_iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// Read one 8-element contribution (lane group l inside sub-block ib32 of
// super-block pointed at by `blk`). Returns the partial accumulator for
// the 8 x[] values at base offset `(ib32*32 + l*8)`.
__device__ __forceinline__ void iq3_dequant_8(const uint8_t *blk,
                                              int ib32, int l,
                                              float *out8) {
    __half dh;
    memcpy(&dh, blk, 2);
    const float d = __half2float(dh);
    const uint8_t *qs = blk + 2;
    const uint8_t *sc  = qs + QK/4;          /* @64..95 within the 98B */
    uint32_t aux; memcpy(&aux, sc + 4*ib32, sizeof(aux));
    const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    const uint8_t signs = d_ksigns_iq2xs[(aux >> 7*l) & 127];
    const uint8_t *qb   = qs + 8*ib32;
    const uint8_t *g1   = (const uint8_t *)(d_iq3xxs_grid + qb[2*l + 0]);
    const uint8_t *g2   = (const uint8_t *)(d_iq3xxs_grid + qb[2*l + 1]);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        out8[j+0] = db * (float)g1[j] * (signs & d_kmask_iq2xs[j+0] ? -1.f : 1.f);
        out8[j+4] = db * (float)g2[j] * (signs & d_kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

// IQ3_XXS GEMV: y[r] = sum_k dequant(W[r,k]) * x[k].
// One warp per output row; each lane strides super-blocks, dequant-inlines
// 8 values, multiplies by x[].
__global__ void k_gemv_iq3_xxs(const uint8_t *__restrict__ W,
                               const float  *__restrict__ x,
                               float        *__restrict__ y,
                               int M, int K) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int nb   = K / QK;                    /* super-blocks per row */
    const uint8_t *rw = W + (long)row * nb * IQ3_BLK;
    float s = 0.0f;
    float tmp[8];
    for (int b = lane; b < nb; b += 32) {
        const uint8_t *blk = rw + b * IQ3_BLK;
        #pragma unroll
        for (int ib32 = 0; ib32 < NB8; ++ib32) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                iq3_dequant_8(blk, ib32, l, tmp);
                const int base = b * QK + ib32*32 + l*8;
                #pragma unroll
                for (int j = 0; j < 8; ++j) s += tmp[j] * x[base + j];
            }
        }
    }
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

// Shared-mem x[] staging across rows in a block: each block handles
// WARPS_PER_BLOCK rows; load the whole x[] (K floats) once into shmem
// and let every warp reuse it.
template <int WARPS_PER_BLOCK>
__global__ void k_gemv_iq3_xxs_shm(const uint8_t *__restrict__ W,
                                   const float  *__restrict__ x,
                                   float        *__restrict__ y,
                                   int M, int K) {
    extern __shared__ float xs[];                /* K floats */
    const int tid  = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth  = blockDim.x * blockDim.y;
    for (int i = tid; i < K; i += nth) xs[i] = x[i];
    __syncthreads();

    const int row = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int nb   = K / QK;
    const uint8_t *rw = W + (long)row * nb * IQ3_BLK;
    float s = 0.0f;
    float tmp[8];
    for (int b = lane; b < nb; b += 32) {
        const uint8_t *blk = rw + b * IQ3_BLK;
        #pragma unroll
        for (int ib32 = 0; ib32 < NB8; ++ib32) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                iq3_dequant_8(blk, ib32, l, tmp);
                const int base = b * QK + ib32*32 + l*8;
                #pragma unroll
                for (int j = 0; j < 8; ++j) s += tmp[j] * xs[base + j];
            }
        }
    }
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

// ====================== launchers ======================
extern "C" int gemv_iq3xxs(const void *W, const float *x, float *y, int M, int K) {
    if (K % QK != 0) return -1;
    const int WARPS_PER_BLOCK = 4;
    dim3 b(32, WARPS_PER_BLOCK);
    dim3 g((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    k_gemv_iq3_xxs<<<g, b>>>((const uint8_t *)W, x, y, M, K);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}
extern "C" int gemv_iq3xxs_shm(const void *W, const float *x, float *y, int M, int K) {
    if (K % QK != 0) return -1;
    const int WARPS_PER_BLOCK = 4;
    dim3 b(32, WARPS_PER_BLOCK);
    dim3 g((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    size_t shm = (size_t)K * sizeof(float);
    k_gemv_iq3_xxs_shm<WARPS_PER_BLOCK><<<g, b, shm>>>((const uint8_t *)W, x, y, M, K);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}
extern "C" int dequant_iq3xxs(const void *W, float *y, int K) {
    if (K % QK != 0) return -1;
    host_dequant_iq3_xxs((const uint8_t *)W, y, K);
    return 0;
}

// ====================== timing + self-test ======================
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int run_self_test(int M, int K) {
    /* K must be multiple of 256 */
    if (K % QK != 0) { fprintf(stderr, "K must be %%256\n"); return 1; }
    const int64_t nelem = (int64_t)M * K;
    const int64_t wbytes = (nelem / QK) * IQ3_BLK;
    fprintf(stderr, "[self] M=%d K=%d  W=%.2f MB  x=%.2f MB  y=%.2f KB\n",
            M, K, wbytes/1e6, nelem*4.0/1e6, M*4.0/1e3);
    uint8_t  *W   = (uint8_t  *)malloc(wbytes);
    float    *x   = (float    *)malloc(nelem * sizeof(float));
    float    *yref= (float    *)malloc(M * sizeof(float));
    float    *ygpu= (float    *)malloc(M * sizeof(float));
    if (!W || !x || !yref || !ygpu) { fprintf(stderr, "OOM host\n"); return 2; }

    seed_state = 0xC0FFEE;
    fabricate_iq3_xxs(W, nelem);
    seed_state = 0xBEEF;
    for (int64_t i = 0; i < nelem; ++i) x[i] = ((float)((lcg() & 0xfff)) - 2048.f) * 0.01f;

    /* host reference: dequant row 0 then dot with x[0..K-1], etc. */
    /* we'll dequant the whole matrix and matmul on host for reference */
    float *Wfp32 = (float *)malloc(nelem * sizeof(float));
    if (!Wfp32) { fprintf(stderr, "OOM Wfp32\n"); return 2; }
    fprintf(stderr, "[self] host dequant %lld elts ...\n", (long long)nelem);
    double t0 = now_sec();
    host_dequant_iq3_xxs(W, Wfp32, nelem);
    double t1 = now_sec();
    fprintf(stderr, "[self]   dequant: %.3f s (%.1f MB/s)\n",
            t1 - t0, wbytes / ((t1 - t0) * 1e6));

    /* reference y[r] = sum_k Wfp32[r,k] * x[k] */
    fprintf(stderr, "[self] host matvec reference ...\n");
    t0 = now_sec();
    for (int r = 0; r < M; ++r) {
        float s = 0.0f;
        const float *wr = Wfp32 + (int64_t)r * K;
        for (int k = 0; k < K; ++k) s += wr[k] * x[k];
        yref[r] = s;
    }
    t1 = now_sec();
    fprintf(stderr, "[self]   matvec: %.3f s (%.1f GB/s effective)\n",
            t1 - t0, (double)M * K * 4.0 / ((t1 - t0) * 1e9));

    /* GPU GEMV */
    uint8_t *dW; float *dx, *dy;
    cudaMalloc(&dW, wbytes);
    cudaMalloc(&dx, nelem * 4);
    cudaMalloc(&dy, M * 4);
    cudaMemcpy(dW, W, wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, nelem * 4, cudaMemcpyHostToDevice);
    cudaMemset(dy, 0, M * 4);

    /* warmup + correctness (small variant) */
    int rc = gemv_iq3xxs(dW, dx, dy, M, K);
    if (rc) { fprintf(stderr, "kernel launch failed: %d\n", rc); return 3; }
    cudaDeviceSynchronize();
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) { fprintf(stderr, "[self] post-launch err: %s\n", cudaGetErrorString(ce)); return 3; }
    cudaMemcpy(ygpu, dy, M * 4, cudaMemcpyDeviceToHost);

    int bad = 0; double max_rel = 0.0; double max_abs = 0.0;
    for (int r = 0; r < M; ++r) {
        float a = fabsf(yref[r]);
        float e = fabsf(ygpu[r] - yref[r]);
        double rel = (a > 1e-3) ? e / a : e;
        if (rel > max_rel) max_rel = rel;
        if (e   > max_abs) max_abs = e;
        if (rel > 5e-3 && e > 1e-2) bad++;
    }
    fprintf(stderr, "[self] GPU vs ref: max_abs=%.4e  max_rel=%.4e  bad_rows=%d/%d\n",
            max_abs, max_rel, bad, M);
    if (bad > M / 100) {
        fprintf(stderr, "[self] FAIL: more than 1%% rows over tolerance\n");
        for (int r = 0; r < 5; ++r) fprintf(stderr, "  r=%d  ref=%.6f  gpu=%.6f\n", r, yref[r], ygpu[r]);
        return 4;
    }

    /* benchmark plain */
    int iters = 50;
    cudaDeviceSynchronize();
    t0 = now_sec();
    for (int it = 0; it < iters; ++it) gemv_iq3xxs(dW, dx, dy, M, K);
    cudaDeviceSynchronize();
    t1 = now_sec();
    double dt = (t1 - t0) / iters;
    double gbs_plain = wbytes / (dt * 1e9);
    double gflops    = (2.0 * (double)M * (double)K) / (dt * 1e9);
    fprintf(stderr, "[bench] plain:   %.3f ms/iter   W: %.1f GB/s   %.1f GFLOPS\n",
            dt * 1e3, gbs_plain, gflops);

    /* benchmark shm variant */
    cudaMemset(dy, 0, M * 4);
    t0 = now_sec();
    for (int it = 0; it < iters; ++it) gemv_iq3xxs_shm(dW, dx, dy, M, K);
    cudaDeviceSynchronize();
    t1 = now_sec();
    dt = (t1 - t0) / iters;
    double gbs_shm = wbytes / (dt * 1e9);
    fprintf(stderr, "[bench] shm:     %.3f ms/iter   W: %.1f GB/s   %.1f GFLOPS\n",
            dt * 1e3, gbs_shm, (2.0 * (double)M * (double)K) / (dt * 1e9));

    free(W); free(x); free(yref); free(ygpu); free(Wfp32);
    cudaFree(dW); cudaFree(dx); cudaFree(dy);
    return 0;
}

int main(int argc, char **argv) {
    /* run two shapes:  M=11008 K=1536  (q4_0 o+mlp shape)
                       M=262144 K=1536  (lm-head shape)                       */
    if (argc >= 3) {
        int M = atoi(argv[1]);
        int K = atoi(argv[2]);
        return run_self_test(M, K);
    }
    int rc = run_self_test(4096, 1536);
    if (rc) return rc;
    rc = run_self_test(11008, 1536);
    if (rc) return rc;
    rc = run_self_test(262144, 1536);
    return rc;
}
