/* M2: AVX2+FMA blocked sgemm (single source of truth).
 *
 * Design:
 *   - Blocked loops: NC (N window) / KC (K panel) / 6x16 register tile.
 *   - A packed once per (jc,kk) into row-major panel Ap[M][KC+APAD]
 *     (padded row stride defeats 4K aliasing between the six broadcast
 *     streams).
 *   - Hand-scheduled AVX2+FMA microkernel (inline asm):
 *       accumulators ymm0-11, B panel ymm12-13, broadcast temp ymm14,
 *       k unrolled x4, hot loop 32-byte aligned, walking C pointer,
 *       12 independent FMA chains saturate both FMA ports.
 *   - OpenMP over i-groups inside the innermost jj/kk loops.
 */
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef MR
#define MR 6
#endif
#ifndef NR
#define NR 16
#endif
#ifndef KC
#define KC 384
#endif
#ifndef NC
#define NC 1024
#endif
#ifndef APAD
#define APAD 16
#endif
#ifndef MC
#define MC 120   /* row-block: 120*(KC+APAD)*4 = 192KB, L2-resident with headroom */
#endif

#ifdef TT_IN_LIB
#include "tensor.h"
#endif

#ifndef PF_A
#define PF_A "prefetcht0 256(%[a0],%[ix],1)\n\t"
#endif
#ifndef PF_B
#define PF_B "prefetcht0 256(%[bp])\n\t"
#endif
/* scalar fallback for the ragged right edge (correctness only) */
static void gemm_tail(int M, int N, int K, const float *A, const float *B,
                      float *C, int n0, int n1) {
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            const float av = A[(long)i * K + k];
            if (av == 0.0f) continue;
            const float *brow = B + (long)k * N;
            float *crow = C + (long)i * N;
            for (int j = n0; j < n1; j++) crow[j] += av * brow[j];
        }
}

/* A points at packed panel row ig, column kk-local 0. Rows advance by
 * lda = KC+APAD floats. Bp is the Ki x NR packed B panel.
 * 6x16 register tile: ymm0-11 accumulators, ymm12/13 packed B pair,
 * ymm14 broadcast temp, k unrolled x8 (loop overhead amortized to stay
 * under the 5-wide rename ceiling of Rocket Lake).
 * NOTE: lda must equal KC+APAD (compile-time constant). */
static void micro_kernel(const float *A, const float *bp0, float *C,
                         long lda, long ldc, int Ki, int first_block) {
    const float *a0 = A;
    const float *a1 = A + lda;
    const float *a2 = A + 2 * lda;
    const float *a3 = A + 3 * lda;
    const float *a4 = A + 4 * lda;
    const float *a5 = A + 5 * lda;
    const float *bp = bp0;
    long k = Ki;
    long ix;
    (void)lda;

    if (first_block) {
        __asm__ __volatile__(
            "vxorps %%ymm0, %%ymm0, %%ymm0\n\t"
            "vxorps %%ymm1, %%ymm1, %%ymm1\n\t"
            "vxorps %%ymm2, %%ymm2, %%ymm2\n\t"
            "vxorps %%ymm3, %%ymm3, %%ymm3\n\t"
            "vxorps %%ymm4, %%ymm4, %%ymm4\n\t"
            "vxorps %%ymm5, %%ymm5, %%ymm5\n\t"
            "vxorps %%ymm6, %%ymm6, %%ymm6\n\t"
            "vxorps %%ymm7, %%ymm7, %%ymm7\n\t"
            "vxorps %%ymm8, %%ymm8, %%ymm8\n\t"
            "vxorps %%ymm9, %%ymm9, %%ymm9\n\t"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n\t"
            "vxorps %%ymm11, %%ymm11, %%ymm11\n\t"
            "test %[k], %[k]\n\t"
            "jz 2f\n\t"
            "xor %[ix], %[ix]\n\t"
            "cmp $8, %[k]\n\t"
            "jl 3f\n\t"
            ".p2align 5\n\t"
            "1:\n\t"
            PF_B
            "vmovaps 0x0(%[bp]), %%ymm12\n\t"
            "vmovaps 0x20(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0x0(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0x0(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0x0(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0x0(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0x0(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0x0(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "vmovaps 0x40(%[bp]), %%ymm12\n\t"
            "vmovaps 0x60(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0x4(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0x4(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0x4(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0x4(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0x4(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0x4(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "vmovaps 0x80(%[bp]), %%ymm12\n\t"
            "vmovaps 0xa0(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0x8(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0x8(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0x8(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0x8(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0x8(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0x8(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "vmovaps 0xc0(%[bp]), %%ymm12\n\t"
            "vmovaps 0xe0(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0xc(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0xc(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0xc(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0xc(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0xc(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0xc(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "vmovaps 0x100(%[bp]), %%ymm12\n\t"
            "vmovaps 0x120(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0x10(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0x10(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0x10(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0x10(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0x10(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0x10(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "vmovaps 0x140(%[bp]), %%ymm12\n\t"
            "vmovaps 0x160(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0x14(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0x14(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0x14(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0x14(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0x14(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0x14(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "vmovaps 0x180(%[bp]), %%ymm12\n\t"
            "vmovaps 0x1a0(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0x18(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0x18(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0x18(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0x18(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0x18(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0x18(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "vmovaps 0x1c0(%[bp]), %%ymm12\n\t"
            "vmovaps 0x1e0(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0x1c(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0x1c(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0x1c(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0x1c(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0x1c(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0x1c(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "add $512, %[bp]\n\t"
            "add $32, %[ix]\n\t"
            "sub $8, %[k]\n\t"
            "cmp $8, %[k]\n\t"
            "jge 1b\n\t"
            "3:\n\t"
            "test %[k], %[k]\n\t"
            "jz 2f\n\t"
            "vmovaps (%[bp]), %%ymm12\n\t"
            "vmovaps 0x20(%[bp]), %%ymm13\n\t"
            "vbroadcastss (%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss (%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss (%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss (%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss (%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss (%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "add $64, %[bp]\n\t"
            "add $4, %[ix]\n\t"
            "dec %[k]\n\t"
            "jnz 3b\n\t"
            "2:\n\t"
            "vmovups %%ymm0, (%[c])\n\t"
            "vmovups %%ymm1, 32(%[c])\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups %%ymm2, (%[c])\n\t"
            "vmovups %%ymm3, 32(%[c])\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups %%ymm4, (%[c])\n\t"
            "vmovups %%ymm5, 32(%[c])\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups %%ymm6, (%[c])\n\t"
            "vmovups %%ymm7, 32(%[c])\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups %%ymm8, (%[c])\n\t"
            "vmovups %%ymm9, 32(%[c])\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups %%ymm10, (%[c])\n\t"
            "vmovups %%ymm11, 32(%[c])\n\t"
            : [bp] "+r"(bp), [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2),
              [a3] "+r"(a3), [a4] "+r"(a4), [a5] "+r"(a5), [k] "+r"(k),
              [ix] "=&r"(ix), [c] "+r"(C)
            : [cs] "r"((long)ldc * 4)
            : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
              "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
              "%ymm13", "%ymm14", "memory", "cc");
    } else {
        __asm__ __volatile__(
            "vmovups (%[c]), %%ymm0\n\t"
            "vmovups 32(%[c]), %%ymm1\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups (%[c]), %%ymm2\n\t"
            "vmovups 32(%[c]), %%ymm3\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups (%[c]), %%ymm4\n\t"
            "vmovups 32(%[c]), %%ymm5\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups (%[c]), %%ymm6\n\t"
            "vmovups 32(%[c]), %%ymm7\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups (%[c]), %%ymm8\n\t"
            "vmovups 32(%[c]), %%ymm9\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups (%[c]), %%ymm10\n\t"
            "vmovups 32(%[c]), %%ymm11\n\t"
            "test %[k], %[k]\n\t"
            "jz 4f\n\t"
            "xor %[ix], %[ix]\n\t"
            "cmp $8, %[k]\n\t"
            "jl 5f\n\t"
            ".p2align 5\n\t"
            "6:\n\t"
            PF_B
            "vmovaps 0x0(%[bp]), %%ymm12\n\t"
            "vmovaps 0x20(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0x0(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0x0(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0x0(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0x0(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0x0(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0x0(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "vmovaps 0x40(%[bp]), %%ymm12\n\t"
            "vmovaps 0x60(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0x4(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0x4(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0x4(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0x4(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0x4(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0x4(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "vmovaps 0x80(%[bp]), %%ymm12\n\t"
            "vmovaps 0xa0(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0x8(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0x8(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0x8(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0x8(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0x8(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0x8(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "vmovaps 0xc0(%[bp]), %%ymm12\n\t"
            "vmovaps 0xe0(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0xc(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0xc(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0xc(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0xc(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0xc(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0xc(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "vmovaps 0x100(%[bp]), %%ymm12\n\t"
            "vmovaps 0x120(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0x10(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0x10(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0x10(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0x10(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0x10(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0x10(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "vmovaps 0x140(%[bp]), %%ymm12\n\t"
            "vmovaps 0x160(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0x14(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0x14(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0x14(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0x14(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0x14(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0x14(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "vmovaps 0x180(%[bp]), %%ymm12\n\t"
            "vmovaps 0x1a0(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0x18(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0x18(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0x18(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0x18(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0x18(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0x18(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "vmovaps 0x1c0(%[bp]), %%ymm12\n\t"
            "vmovaps 0x1e0(%[bp]), %%ymm13\n\t"
            "vbroadcastss 0x1c(%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss 0x1c(%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss 0x1c(%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss 0x1c(%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss 0x1c(%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss 0x1c(%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "add $512, %[bp]\n\t"
            "add $32, %[ix]\n\t"
            "sub $8, %[k]\n\t"
            "cmp $8, %[k]\n\t"
            "jge 6b\n\t"
            "5:\n\t"
            "test %[k], %[k]\n\t"
            "jz 4f\n\t"
            "vmovaps (%[bp]), %%ymm12\n\t"
            "vmovaps 0x20(%[bp]), %%ymm13\n\t"
            "vbroadcastss (%[a0],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"
            "vbroadcastss (%[a1],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"
            "vbroadcastss (%[a2],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"
            "vbroadcastss (%[a3],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"
            "vbroadcastss (%[a4],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"
            "vbroadcastss (%[a5],%[ix],1), %%ymm14\n\t"
            "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t"
            "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"
            "add $64, %[bp]\n\t"
            "add $4, %[ix]\n\t"
            "dec %[k]\n\t"
            "jnz 5b\n\t"
            "4:\n\t"
            "vmovups %%ymm11, 32(%[c])\n\t"
            "vmovups %%ymm10, (%[c])\n\t"
            "sub %[cs], %[c]\n\t"
            "vmovups %%ymm9, 32(%[c])\n\t"
            "vmovups %%ymm8, (%[c])\n\t"
            "sub %[cs], %[c]\n\t"
            "vmovups %%ymm7, 32(%[c])\n\t"
            "vmovups %%ymm6, (%[c])\n\t"
            "sub %[cs], %[c]\n\t"
            "vmovups %%ymm5, 32(%[c])\n\t"
            "vmovups %%ymm4, (%[c])\n\t"
            "sub %[cs], %[c]\n\t"
            "vmovups %%ymm3, 32(%[c])\n\t"
            "vmovups %%ymm2, (%[c])\n\t"
            "sub %[cs], %[c]\n\t"
            "vmovups %%ymm1, 32(%[c])\n\t"
            "vmovups %%ymm0, (%[c])\n\t"
            : [bp] "+r"(bp), [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2),
              [a3] "+r"(a3), [a4] "+r"(a4), [a5] "+r"(a5), [k] "+r"(k),
              [ix] "=&r"(ix), [c] "+r"(C)
            : [cs] "r"((long)ldc * 4)
            : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
              "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
              "%ymm13", "%ymm14", "memory", "cc");
    }
}

int tt_sgemm_rowmajor(int M, int N, int K, const float *A, const float *B,
                       float *C, int nthreads) {
    const int Nfull = (N / NR) * NR;
    const int Msfx = (M / MR) * MR;
    if (Nfull <= 0 || K <= 0) {
        if (Nfull < N) gemm_tail(M, N, K, A, B, C, Nfull, N);
        return 0;
    }

    const size_t bpanel = (size_t)KC * NR;
    float *Bp = (float *)aligned_alloc(32, sizeof(float) * bpanel * NC);
    /* Msfx==0 (M<MR): no main-loop rows; skip Ap, run tails only. */
    float *Ap = Msfx > 0 ? (float *)aligned_alloc(
        32, sizeof(float) * (size_t)((MC < Msfx ? MC : Msfx)) * (KC + APAD))
                          : (float *)1;
    if (!Bp || !Ap) {
        free(Bp);
        if (Msfx > 0) free(Ap);
        return -1;
    }
    if (Msfx == 0) {
        free(Bp);
        gemm_tail(M, N, K, A, B, C, 0, N);
        return 0;
    }

    for (int jc = 0; jc < Nfull; jc += NC) {
        const int jend = (jc + NC < Nfull) ? jc + NC : Nfull;
        const int npj = (jend - jc) / NR;          /* NR-col panels in block */
        for (int kk = 0; kk < K; kk += KC) {
            const int Ki = (kk + KC <= K) ? KC : K - kk;
            /* pack all B panels of this (jc,kk) block once */
            for (int p = 0; p < npj; p++)
                for (int k2 = 0; k2 < Ki; k2++)
                    memcpy(Bp + p * bpanel + (size_t)k2 * NR,
                           B + (long)(kk + k2) * N + jc + (long)p * NR,
                           sizeof(float) * NR);
            for (int ic = 0; ic < Msfx; ic += MC) {
                const int mi = (ic + MC <= Msfx) ? MC : Msfx - ic;
                for (int r = 0; r < mi; r++)       /* pack A row panel */
                    memcpy(Ap + (size_t)r * (KC + APAD),
                           A + (long)(ic + r) * K + kk,
                           sizeof(float) * Ki);
                if (nthreads == 1) {
                    for (int p = 0; p < npj; p++)
                        for (int ig = 0; ig < mi; ig += MR)
                            micro_kernel(Ap + (size_t)ig * (KC + APAD),
                                         Bp + p * bpanel,
                                         C + (long)(ic + ig) * N + jc + (long)p * NR,
                                         KC + APAD, N, Ki, kk == 0);
                } else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(static) collapse(2)
#endif
                    for (int p = 0; p < npj; p++)
                        for (int ig = 0; ig < mi; ig += MR)
                            micro_kernel(Ap + (size_t)ig * (KC + APAD),
                                         Bp + p * bpanel,
                                         C + (long)(ic + ig) * N + jc + (long)p * NR,
                                         KC + APAD, N, Ki, kk == 0);
                }
            }
        }
    }
    free(Bp);
    free(Ap);
    if (Msfx < M)                      /* tail rows, full width */
        gemm_tail(M - Msfx, N, K, A + (long)Msfx * K, B,
                  C + (long)Msfx * N, 0, Nfull);
    if (Nfull < N)                     /* ragged right edge, full K */
        gemm_tail(M, N, K, A, B, C, Nfull, N);
    return 0;
}

#ifdef TT_IN_LIB
static Tensor *matmul_checked(const Tensor *a, const Tensor *b, int nthr) {
    if (!a || !b || a->ndim != 2 || b->ndim != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;
    const long shp[2] = {a->shape[0], b->shape[1]};
    Tensor *out = tt_new(shp, 2);
    if (!out) return NULL;
    if (tt_sgemm_rowmajor((int)out->shape[0], (int)out->shape[1], (int)a->shape[1],
                      a->data, b->data, out->data, nthr) != 0) {
        tt_release(out);
        return NULL;
    }
    return out;
}

Tensor *tt_matmul_fast(const Tensor *a, const Tensor *b) {
    return matmul_checked(a, b, 1);
}

Tensor *tt_matmul_omp(const Tensor *a, const Tensor *b, int nthreads) {
    if (nthreads < 1) nthreads = 1;
#ifdef _OPENMP
    int maxt = omp_get_max_threads();
    if (nthreads > maxt) nthreads = maxt;
#else
    nthreads = 1;
#endif
    return matmul_checked(a, b, nthreads);
}
#endif /* TT_IN_LIB */
