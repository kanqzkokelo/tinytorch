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
#define NC 384
#endif
#ifndef APAD
#define APAD 16
#endif

#ifdef TT_IN_LIB
#include "tensor.h"
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
 * lda = KC+APAD floats. Bp is the Ki x NR packed B panel. */
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
        "vxorps %%ymm11, %%ymm11, %%ymm11\n\t"
        "test %[k], %[k]\n\t"
        "jz 2f\n\t"
        "xor %[ix], %[ix]\n\t"
        "cmp $4, %[k]\n\t"
        "jl 3f\n\t"
        ".p2align 5\n\t"
        "1:\n\t"
        "prefetcht0 256(%[bp])\n\t"
        "prefetcht0 256(%[a0],%[ix],1)\n\t"
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
        "add $256, %[bp]\n\t"
        "add $16, %[ix]\n\t"
        "sub $4, %[k]\n\t"
        "cmp $4, %[k]\n\t"
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
            "add %[cs], %[c]\n\t"
            "add %[cs], %[c]\n\t"
            "add %[cs], %[c]\n\t"
            "add %[cs], %[c]\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups %%ymm10, (%[c])\n\t"
        "vmovups %%ymm11, 0x20(%[c])\n\t"
        "sub %[cs], %[c]\n\t"
        "vmovups %%ymm8, (%[c])\n\t"
        "vmovups %%ymm9, 0x20(%[c])\n\t"
        "sub %[cs], %[c]\n\t"
        "vmovups %%ymm6, (%[c])\n\t"
        "vmovups %%ymm7, 0x20(%[c])\n\t"
        "sub %[cs], %[c]\n\t"
        "vmovups %%ymm4, (%[c])\n\t"
        "vmovups %%ymm5, 0x20(%[c])\n\t"
        "sub %[cs], %[c]\n\t"
        "vmovups %%ymm2, (%[c])\n\t"
        "vmovups %%ymm3, 0x20(%[c])\n\t"
        "sub %[cs], %[c]\n\t"
        "vmovups %%ymm0, (%[c])\n\t"
        "vmovups %%ymm1, 0x20(%[c])\n\t"
        : [bp] "+r"(bp), [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2),
          [a3] "+r"(a3), [a4] "+r"(a4), [a5] "+r"(a5), [k] "+r"(k),
          [ix] "=&r"(ix), [c] "+r"(C)
        : [cs] "r"((long)ldc * 4)
        : "memory");
    } else {
        __asm__ __volatile__(
            "vmovups (%[c]), %%ymm0\n\t"
            "vmovups 0x20(%[c]), %%ymm1\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups (%[c]), %%ymm2\n\t"
            "vmovups 0x20(%[c]), %%ymm3\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups (%[c]), %%ymm4\n\t"
            "vmovups 0x20(%[c]), %%ymm5\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups (%[c]), %%ymm6\n\t"
            "vmovups 0x20(%[c]), %%ymm7\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups (%[c]), %%ymm8\n\t"
            "vmovups 0x20(%[c]), %%ymm9\n\t"
            "add %[cs], %[c]\n\t"
            "vmovups (%[c]), %%ymm10\n\t"
            "vmovups 0x20(%[c]), %%ymm11\n\t"
            "test %[k], %[k]\n\t"
            "jz 4f\n\t"
            "xor %[ix], %[ix]\n\t"
            "cmp $4, %[k]\n\t"
            "jl 5f\n\t"
            ".p2align 5\n\t"
            "6:\n\t"
            "prefetcht0 256(%[bp])\n\t"
            "prefetcht0 256(%[a0],%[ix],1)\n\t"
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
            "add $256, %[bp]\n\t"
            "add $16, %[ix]\n\t"
            "sub $4, %[k]\n\t"
            "cmp $4, %[k]\n\t"
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
            "vmovups %%ymm10, (%[c])\n\t"
            "vmovups %%ymm11, 0x20(%[c])\n\t"
            "sub %[cs], %[c]\n\t"
            "vmovups %%ymm8, (%[c])\n\t"
            "vmovups %%ymm9, 0x20(%[c])\n\t"
            "sub %[cs], %[c]\n\t"
            "vmovups %%ymm6, (%[c])\n\t"
            "vmovups %%ymm7, 0x20(%[c])\n\t"
            "sub %[cs], %[c]\n\t"
            "vmovups %%ymm4, (%[c])\n\t"
            "vmovups %%ymm5, 0x20(%[c])\n\t"
            "sub %[cs], %[c]\n\t"
            "vmovups %%ymm2, (%[c])\n\t"
            "vmovups %%ymm3, 0x20(%[c])\n\t"
            "sub %[cs], %[c]\n\t"
            "vmovups %%ymm0, (%[c])\n\t"
            "vmovups %%ymm1, 0x20(%[c])\n\t"
            : [bp] "+r"(bp), [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2),
              [a3] "+r"(a3), [a4] "+r"(a4), [a5] "+r"(a5), [k] "+r"(k),
              [ix] "=&r"(ix), [c] "+r"(C)
            : [cs] "r"((long)ldc * 4)
            : "memory");
    }
}

void tt_sgemm_rowmajor(int M, int N, int K, const float *A, const float *B,
                       float *C, int nthreads) {
    float *Bp = (float *)aligned_alloc(32, sizeof(float) * KC * NR);
    float *Ap = (float *)aligned_alloc(32,
                                       sizeof(float) * (size_t)M * (KC + APAD));
    if (!Bp || !Ap) {
        free(Bp);
        free(Ap);
        return;
    }

    const int Nfull = (N / NR) * NR;
    const int Msfx = (M / MR) * MR;

    for (int jc = 0; jc < Nfull; jc += NC) {
        const int jend = (jc + NC < Nfull) ? jc + NC : Nfull;
        for (int kk = 0; kk < K; kk += KC) {
            const int Ki = (kk + KC <= K) ? KC : K - kk;
            for (int i = 0; i < M; i++)   /* pack A panel */
                memcpy(Ap + (size_t)i * (KC + APAD), A + (long)i * K + kk,
                       sizeof(float) * Ki);
            for (int jj = jc; jj < jend; jj += NR) {
                for (int k2 = 0; k2 < Ki; k2++)   /* pack B panel */
                    memcpy(Bp + (size_t)k2 * NR,
                           B + (long)(kk + k2) * N + jj,
                           sizeof(float) * NR);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
                for (int ig = 0; ig < Msfx; ig += MR)
                    micro_kernel(Ap + (size_t)ig * (KC + APAD), Bp,
                                 C + (long)ig * N + jj, KC + APAD, N, Ki, kk == 0);
                for (int i = Msfx; i < M; i++) {  /* tail rows */
                    float *crow = C + (long)i * N + jj;
                    for (int k2 = 0; k2 < Ki; k2++) {
                        const float av = Ap[(size_t)i * (KC + APAD) + k2];
                        const float *bq = Bp + (size_t)k2 * NR;
                        for (int j = 0; j < NR; j++)
                            crow[j] = (kk == 0 && k2 == 0 ? 0.0f : crow[j]) + av * bq[j];
                    }
                }
            }
        }
    }
    free(Bp);
    free(Ap);
    if (Nfull < N)                     /* ragged right edge, full K */
        gemm_tail(M, N, K, A, B, C, Nfull, N);
}

#ifdef TT_IN_LIB
static Tensor *matmul_checked(const Tensor *a, const Tensor *b, int nthr) {
    if (!a || !b || a->ndim != 2 || b->ndim != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;
    const long shp[2] = {a->shape[0], b->shape[1]};
    Tensor *out = tt_new(shp, 2);
    if (!out) return NULL;
    tt_sgemm_rowmajor(out->shape[0], out->shape[1], a->shape[1],
                      a->data, b->data, out->data, nthr);
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
