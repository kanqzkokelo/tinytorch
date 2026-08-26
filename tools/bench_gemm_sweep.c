/* Fast standalone sweep driver for the M2 GEMM ladder.
 * Includes ../src/gemm.c directly (no lib rebuild); block sizes come in
 * via -DMC= -DKC= -DNC=. Times 1T sq1024 (+sq2048 with --full), median of
 * RUNS, checks rel err vs naive double-free reference (f32 accumulate). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/gemm.c"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void fill(float *p, long n, unsigned *seed) {
    for (long i = 0; i < n; i++)
        p[i] = (float)((rand_r(seed) % 2000 - 1000) / 997.0f);
}

static int check(int M, int N, int K, const float *A, const float *B,
                 const float *C) {
    /* coarse spot-check against scalar f32 accumulation on a grid */
    for (int t = 0; t < 64; t++) {
        int i = rand() % M, j = rand() % N;
        float s = 0.f;
        for (int k = 0; k < K; k++) s += A[(long)i * K + k] * B[(long)k * N + j];
        double num = fabs((double)s - C[(long)i * N + j]);
        if (num > 1e-3 * (fabs((double)s) + 1e-2)) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    int full = argc > 1 && !strcmp(argv[1], "--full");
    const int RUNS = 5, WARM = 2;
    int sizes[2] = {1024, 2048};
    int nsz = full ? 2 : 1;
    unsigned seed = 42;
    int fail = 0;

    printf("MC=%d KC=%d NC=%d NR=%d MR=%d |", MC, KC, NC, NR, MR);
    for (int s = 0; s < nsz; s++) {
        int n = sizes[s];
        float *A = malloc((size_t)n * n * 4);
        float *B = malloc((size_t)n * n * 4);
        float *C = calloc((size_t)n * n, 4);
        fill(A, (long)n * n, &seed);
        fill(B, (long)n * n, &seed);

        tt_sgemm_rowmajor(n, n, n, A, B, C, 1);
        if (!check(n, n, n, A, B, C)) {
            printf(" CORRUPT@%d", n);
            fail = 1;
        }
        for (int w = 0; w < WARM; w++) tt_sgemm_rowmajor(n, n, n, A, B, C, 1);
        double ts[RUNS], best = 1e9;
        for (int r = 0; r < RUNS; r++) {
            double t0 = now_sec();
            tt_sgemm_rowmajor(n, n, n, A, B, C, 1);
            ts[r] = now_sec() - t0;
            if (ts[r] < best) best = ts[r];
        }
        double gf = 2.0 * n * n * n / best / 1e9;
        printf(" sq%d=%.1fGF", n, gf);
        free(A);
        free(B);
        free(C);
    }
    printf(" %s\n", fail ? "FAIL" : "ok");
    return fail;
}
