// M7 task 2: GPU golden GEMV test for every Tier-1 quant type.
//
// For each type x shape in {(128,896),(4864,896),(896,4864)}: carve M*K
// values of REAL weight bytes (block-aligned) from blk.0.ffn_down.weight of
// the matching SmolLM2 quant file, run tt_gemv_typed on device, compare
// against the CPU dequant dot product from src/dequant_ref.c.
// Tolerance: atol = 1e-2 * rowmax (max |y_ref| over rows). Prints PASS/FAIL
// per combo; exits nonzero on any failure.
//
// Usage: build/test_gemv_typed [models_dir]   (default data/testmodels)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "loader_gguf.h"
#include "dequant_ref.h"

extern "C" int tt_gemv_typed(const void *W, int dtype, const float *x,
                             float *y, int M, int K, cudaStream_t stream);

typedef struct {
    const char *name;
    int code;
    const char *file;
} TypeSpec;

static const TypeSpec TYPES[] = {
    {"Q4_0", TTQ_Q4_0, "smollm2-135m-instruct-Q4_0.gguf"},
    {"Q4_1", TTQ_Q4_1, "smollm2-135m-instruct-Q4_1.gguf"},
    {"Q5_0", TTQ_Q5_0, "smollm2-135m-instruct-Q5_0.gguf"},
    {"Q5_1", TTQ_Q5_1, "smollm2-135m-instruct-Q5_1.gguf"},
    {"Q8_0", TTQ_Q8_0, "smollm2-135m-instruct-Q8_0.gguf"},
    {"Q4_K", TTQ_Q4_K, "smollm2-135m-instruct-Q4_K.gguf"},
    {"Q4_K_S", TTQ_Q4_K, "smollm2-135m-instruct-Q4_K_S.gguf"}, /* same layout */
    {"Q5_K", TTQ_Q5_K, "smollm2-135m-instruct-Q5_K.gguf"},
    {"Q5_K_S", TTQ_Q5_K, "smollm2-135m-instruct-Q5_K_S.gguf"}, /* same layout */
    {"Q6_K", TTQ_Q6_K, "smollm2-135m-instruct-Q6_K.gguf"},
};
#define NTYPES ((int)(sizeof(TYPES) / sizeof(TYPES[0])))

static const int SHAPES[][2] = {{128, 896}, {4864, 896}, {896, 4864},
                                {576, 1536}}; /* last: real ffn_down shape */
#define NSHAPES ((int)(sizeof(SHAPES) / sizeof(SHAPES[0])))

static const char *TENSOR_FALLBACK = "blk.0.ffn_down.weight";

/* llama-quantize may re-store individual tensors at a different (compatible)
 * quant (e.g. ffn_down becomes Q6_K inside the Q4_K file). Pick the LARGEST
 * tensor whose on-disk type matches the requested code instead of betting on
 * one name. */
static GGUFTensor *pick_tensor(GGUFModel *m, int code) {
    GGUFTensor *best = NULL;
    for (int i = 0; i < m->tensor_count; i++) {
        GGUFTensor *t = &m->tensors[i];
        if ((int)t->type == code && (!best || t->size_bytes > best->size_bytes))
            best = t;
    }
    return best;
}

/* block geometry per type (must match loader/dequant_ref) */
static void block_geom(int code, long *vals, long *bytes) {
    switch (code) {
        case TTQ_Q4_0: *vals = 32; *bytes = 18; break;
        case TTQ_Q4_1: *vals = 32; *bytes = 20; break;
        case TTQ_Q5_0: *vals = 32; *bytes = 22; break;
        case TTQ_Q5_1: *vals = 32; *bytes = 24; break;
        case TTQ_Q8_0: *vals = 32; *bytes = 34; break;
        case TTQ_Q4_K: *vals = 256; *bytes = 144; break;
        case TTQ_Q5_K: *vals = 256; *bytes = 176; break;
        case TTQ_Q6_K: *vals = 256; *bytes = 210; break;
        default:       *vals = 1;  *bytes = 4;  break;
    }
}

static unsigned rng_state = 0x12345678u;
static float frand(void) { /* deterministic LCG, no libc dependency drift */
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / 8388608.0f) - 1.0f; /* [-1,1) */
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "data/testmodels";
    int fails = 0, runs = 0;

    for (int ti = 0; ti < NTYPES; ti++) {
        const TypeSpec *ts = &TYPES[ti];
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", dir, ts->file);

        GGUFModel *m = gguf_load(path);
        if (!m) { printf("FAIL %-6s load %s\n", ts->name, path); fails++; continue; }
        GGUFTensor *t = gguf_get_tensor(m, TENSOR_FALLBACK);
        if (!t || (int)t->type != ts->code) t = pick_tensor(m, ts->code);
        if (!t) { printf("FAIL %-6s no tensor with type %d\n", ts->name, ts->code); gguf_free(m); fails++; continue; }

        long bvals, bbytes;
        block_geom(ts->code, &bvals, &bbytes);

        for (int si = 0; si < NSHAPES; si++) {
            const int M = SHAPES[si][0], K = SHAPES[si][1];
            long numel = (long)M * K;
            long need_bytes = numel / bvals * bbytes;
            runs++;

            /* K-quant contract: n_per_row must be a multiple of 256. The
             * dispatcher refuses such launches host-side with a clear error
             * (that IS the specified behavior) -> SKIP, not FAIL. */
            if (numel % bvals ||
                ((ts->code == TTQ_Q4_K || ts->code == TTQ_Q5_K || ts->code == TTQ_Q6_K)
                 && (K % 256))) {
                printf("SKIP %-6s M=%-4d K=%-4d  (K %% 256 != 0 for K-quant)\n",
                       ts->name, M, K);
                runs--; continue;
            }
            if ((long)t->size_bytes < bbytes && t->size_bytes % bbytes) {
                printf("FAIL %-6s tensor not whole blocks\n", ts->name);
                fails++; continue;
            }

            /* REAL weight bytes: stream consecutive blocks from the tensor,
             * wrapping around it until need_bytes are filled (tensor holds
             * whole blocks, so wrap stays block-aligned). */
            std::vector<uint8_t> whost(need_bytes);
            {
                long filled = 0;
                const long src = (long)t->size_bytes;
                while (filled < need_bytes) {
                    long chunk = src < need_bytes - filled ? src : need_bytes - filled;
                    memcpy(whost.data() + filled, (const uint8_t *)t->data +
                           (filled % src), (size_t)chunk);
                    filled += chunk;
                }
            }

            /* shared input vector */
            rng_state = 0xC0FFEEu ^ (unsigned)numel;
            std::vector<float> x(K);
            for (int i = 0; i < K; i++) x[i] = frand();

            /* CPU reference: dequant_ref dot */
            std::vector<float> wref(numel);
            if (ttq_dequant(whost.data(), ts->code, numel, wref.data()) < 0) {
                printf("FAIL %-6s CPU dequant\n", ts->name); fails++; continue;
            }
            std::vector<float> yref(M);
            for (int r = 0; r < M; r++) {
                double acc = 0.0;
                const float *wr = wref.data() + (long)r * K;
                for (int i = 0; i < K; i++) acc += (double)wr[i] * x[i];
                yref[r] = (float)acc;
            }
            float rowmax = 0.0f;
            for (int r = 0; r < M; r++) rowmax = rowmax > fabsf(yref[r]) ? rowmax : fabsf(yref[r]);
            const float atol = 1e-2f * (rowmax > 1e-6f ? rowmax : 1e-6f);

            /* device run */
            uint8_t *dW = NULL; float *dx = NULL, *dy = NULL;
            if (cudaMalloc(&dW, need_bytes) != cudaSuccess ||
                cudaMalloc(&dx, (size_t)K * 4) != cudaSuccess ||
                cudaMalloc(&dy, (size_t)M * 4) != cudaSuccess) {
                printf("FAIL %-6s (%d,%d): cudaMalloc\n", ts->name, M, K);
                fails++; continue;
            }
            cudaMemcpy(dW, whost.data(), (size_t)need_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(dx, x.data(), (size_t)K * 4, cudaMemcpyHostToDevice);

            int rc = tt_gemv_typed(dW, ts->code, dx, dy, M, K, 0);
            std::vector<float> ygpu(M);
            cudaMemcpy(ygpu.data(), dy, (size_t)M * 4, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            float maxerr = 0.0f;
            for (int r = 0; r < M; r++) {
                float e = fabsf(ygpu[r] - yref[r]);
                if (e > maxerr) maxerr = e;
            }
            const bool ok = (rc == 0) && (maxerr <= atol);
            printf("%s %-6s M=%-4d K=%-4d  maxerr=%.3e  atol=%.3e [%s]%s\n",
                   ok ? "PASS" : "FAIL", ts->name, M, K, maxerr, atol, t->name,
                   rc ? "  (kernel err)" : "");
            if (!ok) fails++;

            cudaFree(dW); cudaFree(dx); cudaFree(dy);
        }
        gguf_free(m);
    }

    printf("----\n%d/%d combos passed\n", runs - fails, runs);
    return fails ? 1 : 0;
}
