// PROTOTYPE (M9 perf, CREATE-ONLY study): CUDA graph capture for quant-GEMV.
//
// Goal: quantify replay overhead vs direct launch on the actual hardware the
// engine runs on. We need µs/call for `tt_gemv_typed` directly, then the same
// workload captured into a cudaGraph and replayed, to know whether graph
// capture is worth the complexity for our quant-GEMV decode path.
//
// Workloads:
//   A) single GEMV:           11008 x 1536 Q4_0 (mirrors gemma4 logits shape)
//   B) 5 chained GEMVs:       same shape, simulates a forward block
//   C) 1 graph of 5 GEMVs:    same B but captured and replayed as one graph
//
// For each: median of N measured calls. Print speedup, GPU memory used.
//
// Build (env-gated; no Makefile changes):
//   $HOME/mmcuda/bin/nvcc -O2 -arch=sm_86 -std=c++17 \
//       -Iinclude \
//       -o tests/proto_graph_capture \
//       tests/proto_graph_capture.cu \
//       kernels/gemv_typed.cu kernels/gemv_q4_cuda.cu \
//       -L$HOME/mmcuda/lib -lcudart
// Run:
//   LD_LIBRARY_PATH=$HOME/mmcuda/lib ./tests/proto_graph_capture
//
// Env gate:
//   TT_GRAPH_TEST=0   skip the test (default: run)
//
// Self-contained: only the existing typed-GEMV and the q4_0 helper. No
// engine, no loader, no headers beyond what `gemv_typed.cu` and
// `gemv_q4_cuda.cu` already need (`dequant_ref.h` for the TTQ_Q4_0 enum).
//
// Why this lives next to the other proto_*.cu files: it follows the same
// pattern (single-TU, no engine deps, build from command line). It is
// NOT a parity gate, only a perf measurement.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdio>

#include "dequant_ref.h"   /* TTQ_Q4_0 = 2 */

/* typed GEMV we want to measure: declared in gemv_typed.cu, no header */
extern "C" int tt_gemv_typed(const void *W, int dtype,
                             const float *x, float *y,
                             int M, int K, cudaStream_t stream);

/* ---------------- config ---------------- */

struct Config {
    int M;              /* output rows */
    int K;              /* input dim (must be % 32 for q4_0) */
    int dtype;          /* TTQ_Q4_0 = 2 (BlockQ4_0, 18B per 32 vals) */
    int n_chained;      /* number of GEMVs back-to-back in B/C */
    int n_outer;        /* outer loop reps for the median */
    int n_warmup;       /* warmup runs discarded */
};

static const Config CFG = { 11008, 1536, TTQ_Q4_0, 5, 256, 32 };

/* ---------------- q4_0 block layout (mirrors gemv_q4_cuda.cu) --------------- */
typedef struct {
    uint16_t d;         /* fp16 scale (we write as half) */
    uint8_t  qs[16];    /* 32 packed nibbles */
} BlockQ4_0;

/* fp16 bit-conversion without depending on cuda_fp16.h at host scope */
static inline uint16_t f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  e    = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t man  = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) return (uint16_t)sign;     /* quant scales are normal range */
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)e << 10) | (man >> 13));
}

static void fill_q4_0(BlockQ4_0 *W, long n) {
    /* Random nibbles in [0,15] and scale ~ 0.05 (so dequant values land in
     * the range typical of attention/MLP weights: roughly -0.4..+0.4). */
    for (long i = 0; i < n; i++) {
        W[i].d = f2h(0.05f);
        for (int j = 0; j < 16; j++) W[i].qs[j] = (uint8_t)(rand() & 0xFu);
    }
}

/* ---------------- timer helpers ---------------- */

struct GpuTimer {
    cudaEvent_t a, b;
    GpuTimer()  { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start(cudaStream_t s) { cudaEventRecord(a, s); }
    void stop (cudaStream_t s) { cudaEventRecord(b, s); cudaEventSynchronize(b); }
    float ms() { float x = 0; cudaEventElapsedTime(&x, a, b); return x; }
};

static double median_us_per_call(const std::vector<double> &xs) {
    std::vector<double> v(xs);
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/* ---------------- measurement primitives ---------------- */

struct Weights {
    void     *dW;       /* device weight buffer */
    size_t    bytes;
    int       M, K, dtype;
};

static Weights make_weights(int M, int K, int dtype) {
    Weights w;
    w.M = M; w.K = K; w.dtype = dtype;
    if (dtype == TTQ_Q4_0) {
        w.bytes = (size_t)M * (K / 32) * sizeof(BlockQ4_0);
    } else {
        fprintf(stderr, "[proto_graph_capture] unsupported dtype %d\n", dtype);
        exit(1);
    }
    std::vector<BlockQ4_0> hW(M * (K / 32));
    fill_q4_0(hW.data(), hW.size());
    cudaMalloc(&w.dW, w.bytes);
    cudaMemcpy(w.dW, hW.data(), w.bytes, cudaMemcpyHostToDevice);
    return w;
}

/* A) one direct launch of tt_gemv_typed, timed. Returns µs. */
static double time_one_gemv(const Weights &w, float *dx, float *dy, cudaStream_t s) {
    GpuTimer t; t.start(s);
    int rc = tt_gemv_typed(w.dW, w.dtype, dx, dy, w.M, w.K, s);
    t.stop(s);
    if (rc) fprintf(stderr, "[proto_graph_capture] tt_gemv_typed rc=%d\n", rc);
    return t.ms() * 1000.0;
}

/* Median over n_outer calls. */
static double bench_direct(const Weights &w, float *dx, float *dy, cudaStream_t s,
                           int n_chained) {
    std::vector<double> samples;
    samples.reserve(CFG.n_outer);
    for (int i = 0; i < CFG.n_warmup; i++)
        for (int j = 0; j < n_chained; j++)
            tt_gemv_typed(w.dW, w.dtype, dx, dy, w.M, w.K, s);
    cudaStreamSynchronize(s);
    for (int i = 0; i < CFG.n_outer; i++) {
        GpuTimer t; t.start(s);
        for (int j = 0; j < n_chained; j++)
            tt_gemv_typed(w.dW, w.dtype, dx, dy, w.M, w.K, s);
        t.stop(s);
        samples.push_back(t.ms() * 1000.0 / n_chained);
    }
    return median_us_per_call(samples);
}

/* Build a graph containing n_chained tt_gemv_typed launches. Caller owns
 * the graph and graph_exec and must destroy them. Returns µs/launch after
 * instantiation, measured by replaying the graph n_outer times.
 *
 * Uses cudaStreamCaptureModeThreadLocal (matches qwen2_engine_graph_capture
 * at kernels/qwen2_cuda.cu:1801). Warmup launches are performed before
 * capture to satisfy CUDA's "no lazy module load during capture" rule. */
static double bench_graph(const Weights &w, float *dx, float *dy, cudaStream_t s,
                          int n_chained, cudaGraphExec_t *out_exec) {
    /* 1. warmup outside capture (so module is loaded and any lazy init
     *    inside the kernel is done) */
    for (int i = 0; i < CFG.n_warmup; i++)
        for (int j = 0; j < n_chained; j++)
            tt_gemv_typed(w.dW, w.dtype, dx, dy, w.M, w.K, s);
    cudaStreamSynchronize(s);

    /* 2. capture */
    cudaError_t ce = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[proto_graph_capture] cudaStreamBeginCapture failed: %s\n",
                cudaGetErrorString(ce));
        return -1.0;
    }
    for (int j = 0; j < n_chained; j++) {
        tt_gemv_typed(w.dW, w.dtype, dx, dy, w.M, w.K, s);
    }
    cudaGraph_t g = nullptr;
    ce = cudaStreamEndCapture(s, &g);
    if (ce != cudaSuccess || !g) {
        fprintf(stderr, "[proto_graph_capture] cudaStreamEndCapture failed: %s\n",
                cudaGetErrorString(ce));
        if (g) cudaGraphDestroy(g);
        return -1.0;
    }

    /* 3. instantiate (CUDA 12 3-arg form) */
    cudaGraphExec_t exec = nullptr;
    ce = cudaGraphInstantiate(&exec, g, 0);
    cudaGraphDestroy(g);
    if (ce != cudaSuccess || !exec) {
        fprintf(stderr, "[proto_graph_capture] cudaGraphInstantiate failed: %s\n",
                cudaGetErrorString(ce));
        return -1.0;
    }
    *out_exec = exec;

    /* 4. measure replay */
    std::vector<double> samples;
    samples.reserve(CFG.n_outer);
    for (int i = 0; i < CFG.n_outer; i++) {
        GpuTimer t; t.start(s);
        ce = cudaGraphLaunch(exec, s);
        t.stop(s);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[proto_graph_capture] cudaGraphLaunch failed: %s\n",
                    cudaGetErrorString(ce));
            return -1.0;
        }
        samples.push_back(t.ms() * 1000.0 / n_chained);
    }
    return median_us_per_call(samples);
}

/* ---------------- main ---------------- */

int main(int argc, char **argv) {
    const char *gate = getenv("TT_GRAPH_TEST");
    if (gate && atoi(gate) == 0) {
        fprintf(stderr, "[proto_graph_capture] skipped (TT_GRAPH_TEST=0)\n");
        return 0;
    }

    fprintf(stderr, "PROTO: CUDA graph capture perf for tt_gemv_typed "
                    "(M=%d K=%d q4_0, chained=%d, n_outer=%d)\n",
            CFG.M, CFG.K, CFG.n_chained, CFG.n_outer);

    cudaStream_t s;
    cudaStreamCreate(&s);

    /* buffers */
    Weights w = make_weights(CFG.M, CFG.K, CFG.dtype);
    float *dx = nullptr, *dy = nullptr;
    cudaMalloc(&dx, (size_t)CFG.K * sizeof(float));
    cudaMalloc(&dy, (size_t)CFG.M * sizeof(float));
    /* touch x so it isn't lazily faulted in during measurement */
    {
        std::vector<float> hx(CFG.K, 0.0f);
        for (int i = 0; i < CFG.K; i++) hx[i] = (float)((i * 17) % 31 - 15) * 0.01f;
        cudaMemcpy(dx, hx.data(), CFG.K * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemset(dy, 0, CFG.M * sizeof(float));

    /* 1) single direct */
    double one_us = bench_direct(w, dx, dy, s, 1);
    fprintf(stderr, "  [A] 1x direct launch:           %7.2f us/call\n", one_us);

    /* 2) N chained direct */
    double chain_us = bench_direct(w, dx, dy, s, CFG.n_chained);
    fprintf(stderr, "  [B] %dx direct launch (chain):   %7.2f us/call  "
                    "(%7.2f us total)\n",
            CFG.n_chained, chain_us, chain_us * CFG.n_chained);

    /* 3) 1 graph, n_chained launches, measured per-launch inside graph */
    cudaGraphExec_t execA = nullptr;
    double graph_one_us = bench_graph(w, dx, dy, s, 1, &execA);
    if (graph_one_us > 0) {
        fprintf(stderr, "  [A-G] 1x graph replay:          %7.2f us/call  "
                        "(speedup vs direct %.2fx)\n",
                graph_one_us, one_us / graph_one_us);
    } else {
        fprintf(stderr, "  [A-G] 1x graph replay: FAILED\n");
    }
    if (execA) cudaGraphExecDestroy(execA);

    /* 4) 1 graph, n_chained launches, measured per-launch inside graph */
    cudaGraphExec_t execN = nullptr;
    double graph_chain_us = bench_graph(w, dx, dy, s, CFG.n_chained, &execN);
    if (graph_chain_us > 0) {
        fprintf(stderr, "  [B-G] %dx graph replay (chain):  %7.2f us/call  "
                        "(%7.2f us total)  (speedup vs direct %.2fx)\n",
                CFG.n_chained, graph_chain_us, graph_chain_us * CFG.n_chained,
                chain_us / graph_chain_us);
    } else {
        fprintf(stderr, "  [B-G] %dx graph replay: FAILED\n", CFG.n_chained);
    }
    if (execN) cudaGraphExecDestroy(execN);

    /* memory: report weight buffer + working set */
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    fprintf(stderr, "  [mem] weights=%.2f MB  free=%.2f MB / total=%.2f MB\n",
            w.bytes / (1024.0 * 1024.0),
            free_b / (1024.0 * 1024.0),
            total_b / (1024.0 * 1024.0));

    /* 5) one-shot graph-launch total time vs one-shot N-launches total time
     *    (alternative framing: the whole graph runs as one launch, so
     *    end-to-end wallclock for "do N GEMVs" is what matters). */
    if (graph_chain_us > 0) {
        double direct_total = chain_us * CFG.n_chained;
        double graph_total  = graph_chain_us * CFG.n_chained;
        fprintf(stderr, "  [T] %d chained GEMVs direct end-to-end:  %7.2f us\n",
                CFG.n_chained, direct_total);
        fprintf(stderr, "  [T] %d chained GEMVs graph end-to-end:   %7.2f us  "
                        "(speedup %.2fx)\n",
                CFG.n_chained, graph_total, direct_total / graph_total);
    }

    cudaFree(dx); cudaFree(dy); cudaFree(w.dW);
    cudaStreamDestroy(s);
    return 0;
}
