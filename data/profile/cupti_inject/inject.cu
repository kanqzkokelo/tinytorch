// Tiny CUPTI injector: preloaded into the engine via CUDA_INJECTION64_PATH.
// At library-load time it subscribes to the activity API for kernel records,
// buffers them in-process, and dumps a CSV summary to
// $CUPTI_INJECT_OUT (default /tmp/cupti_kernels.csv) at process exit.
//
// Build:
//   /home/mitesh/mmcuda/bin/nvcc -O2 -shared -fPIC -Xcompiler -fvisibility=hidden \
//     -I/home/mitesh/Storage/repos/fight_analytics/.venv/lib/python3.11/site-packages/nvidia/cuda_cupti/include \
//     -L/home/mitesh/Storage/repos/fight_analytics/.venv/lib/python3.11/site-packages/nvidia/cuda_cupti/lib \
//     -o libcupti_inject.so inject.cu -lcudart -lcupti
//
// Use:
//   CUDA_INJECTION64_PATH=/path/to/libcupti_inject.so \
//   LD_LIBRARY_PATH=$HOME/mmcuda/lib:... \
//   CUPTI_INJECT_OUT=/path/to/out.csv \
//   ./build/run_llm_gpu "..." 16

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <mutex>
#include <chrono>
#include <atomic>

#include <cupti.h>

#define CUPTI_CALL(call)                                                    \
    do {                                                                    \
        CUptiResult _st = (call);                                           \
        if (_st != CUPTI_SUCCESS) {                                         \
            fprintf(stderr, "[inject] %s:%d %s failed: %d\n",               \
                    __FILE__, __LINE__, #call, _st);                        \
        }                                                                   \
    } while (0)

struct KernelInfo {
    std::string name;
    uint64_t    start_ns;
    uint64_t    end_ns;
    uint32_t    grid_x, grid_y, grid_z;
    uint32_t    block_x, block_y, block_z;
};

static std::vector<KernelInfo> g_kernels;
static std::mutex              g_mu;
static std::atomic<bool>       g_subscribed{false};
static uint8_t                *g_buf = nullptr;
static size_t                  g_buf_bytes = 0;
static constexpr size_t        kBufSize = 64 * 1024 * 1024; // 64 MB

static const char *out_path() {
    const char *p = getenv("CUPTI_INJECT_OUT");
    return p && *p ? p : "/tmp/cupti_kernels.csv";
}

static void flush_to_disk() {
    FILE *fp = fopen(out_path(), "w");
    if (!fp) { fprintf(stderr, "[inject] cannot open %s\n", out_path()); return; }
    fprintf(fp, "idx,name,start_ns,end_ns,dur_ns,grid_x,grid_y,grid_z,block_x,block_y,block_z\n");
    std::lock_guard<std::mutex> lk(g_mu);
    for (size_t i = 0; i < g_kernels.size(); ++i) {
        const auto &k = g_kernels[i];
        fprintf(fp, "%zu,\"%s\",%lu,%lu,%lu,%u,%u,%u,%u,%u,%u\n",
                i, k.name.c_str(), (unsigned long)k.start_ns, (unsigned long)k.end_ns,
                (unsigned long)(k.end_ns - k.start_ns),
                k.grid_x, k.grid_y, k.grid_z, k.block_x, k.block_y, k.block_z);
    }
    fclose(fp);
    // also a tiny summary on stderr
    if (!g_kernels.empty()) {
        uint64_t sum = 0, mx = 0;
        for (const auto &k : g_kernels) {
            uint64_t d = k.end_ns - k.start_ns;
            sum += d;
            if (d > mx) mx = d;
        }
        double avg = double(sum) / g_kernels.size();
        fprintf(stderr, "[inject] kernels=%zu avg_ns=%.1f max_ns=%lu total_ns=%lu -> %s\n",
                g_kernels.size(), avg, (unsigned long)mx, (unsigned long)sum, out_path());
    } else {
        fprintf(stderr, "[inject] no kernels captured\n");
    }
}

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
                                     size_t *maxNumRecords) {
    *buffer = g_buf;
    *size = g_buf_bytes;
    *maxNumRecords = 0;
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId,
                                     uint8_t *buffer, size_t size,
                                     size_t validSize) {
    CUpti_Activity *record = nullptr;
    while (CUPTI_CALL_SUCCESS(cuptiActivityGetNextRecord(buffer, validSize, &record)) == CUPTI_SUCCESS) {
        CUpti_ActivityKind kind = cuptiActivityGetKind(record);
        if (kind != CUPTI_ACTIVITY_KIND_KERNEL && kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            // advance — getNextRecord already does this; the loop condition is what matters
            // but the macro doesn't exist; use real API
        }
        if (kind == CUPTI_ACTIVITY_KIND_KERNEL || kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            CUpti_ActivityKernel4 *k = (CUpti_ActivityKernel4 *)record;
            KernelInfo ki;
            ki.name    = k->name ? k->name : "";
            ki.start_ns = k->start;
            ki.end_ns   = k->end;
            ki.grid_x = k->gridX; ki.grid_y = k->gridY; ki.grid_z = k->gridZ;
            ki.block_x = k->blockX; ki.block_y = k->blockY; ki.block_z = k->blockZ;
            std::lock_guard<std::mutex> lk(g_mu);
            g_kernels.push_back(std::move(ki));
        }
        int adv = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (adv != CUPTI_SUCCESS) break;
    }
}

static void install() {
    if (g_subscribed.exchange(true)) return;
    g_buf = (uint8_t *)malloc(kBufSize);
    g_buf_bytes = kBufSize;
    if (!g_buf) { fprintf(stderr, "[inject] oom\n"); return; }

    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    atexit(flush_to_disk);
    fprintf(stderr, "[inject] subscribed, out=%s\n", out_path());
}

extern "C" void cuptiActivityRegisterCallbacks_prepare() __attribute__((constructor));
extern "C" void cuptiActivityRegisterCallbacks_prepare() {
    // Defer to first cuda call? Just install now — driver will be lazy.
    install();
}
