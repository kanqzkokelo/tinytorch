/*
 * measure_real_loader.cu — HONEST S0 vs S4 on a REAL GGUF file.
 *
 * S0 mirrors what kernels/qwen2_cuda.cu:upload_w does today:
 *   per-tensor cudaMalloc + synchronous cudaMemcpyHostToDevice from the
 *   mmap'd source.
 *
 * S4 mirrors bench/bench_load_strategies.cu:run_S4:
 *   one cudaMalloc arena sized to the sum of all weight bytes, then
 *   pinned double-buffered async upload per tensor into arena offsets.
 *   The "host source" is the SAME mmap'd page (no extra copy) — this is
 *   the conservative S4 that does NOT prespec the host. The pinned
 *   staging buffer is the only difference vs S0.
 *
 * Both strategies walk the *actual* tensor list of the GGUF (no synthetic
 * set). Cold vs warm controlled via the same posix_fadvise + MADV_DONTNEED
 * trick used in bench/bench_load_strategies.cu.
 *
 * Output: median-of-N wall + median-of-N cudaEvent timing per strategy,
 * broken down into total/allocate/copy/other. Speedup + 95% CI printed.
 *
 * BUILD:
 *   $HOME/mmcuda/bin/nvcc -arch=sm_86 -O2 -I include \
 *       -o build/measure_real_loader bench/measure_real_loader.cu \
 *       src/loader_gguf.c -lpthread
 *
 * RUN:
 *   LD_LIBRARY_PATH=$HOME/mmcuda/lib ./build/measure_real_loader \
 *       data/models/gemma-4-E2B-it-Q4_0.gguf [runs=3] [cold=1]
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <string>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "loader_gguf.h"

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); std::exit(1);} } while(0)

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0){ return std::chrono::duration<double,std::milli>(Clock::now()-t0).count(); }

/* mmap helpers (identical style to bench/bench_load_strategies.cu) */
struct Map { void* p=nullptr; int fd=-1; size_t n=0; };

static Map map_file_ro(const char* path){
    Map m; struct stat st{};
    m.fd=open(path,O_RDONLY); if(m.fd<0){perror("open");std::exit(1);}
    fstat(m.fd,&st); m.n=(size_t)st.st_size;
    m.p=mmap(nullptr,m.n,PROT_READ,MAP_SHARED,m.fd,0);
    if(m.p==MAP_FAILED){perror("mmap");std::exit(1);}
    return m;
}
static void unmap_drop(Map& m, bool evict){
    if(evict){
        madvise(m.p,m.n,MADV_DONTNEED);
        munmap(m.p,m.n);
        /* fsync a temp file that reads the same range, kernel may evict */
        posix_fadvise(m.fd,0,0,POSIX_FADV_DONTNEED);
    } else {
        munmap(m.p,m.n);
    }
    close(m.fd); m={};
}
static double residency(const Map& m){
    size_t pg=(size_t)getpagesize(), npg=m.n/pg;
    size_t sample_n = npg > (1u<<26) ? (1u<<26) : npg;
    size_t sample_bytes = sample_n * pg;
    std::vector<unsigned char> vec(sample_n);
    if(mincore(m.p, sample_bytes, vec.data())) return -1.0;
    size_t res=0;
    for(size_t i=0;i<sample_n;i++) res+=(vec[i]&1);
    return (double)res/sample_n;
}
static void warm_map(const Map& m){
    madvise(m.p,m.n,MADV_WILLNEED);
    volatile uint64_t sink=0; const uint64_t* p=(const uint64_t*)m.p;
    for(size_t i=0;i<m.n;i+=4096) sink+=p[i/8]; (void)sink;
}

/* S0: per-tensor cudaMalloc + sync cudaMemcpy. Mirrors upload_w() exactly. */
struct S0Result { double wall_ms; double alloc_ms; double copy_ms; size_t bytes; int n; };
static S0Result run_S0(const Map& m, const std::vector<GGUFTensor*>& ts, bool warm){
    if(warm) warm_map(m);
    cudaEvent_t aS,aE,cS,cE;
    CK(cudaEventCreate(&aS)); CK(cudaEventCreate(&aE));
    CK(cudaEventCreate(&cS)); CK(cudaEventCreate(&cE));
    auto t0=Clock::now();
    CK(cudaEventRecord(aS));
    size_t total=0; int n=0;
    std::vector<void*> ptrs; ptrs.reserve(ts.size());
    for(auto* t:ts){
        if(!t->data) continue;
        void* d; CK(cudaMalloc(&d,t->size_bytes));
        ptrs.push_back(d); total+=t->size_bytes; n++;
    }
    CK(cudaEventRecord(aE)); CK(cudaEventSynchronize(aE));
    float ams=0; CK(cudaEventElapsedTime(&ams,aS,aE));
    CK(cudaEventRecord(cS));
    size_t i=0;
    for(auto* t:ts){
        if(!t->data) continue;
        CK(cudaMemcpy(ptrs[i++],(const char*)m.p + t->offset,t->size_bytes,cudaMemcpyHostToDevice));
    }
    CK(cudaEventRecord(cE)); CK(cudaEventSynchronize(cE));
    float cms=0; CK(cudaEventElapsedTime(&cms,cS,cE));
    double wall=ms_since(t0);
    for(void* p:ptrs) cudaFree(p);
    CK(cudaEventDestroy(aS)); CK(cudaEventDestroy(aE));
    CK(cudaEventDestroy(cS)); CK(cudaEventDestroy(cE));
    return {wall, (double)ams, (double)cms, total, n};
}

/* S4: one arena + pinned double-buffered async upload per tensor.
 * Identical upload_pinned_dbuf to bench/bench_load_strategies.cu. */
static void upload_pinned_dbuf(const Map& m, const GGUFTensor* t, char* d_base,
                               char* stage[2], size_t slot, cudaStream_t st, cudaEvent_t ev[2]){
    const char* src=(const char*)m.p + t->offset;
    size_t done=0; int buf=0;
    while(done<t->size_bytes){
        size_t c=std::min(slot,t->size_bytes-done);
        CK(cudaEventSynchronize(ev[buf]));
        memcpy(stage[buf], src+done, c);
        CK(cudaMemcpyAsync(d_base+done, stage[buf], c, cudaMemcpyHostToDevice, st));
        CK(cudaEventRecord(ev[buf], st));
        done+=c; buf^=1;
    }
}
struct S4Result { double wall_ms; double alloc_ms; double copy_ms; size_t bytes; int n; };
static S4Result run_S4(const Map& m, const std::vector<GGUFTensor*>& ts, bool warm){
    if(warm) warm_map(m);
    size_t total=0; int n=0;
    for(auto* t:ts){ if(t->data){ total+=t->size_bytes; n++; } }
    const size_t SLOT=32ull<<20;
    char* stage[2]={};
    CK(cudaHostAlloc(&stage[0],SLOT,cudaHostAllocDefault));
    CK(cudaHostAlloc(&stage[1],SLOT,cudaHostAllocDefault));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t ev[2]; CK(cudaEventCreate(&ev[0])); CK(cudaEventCreate(&ev[1]));
    cudaEvent_t aS,aE,cS,cE;
    CK(cudaEventCreate(&aS)); CK(cudaEventCreate(&aE));
    CK(cudaEventCreate(&cS)); CK(cudaEventCreate(&cE));
    auto t0=Clock::now();
    char* arena=nullptr;
    CK(cudaEventRecord(aS));
    CK(cudaMalloc(&arena,total));
    CK(cudaEventRecord(aE)); CK(cudaEventSynchronize(aE));
    float ams=0; CK(cudaEventElapsedTime(&ams,aS,aE));
    CK(cudaEventRecord(cS));
    size_t off=0;
    for(auto* t:ts){
        if(!t->data) continue;
        upload_pinned_dbuf(m, t, arena+off, stage, SLOT, st, ev);
        off += t->size_bytes;
    }
    CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(cE)); CK(cudaEventSynchronize(cE));
    float cms=0; CK(cudaEventElapsedTime(&cms,cS,cE));
    double wall=ms_since(t0);
    CK(cudaFree(arena));
    CK(cudaStreamDestroy(st));
    CK(cudaEventDestroy(ev[0])); CK(cudaEventDestroy(ev[1]));
    CK(cudaEventDestroy(aS)); CK(cudaEventDestroy(aE));
    CK(cudaEventDestroy(cS)); CK(cudaEventDestroy(cE));
    CK(cudaFreeHost(stage[0])); CK(cudaFreeHost(stage[1]));
    return {wall, (double)ams, (double)cms, total, n};
}

static double med(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
static double mean(std::vector<double> v){ double s=0; for(double x:v) s+=x; return s/v.size(); }
static double sd(std::vector<double> v){
    double m=mean(v), s=0; for(double x:v){ double d=x-m; s+=d*d; }
    return std::sqrt(s/(v.size()-1));
}

int main(int argc, char** argv){
    if(argc<2){ fprintf(stderr,"usage: %s <gguf> [runs=3] [cold=1]\n",argv[0]); return 1; }
    const char* path=argv[1];
    int runs = argc>2 ? atoi(argv[2]) : 3;
    int cold = argc>3 ? atoi(argv[3]) : 1;
    int dev=0; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop,dev));
    fprintf(stderr,"[gpu] %s, %.1f GB free\n", prop.name, prop.totalGlobalMem/(1024.0*1024*1024));

    /* load GGUF (re-uses src/loader_gguf.c) */
    GGUFModel* model = gguf_load(path);
    if(!model){ fprintf(stderr,"gguf_load failed\n"); return 1; }
    fprintf(stderr,"[gguf] tensors=%d total=%.2f MiB\n", model->tensor_count, model->mmap_size/(1024.0*1024));

    /* collect every tensor with data+offset (these are upload_w candidates) */
    std::vector<GGUFTensor*> tlist;
    size_t sum_bytes=0;
    for(int i=0;i<model->tensor_count;i++){
        GGUFTensor* t = &model->tensors[i];
        if(t->data && t->size_bytes>0){
            tlist.push_back(t); sum_bytes += t->size_bytes;
        }
    }
    fprintf(stderr,"[plan] %zu upload-candidate tensors, %.2f MiB total weight bytes\n",
            tlist.size(), sum_bytes/(1024.0*1024));
    if(sum_bytes > (size_t)prop.totalGlobalMem * 0.6){
        fprintf(stderr,"[abort] weight bytes (%.0f MiB) exceed 60%% of VRAM (%.0f MiB); refusing to allocate arena\n",
                sum_bytes/(1024.0*1024), prop.totalGlobalMem/(1024.0*1024));
        gguf_free(model); return 1;
    }
    /* Peak VRAM during S0 = sum_bytes (one alloc per tensor held until last
     * memcpy); during S4 = sum_bytes (one arena). Ensure neither exceeds 0.55x
     * free VRAM after the model's own ~300MB of state. We use free memory
     * not total to avoid OOMing a busy GPU. */
    size_t free_b=0,total_b=0; CK(cudaMemGetInfo(&free_b,&total_b));
    if(sum_bytes > free_b * 9 / 10){
        fprintf(stderr,"[abort] weight bytes (%.0f MiB) exceed 90%% of FREE VRAM (%.0f MiB)\n",
                sum_bytes/(1024.0*1024), free_b/(1024.0*1024));
        gguf_free(model); return 1;
    }

    std::vector<double> s0_wall, s0_alloc, s0_copy, s4_wall, s4_alloc, s4_copy;
    for(int r=0;r<runs;r++){
        Map m = map_file_ro(path);
        if(cold){
            /* Evict page cache from previous iteration by madvise on this fresh
             * mapping. MADV_DONTNEED on a fresh SHARED map asks the kernel to
             * drop any cached pages for that range. If file is hot from prior
             * run, this is unreliable; the alternative (drop_caches) requires
             * root. We log residency before AND after to document reality. */
            double pre = residency(m);
            madvise(m.p, m.n, MADV_DONTNEED);
            posix_fadvise(m.fd, 0, 0, POSIX_FADV_DONTNEED);
            double post = residency(m);
            fprintf(stderr,"\n=== run %d (%s) page-cache residency pre=%.0f%% post-evict=%.0f%% ===\n",
                    r, cold?"COLD":"WARM", 100.0*pre, 100.0*post);
        } else {
            fprintf(stderr,"\n=== run %d (WARM) ===\n", r);
        }

        S0Result a = run_S0(m, tlist, !cold);
        fprintf(stderr,"  S0 wall=%.1fms  alloc=%.1fms  copy=%.1fms  bytes=%zu  GBps=%.2f\n",
                a.wall_ms, a.alloc_ms, a.copy_ms, a.bytes,
                a.copy_ms>0 ? a.bytes/1e9/(a.copy_ms/1000.0) : 0.0);
        s0_wall.push_back(a.wall_ms); s0_alloc.push_back(a.alloc_ms); s0_copy.push_back(a.copy_ms);
        CK(cudaDeviceSynchronize());
        /* drop the per-tensor pointers we malloc'd — S0 holds all of them
         * until this point; freeing here returns VRAM before S4 allocates. */
        unmap_drop(m, cold);

        Map m2 = map_file_ro(path);
        if(cold){ madvise(m2.p,m2.n,MADV_DONTNEED); posix_fadvise(m2.fd,0,0,POSIX_FADV_DONTNEED); }
        CK(cudaDeviceSynchronize());
        S4Result b = run_S4(m2, tlist, !cold);
        fprintf(stderr,"  S4 wall=%.1fms  alloc=%.1fms  copy=%.1fms  bytes=%zu  GBps=%.2f\n",
                b.wall_ms, b.alloc_ms, b.copy_ms, b.bytes,
                b.copy_ms>0 ? b.bytes/1e9/(b.copy_ms/1000.0) : 0.0);
        s4_wall.push_back(b.wall_ms); s4_alloc.push_back(b.alloc_ms); s4_copy.push_back(b.copy_ms);
        CK(cudaDeviceSynchronize());
        unmap_drop(m2, cold);
    }

    double m0w=med(s0_wall), m4w=med(s4_wall);
    double s0w=sd(s0_wall), s4w=sd(s4_wall);
    double sp = m0w/m4w;
    /* 95% CI on ratio via log-normal approx: exp(log(sp) ± 1.96*sd_log) */
    std::vector<double> lograt; for(size_t i=0;i<s0_wall.size();i++) lograt.push_back(std::log(s0_wall[i]/s4_wall[i]));
    double mlr=mean(lograt), slr=sd(lograt);
    double ci_lo = std::exp(mlr - 1.96*slr), ci_hi = std::exp(mlr + 1.96*slr);

    printf("\n=========== %s %s — n=%d ===========\n",
           cold?"COLD":"WARM", path, runs);
    printf("  S0 wall:  median=%.1f ms  mean=%.1f  sd=%.1f  CI(95)=[%.1f, %.1f]\n",
           m0w, mean(s0_wall), s0w, m0w-1.96*s0w, m0w+1.96*s0w);
    printf("  S0 alloc: median=%.1f  copy=%.1f\n", med(s0_alloc), med(s0_copy));
    printf("  S4 wall:  median=%.1f ms  mean=%.1f  sd=%.1f  CI(95)=[%.1f, %.1f]\n",
           m4w, mean(s4_wall), s4w, m4w-1.96*s4w, m4w+1.96*s4w);
    printf("  S4 alloc: median=%.1f  copy=%.1f\n", med(s4_alloc), med(s4_copy));
    printf("  Speedup S0/S4: %.2fx  95%% CI [%.2fx, %.2fx]\n", sp, ci_lo, ci_hi);
    printf("  Copy-GBps S0: %.2f  S4: %.2f\n",
           sum_bytes/1e9/(med(s0_copy)/1000.0), sum_bytes/1e9/(med(s4_copy)/1000.0));
    gguf_free(model);
    return 0;
}
