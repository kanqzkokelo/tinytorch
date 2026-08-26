/*
 * bench_load_strategies.cu — PROTOTYPE: model-load-time experiment matrix.
 *
 * Measures host->device upload throughput for a synthetic 2GB tensor set that
 * mimics the real GGUF distribution (many small f32[1536] tensors + large
 * q4_0 blocks up to 226MB), sourced through an mmap like loader_gguf.c does.
 *
 * Strategies:
 *   S0  per-tensor cudaMalloc + synchronous cudaMemcpy from pageable mmap
 *       (current engine behavior, see kernels/qwen2_cuda.cu:72-73)
 *   S1  ONE pre-pinned 64MB staging buffer (cudaHostAlloc, 2x32MB slots),
 *       double-buffered mmap->pinned->cudaMemcpyAsync per chunk
 *   S2  cudaMemcpyAsync issued directly from mmap'd pages (no pinning;
 *       often silently degrades to sync — we measure what actually happens)
 *   S3  cudaMallocManaged + CPU memcpy + GPU touch kernel (UVM demand-fault)
 *   S4  ONE cudaMalloc arena for the whole model + S1-style pinned staging
 *       into arena offsets (eliminates per-tensor cudaMalloc/Free churn,
 *       which profiling showed dominates S0)
 *
 * Cache-state control: between runs we munmap, fsync, and issue
 * posix_fadvise(POSIX_FADV_DONTNEED) on the backing file to evict it from
 * page cache; mincore() sampling reports estimated residency before each run
 * so cold vs warm numbers are labeled honestly. Each strategy also runs once
 * explicitly WARM (madvise MADV_WILLNEED + pre-read) for comparison.
 *
 * BUILD:
 *   $HOME/mmcuda/bin/nvcc -arch=sm_86 -O2 -o bench_load_strategies \
 *       bench/bench_load_strategies.cu
 *
 * RUN:
 *   LD_LIBRARY_PATH=$HOME/mmcuda/lib ./bench_load_strategies [tensor_file]
 *   (default file: /tmp/bench_tensor_set_2g.bin, created if absent, 2GiB)
 *
 * Output: median-of-3 wall-clock per strategy (cold + warm), effective GB/s,
 * projected 5GB-model load time. VRAM footprint ~2GB (+overhead) <= 2.5GB.
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

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0){ return std::chrono::duration<double,std::milli>(Clock::now()-t0).count(); }

/* ---------- synthetic tensor set ---------- */

struct TensorSpec { size_t off; size_t size; bool q4; };

static std::vector<TensorSpec> g_specs;
static size_t g_total;
static void build_specs(){
    g_specs.clear();
    size_t off = 0;
    /* 8 large q4_0 blocks x 226MB = 1808MB */
    const size_t BIG = 226ull<<20;
    for(int i=0;i<8;i++){ g_specs.push_back({off,BIG,true}); off+=BIG; }
    /* 4 medium q4_0 blocks x 64MB = 256MB */
    const size_t MED = 64ull<<20;
    for(int i=0;i<4;i++){ g_specs.push_back({off,MED,true}); off+=MED; }
    /* 3000 small f32[1536] = ~18MB */
    const size_t SMALL = 1536*sizeof(float);
    for(int i=0;i<3000;i++){ g_specs.push_back({off,SMALL,false}); off+=SMALL; }
    g_total = off;
    fprintf(stderr,"[set] %zu tensors, %.2f GiB\n", g_specs.size(), off/(1024.0*1024*1024));
}

static const char* FILEPATH = "/tmp/bench_tensor_set_2g.bin";

static void ensure_file(size_t total){
    struct stat st{};
    if(::stat(FILEPATH,&st)==0 && (size_t)st.st_size==total) return;
    fprintf(stderr,"[file] writing %s (%zu MiB)...\n",FILEPATH,total>>20);
    int fd=open(FILEPATH,O_RDWR|O_CREAT|O_TRUNC,0644);
    if(fd<0){perror("open");exit(1);}
    if(ftruncate(fd,total)){perror("ftruncate");exit(1);}
    /* fill sparsely-mapped regions with real data so cold page-in hits disk */
    char* buf=(char*)malloc(1<<20); memset(buf,0xAB,1<<20);
    if(lseek(fd,0,SEEK_SET)<0){perror("lseek");exit(1);}
    for(size_t w=0;w<total;w+=(1<<20)) {
        ssize_t n=write(fd,buf,1<<20); if(n!=(1<<20)){perror("write");exit(1);} }
    free(buf); close(fd);
}

/* mmap helpers + residency probe */
struct Map { void* p=nullptr; int fd=-1; size_t n=0; };

static Map map_file(size_t n){
    Map m; m.n=n;
    m.fd=open(FILEPATH,O_RDONLY); if(m.fd<0){perror("open ro");exit(1);}
    m.p=mmap(nullptr,n,PROT_READ,MAP_SHARED,m.fd,0);
    if(m.p==MAP_FAILED){perror("mmap");exit(1);}
    return m;
}
static void unmap_drop(Map& m){
    madvise(m.p,m.n,MADV_DONTNEED);
    munmap(m.p,m.n);
    fsync(m.fd);
    posix_fadvise(m.fd,0,0,POSIX_FADV_DONTNEED);  /* best-effort cache evict */
    close(m.fd); m={};
}
/* fraction of pages currently resident (sampled every 64 pages) */
static double residency(const Map& m){
    size_t pg=(size_t)getpagesize(), npg=m.n/pg;
    std::vector<unsigned char> vec(npg);   /* mincore: one byte PER PAGE */
    if(mincore(m.p,m.n,vec.data())) return -1.0;
    size_t res=0,tot=0;
    for(size_t i=0;i<npg;i++){ tot++; res+=(vec[i]&1); }
    return tot? (double)res/tot : 0.0;
}
static void warm_map(const Map& m){
    madvise(m.p,m.n,MADV_WILLNEED);
    volatile uint64_t sink=0; const uint64_t* p=(const uint64_t*)m.p;
    for(size_t i=0;i<m.n;i+=4096) sink+=p[i/8]; (void)sink;
}

/* ---------- strategies ---------- */

static double run_S0(const Map& m,const std::vector<TensorSpec>& ts,bool warm,size_t total) {
    if(warm) warm_map(m);
    auto t0=Clock::now();
    for(const auto& t:ts){
        void* d; CK(cudaMalloc(&d,t.size));
        CK(cudaMemcpy(d,(const char*)m.p+t.off,t.size,cudaMemcpyHostToDevice));
        CK(cudaFree(d));
    }
    CK(cudaDeviceSynchronize());
    return ms_since(t0);
}

static double run_S1(const Map& m,const std::vector<TensorSpec>& ts,bool warm,size_t total) {
    if(warm) warm_map(m);
    const size_t SLOT=32ull<<20, NBUF=2;
    char* stage[NBUF]={};
    CK(cudaHostAlloc(&stage[0],SLOT,cudaHostAllocDefault));
    CK(cudaHostAlloc(&stage[1],SLOT,cudaHostAllocDefault));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t ev[NBUF]; CK(cudaEventCreate(&ev[0])); CK(cudaEventCreate(&ev[1]));
    auto t0=Clock::now();
    for(const auto& t:ts){
        void* d; CK(cudaMalloc(&d,t.size));
        const char* src=(const char*)m.p+t.off;
        size_t done=0; int buf=0;
        while(done<t.size){
            size_t c=std::min(SLOT,t.size-done);
            CK(cudaEventSynchronize(ev[buf]));          /* slot free? */
            memcpy(stage[buf],src+done,c);              /* mmap -> pinned */
            CK(cudaMemcpyAsync((char*)d+done,stage[buf],c,cudaMemcpyHostToDevice,st));
            CK(cudaEventRecord(ev[buf],st));
            done+=c; buf^=1;
        }
        CK(cudaStreamSynchronize(st));
        CK(cudaFree(d));
    }
    double el=ms_since(t0);
    CK(cudaStreamDestroy(st)); CK(cudaEventDestroy(ev[0])); CK(cudaEventDestroy(ev[1]));
    CK(cudaFreeHost(stage[0])); CK(cudaFreeHost(stage[1]));
    return el;
}

static double run_S2(const Map& m,const std::vector<TensorSpec>& ts,bool warm,size_t total) {
    if(warm) warm_map(m);
    auto t0=Clock::now();
    for(const auto& t:ts){
        void* d; CK(cudaMalloc(&d,t.size));
        /* async issued straight from pageable mmap-backed address */
        CK(cudaMemcpyAsync(d,(const char*)m.p+t.off,t.size,cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());   /* sync per tensor mirrors engine loop */
        CK(cudaFree(d));
    }
    return ms_since(t0);
}

/* ---------- strategy shared pieces ---------- */

/* stream chunked mmap->pinned->device upload used by S1 and S4 */
static void upload_pinned_dbuf(const Map& m,const TensorSpec& t,char* d_base,
                               char* stage[2],size_t slot,cudaStream_t st,cudaEvent_t ev[2]){
    const char* src=(const char*)m.p+t.off;
    size_t done=0; int buf=0;
    while(done<t.size){
        size_t c=std::min(slot,t.size-done);
        CK(cudaEventSynchronize(ev[buf]));          /* slot free? */
        memcpy(stage[buf],src+done,c);              /* mmap -> pinned */
        CK(cudaMemcpyAsync(d_base+done,stage[buf],c,cudaMemcpyHostToDevice,st));
        CK(cudaEventRecord(ev[buf],st));
        done+=c; buf^=1;
    }
}

__global__ void touch_kernel(const char* p,size_t n,unsigned long long* out){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    size_t stride=(size_t)gridDim.x*blockDim.x;
    unsigned long long s=0;
    for(;i<n;i+=stride) s+=(unsigned char)p[i];
    if(s==0xDEADBEEFDEADBEEFull) *out=s;  /* never true; defeats DCE */
}

static double run_S3(const Map& m,const std::vector<TensorSpec>& ts,bool warm,size_t total) {
    if(warm) warm_map(m);
    unsigned long long* guard; CK(cudaMalloc(&guard,8));
    auto t0=Clock::now();
    for(const auto& t:ts){
        char* u; CK(cudaMallocManaged(&u,t.size));
        memcpy(u,(const char*)m.p+t.off,t.size);      /* CPU populates */
        touch_kernel<<<512,256>>>(u,t.size,guard);     /* GPU first-touch faults */
        CK(cudaDeviceSynchronize());
        CK(cudaFree(u));
    }
    double el=ms_since(t0);
    CK(cudaFree(guard));
    return el;
}

/* S4: one arena alloc for whole set, pinned-staged async uploads into it.
 * Alloc/free cost paid ONCE instead of per tensor. */
static double run_S4(const Map& m,const std::vector<TensorSpec>& ts,size_t total,bool warm){
    if(warm) warm_map(m);
    const size_t SLOT=32ull<<20;
    char* stage[2]={};
    CK(cudaHostAlloc(&stage[0],SLOT,cudaHostAllocDefault));
    CK(cudaHostAlloc(&stage[1],SLOT,cudaHostAllocDefault));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t ev[2]; CK(cudaEventCreate(&ev[0])); CK(cudaEventCreate(&ev[1]));
    auto t0=Clock::now();
    char* arena; CK(cudaMalloc(&arena,total));
    size_t off=0;
    for(const auto& t:ts){
        upload_pinned_dbuf(m,t,arena+off,stage,SLOT,st,ev);
        CK(cudaStreamSynchronize(st));
        off+=t.size;
    }
    double el=ms_since(t0);
    CK(cudaFree(arena));
    CK(cudaStreamDestroy(st)); CK(cudaEventDestroy(ev[0])); CK(cudaEventDestroy(ev[1]));
    CK(cudaFreeHost(stage[0])); CK(cudaFreeHost(stage[1]));
    return el;
}

/* uniform-signature wrapper for harness table */
static double run_S4_(const Map& m,const std::vector<TensorSpec>& ts,bool warm,size_t total){
    return run_S4(m,ts,total,warm);
}

/* ---------- harness ---------- */

struct Result { const char* name; bool warm; double med_ms; };

typedef double (*StrategyFn)(const Map&,const std::vector<TensorSpec>&,bool,size_t);

static double median(std::vector<double> v){
    std::sort(v.begin(),v.end()); return v[v.size()/2];
}

int main(int argc,char** argv){
    if(argc>1) FILEPATH=argv[1];

    int dev=0; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop,dev));
    printf("[gpu] %s\n",prop.name);

    build_specs();
    ensure_file(g_total);                 /* must precede any map of g_total */
    const double GB=g_total/(1024.0*1024*1024);

    Result results[20]; int nr=0;
    struct Entry { const char* name; StrategyFn fn; };
    Entry entries[]={
        {"S0 pageable-sync",run_S0},
        {"S1 pinned-dbuf ",run_S1},
        {"S2 async-mmap  ",run_S2},
        {"S3 UVM-touch   ",run_S3},
        {"S4 arena-pinned",run_S4_},
    };

    for(bool warm:{false,true}){
        printf("\n=== %s runs ===\n",warm?"WARM":"COLD"); fflush(stdout);
        for(auto& e:entries){
            std::vector<double> runs;
            for(int r=0;r<3;r++){
                Map m=map_file(g_total);
                if(!warm) printf("  [%s run%d] page-cache residency ~%.0f%%\n",
                                 e.name,r,100.0*residency(m)), fflush(stdout);
                double ms=e.fn(m,g_specs,warm,g_total);
                unmap_drop(m);
                runs.push_back(ms);
                printf("  [%s run%d] %.1f ms (%.2f GB/s)\n",e.name,r,ms,GB/(ms/1000.0));
                CK(cudaDeviceSynchronize());
            }
            double med=median(runs);
            results[nr++]={e.name,warm,med};
        }
    }

    printf("\n%-18s %-5s %10s %10s %12s\n","strategy","cache","median-ms","GB/s","5GB-proj(s)");
    for(int i=0;i<nr;i++){
        Result& r=results[i];
        printf("%-18s %-5s %10.1f %10.2f %12.1f\n",r.name,r.warm?"warm":"cold",
               r.med_ms,GB/(r.med_ms/1000.0),r.med_ms*(5.0/GB)/1000.0);
    }
    printf("\nNote: 'cold' = best-effort eviction via posix_fadvise(DONTNEED); residency\n"
           "printed per run. Kernel/page-table effects not fully droppable from userspace.\n");
    return 0;
}
