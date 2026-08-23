# PRD: Milestone 5 — High-Speed 2B LLM Engine & Zero-Overhead C Terminal Streaming

## 1. Executive Summary & Objective

The goal of Milestone 5 is to elevate `tinytorch` from a neural network framework into a production-grade, zero-overhead LLM inference engine. 

While `tinytorch`'s pure GPU CUDA Graph execution pipeline already generates **277.7 tokens/sec** on NVIDIA GeForce RTX 3050 Laptop GPU (sm_86), end-to-end terminal output speed drops to **~103.7 tokens/sec** due to Python GIL string allocation and blocking POSIX `fflush(stdout)` I/O system calls.

This PRD specifies the complete architectural, algorithmic, and implementation blueprint to eliminate CPU-side I/O bottlenecks and deliver **> 270 tokens/sec live terminal streaming speed**, outperforming `llama.cpp` CUDA GPU (**262.6 tokens/sec**).

---

## 2. Hardware & Baseline Technical Context

### Target Hardware Profile
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (Ampere architecture, `compute_86`, `sm_86`).
- **GPU SMs**: 14 Streaming Multiprocessors (1792 FP32 CUDA cores, 56 Tensor Cores).
- **VRAM Capacity & Bandwidth**: 4.0 GB GDDR6 @ **192.0 GB/s peak memory bandwidth**.
- **CPU / RAM**: Intel Core i5-11400H (12 logical threads, AVX2 + FMA, 16GB DDR4 RAM).
- **Storage Directory**: `/home/mitesh/Storage/repos/nnfromscratch` (230 GB free).

### Benchmark Model Profile
- **Model**: `Qwen2.5-0.5B-Instruct-Q4_0.gguf` (also compatible with `Llama-3.2-1B-Instruct-Q4_0.gguf`).
- **Layers**: $N_{\text{layers}} = 24$.
- **Embedding Dim ($d$)**: 896.
- **FFN Hidden Dim ($d_{\text{ffn}}$)**: 4864.
- **Attention Heads**: $N_{\text{heads}} = 14$, $N_{\text{kv\_heads}} = 2$, Head Dim $d_k = 64$.
- **Quantization**: `q4_0` (32 FP16 values stored in 16 bytes of nibbles + 1 FP16 scale = 18 bytes/block).
- **Weight Footprint in VRAM**: **253.5 MB**.
- **Theoretical Max VRAM Generation Throughput**: $\frac{192.0 \text{ GB/s}}{0.2535 \text{ GB}} = \mathbf{757.4 \text{ tokens/sec}}$.

---

## 3. Bottleneck Diagnosis & Root Cause Analysis

| Bottleneck Component | Baseline Failure Mechanism | Target Architectural Fix | Expected Improvement |
| :--- | :--- | :--- | :--- |
| **Python GIL & String Alloc** | Python string manipulation & UTF-8 decoding during loop adds ~3.8ms/token delay. | Native C BPE Tokenizer (`src/tokenizer_bpe.c`) with pre-allocated string byte offset tables. | Eliminates 100% of Python overhead (< 10 ns lookup). |
| **Blocking `fflush(stdout)`** | Synchronous POSIX `write(1, ...)` syscall halts CPU thread every token (~2.5ms delay). | Lock-Free Double-Buffered 16KB Async I/O Printer (`src/async_printer.c`). | Reduces terminal I/O latency from 2.5ms to < 0.01ms. |
| **SM Block Scheduling** | Launching 304 blocks per kernel causes 810 waves of block scheduling per token. | Grid-tiled block grid size ($14$ SM blocks matching RTX 3050 SM count). | Reduces kernel launch overhead from 9.7ms to 0.3ms. |
| **CUDA Driver API Latency** | 168 individual `cudaLaunchKernel` calls per token step incur ~2.5ms driver API delay. | Single-pass CUDA Graph stream (`cudaGraphInstantiate` / `cudaGraphLaunch`). | Driver launch overhead drops to 0.00ms. |

---

## 4. Technical Architecture Specifications

### 4.1 Native C BPE Tokenizer (`include/tokenizer_bpe.h`, `src/tokenizer_bpe.c`)
Must parse vocabulary tables directly from GGUF metadata (`tokenizer.ggml.tokens`, `tokenizer.ggml.scores`) during `gguf_load()`.

```c
// Struct layout in include/tokenizer_bpe.h
typedef struct {
    int vocab_size;
    char **tokens;        // Flat array of string pointers
    int *token_lens;      // Array of token string lengths
    float *scores;        // Token scores for BPE merge ranking
    int bos_id;
    int eos_id;
} BPETokenizer;

BPETokenizer *bpe_tokenizer_init(const GGUFModel *model);
const char *bpe_decode_token(const BPETokenizer *tok, int token_id, int *out_len);
void bpe_tokenizer_free(BPETokenizer *tok);
```

### 4.2 Lock-Free Async Terminal Printer (`include/async_printer.h`, `src/async_printer.c`)
Must execute in a background POSIX thread without mutex locks, using C11 `stdatomic.h` primitives and a 16KB double-buffer to batch Linux `write(1, ...)` syscalls.

```c
// Struct layout in include/async_printer.h
#define ASYNC_BUF_SIZE 16384
#define ASYNC_QUEUE_CAP 1024

typedef struct {
    const char *queue_data[ASYNC_QUEUE_CAP];
    int queue_lens[ASYNC_QUEUE_CAP];
    _Atomic int head;
    _Atomic int tail;
    _Atomic int done;
    pthread_t thread;
} AsyncPrinter;

AsyncPrinter *async_printer_start(void);
void async_printer_push(AsyncPrinter *ap, const char *str, int len);
void async_printer_stop_and_flush(AsyncPrinter *ap);
```

### 4.3 Fused CUDA Kernels (`kernels/gemv_q4_cuda.cu`, `kernels/flash_attn_decode_cuda.cu`)

#### A. Fused `q4_0` GEMV + SwiGLU Kernel
Computes $\text{SwiGLU}(W_{\text{gate}} x, W_{\text{up}} x)$ in registers during a single global memory sweep.

```cuda
__global__ void k_fused_swiglu_q4_0(const BlockQ4_0 *__restrict__ W_gate,
                                    const BlockQ4_0 *__restrict__ W_up,
                                    const float *__restrict__ x,
                                    float *__restrict__ out,
                                    int M, int K);
```

#### B. FlashAttention-2 In-Register Decode Kernel
Pins Query vector $Q \in \mathbb{R}^{1 \times 64}$ into warp registers and streams $K, V$ cache tiles from VRAM with online softmax tracking running max $m$ and sum $\ell$ in registers.

```cuda
__global__ void k_flash_attn_decode(const float *__restrict__ Q,
                                    const float *__restrict__ K_cache,
                                    const float *__restrict__ V_cache,
                                    float *__restrict__ Out,
                                    int seq_len, float scale);
```

---

## 5. Implementation Roadmap & Milestones

1. **Step 1: Native C BPE Tokenizer (`src/tokenizer_bpe.c`)**
   - Extract string token array from GGUF metadata.
   - Implement zero-copy `bpe_decode_token()` lookup returning token text in < 10 ns.

2. **Step 2: Lock-Free Async Printer (`src/async_printer.c`)**
   - Implement `stdatomic.h` ring buffer with 16KB IO buffer.
   - Verify zero mutex contention during GPU execution loop.

3. **Step 3: CUDA Graph Driver Wrapper (`src/llm_engine_cuda.cu`)**
   - Wrap 24 Transformer layers into a single pre-recorded CUDA Graph (`cudaGraphExec_t`).
   - Eliminate per-token CPU kernel launch overhead.

4. **Step 4: Executable Runner (`examples/run_llm_gpu.c`)**
   - Build native C executable `build/run_llm_gpu`.
   - Run live prompt generation with background terminal streaming.

---

## 6. Verification Protocol & Acceptance Criteria

### Acceptance Gate Criteria
1. **Correctness (Gate A)**: The generated text output matches valid English response to prompt `"Explain quantum computing in one sentence."`
2. **Terminal Output Speed (Gate B)**: Live terminal streaming output speed must exceed **270 tokens/sec** as measured by `clock_gettime(CLOCK_MONOTONIC)` across 100 generated tokens.
3. **Head-to-Head Victory (Gate C)**: `tinytorch` live terminal throughput must be strictly greater than `llama.cpp` CUDA GPU (**262.6 tokens/sec**).
