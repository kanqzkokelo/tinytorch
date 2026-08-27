CXX     ?= g++
CFLAGS  ?= -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -std=c11 -fPIC -DTT_IN_LIB
BUILD   := build
SRCS    := $(wildcard src/*.c)
HDRS    := $(wildcard include/*.h)

PYBIND_INC := $(shell python3 -m pybind11 --includes 2>/dev/null)
EXT_SUFFIX := $(shell python3-config --extension-suffix 2>/dev/null || echo ".so")

$(BUILD)/libtinytorch.so: $(SRCS) $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -Iinclude -shared -o $@ $(SRCS) -lm

# async_printer.c uses C11 _Atomic; g++ (pybind link) cannot parse it and
# the pybind module never calls it, so exclude it from this target.
PYBIND_SRCS := $(filter-out src/async_printer.c,$(SRCS))

$(BUILD)/tinytorch_pybind$(EXT_SUFFIX): src/bindings.cpp $(PYBIND_SRCS) $(HDRS) | $(BUILD)
	$(CXX) $(CFLAGS) -std=c++17 $(PYBIND_INC) -Iinclude -shared -fPIC -o $@ src/bindings.cpp $(PYBIND_SRCS) -lm

$(BUILD):
	mkdir -p $(BUILD)

lib: $(BUILD)/libtinytorch.so
pybind: $(BUILD)/tinytorch_pybind$(EXT_SUFFIX)

clean:
	rm -rf $(BUILD)

.PHONY: lib clean

NVCC ?= $(HOME)/mmcuda/bin/nvcc
CUDA_INC := $(HOME)/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/include

$(BUILD)/libtinytorch_cuda.so: kernels/gemm_cuda.cu kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu | $(BUILD)
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 \
	  -I$(CUDA_INC) -Iinclude -shared -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ kernels/gemm_cuda.cu kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu \
	  -Lbuild -lcudart

cuda: $(BUILD)/libtinytorch_cuda.so

.PHONY: cuda

CUBLAS_INC := $(HOME)/.local/lib/python3.12/site-packages/nvidia/cublas/include
CUBLAS_LIB := $(HOME)/.local/lib/python3.12/site-packages/nvidia/cublas/lib

$(BUILD)/libtt_cublas.so: kernels/cublas_ref.cu | $(BUILD)
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 \
	  -I$(CUDA_INC) -I$(CUBLAS_INC) -shared -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib:$(CUBLAS_LIB) \
	  -o $@ kernels/cublas_ref.cu \
	  -Lbuild -L$(CUBLAS_LIB) -lcudart -lcublas

cublas: $(BUILD)/libtt_cublas.so

.PHONY: cublas

$(BUILD)/run_llm_gpu: examples/run_llm_gpu.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/tokenizer_bpe.c src/async_printer.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  examples/run_llm_gpu.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/tokenizer_bpe.c src/async_printer.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread

run_llm_gpu: $(BUILD)/run_llm_gpu

$(BUILD)/chat_llm_gpu: examples/chat_llm_gpu.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/tokenizer_bpe.c src/async_printer.c src/chat_template.c src/samplers.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  examples/chat_llm_gpu.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/tokenizer_bpe.c src/async_printer.c src/chat_template.c src/samplers.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread

chat_llm_gpu: $(BUILD)/chat_llm_gpu

# Universal Speculative Engine orchestrator: N-gram drafter (host) +
# batched verify_speculative() (CUDA). Source list mirrors run_llm_gpu
# plus src/ngram_lookup.c.
$(BUILD)/spec_llm_gpu: examples/spec_llm_gpu.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/tokenizer_bpe.c src/async_printer.c src/ngram_lookup.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  examples/spec_llm_gpu.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/tokenizer_bpe.c src/async_printer.c src/ngram_lookup.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread

spec_llm_gpu: $(BUILD)/spec_llm_gpu

.PHONY: spec_llm_gpu

# Oracle logits tool against the vendored llama.cpp build (parity fixtures).
$(BUILD)/oracle_logits: tools/oracle_logits.c | $(BUILD)
	gcc -O2 -Wno-deprecated-declarations \
	  -I oracle/llama.cpp/include -I oracle/llama.cpp/ggml/include \
	  -o $@ tools/oracle_logits.c \
	  -L oracle/llama.cpp/build/bin -lllama \
	  -Wl,-rpath=$(CURDIR)/oracle/llama.cpp/build/bin

oracle_logits: $(BUILD)/oracle_logits

.PHONY: run_llm_gpu chat_llm_gpu

$(BUILD)/dump_logits: tools/dump_logits.c src/loader_gguf.c src/dequant_ref.c src/arch_registry.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tools/dump_logits.c src/loader_gguf.c src/dequant_ref.c src/arch_registry.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread

dump_logits: $(BUILD)/dump_logits

.PHONY: dump_logits

$(BUILD)/bench_prefill: tools/bench_prefill.c src/loader_gguf.c src/dequant_ref.c src/arch_registry.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tools/bench_prefill.c src/loader_gguf.c src/dequant_ref.c src/arch_registry.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread

bench_prefill: $(BUILD)/bench_prefill

.PHONY: bench_prefill

$(BUILD)/profile_step: tools/profile_step.cu src/loader_gguf.c src/arch_registry.c src/dequant_ref.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tools/profile_step.cu src/loader_gguf.c src/arch_registry.c src/dequant_ref.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread

profile_step: $(BUILD)/profile_step

.PHONY: profile_step

# CPU golden dequant CLI (validates GGUF quant formats against gguf-py goldens)
$(BUILD)/dequant_ref: src/dequant_ref.c src/loader_gguf.c include/dequant_ref.h include/loader_gguf.h | $(BUILD)
	gcc $(CFLAGS) -DTTQ_MAIN -Iinclude -o $@ src/dequant_ref.c src/loader_gguf.c -lm

dequant_ref: $(BUILD)/dequant_ref

.PHONY: dequant_ref

# M7 task 2: GPU golden GEMV grid for all Tier-1 quant types
$(BUILD)/test_gemv_typed: tools/test_gemv_typed.cu src/loader_gguf.c src/dequant_ref.c \
		kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu | $(BUILD)
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 \
	  -I$(CUDA_INC) -Iinclude -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tools/test_gemv_typed.cu src/loader_gguf.c src/dequant_ref.c \
	  kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lm

test_gemv_typed: $(BUILD)/test_gemv_typed

.PHONY: test_gemv_typed

# Speculative-decode verify test: compares batched verify(N) logits against
# N sequential single-token forwards (bit-exact on qwen2.5-0.5b-q4_0).
$(BUILD)/test_spec_verify: tests/test_spec_verify.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tests/test_spec_verify.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c \
	  kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread

test_spec_verify: $(BUILD)/test_spec_verify

.PHONY: test_spec_verify

# CI: cheap CPU-only sanity (no GPU needed) -- same checks as GitHub CI.
ci:
	./scripts/ci_local.sh

.PHONY: ci
