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

$(BUILD)/libtinytorch_cuda.so: kernels/gemm_cuda.cu kernels/gemv_q4_cuda.cu | $(BUILD)
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 \
	  -I$(CUDA_INC) -Iinclude -shared -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ kernels/gemm_cuda.cu kernels/gemv_q4_cuda.cu \
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

$(BUILD)/run_llm_gpu: examples/run_llm_gpu.c src/loader_gguf.c src/tokenizer_bpe.c src/async_printer.c kernels/gemv_q4_cuda.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  examples/run_llm_gpu.c src/loader_gguf.c src/tokenizer_bpe.c src/async_printer.c kernels/gemv_q4_cuda.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread

run_llm_gpu: $(BUILD)/run_llm_gpu

$(BUILD)/chat_llm_gpu: examples/chat_llm_gpu.c src/loader_gguf.c src/tokenizer_bpe.c src/async_printer.c kernels/gemv_q4_cuda.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  examples/chat_llm_gpu.c src/loader_gguf.c src/tokenizer_bpe.c src/async_printer.c kernels/gemv_q4_cuda.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread

chat_llm_gpu: $(BUILD)/chat_llm_gpu

.PHONY: run_llm_gpu chat_llm_gpu
