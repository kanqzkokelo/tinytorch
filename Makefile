CC      ?= gcc
CFLAGS  ?= -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -std=c11 -fPIC -DTT_IN_LIB
BUILD   := build
SRCS    := $(wildcard src/*.c)
HDRS    := $(wildcard include/*.h)

$(BUILD)/libtinytorch.so: $(SRCS) $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -Iinclude -shared -o $@ $(SRCS) -lm

$(BUILD):
	mkdir -p $(BUILD)

lib: $(BUILD)/libtinytorch.so

clean:
	rm -rf $(BUILD)

.PHONY: lib clean

NVCC ?= $(HOME)/mmcuda/bin/nvcc
CUDA_INC := $(HOME)/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/include

$(BUILD)/libtinytorch_cuda.so: kernels/gemm_cuda.cu | $(BUILD)
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 \
	  -I$(CUDA_INC) -Iinclude -shared -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ kernels/gemm_cuda.cu \
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
