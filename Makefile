CC      ?= gcc
CXX     ?= g++
CFLAGS  ?= -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -std=c11 -fPIC -DTT_IN_LIB
BUILD   := build
SRCS    := $(wildcard src/*.c)
HDRS    := $(wildcard include/*.h)

PYBIND_INC := $(shell python3 -m pybind11 --includes 2>/dev/null)
EXT_SUFFIX := $(shell python3-config --extension-suffix 2>/dev/null || echo ".so")

$(BUILD)/libtinytorch.so: $(SRCS) $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -Iinclude -shared -o $@ $(SRCS) -lm

$(BUILD)/tinytorch_pybind$(EXT_SUFFIX): src/bindings.cpp $(SRCS) $(HDRS) | $(BUILD)
	$(CXX) $(CFLAGS) -std=c++17 $(PYBIND_INC) -Iinclude -shared -fPIC -o $@ src/bindings.cpp $(SRCS) -lm

$(BUILD):
	mkdir -p $(BUILD)

lib: $(BUILD)/libtinytorch.so
pybind: $(BUILD)/tinytorch_pybind$(EXT_SUFFIX)

# M2 GEMM sweep driver (block sizes via -DMC= -DKC= -DNC=).
$(BUILD)/bench_gemm_sweep: tools/bench_gemm_sweep.c src/gemm.c | $(BUILD)
	$(CC) -O3 -mavx2 -mfma -fopenmp -Wall -Iinclude -o $@ tools/bench_gemm_sweep.c -lm

bench-gemm-sweep: $(BUILD)/bench_gemm_sweep

# CPU-only sanity: syntax-check every source + rebuild lib.
ci:
	mkdir -p $(BUILD)/ci_logs
	for f in src/*.c; do \
	  $(CC) -std=c11 -fsyntax-only -DTT_IN_LIB -Iinclude -Isrc "$$f" || exit 1; \
	done
	$(MAKE) -s lib

clean:
	rm -rf $(BUILD)

.PHONY: lib pybind bench-gemm-sweep ci clean
