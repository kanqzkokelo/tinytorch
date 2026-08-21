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
