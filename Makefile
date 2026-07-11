UNAME_S := $(shell uname -s)

CC ?= cc
CFLAGS ?= -O3 -ffast-math -g -Wall -Wextra -std=c99 -D_GNU_SOURCE
LDLIBS ?= -lm -lpthread

# -march=native is an x86/GCC-ism; Apple Silicon clang doesn't want it (the
# compiler already targets the host CPU) and some clang builds reject it.
ifneq ($(UNAME_S),Darwin)
CFLAGS += -march=native
endif

# OpenMP parallelizes the CPU matmul/dequant loops (mul_mat_f32,
# dequantize_row_q4_K/q2_K in hy3.c) and the quantizer in hy3_convert.c.
# Linux (gcc/clang+libomp-dev) normally has this out of the box. Apple's
# bundled clang does not ship OpenMP; if Homebrew's libomp is installed we
# use it, otherwise we build without OpenMP (correct, just single-threaded
# on any CPU-resident layers -- irrelevant when running --metal, since all
# layers are Metal-resident there).
ifeq ($(UNAME_S),Darwin)
  LIBOMP_PREFIX := $(firstword $(wildcard /opt/homebrew/opt/libomp /usr/local/opt/libomp))
  ifneq ($(LIBOMP_PREFIX),)
    CFLAGS  += -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
    LDLIBS  += -L$(LIBOMP_PREFIX)/lib -lomp
  else
    $(info hy3: libomp not found (brew install libomp) -- building without OpenMP; \
           this only affects CPU-resident-layer speed, not --metal)
  endif
else
  CFLAGS  += -fopenmp
  LDLIBS  += -fopenmp
endif

NVCC ?= nvcc
# Blackwell (B200/B300, compute capability 10.x): sm_100 is this toolkit's
# newest officially-named real target (CUDA 12.8's nvcc has no explicit
# `sm_103` for "Blackwell Ultra" B300, which `nvidia-smi --query-gpu=compute_cap`
# reports as 10.3) -- so we also embed compute_100 *PTX* alongside the
# sm_100 cubin, and the driver JIT-compiles that PTX to real B300 SASS at
# load time. sm_90 (Hopper H100/H200) is kept for the existing README
# sizing guidance ("H200 (144GB) ... --gpu-layers 40-47").
#
# IMPORTANT: use the plain `sm_90`/`sm_100`/`compute_100` targets, NOT the
# family-specific "a"-suffixed ones (`sm_90a`, `sm_100a`, `compute_100a`).
# Empirically verified on a real NVIDIA B300 SXM6 (driver 580.126.09, CUDA
# 12.8) with the real ~162GB hy3 checkpoint: building with ANY "a"-suffixed
# target -- whether real `sm_100a`/`sm_90a` cubin or JIT-compiled from
# `compute_100a`/`compute_90a` PTX -- launches without any CUDA error but
# silently corrupts every kernel's output (all-zero logits from the very
# first token, on every layer). The plain (non-"a") `sm_90`/`sm_100`
# targets produce correct, verified logits (matching the CPU backend's
# output token-for-token on "11+22+33=?") on the exact same hardware. Do
# not add "a" suffixes back without re-verifying end to end on real
# hardware with a real checkpoint -- this is not a hypothetical concern,
# it silently breaks inference with no error message.
NVCC_ARCH_FLAGS ?= -gencode arch=compute_90,code=sm_90 \
                    -gencode arch=compute_100,code=sm_100 \
                    -gencode arch=compute_100,code=compute_100
NVCC_FLAGS ?= -O3 $(NVCC_ARCH_FLAGS) -Xcompiler -fopenmp -Xcompiler -pthread
CUDA_LIBS ?= -lcublas -lcudart

.PHONY: all clean

all: hy3 hy3-cli hy3-convert hy3-agent

# --- Backend selection -----------------------------------------------
# Linux + CUDA toolkit present -> CUDA backend (hy3_gpu.cu, --gpu-layers).
# macOS                        -> Metal backend (hy3_metal.m + hy3.metal,
#                                  --metal offloads all 80 layers using
#                                  unified memory; see run_metal.sh).
# Anything else                -> CPU-only.
ifneq ($(wildcard /usr/local/cuda/include/cuda_runtime.h),)
HY3_CUDA ?= 1
endif
ifeq ($(UNAME_S),Darwin)
HY3_METAL ?= 1
endif

ifeq ($(HY3_CUDA),1)
CFLAGS += -DHY3_CUDA

hy3: hy3_cuda.o hy3.o hy3_cli.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_LIBS) -lm -lpthread

hy3-cli: hy3_cuda.o hy3.o hy3_cli.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_LIBS) -lm -lpthread

hy3-agent: hy3_cuda.o hy3.o hy3_agent.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_LIBS) -lm -lpthread

hy3_agent.o: hy3_agent.c hy3.h
	$(CC) $(CFLAGS) -c -o $@ hy3_agent.c

hy3_cuda.o: hy3_gpu.cu hy3.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ hy3_gpu.cu

else ifeq ($(HY3_METAL),1)
CFLAGS += -DHY3_METAL '-DHY3_METAL_SHADER_PATH="$(CURDIR)/hy3.metal"'
OBJC_FLAGS = -O3 -g -fobjc-arc -Wall -Wextra -DHY3_METAL '-DHY3_METAL_SHADER_PATH="$(CURDIR)/hy3.metal"'
METAL_LIBS = -framework Metal -framework Foundation

hy3: hy3_metal.o hy3.o hy3_cli.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) $(METAL_LIBS)

hy3-cli: hy3_metal.o hy3.o hy3_cli.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) $(METAL_LIBS)

hy3-agent: hy3_metal.o hy3.o hy3_agent.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) $(METAL_LIBS)

hy3_metal.o: hy3_metal.m hy3.h
	$(CC) $(OBJC_FLAGS) -c -o $@ hy3_metal.m

else
hy3: hy3.o hy3_cli.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

hy3-cli: hy3.o hy3_cli.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

hy3-agent: hy3.o hy3_agent.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
endif

hy3-convert: hy3_convert.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

hy3.o: hy3.c hy3.h
	$(CC) $(CFLAGS) -c -o $@ hy3.c

hy3_cli.o: hy3_cli.c hy3.h
	$(CC) $(CFLAGS) -c -o $@ hy3_cli.c

# Standalone Metal kernel experiment bed (macOS only). Not part of `all`.
fast_metal: fast_metal.m
	$(CC) -O3 -fobjc-arc -o $@ fast_metal.m -framework Metal -framework Foundation

clean:
	rm -f hy3 hy3-cli hy3-convert hy3-agent fast_metal *.o
