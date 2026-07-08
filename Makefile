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
NVCC_FLAGS ?= -O3 -arch=sm_90 -Xcompiler -fopenmp -Xcompiler -pthread
CUDA_LIBS ?= -lcublas -lcudart

.PHONY: all clean

all: hy3 hy3-cli hy3-convert

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

hy3_metal.o: hy3_metal.m hy3.h
	$(CC) $(OBJC_FLAGS) -c -o $@ hy3_metal.m

else
hy3: hy3.o hy3_cli.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

hy3-cli: hy3.o hy3_cli.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
endif

hy3-convert: hy3_convert.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

hy3.o: hy3.c hy3.h
	$(CC) $(CFLAGS) -c -o $@ hy3.c

hy3_cli.o: hy3_cli.c hy3.h
	$(CC) $(CFLAGS) -c -o $@ hy3_cli.c

clean:
	rm -f hy3 hy3-cli hy3-convert *.o
