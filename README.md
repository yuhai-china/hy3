# hy3 — tencent/Hy3 GGUF Converter and Inference Engine

`hy3` is a from-scratch C inference engine and GGUF converter for **tencent/Hy3**
(`HYV3ForCausalLM`, `model_type: hy_v3`), a 295B-parameter / 21B-active-parameter
Mixture-of-Experts model released by **Tencent's Hunyuan ("Hy") team**
(`tencent/Hy3` on Hugging Face).

The project is called **hy3**.

> **Derivative project.** hy3 is a third-party derivative of the official
> **[`tencent/Hy3`](https://huggingface.co/tencent/Hy3)** model — it re-implements
> loading/inference from scratch and is not affiliated with or endorsed by Tencent.
> Pre-converted GGUF weights produced by this project's converter can be
> downloaded from **[`cloudyu/hy3-gguf`](https://huggingface.co/cloudyu/hy3-gguf)**.

> **⚠️ Testing scope:** All recent performance work (single-command-buffer
> forward, concurrent encoder, GPU-resident MoE routing, fused per-head RMS
> norm, SIMD-group matmul kernels, `MTLResidencySet` warm-up) and all benchmark
> numbers were developed and verified **only on macOS / Apple Silicon (the Metal
> backend)** — measured on an M2 Ultra. The runtime top-k experts parameter
> (`-experts` / `HY3_TOP_K_EXPERTS`) was also wired into the CPU (`hy3.c`)
> backend for consistency, but that path was **not compiled or run** as part
> of this work — treat CPU as untested here.
>
> **CUDA backend (`hy3_gpu.cu`) update:** a later round of work targeted a
> real **NVIDIA B300 (Blackwell Ultra, `sm_103`/compute capability 10.3)**,
> fixed the O(n²) KV-cache re-upload described below, fixed a **silent
> wrong-output bug** in the Makefile's arch flags (see below — this one
> matters if you build this yourself), and enabled TF32 tensor-core cuBLAS
> math. This was tested end to end against a real ~162GB checkpoint on
> that B300 (not just synthetic weights): `--gpu-layers` up to ~20-40
> produces correct output but is currently *not* faster than the CPU
> backend on this checkpoint/hardware, and `--gpu-layers` above that gets
> faster than CPU but **increasingly numerically wrong** (a pre-existing
> issue, not a regression from this round — reproduced on the unmodified
> original code too). See "CUDA backend (NVIDIA, Blackwell/Hopper)" further
> down for the full data and root-cause hypothesis. Bottom line: don't
> trust `--gpu-layers` above ~20-40 on this checkpoint yet, and don't add
> back "a"-suffixed nvcc arch flags without re-verifying on real hardware.

## Model facts

| | |
|---|---|
| Architecture | `HYV3ForCausalLM` (`hy_v3`) |
| Layers | 80 (layer 0 dense, layers 1-79 sparse/MoE) |
| Hidden size | 4096 |
| Attention heads | 64 (GQA, 8 KV heads, head_dim 128 — note `n_head*head_dim=8192 != hidden_size`) |
| Experts | 192 routed (top-8 activated) + 1 shared (always active) |
| Expert intermediate size | 1536 |
| Dense (layer 0) intermediate size | 13312 |
| Vocab size | 120832 (120818 real tokens + padding to a multiple of 128) |
| RoPE | theta 11158840, **`rotate_half` pairing** (dim `d` with `d+head_dim/2`) — NOT the interleaved `(2i,2i+1)` GPT-NeoX pairing. Verified against `transformers.models.hy_v3.modeling_hy_v3.apply_rotary_pos_emb`. |
| QK norm | RMSNorm applied per-head to Q and K, before RoPE |
| MoE routing | `sigmoid(router_logits)`; top-8 selected by `sigmoid + expert_bias`, but combined using the **unbiased** sigmoid weights, renormalized to sum 1, then scaled by `router_scaling_factor = 2.826` |
| MTP | `model.layers.80.*` is a multi-token-prediction layer, not used by this engine (matches upstream `transformers`, which also ignores it: `_keys_to_ignore_on_load_unexpected = [r"model\.layers\.80.*"]`) |

Reference implementation used to verify all of the above:
`transformers/models/hy_v3/modeling_hy_v3.py` (this environment has a
`transformers` build new enough to include native `hy_v3` support — no
`trust_remote_code` model file was needed).

## Project layout

```
hy3.c            Core inference engine: GGUF parsing, tokenizer, CPU forward
                 pass, sampling, generation loop.
hy3.h            Shared types, architecture constants, public API.
hy3_cli.c        CLI: `hy3` (interactive) and `hy3-cli` (prompt/batch) are the
                 exact same binary under two names (see Makefile).
hy3_convert.c    HuggingFace safetensors -> GGUF converter.
hy3_gpu.cu       CUDA backend (NVIDIA, Linux). --gpu-layers N offloads N
                 layers to one GPU's VRAM; remaining layers run on CPU.
hy3_cuda.cu/.h   Unused/dead code, not part of the build (Makefile builds
                 hy3_gpu.cu into hy3_cuda.o, not this file — kept only
                 because it predates the current build and nothing
                 references it).
hy3_metal.m      Metal backend (Apple Silicon, macOS). All 80 layers run on
                 Metal using zero-copy unified-memory buffers; no CPU split.
hy3.metal        Metal Shading Language compute kernels for hy3_metal.m.
run_metal.sh     Convenience build+run script for the Metal backend.
patch_gguf.py    Ad-hoc GGUF metadata patcher (older tool, kept for reference).
tests/
  hy3_test.c       Basic unit test.
  hy3_eval.c       6-question quality eval (GPQA Diamond / SuperGPQA / AIME
                   2025 excerpts), CPU only, greedy decode, answer-graded.
  hy3_eval_fast.c  2-question subset of the above, CPU only.
  hy3_eval_gpu.c   Same idea as hy3_eval.c, plus optional CUDA offload and a
                   configurable model path / thread count (see -h/usage in
                   the file); this is the one to adapt for Metal testing too,
                   since hy3_eval_metal/hy3_metal_init share the same API
                   shape hy3_gpu_init/hy3_eval_gpu use.
  hy3_quick_check.c  Minimal 4-prompt smoke test (arithmetic, factual
                     recall, one longer reasoning question, one arithmetic
                     word problem), meant to run in well under a minute.
```

## Building

The Makefile auto-detects the platform:

- **Linux with the CUDA toolkit installed** (`/usr/local/cuda/include/cuda_runtime.h`
  present): builds the CUDA backend (`hy3_gpu.cu`), `--gpu-layers` CLI flag
  available. Builds fat binaries for both Hopper (`sm_90a`) and Blackwell
  (`sm_100a` + `compute_100a` PTX, which the driver JIT-compiles to real
  SASS on Blackwell Ultra/B300 and other same-generation chips newer than
  this `nvcc`'s named targets — see `NVCC_ARCH_FLAGS` in the Makefile to
  override). If your `nvcc` isn't on `$PATH` as a symlink resolved from its
  real CUDA install directory (e.g. `/usr/bin/nvcc -> /usr/local/cuda/bin/nvcc`
  but invoked as `/usr/bin/nvcc`), it may fail to find `cicc`/`nvvm`
  internally (`... /usr/bin/../nvvm/bin/cicc: not found`) — pass
  `make NVCC=/usr/local/cuda/bin/nvcc` (its real path) to work around this.
- **macOS** (`Darwin`): builds the Metal backend (`hy3_metal.m` +
  `hy3.metal`), `--metal` CLI flag available. Needs Xcode command line tools
  (`xcode-select --install`). OpenMP is optional on macOS (only affects the
  speed of any CPU-side code, not Metal correctness) — install via
  `brew install libomp` if you want it; the build falls back to
  single-threaded CPU code cleanly if it's absent.
- **Anything else**: CPU-only build.

```bash
make -j$(nproc)              # auto-detect backend
make HY3_CUDA=0 -j$(nproc)   # force CPU-only on a CUDA-capable Linux box
```

Produces three binaries: `hy3` (interactive REPL), `hy3-cli` (same binary,
prompt/batch mode — see hy3_cli.c, they're built from identical objects),
and `hy3-convert` (the GGUF converter).

On macOS, prefer `./run_metal.sh` over calling `make`/`./hy3-cli` directly —
it rebuilds automatically when the Metal sources change and adds `--metal`.

## Converting a HuggingFace checkpoint to GGUF

```bash
./hy3-convert -i /path/to/Hy3 -o hy3.gguf -t q4_k
```

`-t f32|q8_0|q4_k` picks a **precision scheme**, not a uniform dtype for
every tensor. Since a change requested mid-project (see changelog below),
`q8_0` and `q4_k` are equivalent and both select a mixed-precision layout
(`select_ggml_type()` in hy3_convert.c is the single source of truth):

| Tensor | dtype | Why |
|---|---|---|
| Routed experts (`ffn_{gate,up,down}_exps`) | Q4_K | Bulk of the model (45504 of 47138 tensors); sparsely activated (8/192 per token), so lower precision here has limited per-token impact. Always Q4_K regardless of `-t` (except `-t f32`, see below). |
| `token_embd.weight` | F16 | Large (495M elements) but a straight lookup table, not matrix-multiplied; F16 is effectively lossless for embedding magnitudes and halves its size vs F32. |
| `output.weight`, attention `q/k/v/o`, shared-expert FFN, dense-layer (0) FFN | Q8_0 | These are *always active* every token (unlike routed experts), and `output.weight` directly determines the final logits — worth the extra bits. |
| Norms, router gate, expert bias, unused MTP tensors (`eh_proj`/`enorm`/`hnorm`/`final_norm`) | F32 | Tiny; precision-critical; not worth quantizing. |

`-t f32` is a debug/reference escape hatch: everything is F32 except routed
experts, which are *always* Q4_K regardless of `-t` (uncompressed experts
would make the file >1TB).

hy3's Q8_0 is **not** upstream ggml's Q8_0 — it uses an F32 block scale (36
bytes/32 elements) instead of ggml's F16 scale (34 bytes/32 elements). This
is an internal format; don't expect other GGUF-reading tools to load these
files.

## Running inference

```bash
# CPU (all layers). -t sets the OpenMP thread count for CPU matmuls — pass
# something close to nproc, the default of 4 is very conservative.
./hy3-cli -m hy3.gguf -t "$(nproc)" -p "11+22+33=?" -n 32 -temp 0

# CUDA: offload N layers to one GPU's VRAM, rest run on (now-parallel) CPU.
./hy3-cli -m hy3.gguf --gpu-layers 40 -p "11+22+33=?" -n 32 -temp 0

# Metal (macOS): all layers run on Metal via unified memory.
./run_metal.sh -m hy3.gguf -p "11+22+33=?" -n 32 -temp 0

# Interactive mode: omit -p.
./hy3 -m hy3.gguf --gpu-layers 40
```

Key flags (`./hy3-cli -h` for the full list): `-n` tokens to generate,
`-temp` sampling temperature (`-temp 0` = greedy/deterministic, useful for
reproducible comparisons), `-top_k`/`-top_p`, `-t` CPU thread count,
`-experts` MoE experts activated per token.

### Recommended: `-experts 4`

The checkpoint is natively top-8, but **we recommend running with `-experts 4`**
— it activates only the 4 highest-scoring routed experts per token instead of 8,
which roughly halves the routed Q4_K matmul work (the dominant decode cost) for a
noticeable speedup, with **no measurable quality loss on our test suites**:

| Suite (CUDA, `--gpu-layers 80`, think off) | `-experts 8` | `-experts 4` |
|--------------------------------------------|:------------:|:------------:|
| Reasoning/coding benchmark (13 tasks, `temp 1.0`) | 9/13 | 10/13 |
| Tool-calling (6 cases, `temp 0` greedy) | 5/6 | 5/6 |

The tool-calling result is identical (deterministic greedy), and the 9-vs-10
benchmark difference is within `temp=1.0` sampling noise — i.e. the two settings
are statistically indistinguishable in quality while `-experts 4` is faster
(fewer routed experts = less Q4_K matmul per token). Run the suites yourself with
`HY3_EVAL_EXPERTS=4` / `HY3_TOOL_EXPERTS=4` (see `eval/`). Use `-experts 8` if you
want the model's native routing; drop to `-experts 2`/`1` for maximum speed at
some quality cost.

### Sizing GPU/Metal offload

CUDA (discrete VRAM, must fit weights + KV cache + scratch in one GPU):
roughly ~2.4-3GB per MoE layer with the mixed-precision GGUF (Q4_K-compressed
experts + F32-dequantized dense weights), ~1GB for the dense layer 0. An
H200 (144GB) comfortably fits `--gpu-layers 40-47`; going higher risks OOM.
The remaining CPU-resident layers are now OpenMP-parallelized (see
changelog) so they're no longer the dominant bottleneck they used to be.

Metal (unified memory): no separate VRAM budget, so all 80 layers always run
on Metal — the constraint is just total system memory vs. the GGUF's size
plus a KV cache (default 8192 tokens ≈ 5.4GB, `HY3_METAL_CTX_TOKENS` env var
to change) plus small scratch buffers. On a 192GB Mac with the 173.78GB
mixed-precision GGUF this leaves headroom, but it's not huge — close other
memory-heavy applications.

## Testing

```bash
cd tests
gcc -O2 -fopenmp -I.. -o hy3_quick_check hy3_quick_check.c ../hy3.o ../hy3_cuda.o \
    -lm -lpthread -L/usr/local/cuda/lib64 -lcudart -lcublas   # Linux/CUDA build
./hy3_quick_check /path/to/hy3.gguf 90     # model path, thread count

gcc -O2 -fopenmp -I.. -o hy3_eval_gpu hy3_eval_gpu.c ../hy3.o ../hy3_cuda.o \
    -lm -lpthread -L/usr/local/cuda/lib64 -lcudart -lcublas
./hy3_eval_gpu <gpu_layers> /path/to/hy3.gguf [n_threads]
```

(Link against `../hy3.o` only, without `../hy3_cuda.o`/CUDA libs, for a
CPU-only test binary; adjust for Metal by linking `../hy3_metal.o` + the
`Metal`/`Foundation` frameworks instead.)

## Bug fix changelog

This codebase had several serious correctness bugs when this round of work
started (symptom: greedy-decoded `11+22+33=?` did not return `66`). All are
fixed.

1. **RoPE used the wrong rotation pairing.** `rope()` (hy3.c) and
   `rope_kernel` (hy3_gpu.cu) rotated adjacent dimension pairs `(d, d+1)`
   (GPT-NeoX/"interleaved" style). HYV3 actually uses HF's `rotate_half`
   convention, pairing `d` with `d + head_dim/2`. Confirmed by computing
   rotations with the real `apply_rotary_pos_emb` from `modeling_hy_v3.py`
   and diffing against the C code numerically — completely different
   vectors for any position > 0. This corrupted attention for every token
   beyond the first.

2. **Q4_K dequantization didn't match the quantizer's bit layout.**
   `dequantize_row_q4_K()` (hy3.c) used an incorrect nibble/byte indexing
   scheme that only ever read 16 of the 128 `qs` bytes per 256-value block
   (repeating them across all 8 sub-blocks), discarding ~87.5% of every
   quantized weight. Verified with a standalone round-trip test: relative
   RMSE was 1.44 (worse than noise) before the fix, ~0.06 (normal 4-bit
   quantization error) after.

3. **GPU dense-weight upload silently zeroed tensors.**
   `upload_weight_dense()` (hy3_gpu.cu) had cases for F32/F16/Q8_0 but not
   Q4_K; when `-t q4_k` conversion also made attention/shared-FFN weights
   Q4_K, they fell into the `default` branch, which `cudaMemset`'d them to
   zero. Any `--gpu-layers N` run had non-functional attention and
   dense/shared-FFN paths for the first N layers.

4. **CPU KV-cache capacity check compared mismatched units.** In the
   CPU-only `hy3_eval()`, the growth check compared `needed_slots *
   floats_per_slot` against `ctx_size` (itself already a slot count),
   causing the cache to roughly double on almost every token. Fixed to
   compare slot counts consistently.

5. **GPU KV-cache buffer was undersized by 80x.** `d_k_cache`/`d_v_cache`
   were allocated for a fixed 8192 *slots*, but the KV cache is interleaved
   by layer (`slot = token_idx * 80 + layer_id`), so 8192 slots only covers
   ~102 *tokens* of context, not 8192. Any prompt longer than that crashed
   with `CUDA error: invalid argument` from an out-of-bounds `cudaMemcpy`.
   Fixed by sizing for 8192 *tokens* (×80 layers) and adding growth-on-demand.

6. **CPU compute was single-threaded despite linking OpenMP.**
   `mul_mat_f32()`'s per-row loops and the Q4_K/Q2_K dequantizers are
   embarrassingly parallel (independent output rows / blocks) but had no
   `#pragma omp` anywhere, and `hy3_model_load()`'s `n_threads` parameter
   was stored but never passed to `omp_set_num_threads()`. Any CPU-resident
   layer (the whole model with no `--gpu-layers`, or the tail layers beyond
   it) ran on exactly one core regardless of `-t`. Fixed; measured ~50-60x
   wall-clock speedup on a 96-core box with `-t 90`.

7. **CPU Q8_0 dot product crushed activation precision.** The Q8_0 matmul
   case scaled activations by a fixed factor of 8 and truncated to `int32`
   before the dot product — effectively 3-bit fixed-point encoding of the
   activation, independent of the weight's own (correct) 8-bit precision.
   Measured ~70% relative error per row on typical post-RMSNorm magnitudes.
   This mattered once Q8_0 became the format for precision-sensitive tensors
   (see the mixed-precision conversion above). Replaced with a direct
   dequantize-and-multiply in float.

8. **KV-cache-mixing bug (fixed before this round, verified correct).** An
   earlier version of this codebase wrote every layer's K/V into the same
   flat, non-interleaved cache buffer, so attention at layer N would attend
   over a mix of layers 0..N's keys/values instead of just its own layer's
   history. The interleaved-by-layer scheme (`slot = token*80 + layer`)
   used throughout hy3.c/hy3_gpu.cu/hy3_metal.m today is the fix; verified
   by re-deriving the write/read indexing algebraically and confirming
   `attention()`'s token count computation reduces to the correct value.

None of the MoE routing math, GQA head-grouping, RMSNorm formula, or overall
transformer block structure needed fixing — those were cross-checked against
`modeling_hy_v3.py` and matched from the start.

### Mixed-precision GGUF conversion

Originally, `-t q4_k` conversion quantized only routed experts to Q4_K and
left every other tensor (attention, output, embeddings, shared/dense FFN,
norms) as full F32 — despite the `hy3_q4k.gguf` filename suggesting
otherwise. Added a proper per-tensor-category precision scheme (table
above): moved attention/output/shared-FFN/dense-FFN to Q8_0 and the
embedding to F16, while leaving norms/router/bias at F32. Net effect on the
existing checkpoint: **198.58GB -> 173.78GB**, i.e. smaller *and* higher
average precision on the always-active tensors than before, because the
previous "everything but experts is F32" baseline was more wasteful than it
looked from the filename.

### Metal backend (macOS / Apple Silicon)

The Metal backend (`hy3_metal.m` + `hy3.metal`) targets Apple Silicon.
Unlike the CUDA backend, which must dequantize weights on the CPU and
`cudaMemcpy` a copy into a discrete GPU's limited VRAM, Apple Silicon's
unified memory lets the same mmap'd GGUF pages be wrapped as zero-copy
`MTLBuffer`s (`newBufferWithBytesNoCopy`), so all 80 layers can be
Metal-resident with no CPU/GPU layer split and no memory duplication (this
model would not fit in RAM twice on a 192GB machine). Quantized formats
dequantize inline inside the matmul kernels rather than being pre-expanded.
All kernel math is a direct port of the validated CPU/CUDA formulas (RoPE
convention, KV-cache interleaving, Q4_K bit layout, MoE routing).

The backend has been **built and tested on Apple Silicon (macOS, clang +
Metal toolchain)**:

- Fixed a compile-blocking bug in `hy3.c`: `hy3_model_free()` called
  `hy3_metal_free()` before its forward declaration (the prototype sat ~400
  lines later, at the old `hy3_eval()` site). Under macOS clang, implicit
  function declarations are a hard error (ISO C99+), so the `HY3_METAL`
  build failed outright. The `hy3_gpu_free`/`hy3_metal_free` prototypes are
  now declared above `hy3_model_free()`.
- `hy3_metal.m` (`-fobjc-arc`) and `hy3.metal` (`xcrun metal`) both compile
  with zero warnings/errors, and the `hy3` / `hy3-cli` Metal binaries build
  and link cleanly as native arm64 executables.
- Every kernel in `hy3.metal` / `hy3_metal.m` was cross-checked against the
  validated CPU path in `hy3.c` (RoPE `rotate_half` pairing, Q4_K 144-byte
  block layout + nibble packing, hy3's 36-byte F32-scale Q8_0, layer-
  interleaved KV cache, sigmoid/top-8/×2.826 MoE routing) and matches.

An earlier KV-cache-write bug (a CPU-side `memcpy` that raced ahead of an
unexecuted GPU kernel) was caught via review and fixed by making the cache
write a GPU kernel; see `run_metal.sh` for the build/run steps.

### CUDA backend (NVIDIA, Blackwell/Hopper)

A later round of work targeted **NVIDIA Blackwell** (B200/B300) on Linux,
building and running `hy3_gpu.cu` on a real **B300 SXM6** (`nvidia-smi`
reports compute capability 10.3, i.e. "Blackwell Ultra"; CUDA 12.8's `nvcc`
has no explicit `sm_103` target for it yet):

- **Build:** `NVCC_ARCH_FLAGS` in the Makefile now generates real SASS for
  both `sm_90a` (Hopper H100/H200, matching the existing README VRAM sizing
  guidance) and `sm_100a` (Blackwell B200), plus embeds `compute_100a` PTX
  so the driver JIT-compiles forward-compatible SASS for B300 and any other
  same-generation chip newer than the toolkit's named targets. Confirmed:
  `make` builds cleanly and the resulting binary launches CUDA kernels
  successfully on the B300 in this environment.
- **Fixed the O(n²) KV-cache-reupload perf bug** called out in "Known
  limitations" below: attention used to `cudaMemcpy` the *entire* K/V
  history from the CPU cache to the GPU before every attention call, every
  layer, every token. The GPU-resident cache is now the source of truth
  and is updated with a single O(1) device-side write of just the new
  token's K/V per layer (cache growth-on-demand was also fixed to carry
  old slots forward, since it no longer gets "refilled" by the next full
  reupload). Also removed a stray `cudaDeviceSynchronize()` inside the
  per-expert MoE loop that serialized every expert's GPU work behind an
  unnecessary host round trip, and stopped uploading `expert_bias` to the
  GPU only to immediately copy it straight back to the host every MoE
  layer of every token (routing happens on the CPU; the bias already lives
  in host memory in `m->w`). Enabled cuBLAS's TF32 tensor-core math mode
  (`cublasSetMathMode(..., CUBLAS_TF32_TENSOR_OP_MATH)`), used by every
  dense `gpu_mul_mat()` call (attention QKVO, dense/shared FFN, output
  projection).
- **KV-cache fix, initial verification (synthetic model):** before a real
  checkpoint was available, the fix above was validated with a synthetic
  model at reduced dimensions (8 layers, 1024 hidden size, 32 experts —
  same struct layout and kernels as production, just smaller constants).
  It ran `hy3_gpu_init()` + thousands of `hy3_eval_gpu()` calls with no
  CUDA errors and no NaNs, both before and after the fix, and measured
  wall-clock decode latency over a 3000-token synthetic generation:

  | | token 10 | token 500 | token 1500 | token 2999 | total (3000 tok) |
  |---|---|---|---|---|---|
  | Before (full reupload) | 1.03ms | 2.43ms | 5.67ms | 11.68ms | 18.12s |
  | After (O(1) update) | 0.72ms | 0.73ms | 0.73ms | 0.74ms | 2.36s |

  i.e. flat per-token latency instead of growing with context, ~7.7x less
  total wall-clock time for that synthetic run. (Harness not part of the
  repo; superseded by the real-checkpoint testing below.)

- **Real-checkpoint verification, and an important correctness caveat
  found by it:** a real ~162GB `hy3_q4k_mixed.gguf` (the `-t q4_k`
  mixed-precision conversion described above) was later tested end to end
  on the same B300, greedy-decoding `"11+22+33=?"` and checking the answer
  reaches `66`, exactly as the CPU-backend bug-fix changelog above does.
  Findings:

  1. **The build-flag fix above (dropping "a"-suffixed archs) was not
     optional — it was hiding a silent correctness bug.** The original
     `-arch=sm_100a`-style ("family-specific") targets built and ran with
     no CUDA error on this real B300, but produced all-zero logits from
     the very first token, every time, with any GPU layers offloaded —
     confirmed both for `compute_100a`/`sm_100a` (Blackwell) and, more
     surprisingly, for `compute_90a`/`sm_90a` (Hopper) on this same
     hardware/driver/toolkit combination (driver 580.126.09, CUDA 12.8).
     The plain, non-"a" `sm_90`/`sm_100`/`compute_100` targets (what the
     Makefile now uses) produce correct output, matching the CPU
     backend's `66` answer token-for-token in the low-`--gpu-layers`
     regime tested. **Do not add "a"-suffixed archs back without
     re-verifying end to end on real hardware with a real checkpoint** —
     this fails silently, with no error message anywhere.
  2. **`--gpu-layers` correctness degrades as the layer count increases,
     and this predates all of this round's changes** (reproduced
     identically on the unmodified pre-existing code from before this
     round, with the same real checkpoint, same hardware). Measured
     (greedy, `-temp 0`, prompt `"11+22+33=?"`):

     | `--gpu-layers` | Output | Speed (gen) |
     |---|---|---|
     | 0 (CPU only) | Correct: reaches `66`, stops cleanly at EOS | 3.17 tok/s |
     | 5, 20 | Correct: reaches `66` (20: `\boxed{66}`) | 1.9–2.3 tok/s |
     | 40, 60 | Starts drifting: correct early arithmetic (`11+22=33`) but loops/repeats instead of concluding | 2.7–3.5 tok/s |
     | 79, 80 (all layers) | Diverges into repetition and eventually gibberish within ~20-80 tokens | 4.6–4.9 tok/s |

     The divergence gets monotonically worse with more GPU-resident MoE
     layers, and is present in the unmodified original code too — it is
     **not** a regression introduced by this round's fixes. Ruled out
     TF32 (disabling `CUBLAS_TF32_TENSOR_OP_MATH` at `--gpu-layers 80`
     produced equally-diverged, not-meaningfully-better output) and ruled
     out CPU floating-point non-associativity as the general explanation
     (CPU `-t 1` vs `-t 30` produce byte-identical correct output at 80
     generated tokens). The leading hypothesis, not yet confirmed by
     tracing individual expert-routing decisions: the GPU MoE kernels
     (`moe_gate_up_q4K_qwarp32_kernel`/`moe_down_q4K_qwarp32_kernel`)
     quantize the *activation* side to Q8_K (int8) before every routed-
     expert dot product, whereas the CPU path dequantizes Q4_K weights to
     F32 and does a direct F32×F32 dot product with the un-quantized
     activation — the GPU path's extra activation-quantization noise is a
     plausible source of the small per-layer hidden-state drift that
     could flip a top-8 MoE routing decision (a hard, discontinuous
     choice) in a way that compounds across many sequential MoE layers.
     Not fixed in this round; needs further investigation (e.g. comparing
     per-layer GPU vs CPU hidden states/router logits directly) before
     `--gpu-layers` above roughly 20-40 should be trusted for anything
     but a speed smoke-test.
  3. **Net practical implication:** in the layer-count range that's
     actually *correct* (`--gpu-layers` up to ~20-40 on this checkpoint),
     the current GPU backend is not faster than the 30-core CPU backend
     for single-token decode (2.2-3.5 vs 3.17 tok/s) — the extra
     CPU↔GPU round trips per layer (KV cache append, MoE routing) aren't
     amortized by enough GPU-resident work yet. Raw throughput only
     exceeds CPU once most/all layers are GPU-resident (4.6-4.9 tok/s at
     `--gpu-layers 79-80`), which is exactly the regime that's not
     numerically trustworthy right now. This round's fixes (KV-cache
     O(n²)→O(n), removed redundant sync/copies, TF32) are real and
     measured improvements to the GPU backend's *existing* per-token
     work, but they don't yet add up to a `--gpu-layers` setting that is
     simultaneously correct and faster than CPU-only on this checkpoint —
     that requires fixing point 2 above first.

### CUDA backend, round 2: Metal-inspired GPU-resident MoE + a real determinism bug

A further round of work ported several of the Metal backend's key ideas
(`hy3_metal.m`'s "fast path": `router_topk` + `matmul_q4_k_id` +
`moe_swiglu_id` + `moe_combine_id`, and `rms_norm_heads`) to `hy3_gpu.cu`,
targeting the single biggest remaining bottleneck for batch-1 decode: sheer
kernel-launch count and host↔device round trips (a single MoE layer was
issuing ~57 kernel launches plus a synchronous CPU round-trip for routing;
80 layers/token meant thousands of tiny launches, each paying host-side
dispatch latency that a B300 can't hide when the GPU work per launch is
this small). Changes, all in `hy3_gpu.cu`:

- **`router_topk_kernel`**: MoE top-k expert selection (sigmoid + bias,
  select top-8, renormalize the *unbiased* sigmoid weights, scale by
  2.826) now runs entirely on the GPU, writing directly to device buffers
  the next kernels read — no more per-MoE-layer `cudaMemcpy` of the
  router logits back to the CPU for a host-side sort.
- **`moe_matmul_q4k_id_kernel`**: one kernel launch computes *all* K
  selected experts for a given projection (gate, up, or down), indexed by
  a small device-side pointer table per expert (`d_gate_ptrs`/
  `d_up_ptrs`/`d_down_ptrs`, built once at init) instead of one launch per
  expert. It also dequantizes Q4_K weights and dots them directly against
  the plain F32 activation (`dev_dot_q4_K_f32_block`) instead of
  quantizing the activation to Q8_K (int8) first like the old per-expert
  kernels did — matching the CPU path's precision and hy3.metal's
  `matmul_q4_k_id` exactly.
- **`rms_norm_heads_kernel`**: fused per-head QK-norm into one launch per
  tensor (grid = n_heads) instead of a host-side loop issuing up to
  64+8=72 separate tiny launches per attention layer.
- Net effect: MoE went from ~57 launches + 1 blocking host round-trip per
  layer to ~13 launches + 0 host round-trips; QK-norm went from up to 72
  launches to 2.

**Measured on the real B300 + real 162GB checkpoint** (greedy, prompt
`"11+22+33=?"`), before vs. after this round, at several `--gpu-layers`:

| `--gpu-layers` | Before (round 1) | After (round 2) |
|---|---|---|
| 5 | ~2.0 tok/s | ~2.0-2.4 tok/s |
| 40 | ~2.7-3.5 tok/s | ~3.1 tok/s (≈ CPU's 3.17) |
| 50 | (not separately measured) | **~3.7-3.9 tok/s, correct — beats CPU** |
| 60 | ~3.5 tok/s | ~4.5 tok/s |
| 80 | ~4.6-4.9 tok/s | **~9.0-9.6 tok/s** (~2x) |

Roughly a 2x speedup at every layer count, and `--gpu-layers 50` is now a
genuinely useful operating point: correct output *and* faster than the
30-core CPU backend (previously, no setting was both). Switching the MoE
math to direct F32 dequant-dot (matching CPU/Metal) did **not**, on its
own, fix the "more GPU layers = more likely to diverge from the CPU's
answer" pattern from round 1 — it's still present at `--gpu-layers` 60+
after this round, ruling out Q8_K activation-quantization noise as the
(sole) explanation. TF32 was re-tested and ruled out again too (disabling
it at `--gpu-layers 80` with the new kernels produced equally-diverged
output). The remaining leading hypothesis is genuine GPU-vs-CPU
floating-point non-associativity in cuBLAS's GEMM reduction (vs. the
CPU's own reduction order) occasionally flipping a top-8 MoE routing
decision — a hard, discontinuous choice — which then compounds across
however many further GPU-resident MoE layers follow. Still not fixed;
same practical guidance as round 1 (trust `--gpu-layers` up to ~40-50 on
this checkpoint), just faster at every point on that curve, and now with
one more setting (50) firmly in "correct and faster than CPU" territory.

**A real, separate bug found and fixed in this round: `hy3_eval_gpu` was
non-deterministic run-to-run**, even at `-temp 0` (greedy) with identical
weights/prompt/settings. Verified via repeated runs at a fixed
`--gpu-layers` (5, well below the layer-count-correlated divergence
above): 2 of 3 runs matched, 1 diverged onto a different (but still
locally-sensible) continuation after the correct `11+22=33` step — and
this was reproducible down to *only* the KV-cache O(1) round-1 fix being
applied (no TF32, no fast MoE path), isolated by rebuilding that fix alone
on top of the pristine pre-round-1 code and re-running it 3x. The
pre-round-1 code (unmodified) was verified fully deterministic (3/3
identical runs) under the same test. Root cause was not fully isolated —
`compute-sanitizer --tool racecheck` doesn't support this GPU yet
("Device not supported"), and `--tool initcheck` flagged the same
"uninitialized access on cudaMemcpy source" warning symmetrically in both
the deterministic pre-round-1 code and the non-deterministic version, so
it isn't the differentiator either (most likely a sanitizer limitation
around memory cuBLAS writes, not a real bug: same warning, same count
order of magnitude, in code that's provably deterministic). Empirically,
adding an explicit `cudaDeviceSynchronize()` immediately after the KV-
cache's device-to-device append (see the comment at that call site)
reliably restored deterministic output (verified >=4 identical repeated
runs after the fix, vs. divergence within 1-3 runs before it) and was kept
in the shipped code on that basis, even though plain synchronous
`cudaMemcpy` should already stream-order correctly against the following
kernels without it per normal CUDA semantics — something about this real
B300/driver/CUDA-12.8 combination needs the extra, stronger sync. This is
worth revisiting with proper profiling tools (e.g. Nsight Systems) rather
than the trial-and-error used here, since "add a sync and the symptom
went away" is a fix, not necessarily a full explanation.

### CUDA backend, round 3: single-stream + CUDA graph + QKV fusion, and the batch-1 latency wall

Round 3 targeted raw decode throughput (goal: 40 tok/s at full 80-layer
offload on the B300). What was done, all in `hy3_gpu.cu`:

- **Single stream for everything.** All kernels and every cuBLAS call now
  run on one `ctx->stream` (via `cublasSetStream`), so stream-ordering
  alone gives correctness with no device-wide sync mid-token (the per-layer
  `cudaDeviceSynchronize` from round 2 is gone), and the whole per-token
  forward is graph-capturable.
- **CUDA graph decode.** The full 80-layer forward is captured once into a
  `cudaGraphExec_t` and replayed per token with just a device-side position
  update (`d_pos`) — the token id and position are the only per-token
  variables, so position-dependent kernels (`rope`, KV-append, attention)
  were given device-position variants (`*_dyn`) that read `*d_pos`. A
  one-token eager "warmup" runs first so cuBLAS allocates its workspace
  before capture (no `cudaMalloc` is allowed inside a capture region). The
  graph is re-captured only if the KV cache is reallocated.
- **Fused QKV projection** (3 GEMVs → 1; weights concatenated at load).
- Fused per-head Q/K norm (round 2) and batched multi-expert MoE (round 2)
  carry over.

**Measured on the real B300 + 162GB checkpoint, full 80-layer offload:**
decode went from ~4.6 tok/s (before this whole line of work) to ~9.6 tok/s
now — a ~2x gain, but **well short of the 40 tok/s goal, and I want to be
precise about why**, because it's a fundamental property of the workload,
not a missing tweak:

- The CUDA graph works (captured exactly once, replayed) and removed host
  launch overhead, but only moved decode from ~9.2 to ~9.6 tok/s. QKV
  fusion added ~0. Dropping from 8 experts to 1 (`-experts 1`) also changed
  throughput by ~0 (11.08 vs 10.92 tok/s).
- During generation, `nvidia-smi dmon` shows **sm ≈ 25%, mem ≈ 0%** — the
  GPU is neither compute- nor bandwidth-bound. It is **latency-bound**: a
  single decoded token is ~1760 *sequential, dependency-chained* kernels
  (~22/layer × 80), each of which is a batch-1 GEMV/GEMM that only fills a
  fraction of the B300's 148 SMs and finishes before compute or memory
  saturates. They run one-after-another on one stream, so the SMs sit idle
  ~75% of the time waiting on the chain. Bandwidth math confirms it: the
  per-token weight traffic (~36 GB) would take only ~4.6 ms at ~8 TB/s
  (≈200 tok/s) if bandwidth-bound; we're 20x off that, i.e. not moving data,
  just waiting.
- **This is why converting the GGUF to a different format does not help the
  speed here.** Format/precision changes attack bandwidth and memory
  footprint; the wall is per-kernel launch/scheduling *latency* on a serial
  chain (proven by `-experts 1` and `mem%≈0` above). Lower-precision dense
  weights (e.g. F16) would halve their footprint but not shorten the
  latency chain. Higher-precision experts (F16/Q8_0, which might help the
  separate high-layer-count *correctness* drift) don't fit: F16 experts
  alone would be ~320 GB > the B300's 275 GB, and Q8_0 ~300 GB — the
  routed experts have to stay Q4_K to fit at all.
- **What would actually reach ~40 tok/s** is the technique the Metal
  backend already uses to hit 15 tok/s on an M2 Ultra: run the *independent*
  kernels within a layer concurrently (Metal's `concurrent` encoder with
  `memoryBarrier`), i.e. on CUDA, capture multiple streams into the graph
  with event fork/join so shared-expert vs routed-expert branches, Q/K
  norms, etc. overlap and fill the idle SMs — plus deeper fusion (ideally a
  fused/persistent per-layer megakernel) to cut the chain length. That is a
  large, high-risk rewrite (multi-stream cuBLAS under graph capture, careful
  event dependencies) with a payoff bounded by available intra-layer
  parallelism (~2-3x), and it was **not** attempted here because (a) the
  80-layer path is still numerically wrong (see round 1/2: `--gpu-layers`
  above ~50 diverges), so faster-but-wrong isn't useful yet, and (b) it
  needs proper profiling (Nsight) to guide it rather than the trial-and-error
  this environment allowed. Recommended order for a future round: fix the
  high-layer-count divergence first, then add multi-stream concurrency.

Net: this round is real infrastructure (single-stream, a working decode
CUDA graph, fused QKV) and a ~2x decode speedup, with an honest, measured
account of why 40 tok/s needs the concurrency rewrite above rather than a
format change. `HY3_TIMING=1` enables the per-token eval/sample breakdown
and a one-time "captured decode graph" log line used for this analysis.

## Known limitations

- **`hy3_gpu.cu`'s `--gpu-layers` is only numerically trustworthy up to
  roughly 20-40 layers on the real checkpoint tested (out of 80); higher
  values run faster but produce increasingly wrong output (repetition,
  eventually gibberish), and this predates all work in this round — see
  "CUDA backend (NVIDIA, Blackwell/Hopper)" below for the measured
  per-layer-count data and a root-cause hypothesis (likely the GPU MoE
  kernels' Q8_K activation quantization perturbing hard top-8 routing
  decisions enough to compound over many layers). Also in that range,
  the GPU backend is currently *slower* than CPU-only on this
  hardware/checkpoint — the layer-count range that's fast isn't the one
  that's correct yet. Not fixed in this round.
- The CUDA and Metal backends both re-derive the current token's absolute
  position from `cache_len / HY3_N_LAYER` and recompute per-layer KV slot
  indices on every call rather than tracking them incrementally; fine for
  correctness, a little wasteful, not worth the complexity of changing
  without a measured need.
- ~~`hy3_gpu.cu`'s per-layer attention re-uploads the *entire* KV cache
  history from CPU to GPU before every attention call (O(n²) total data
  movement over a sequence) rather than only the new token~~ — **fixed**
  (see "CUDA backend (NVIDIA, Blackwell/Hopper)" below): the GPU-resident
  KV cache is now updated with a single O(1) per-token device write, and
  growth-on-demand carries old slots forward instead of relying on a full
  reupload. Measured on a real B300 with a reduced-scale synthetic model
  (see that section for why): flat ~0.74ms/token out to 3000 tokens vs.
  the old code's 1.03ms → 11.68ms/token (linearly growing) over the same
  range — 7.7x less total wall-clock time end to end.
- `hy3_gpu.cu` still re-derives the KV slot for cache growth headroom as a
  fixed "+8192 more tokens" rather than e.g. doubling; fine in practice
  (growth is already an O(cache-size) amortized-rare event, not a per-token
  cost), just not maximally tuned.
- `hy3_metal.m` commits and waits on a command buffer once or twice per
  layer (needed for the CPU-side MoE top-k routing step) rather than
  batching many layers into fewer command buffers. Correct, not maximally
  fast; a reasonable place to optimize once correctness is confirmed on
  real hardware.
- The tokenizer (`hy3_tokenize` in hy3.c) is a greedy longest-prefix-match
  over the whole vocabulary, not a proper BPE merge algorithm. This happens
  to agree with the real tokenizer for plain ASCII prompts (verified for
  `"11+22+33=?"`), but is not guaranteed to match in general, particularly
  for text requiring the GPT-2-style byte-level unicode escaping.
- `attention()`/`attention_kernel`/`attention` (Metal) cap attended history
  at 8192 tokens *per layer* by truncating to the first 8192 rather than a
  sliding window; sequences longer than that will attend incorrectly (not
  to "recent" context) rather than erroring out. Matches the model's
  `max_position_embeddings` only up to that length; long-context use beyond
  8192 tokens needs real work, not just raising the constant.
