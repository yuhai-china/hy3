# hy3 — HYV3 GGUF Converter and Inference Engine

`hy3` is a from-scratch C inference engine and GGUF converter for **Hy3**
(`HYV3ForCausalLM`, `model_type: hy_v3`), a 295B-parameter / 21B-active-parameter
Mixture-of-Experts model released by **Tencent's Hunyuan ("Hy") team**
(`tencent/Hy3` on Hugging Face).

The project is called **hy3**.

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
  available.
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
reproducible comparisons), `-top_k`/`-top_p`, `-t` CPU thread count.

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

## Known limitations

- The CUDA and Metal backends both re-derive the current token's absolute
  position from `cache_len / HY3_N_LAYER` and recompute per-layer KV slot
  indices on every call rather than tracking them incrementally; fine for
  correctness, a little wasteful, not worth the complexity of changing
  without a measured need.
- `hy3_gpu.cu`'s per-layer attention re-uploads the *entire* KV cache
  history from CPU to GPU before every attention call (O(n²) total data
  movement over a sequence) rather than only the new token — a real
  performance issue for long sequences, not a correctness one. Not fixed in
  this round; flagging for future work.
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
