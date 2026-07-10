# Optimizing GGUF Inference on CUDA — Step by Step

This document records, in order, how we took the CUDA decode path of the Hy3
GGUF runtime (`HYV3ForCausalLM`, a 295B-parameter / 21B-active MoE model in
Q4_K_M) from **4.6 tok/s to 44 tok/s** at full 80-layer GPU offload on a single
NVIDIA **B300** (Blackwell Ultra, compute capability 10.3), CUDA 12.8.

The intent is that a future engineer can reproduce the reasoning, understand
*why* each step helped, and avoid the traps we hit.

---

## 0. Environment & ground rules

| Item | Value |
|------|-------|
| GPU | NVIDIA B300 SXM6, compute capability 10.3 |
| Toolkit | CUDA 12.8 (`/usr/local/cuda/bin/nvcc`), driver 13.0 |
| Model | `hy3_q4k_mixed.gguf`, 162 GB, 80 layers, 4096 embd, 64 heads / 8 KV heads, 8 active experts of 256 |
| Build | `make NVCC=/usr/local/cuda/bin/nvcc -j4` |
| Backend source | `hy3_gpu.cu`, `hy3_gpu.h`; driver loop in `hy3.c` |

**Ground rules learned the hard way (read before touching anything):**

1. **Invoke nvcc by absolute path** (`/usr/local/cuda/bin/nvcc`). A bare `nvcc`
   on this box fails to locate `cicc`.
2. **Use plain arch flags only** — `sm_90`, `sm_100`, `compute_100`. The
   family-specific "a"-suffixed targets (`sm_90a`, `sm_100a`, `compute_100a`)
   compile and launch with **no CUDA error but produce all-zero logits** on this
   B300 / driver 13.0 / CUDA 12.8 combination. We embed `sm_100` cubin plus
   `compute_100` PTX so the driver JITs real B300 SASS at load. See the long
   comment block in the `Makefile`.
3. **Commit every working state.** We once destroyed hours of work with a stray
   `git checkout` of uncommitted changes. Commit before each experiment.
4. **Verify correctness after every change**, not just speed (see §11).

---

## 1. Baseline and how we measured

Baseline full-offload decode was **~4.6 tok/s**. Before optimizing anything we
added instrumentation, because you cannot optimize what you cannot see:

- `HY3_TIMING=1` — per-token wall-clock breakdown (eval vs. sample) printed from
  `hy3.c`.
- `HY3_GRAPH_BENCH=1` — times 50 back-to-back CUDA-graph replays and reports
  pure-GPU ms/token, isolating device work from host/sampling overhead.
- `HY3_SKIP_ATTN` / `HY3_SKIP_FFN` / `HY3_SKIP_EXP` — skip a phase of the layer
  so we can attribute time differentially (run with/without and subtract).

**Key early finding** from differential profiling: decode was *latency-bound*,
not compute-bound (SM utilization ~25%, memory controller ~0%). Skipping the
routed-expert matmuls (`HY3_SKIP_EXP=1`) collapsed a ~70 ms token to ~10 ms —
i.e. **the Q4_K routed-expert matmuls were ~85% of decode time.** Every step
below is aimed, directly or indirectly, at that fact.

---

## 2. Step 1 — Blackwell build + O(n²)→O(n) KV cache

Commit `bdbb2a8` — *Blackwell/B300 support + 2.9x faster CUDA decode.*

Two independent wins bundled together:

1. **Correct build for B300** (arch flags above). This alone made the GPU path
   produce correct logits instead of zeros.
2. **KV cache made O(n) per token.** The original path recomputed attention over
   a growing buffer with O(n²) work and re-read all of K/V every token. We
   switched to a persistent device K/V cache indexed by position, so each decode
   step appends one K/V and reads the cache once.

Result: **4.6 → ~13.2 tok/s.**

---

## 3. Step 2 — FP16 KV cache

Commit `757f261` — *fp16 KV cache.*

Store K/V as `half` instead of `float`. This **halves KV memory** and, more
importantly for a latency/bandwidth-bound decode, **halves the attention
read bandwidth**. The online-softmax attention kernel
(`attention_kernel_online`, `hy3_gpu.cu:173`) reads the half cache directly and
accumulates in FP32, so no accuracy is lost in practice.

---

## 4. Step 3 — Fused attention & routing kernels

Bundled across the Blackwell rewrite. The per-layer attention path was collapsed
into a few fused kernels to cut launch count and intermediate global-memory
round trips:

- `qk_norm_rope_fused_kernel` (`hy3_gpu.cu:144`) — RMS-norm of Q and K **plus**
  RoPE in one kernel, one warp/block per head.
- `attention_kernel_online` (`hy3_gpu.cu:173`) — online-softmax (flash-style)
  attention, float4-tiled, one warp per head, reads the FP16 KV cache, no
  materialized score matrix.
- `router_topk_kernel` (`hy3_gpu.cu:81`) — MoE top-k routing done **on device**
  (previously a CPU round-trip per layer). Includes a `best=0` NaN guard so a
  degenerate router score can never select expert 0 spuriously.
- `add_rmsnorm_kernel` (`hy3_gpu.cu:218`) — fused residual-add + RMS-norm.

Differential profiling here confirmed attention was now only ~5 ms of the token;
FFN/experts were ~65 ms. Attention was no longer worth chasing.

---

## 5. Step 4 — FP16 dense weights via `cublasGemmEx`

Commit `e89e36e` — *FP16 dense weights via cublasGemmEx.*

The dense (non-expert) projections — QKV, shared gate/up, down, output — were
converted to FP16 and run through `cublasGemmEx` as GEMV
(`hy3_gpu.cu:380`). This uses the tensor cores' FP16 path and **halves the
weight-read bandwidth** for the dense matmuls. GEMVs are memory-bound, so this
is a direct bandwidth win with negligible accuracy impact (FP32 accumulation).

---

## 6. Step 5 — CUDA graph decode

Commit `b12941a` — *CUDA graph decode for full-GPU path.*

At ~13 tok/s the decode was launching dozens of tiny kernels per layer × 80
layers per token. Per-launch overhead dominated the latency-bound token. We
captured the entire 80-layer decode into a **CUDA graph** and replay it each
step:

- All position-dependent kernels read the token position from a device pointer
  `d_pos` (updated between replays with a tiny copy), so the *graph topology is
  fixed* and can be captured once.
- Flow: **warmup** (one eager pass to populate lazily-allocated state) →
  **capture** → **replay** every token (`gpu_decode_graph`, `hy3_gpu.cu:446`).

This removed almost all host-side launch latency. Pure graph replay at this
point was ~42 ms/token. Speed: **~13.2 → 13.84 tok/s** end-to-end (the graph
mostly removes host overhead that overlapped anyway, but it is the enabler for
measuring and attacking pure device time cleanly).

---

## 7. Step 6 — Warp-per-row Q4_K expert matmul

Commit `6a6ef1e` — *warp-per-row Q4_K expert matmul → 13.8→20.4 tok/s.*

Now the profiler said experts were 85% of the token, and the expert matmul was
massively under-occupying the GPU. The original expert kernel used roughly one
*thread* per output row with only a couple dozen blocks — the B300's ~148 SMs
were nearly idle.

Redesign: **one warp (32 lanes) computes one output row** via a full-warp
shuffle reduction. Grid is `grid.x = ceil(M / MOE_WPB)`, `grid.y = slot`
(the routed expert instance), with `MOE_WPB = 8` warps per block. For the gate
projection this yields ~1536 blocks instead of ~24 — full SM occupancy, which
hides the Q4_K dequant + weight-read latency.

Result: **13.84 → 20.4 tok/s.**

---

## 8. Step 7 — Q4_K sub-block parallelism

Commit `16211db` — *Q4_K sub-block parallelism → 22.7 tok/s.*

A warp-per-row still left lanes idle when the row's block count `nb` was small:
work was distributed one 256-element Q4_K super-block per stride. We introduced
`dev_dot_q4_K_sub`, splitting each row's work across its **`nb*8` sub-blocks**
(32-element Q4_K sub-blocks) so all 32 lanes stay busy regardless of `nb`.

Result: **20.4 → 22.68 tok/s.** Pure graph replay ~42.3 ms.

At this point differential profiling was blunt: `HY3_SKIP_EXP=1` gave 9.98 ms
graph replay (~100 tok/s) vs. 42 ms with experts — experts were still ~32 ms
(76%) of the token. The experts read ~5.9 GB of Q4_K weights per token; at 42 ms
that is only **~140 GB/s effective, roughly 56× below** the B300's ~8 TB/s
floor. The reads were **uncoalesced**: the warp's 32 lanes were fetching
scattered sub-blocks rather than one contiguous transaction.

---

## 9. Step 8 — Coalesced warp-cooperative Q4_K row dot (the big one)

Commit `bb2b33b` — *Coalesced warp-cooperative Q4_K expert matmul.*

This is the decisive memory-coalescing fix. New device function
`warp_row_dot_q4k` (`hy3_gpu.cu:51`), used by both
`moe_matmul_q4k_id_kernel` (down projection) and
`moe_matmul_q4k_id_gate_up_kernel` (gate+up).

### The Q4_K block layout (why the mapping is fiddly)

A Q4_K super-block holds 256 quantized weights in a 144-byte struct: an FP16
`d`, FP16 `dmin`, 12 packed 6-bit scale/min bytes, and a **128-byte `qs`** of
4-bit nibbles. The 256 weights are 8 sub-blocks of 32. Sub-block pair
`p` (sub-blocks `2p`, `2p+1`) occupies `qs` bytes `[p*32 .. p*32+31]`, with the
**low** nibble belonging to sub-block `2p` and the **high** nibble to `2p+1`.

### The coalesced access pattern

Have the 32 lanes read each block's 128-byte `qs` as **32 contiguous `uint32`s
in one coalesced transaction**: `((const uint32_t*)blk->qs)[lane]`. Then:

- Lane `L` owns byte range `[L*4, L*4+3]`, which lands exactly in pair
  `p = L/8`, element offset `e0 = (L%8)*4` (proof: `p*32 + e0 == 4*L`).
- Each lane's 4 bytes yield 4 low nibbles (→ sub-block `2p`, elements
  `e0..e0+3`) and 4 high nibbles (→ sub-block `2p+1`).
- Because a scale/min is constant across a sub-block and the same for all 8
  lanes of pair `p`, each lane can pre-multiply its partial dot by its own
  sub-block scale/min; the final warp shuffle-reduce sums everything correctly:
  `sc·d·Σ(q·x) − m·dmin·Σx`, distributed across lanes.

Alignment holds: `cuda_block_q4_K` is 144 = 9×16 bytes, `qs` sits at offset 16,
`cudaMalloc` is 256-aligned, so the `uint32` (and 16-byte) accesses are aligned.

For the gate+up kernel the same activation `x` is reused for both weight
matrices, so `warp_row_dot_q4k` is called twice per row without re-reading `x`
from a second pass of logic.

### Result

- Pure graph replay: **42.3 → 22.3 ms/token.**
- End-to-end at 80 layers: **22.68 → 44 tok/s** (≈2×).

This single coalescing change is the largest late-stage win, because it attacked
the exact bottleneck the profiler identified: uncoalesced Q4_K weight reads.

---

## 9b. Step 9 — Overlap shared-expert GEMV with routed-expert matmuls (2nd stream)

Commit `a696f8f`.

Each MoE layer runs two independent bodies of work: the **routed** experts (8
Q4_K matmuls — the dominant cost) and the small **shared** expert (FP16
cublas GEMVs). They were serialized on one stream. We now fork the routed
matmuls onto a second CUDA stream (`stream2`) while the shared-expert GEMVs run
on the main stream, then join before the final combine:

- Fork after routing: `cudaEventRecord(ev_fork, main)` →
  `cudaStreamWaitEvent(stream2, ev_fork)`.
- Routed gate_up / silu / down on `stream2`; shared gate_up / silu / down on
  main. Both read the layer input `s` **read-only**; the shared-down output is
  retargeted from `s` to the free `ao` scratch to remove a write-after-read
  hazard on `s`.
- Join before combine: `cudaEventRecord(ev_join, stream2)` →
  `cudaStreamWaitEvent(main, ev_join)`.

This multi-stream fork/join is captured *into the CUDA graph* via events (the
standard multi-stream capture pattern), so decode still replays as one graph.

**Result: graph replay 22.3 → 20.8 ms/token (45 → 48 tok/s pure GPU).** The gain
is modest (~7%) and bounded: the routed matmuls launch ~1500–4000 blocks and
already saturate the ~148 SMs, so the shared GEMV has little spare occupancy to
overlap into. Still a free, correctness-preserving win.

## 9c. Step 10 — FP32 router GEMV for routing stability across depth

Commit `5285365`.

**Investigation.** The earlier "diverges above ~50 layers" caveat was
re-examined. With greedy (temp 0) decoding, outputs at 40/50/60/70/80 layers are
*identical* for short prompts; the divergence only appears in long generations
and is **benign** (e.g. 20-layer chooses `\( … \)` inline math where 80-layer
chooses `\[ … \]` display math — same content, both correct). It is never a
wrong answer; it is a tie-break flip in the hard top-k router amplified by
floating-point drift accumulated across more GPU layers.

**Root cause.** The router projection (`ffn_gate_inp`, a tiny `[192 × 4096]`
matrix) was an **FP16** `cublasGemmEx`. The router's `argmax` over near-equal
expert scores is the single most precision-sensitive step in the model, so FP16
rounding there is what flips routes.

**Fix.** Store the router weight in FP32 and compute its GEMV in full FP32 via
`cublasSgemv` (`gpu_mul_mat_f32`). Everything else stays FP16.

**Result.** On a long greedy generation, the identical-output prefix between
20-layer and 80-layer offload extended **from 243 → 1055 characters (4.3×)**;
the remaining split is again benign phrasing. No performance regression (graph
replay 20.8 → 20.3 ms/token — the SGEMV is tiny and we drop the router's
f32→f16 conversion). Cost: +240 MB of weights.

---

## 10. Overall progression

| Step | Change | tok/s (80 layers) |
|-----:|--------|------------------:|
| 0 | Baseline | 4.6 |
| 1 | Blackwell build + O(n) KV cache | ~13.2 |
| 2 | FP16 KV cache | (folded in) |
| 3 | Fused attention / on-device routing | (folded in) |
| 4 | FP16 dense weights (`cublasGemmEx`) | (folded in) |
| 5 | CUDA graph decode | 13.84 |
| 6 | Warp-per-row Q4_K expert matmul | 20.4 |
| 7 | Q4_K sub-block parallelism | 22.68 |
| 8 | **Coalesced warp-cooperative Q4_K row dot** | **44** |
| 9 | Shared/routed expert overlap (2nd stream) | ~48 (pure GPU) |
| 10 | FP32 router GEMV (stability; ~neutral speed) | ~49 (pure GPU) |

Pure GPU graph replay ended at ~20.3 ms/token (~49 tok/s). Peak resident memory
~192 GB (+240 MB after the FP32 router).

---

## 11. How to verify correctness (do this every change)

Speed is meaningless if the logits are wrong (recall the all-zero-logits trap).
Two quick checks:

1. **Arithmetic, ≤ ~40–50 layers** (fully correct regime):
   ```
   ./hy3-cli -m /home/user/hy3-gguf/hy3_q4k_mixed.gguf --gpu-layers 20 -p "11+22+33=?" -n 60
   ```
   Must reach `66` / `\boxed{66}`.

2. **Coherence at full 80 layers:**
   ```
   ./hy3-cli -m /home/user/hy3-gguf/hy3_q4k_mixed.gguf --gpu-layers 80 -p "The capital of France is" -n 40
   ```
   Must answer "Paris" and stay coherent.

**Routing divergence (benign, and much reduced — see Step 10):** across GPU
offload depths the greedy output can eventually diverge, but only as *benign
phrasing/formatting* (never a wrong answer): a hard top-k tie-break flips under
floating-point drift accumulated across more layers. Moving the router GEMV to
FP32 pushed the identical 20-vs-80-layer greedy prefix from ~243 to ~1055
characters. Short-prompt greedy output is identical across 40–80 layers.

---

## 12. Benchmarking recipes

```bash
# Build
make NVCC=/usr/local/cuda/bin/nvcc -j4

# Pure-GPU device time (50 replays/token), isolates kernel work
HY3_GRAPH_BENCH=1 ./hy3-cli -m .../hy3_q4k_mixed.gguf --gpu-layers 80 \
  -p "The capital of France is" -n 40 2>&1 | grep "graph replay"

# End-to-end throughput (this is the real number; do NOT use with GRAPH_BENCH)
./hy3-cli -m .../hy3_q4k_mixed.gguf --gpu-layers 80 \
  -p "Explain quantum entanglement in three sentences." -n 120 2>&1 | tail -1

# Per-token eval-vs-sample breakdown
HY3_TIMING=1 ...
```

> Historical note: differential phase-skipping env vars (`HY3_SKIP_ATTN` /
> `HY3_SKIP_FFN` / `HY3_SKIP_EXP`) were used during development to attribute
> time to each layer phase. They were removed once the routed-expert matmul
> (the confirmed 85% bottleneck) was optimized; `git log` has the details.

---

## 13. General lessons (transferable to other GGUF/CUDA work)

1. **Profile before optimizing.** Differential phase-skipping (temporary
   `HY3_SKIP_*` env vars) told us experts were 85% of the token — everything
   else was a distraction.
2. **Decode is latency/bandwidth-bound, not FLOP-bound.** GEMV with batch 1
   reads weights once and does little arithmetic; the wins are occupancy, fewer
   launches (CUDA graphs), smaller/half-precision reads, and *coalescing*.
3. **Occupancy first, then coalescing.** Warp-per-row (occupancy) then
   sub-block (idle-lane) then coalesced `uint32` reads (bandwidth) — each
   unlocked the next bottleneck. A ~56×-off-peak effective bandwidth number is
   the smoking gun for uncoalesced access.
4. **Know your quant layout at the byte level.** The whole final win rests on
   the identity `p*32 + e0 == 4*L`, letting 32 lanes read 128 contiguous bytes
   while still mapping cleanly onto Q4_K's sub-block/nibble structure.
5. **CUDA graphs need device-side position state** (`d_pos`) so the captured
   topology is invariant across tokens.
6. **Verify accuracy on every change**, and know your model's inherent
   divergence regime so you do not chase a "bug" that is just FP-reduction
   ordering.

---

## 14. Follow-ups (done)

The three follow-ups originally listed here are now complete:

- **Done** — removed the `HY3_SKIP_*` profiling scaffolding and the dead device
  functions (`dev_dot_q4_K_sub`, `qwarp_sum_f32`, `dev_dot_q4_K_f32_block`).
- **Done** — overlapped the shared-expert GEMV with the routed-expert matmuls on
  a second stream (Step 9; commit `a696f8f`).
- **Done** — the router-stability investigation led to the FP32 router GEMV
  (Step 10; commit `5285365`), which substantially reduces routing divergence.

## 15. Remaining opportunities

- The routed matmuls saturate the SMs, capping stream-overlap gains; a
  persistent-kernel / megakernel MoE that fuses gate_up→silu→down without
  round-trips to global memory could recover more.
- A fully deterministic (fixed-reduction-order) expert matmul would make GPU
  output bitwise-stable across offload depth, eliminating even the benign
  phrasing divergence.
