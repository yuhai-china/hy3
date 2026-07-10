---
license: apache-2.0
base_model: tencent/Hy3
tags:
- hy3
- hunyuan
- hy_v3
- gguf
- mixture-of-experts
- moe
language:
- en
- zh
library_name: hy3
pipeline_tag: text-generation
---

# hy3-gguf — GGUF weights for Tencent Hy3 (HYV3)

GGUF-format weights for **Hy3** (`HYV3ForCausalLM`, `model_type: hy_v3`), a
295B-parameter / 21B-active-parameter Mixture-of-Experts model from
**Tencent's Hunyuan ("Hy") team** ([`tencent/Hy3`](https://huggingface.co/tencent/Hy3)).

These files were produced by the **[`hy3`](https://github.com/yuhai-china/hy3)**
converter (`hy3-convert`) and are meant to be run with the **`hy3` inference
engine**, a from-scratch C/Metal/CUDA implementation.

> ## ⚠️ This GGUF does NOT work with llama.cpp
>
> Despite the `.gguf` extension, these files are **only usable by the
> [`hy3`](https://github.com/yuhai-china/hy3) engine**. `llama.cpp`,
> `ollama`, `LM Studio`, `text-generation-webui`, `koboldcpp`, and any other
> llama.cpp-based tool **cannot load these files**. Three independent reasons:
>
> 1. **Unknown architecture.** The metadata declares
>    `general.architecture = "hy_v3"`. llama.cpp only knows `hunyuan-moe`,
>    `hunyuan-dense`, `hunyuan_vl` — loading aborts with
>    `unknown model architecture: 'hy_v3'`.
> 2. **Custom metadata keys.** All hyperparameters use the `hy_v3.*` prefix
>    (`hy_v3.block_count`, `hy_v3.expert_count`, …), which llama.cpp does not
>    look up.
> 3. **Non-fused expert tensors.** Experts are stored **one tensor per expert**
>    (`blk.N.ffn_gate_exps.0.gate_proj.weight`, `…1…`, … — 46080 tensors),
>    whereas llama.cpp expects experts fused into a single stacked 3D tensor per
>    layer. This is a fundamentally different on-disk layout.
>
> This is a **custom GGUF** readable only by the `hy3` loader. Do not open
> issues against llama.cpp for these files.

## How to run

Use the `hy3` engine: <https://github.com/yuhai-china/hy3>

```bash
git clone https://github.com/yuhai-china/hy3
cd hy3
make            # macOS builds the Metal backend automatically

# download a GGUF from this repo, then:
./run_metal.sh -m /path/to/hy3_q4k_mixed.gguf -p "The capital of France is" -experts 8
```

> **Testing scope:** the `hy3` engine's performance work and benchmarks were
> developed and verified **only on macOS / Apple Silicon (Metal backend)**,
> measured on an M2 Ultra (~20–27 tok/s decode depending on `-experts`). The
> CPU and CUDA backends exist in the source but were not exercised as part of
> that work — treat them as untested.

## Files / quantization

The mixed-precision GGUF follows this scheme (see `hy3_convert.c`):

| Tensor group | Type |
|---|---|
| Routed experts (`ffn_{gate,up,down}_exps`) — the bulk of the model | **Q4_K** |
| Attention q/k/v/o projections, shared-expert & dense FFN, `output.weight` | **Q8_0** |
| Norms, router (`ffn_gate_inp`), biases | **F32** |
| `token_embd.weight` | **F16** |

## Model facts

| | |
|---|---|
| Architecture | `HYV3ForCausalLM` (`hy_v3`) |
| Layers | 80 (layer 0 dense, layers 1–79 MoE) |
| Hidden size | 4096 |
| Attention | 64 heads, GQA with 8 KV heads, head_dim 128 |
| Experts | 192 routed (top-8 activated) + 1 shared (always active) |
| Expert intermediate size | 1536 |
| Dense (layer 0) intermediate size | 13312 |
| Vocab size | 120832 (120818 real tokens + padding) |
| RoPE | theta 11158840, `rotate_half` pairing |
| QK norm | per-head RMSNorm on Q and K, before RoPE |
| MoE routing | `sigmoid(router_logits)`; top-8 by `sigmoid + expert_bias`, combined using **unbiased** sigmoid weights, renormalized to sum 1, scaled by `router_scaling_factor = 2.826` |

The engine supports a runtime **top-k experts** override (`-experts 1..8`) to
trade quality for speed. On a small 13-question code/reasoning eval (greedy,
no-think): **experts=8 → 10/13**, **experts=4 → 7/13**. Default is 8.

## Chat template

Hy3 is instruction-tuned and expects the Hunyuan V3 chat format (the `hy3`
engine applies it automatically; use `--raw` to bypass). Single user turn,
no-think:

```
<｜hy_begin_of_sentence:opensource｜><｜reasoning_mode:opensource｜>reasoning_effort:no_think<｜hy_User:opensource｜>{prompt}<｜hy_Assistant:opensource｜><think:opensource></think:opensource>
```

Generation stops on `<｜hy_eos:opensource｜>` (120025),
`<｜hy_endofsentence｜>` (120001), or `<｜hy_EOT｜>` (120008).

## License & attribution

Weights derive from [`tencent/Hy3`](https://huggingface.co/tencent/Hy3); refer
to the upstream repository for the governing model license. This is an
unofficial community conversion, not affiliated with or endorsed by Tencent.
