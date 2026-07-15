#!/usr/bin/env python3
"""Needle-in-a-haystack long-context retrieval test for hy3.

Generates a long filler "haystack" with a unique "needle" fact embedded at a
given depth, asks the model to retrieve it, and checks the answer. Used to
measure whether YaRN RoPE scaling (env/CLI) preserves retrieval past the
model's native 262144-token context.

Usage:
  python3 eval/hy3_needle.py --bin ./hy3-cli --model hy3-gguf/hy3_heretic_q4k.gguf \
      --gpu-layers 80 --tokens 16000 --depth 0.5 [--yarn-factor 4] [--kv-int4]

Env passed through to hy3-cli: HY3_ROPE_FACTOR / HY3_KV_INT4 are set from flags.
"""
import argparse, os, re, subprocess, sys, time

# A small pool of filler sentences; cycling avoids pure repetition (which would
# make retrieval trivially easy) while staying deterministic.
FILLER = [
    "The morning sun cast long shadows across the quiet valley.",
    "Rivers wind slowly through the ancient forests of the north.",
    "Merchants gathered in the square to trade grain and cloth.",
    "A gentle wind carried the scent of pine over the hills.",
    "Scholars debated the meaning of the old inscriptions for hours.",
    "The lighthouse blinked steadily against the darkening sky.",
    "Farmers counted the seasons by the flight of migrating birds.",
    "Snow settled softly on the rooftops of the sleeping town.",
    "The blacksmith's hammer rang out across the cobbled street.",
    "Children chased kites along the windy edge of the meadow.",
]

def build_prompt(target_tokens, depth, needle_val):
    # ~0.75 words per token is a rough English BPE ratio; we overshoot a little
    # and rely on the engine's own "prompt N tok" report for the true count.
    approx_words = int(target_tokens * 0.75)
    needle = f"The secret passcode is {needle_val}. Remember it."
    sentences, words = [], 0
    i = 0
    inserted = False
    while words < approx_words:
        # insert the needle once we pass the requested depth fraction
        if not inserted and words >= approx_words * depth:
            sentences.append(needle)
            inserted = True
        s = f"({i}) " + FILLER[i % len(FILLER)]
        sentences.append(s)
        words += len(s.split())
        i += 1
    if not inserted:
        sentences.append(needle)
    body = " ".join(sentences)
    return (body +
            "\n\nQuestion: What is the secret passcode mentioned somewhere in the "
            "text above? Reply with only the number.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./hy3-cli")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu-layers", type=int, default=80)
    ap.add_argument("--threads", type=int, default=30)
    ap.add_argument("--tokens", type=int, default=16000)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--yarn-factor", type=float, default=0.0)
    ap.add_argument("--kv-int4", action="store_true")
    ap.add_argument("--pos-stride", type=int, default=1,
                    help="RoPE position multiplier: spread N tokens across N*stride positions "
                         "to probe long-context extrapolation without dense prefill")
    ap.add_argument("--needle", default="92183")
    ap.add_argument("--n-predict", type=int, default=16)
    args = ap.parse_args()

    prompt = build_prompt(args.tokens, args.depth, args.needle)
    env = dict(os.environ)
    if args.yarn_factor and args.yarn_factor > 1.0:
        env["HY3_ROPE_FACTOR"] = str(args.yarn_factor)
    if args.kv_int4:
        env["HY3_KV_INT4"] = "1"
    if args.pos_stride and args.pos_stride > 1:
        env["HY3_POS_STRIDE"] = str(args.pos_stride)

    # Long prompts blow past the argv length limit, so feed via a --batch file
    # (one prompt per line, real newlines escaped to \n; run_batch unescapes).
    import tempfile
    bf = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    bf.write(prompt.replace("\\", "\\\\").replace("\n", "\\n") + "\n")
    bf.close()
    cmd = [args.bin, "-m", args.model, "--gpu-layers", str(args.gpu_layers),
           "-t", str(args.threads), "-n", str(args.n_predict), "-temp", "0",
           "--batch", bf.name]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    os.unlink(bf.name)
    raw_out, err = p.stdout, p.stderr
    # Extract the answer framed by <<<HY3_BEGIN 0>>> ... <<<HY3_END>>>
    mm = re.search(r"<<<HY3_BEGIN \d+>>>\n?(.*?)\n?<<<HY3_END>>>", raw_out, re.S)
    out = mm.group(1) if mm else raw_out

    m = re.search(r"prompt (\d+) tok", err)
    ptok = int(m.group(1)) if m else -1
    yarn = "YaRN RoPE enabled" in err
    kv = "INT4 KV" if "INT4 KV" in err else ("INT8 KV" if "INT8 KV" in err else "?")
    found = args.needle in out
    maxpos = ptok * args.pos_stride if ptok > 0 else -1
    print(f"[needle] req_tokens={args.tokens} actual_prompt_tok={ptok} stride={args.pos_stride} "
          f"max_rope_pos~{maxpos} depth={args.depth} "
          f"yarn={'on' if yarn else 'off'}(factor={args.yarn_factor}) kv={kv} "
          f"wall={dt:.1f}s -> {'PASS' if found else 'FAIL'}")
    print(f"  model_output: {out.strip()[:200]!r}")
    tl = re.search(r"timing \|.*", err)
    if tl: print("  " + tl.group(0))
    sys.stdout.flush()
    return 0 if found else 1

if __name__ == "__main__":
    sys.exit(main())
