#!/usr/bin/env python3.12
"""
hy3 reasoning/coding benchmark (borrowed from ds4-eval style).

Drives hy3-cli ONCE in --batch mode so the ~140s model load is paid a single
time for all 13 questions. Each answer is code that we extract, execute in a
subprocess, and auto-grade.

Categories: Math Reasoning | Coding | Logic | Algorithm | Science
"""

import os
import re
import sys
import json
import time
import subprocess
import tempfile
from datetime import datetime
from typing import Optional, Callable

# ─── Configuration ────────────────────────────────────────────────────────────

HY3_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HY3_CLI      = os.path.join(HY3_DIR, "hy3-cli")

def _find_model() -> str:
    if os.environ.get("HY3_EVAL_MODEL"):
        return os.environ["HY3_EVAL_MODEL"]
    candidates = [
        os.path.join(HY3_DIR, "hy3-gguf", "hy3_q4k_mixed.gguf"),
        os.path.join(os.path.dirname(HY3_DIR), "hy3-gguf", "hy3_q4k_mixed.gguf"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

MODEL_PATH   = _find_model()
EXEC_TIMEOUT = 60      # seconds per code execution
MAX_TOKENS   = int(os.environ.get("HY3_EVAL_MAX_TOKENS", "8000"))  # -n per question
EXPERTS      = int(os.environ.get("HY3_EVAL_EXPERTS", "8"))  # MoE experts per token
TEMP         = float(os.environ.get("HY3_EVAL_TEMP", "1.0")) # sampling temperature
THINK        = os.environ.get("HY3_EVAL_THINK", "low")       # reasoning: off | low | high
# Backend: "cuda" (default, NVIDIA) drives --gpu-layers N; "metal" drives --metal.
BACKEND      = os.environ.get("HY3_EVAL_BACKEND", "cuda").lower()
# For CUDA: how many transformer layers to offload to the GPU (80 = full offload).
GPU_LAYERS   = int(os.environ.get("HY3_EVAL_GPU_LAYERS", "80"))

# ─── Code Execution Engine ────────────────────────────────────────────────────

def extract_code_blocks(text: str) -> list:
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    return [b.strip() for b in blocks]


def run_code_in_subprocess(code: str, timeout: int = EXEC_TIMEOUT) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, encoding="utf-8") as f:
        f.write(code)
        fname = f.name
    try:
        proc = subprocess.run([sys.executable, fname],
                              capture_output=True, text=True, timeout=timeout)
        return {"stdout": proc.stdout, "stderr": proc.stderr,
                "returncode": proc.returncode, "timed_out": False}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "TIMEOUT",
                "returncode": -1, "timed_out": True}
    finally:
        os.unlink(fname)


def execute_response(response: str, timeout: int = EXEC_TIMEOUT) -> dict:
    blocks = extract_code_blocks(response)
    if not blocks:
        return {"ran": False, "code": "", "stdout": "",
                "stderr": "No code block found.", "returncode": -1,
                "timed_out": False}
    result = run_code_in_subprocess(blocks[-1], timeout)
    result["ran"] = True
    result["code"] = blocks[-1]
    return result


# ─── Grading Helpers ──────────────────────────────────────────────────────────

def grade_contains(stdout: str, expected: str) -> bool:
    return expected.strip() in stdout

def grade_int_anywhere(stdout: str, expected: int) -> bool:
    return str(expected) in stdout


# ─── Test Suite (13 questions) ─────────────────────────────────────────────────

TEST_SUITE = [
    {
        "id": "math_02",
        "category": "Math · Number Theory",
        "difficulty": "3/5",
        "title": "Euler Totient Sum — last 6 digits",
        "prompt": """Write Python code to compute the last 6 digits of  S = sum_{k=1}^{10^6} phi(k)
where phi is Euler's totient function.

Requirements:
1. Use a linear sieve (O(n)) — absolutely no trial division per element
2. First verify: sum_{k=1}^{10} phi(k) = 32  (print this check)
3. Print the last 6 digits of S, zero-padded: e.g.  "Answer: 552392"

Your final print must match the format: "Answer: XXXXXX\"""",
        "grade": lambda r: grade_contains(r["stdout"], "552392"),
        "answer_note": "Last 6 digits: 552392  (total S = 303963552392)",
    },
    {
        "id": "math_03",
        "category": "Math · Combinatorics",
        "difficulty": "4/5",
        "title": "Lattice Paths Avoiding the Anti-Diagonal",
        "prompt": """Write Python code to solve:

Count monotone lattice paths (right/up steps only) from (0,0) to (n,n)
that do NOT pass through any point (i, n-i) for i = 1, 2, ..., n-1.
(These are the n-1 interior points of the anti-diagonal x+y=n; the corners
(0,n) and (n,0) are NOT excluded.)

1. Implement with dynamic programming
2. Print results for n = 1 through 10 (one line each: "n=k: <count>")
3. Print the result for n=20 as: "n=20: <count>\"""",
        "grade": lambda r: "n=20:" in r["stdout"] and r["returncode"] == 0,
        "answer_note": "Must output n=20 result without crashing",
    },
    {
        "id": "math_04",
        "category": "Math · Prime",
        "difficulty": "2/5",
        "title": "Segmented Sieve in [10^12, 10^12 + 10^6]",
        "prompt": """Implement a segmented Sieve of Eratosthenes in Python to find all primes
in the range [10^12, 10^12 + 10^6].

Requirements:
1. Memory-efficient segmented sieve (segment size <= 10^6)
2. Print: "Prime count: <N>"
3. Print: "First 5: <list>"
4. Print: "Last 5: <list>"
5. Print: "Max gap: <G>"

The known answer is 36249 primes. Must run in under 60 seconds.""",
        "grade": lambda r: grade_int_anywhere(r["stdout"], 36249),
        "answer_note": "36249 primes in [10^12, 10^12 + 10^6]",
    },
    {
        "id": "code_01",
        "category": "Coding · Algorithm",
        "difficulty": "3/5",
        "title": "Median of Two Sorted Arrays — O(log n)",
        "prompt": """Implement `find_median(nums1, nums2)` with O(log(m+n)) binary search (NOT merge).

Assert these test cases and print "ALL TESTS PASSED" if all correct:
  find_median([1,3], [2])            == 2.0
  find_median([1,2], [3,4])          == 2.5
  find_median([], [1])               == 1.0
  find_median([1,1,1], [1,1])        == 1.0
  find_median([1,2,3,4,5], [6,7,8])  == 4.5

Print "ALL TESTS PASSED" if assertions succeed.""",
        "grade": lambda r: grade_contains(r["stdout"], "ALL TESTS PASSED"),
        "answer_note": 'Must print "ALL TESTS PASSED"',
    },
    {
        "id": "code_02",
        "category": "Coding · System Design",
        "difficulty": "4/5",
        "title": "Thread-Safe LRU Cache with TTL",
        "prompt": """Implement a production-grade LRU cache class in Python:
1. O(1) get/put using OrderedDict
2. Thread-safe: shard by key hash into 8 sub-caches, each with its own RLock
3. Optional per-entry TTL (seconds); expired = miss
4. hit_rate() and eviction_count() stats

Stress test:
- 8 threads, each doing 100000 random get/put on capacity=1000 cache
- Print: "ops/sec: <N>", "hit_rate: <X>%", "evictions: <N>"
- Print "STRESS TEST DONE" at the very end""",
        "grade": lambda r: grade_contains(r["stdout"], "STRESS TEST DONE"),
        "answer_note": 'Must print "STRESS TEST DONE"',
    },
    {
        "id": "code_03",
        "category": "Coding · Data Structures",
        "difficulty": "4/5",
        "title": "Persistent Segment Tree — K-th Smallest",
        "prompt": """Implement a Persistent Segment Tree in Python.

Array A = [3,1,4,1,5,9,2,6,5,3,5]  (0-indexed, values 1-9)

The tree should support:
1. Build initial version from A
2. Point update -> new version
3. Count elements <= v in range [l,r] on any version
4. K-th smallest in range [l,r]

Answer and print (one per line):
  Q1: count elements in A[2..8] <= 4            # expected 4
  Q2: 3rd smallest in A[0..6]                   # expected 3
  Q3: direct sum A[3..9]                        # expected 31
  Q4: after setting A[5]=1, 2nd smallest in A[0..10]

Print "SEGMENT TREE OK" after all queries.""",
        "grade": lambda r: grade_contains(r["stdout"], "SEGMENT TREE OK"),
        "answer_note": 'Must print "SEGMENT TREE OK"',
    },
    {
        "id": "code_04",
        "category": "Coding · ML",
        "difficulty": "4/5",
        "title": "Multi-Head Attention + RoPE (NumPy only)",
        "prompt": """Implement from scratch using ONLY numpy (no torch/jax/etc.):

1. scaled_dot_product_attention(Q, K, V, mask=None)  — stable softmax
2. multi_head_attention(x, W_Q, W_K, W_V, W_O, n_heads, causal=False)
3. apply_rope(x, head_dim)  — Rotary Position Embedding

Test:
- batch=2, seq_len=8, d_model=64, n_heads=4
- Run forward pass with causal=True
- Assert output.shape == (2, 8, 64)
- For causal mask: each position i should have ~0 attention weight on j>i
  Print: "max_future_attn: <value>"  (should be < 1e-6)
- Print "ATTENTION OK" if shape is correct and max_future_attn < 1e-6""",
        "grade": lambda r: grade_contains(r["stdout"], "ATTENTION OK"),
        "answer_note": 'Must print "ATTENTION OK"',
    },
    {
        "id": "code_05",
        "category": "Coding · Graph",
        "difficulty": "3/5",
        "title": "Dijkstra vs A* on Large Random Graph",
        "prompt": """Implement Dijkstra and A* (Euclidean heuristic) in Python.

Setup (use random seed 42 throughout):
- 5000 nodes, each with random (x,y) in [0,1000]^2
- 20000 random weighted edges, weights uniform in [1,100]
- Make the graph undirected

Tasks:
1. Find shortest path node 0 -> node 4999 with both algorithms
2. Assert both give the same distance (print "SAME DISTANCE: <d>")
3. Print: "Dijkstra: <time>s  A*: <time>s  Speedup: <ratio>x"
4. Print "GRAPH OK\"""",
        "grade": lambda r: grade_contains(r["stdout"], "GRAPH OK"),
        "answer_note": 'Must print "GRAPH OK"',
    },
    {
        "id": "logic_01",
        "category": "Logic · SAT",
        "difficulty": "3/5",
        "title": "Knights & Knaves — Exhaustive Solver",
        "prompt": """Solve this logic puzzle by brute-force enumeration in Python (2^5 = 32 cases):

Five people A,B,C,D,E — each is Knight (always true) or Knave (always lies).
  A: "Exactly two of the five of us are knights."
  B: "A is a knave."
  C: "B and D are both knights."
  D: "C is a knave."
  E: "The number of knights among A,B,C,D is even."

Consistency rule: if speaker is Knight, their statement must be True; if Knave, False.

Print all valid assignments (format: "A=Knight B=Knave ...") or "No solution".
Print total: "Solutions found: <N>"
Print "PUZZLE SOLVED\"""",
        "grade": lambda r: grade_contains(r["stdout"], "PUZZLE SOLVED"),
        "answer_note": 'Must print "PUZZLE SOLVED"',
    },
    {
        "id": "logic_02",
        "category": "Logic · Proof",
        "difficulty": "4/5",
        "title": "Verify Three Mathematical Claims with Python",
        "prompt": """For each claim, write Python to investigate and print a verdict.

Claim 1: For all n >= 2, n! + 1 is never a perfect square.
  - Check n=2..20, print any counterexample or "Claim 1: No counterexample n=2..20"

Claim 2: The product of any 4 consecutive integers is always divisible by 24.
  - Check all starting integers -100 to 100
  - Print "Claim 2: TRUE" or "Claim 2: FALSE, counterexample: ..."

Claim 3: There is no prime p such that both p and p^2+2 are prime.
  - Check all primes up to 10^6
  - Print any (p, p^2+2) counterexample or result

Print "CLAIMS DONE" at the very end.""",
        "grade": lambda r: grade_contains(r["stdout"], "CLAIMS DONE"),
        "answer_note": "Claim 1 FALSE (n=4,5,7); Claim 2 TRUE; Claim 3 FALSE (p=3).",
    },
    {
        "id": "sci_01",
        "category": "Science · Simulation",
        "difficulty": "3/5",
        "title": "Figure-8 Three-Body Orbit — Energy Conservation",
        "prompt": """Implement the Leapfrog integrator for 3-body gravity in Python (numpy only).

Use the Chenciner-Montgomery figure-8 initial conditions (G=1, all masses=1):
  positions = [[-0.97000436, 0.24308753],
               [ 0.97000436,-0.24308753],
               [ 0.0,        0.0       ]]
  velocities = [[ 0.466203685, 0.43236573],
                [ 0.466203685, 0.43236573],
                [-0.93240737, -0.86473146]]

Simulate for T=6.3259 (approx 1 period), dt=0.0001.

At each step track total energy E = KE + PE.
Print:
  "E0: <initial energy>"
  "Ef: <final energy>"
  "Drift: <|Ef-E0|/|E0|>"
  "SIMULATION OK" if drift < 0.01, else "SIMULATION FAILED\"""",
        "grade": lambda r: grade_contains(r["stdout"], "SIMULATION OK"),
        "answer_note": 'Must print "SIMULATION OK" with <1% energy drift',
    },
    {
        "id": "sci_03",
        "category": "Science · Optimization",
        "difficulty": "3/5",
        "title": "Optimizer Comparison on Rosenbrock",
        "prompt": """Implement and compare 4 optimizers on the Rosenbrock function
  f(x,y) = (1-x)^2 + 100*(y-x^2)^2   (minimum at (1,1), f=0)

All start from (-1.0, 1.0).

Implement from scratch (numpy only for 1-3):
  1. Vanilla GD:  lr=0.001, 20000 steps
  2. Momentum:    lr=0.001, beta=0.9, 20000 steps
  3. Adam:        lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, 5000 steps
  4. scipy L-BFGS-B (allowed to use scipy here)

For each print: "Name | f_final | steps_to_0.001 | time_ms"
Print "OPTIMIZATION OK" at the end.""",
        "grade": lambda r: grade_contains(r["stdout"], "OPTIMIZATION OK"),
        "answer_note": 'Must print "OPTIMIZATION OK"',
    },
    {
        "id": "algo_01",
        "category": "Algorithm · DP",
        "difficulty": "3/5",
        "title": "Edit Distance with Path Reconstruction",
        "prompt": """Implement Levenshtein edit distance with full operation reconstruction.

edit_distance(s1, s2) should return (distance, operations) where operations is
a list of the edits (insert/delete/substitute/match) to turn s1 into s2.

Test and assert:
  edit_distance("kitten", "sitting")[0]   == 3
  edit_distance("", "abc")[0]             == 3
  edit_distance("abc", "abc")[0]          == 0
  edit_distance("sunday", "saturday")[0]  == 3

Verify that applying the returned operations to s1 actually produces s2 for
each test case. Print "EDIT DISTANCE OK" if all pass.""",
        "grade": lambda r: grade_contains(r["stdout"], "EDIT DISTANCE OK"),
        "answer_note": 'Must print "EDIT DISTANCE OK"',
    },
]


# ─── Batch driver ─────────────────────────────────────────────────────────────

BEGIN_RE = re.compile(r"^<<<HY3_BEGIN (\d+)>>>$")
END_MARK = "<<<HY3_END>>>"

def run_hy3_batch(tests: list) -> list:
    """Write all prompts to a batch file, call hy3-cli once, stream stdout live,
    and grade each question the moment its <<<HY3_END>>> marker arrives (so you
    see PASS/FAIL right after each answer instead of only at the end)."""
    if not os.path.exists(HY3_CLI):
        sys.exit(f"hy3-cli not found: {HY3_CLI}  (run `make hy3-cli`)")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"model not found: {MODEL_PATH}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as bf:
        for t in tests:
            bf.write(t["prompt"].replace("\\", "\\\\").replace("\n", "\\n") + "\n")
        batch_path = bf.name

    cmd = [HY3_CLI, "-m", MODEL_PATH,
           "-n", str(MAX_TOKENS), "-temp", str(TEMP),
            "-experts", str(EXPERTS), "--batch", batch_path]
    if BACKEND == "cuda":
        cmd += ["--gpu-layers", str(GPU_LAYERS)]
    elif BACKEND == "metal":
        cmd.append("--metal")
    else:
        sys.exit(f"unknown HY3_EVAL_BACKEND={BACKEND!r} (expected 'cuda' or 'metal')")
    if THINK == "high":
        cmd.append("--think")
    elif THINK == "low":
        cmd.append("--think-low")

    backend_desc = (f"CUDA (--gpu-layers {GPU_LAYERS})" if BACKEND == "cuda"
                    else "Metal (--metal)")
    print(f"[hy3] backend: {backend_desc}")
    print(f"[hy3] launching: {' '.join(cmd)}")
    print(f"[hy3] loading model (this pays the model cold start once)…")
    print(f"[hy3] hy3-cli stderr (load progress + per-question timing) streams below.\n")

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1,
                            encoding="utf-8", errors="replace")

    results = [None] * len(tests)
    cur_idx = None          # question currently being received, or None
    cur_body = []           # accumulated lines of the current answer

    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

            stripped = line.rstrip("\n")
            m = BEGIN_RE.match(stripped)
            if m:
                cur_idx = int(m.group(1))
                cur_body = []
                continue

            if cur_idx is not None and stripped == END_MARK:
                response = "".join(cur_body).strip()
                # grade this question immediately, inline
                if 0 <= cur_idx < len(tests):
                    results[cur_idx] = grade_one(tests[cur_idx], response)
                cur_idx = None
                cur_body = []
                continue

            if cur_idx is not None:
                cur_body.append(line)
    except KeyboardInterrupt:
        proc.kill()
        raise
    proc.wait()
    elapsed = time.time() - t0
    os.unlink(batch_path)

    if proc.returncode != 0:
        sys.exit(f"\nhy3-cli failed (rc={proc.returncode})")

    # fill in any question that never produced output
    for i, t in enumerate(tests):
        if results[i] is None:
            results[i] = {"id": t["id"], "title": t["title"], "category": t["category"],
                          "difficulty": t["difficulty"], "verdict": "NO_CODE",
                          "passed": False, "returncode": -1, "stdout": "",
                          "stderr": "no output produced", "response": ""}

    print(f"\n[hy3] batch finished in {elapsed:.0f}s\n")
    return results


# ─── Runner ───────────────────────────────────────────────────────────────────

VERDICT = {"PASS": "[PASS]   ", "FAIL": "[FAIL]   ", "ERROR": "[ERROR]  ",
           "TIMEOUT": "[TIMEOUT]", "NO_CODE": "[NO CODE]"}

def grade_one(test: dict, response: str) -> dict:
    sep = "=" * 72
    print(f"\n{sep}\n  [{test['category']}]  {test['difficulty']}  |  {test['title']}\n{sep}")
    print(f"[Preview] {response[:300].strip()}\n")

    exec_result = execute_response(response)
    grader: Optional[Callable] = test.get("grade")

    if not exec_result["ran"]:
        verdict, passed = "NO_CODE", False
    elif exec_result["timed_out"]:
        verdict, passed = "TIMEOUT", False
    elif exec_result["returncode"] != 0:
        verdict, passed = "ERROR", False
    elif grader is not None:
        try:
            passed = grader(exec_result)
        except Exception:
            passed = False
        verdict = "PASS" if passed else "FAIL"
    else:
        passed, verdict = True, "PASS"

    print(f"[Exec] rc={exec_result['returncode']} timed_out={exec_result['timed_out']}")
    if exec_result["stdout"]:
        print(f"[stdout]\n{exec_result['stdout'][:600].rstrip()}")
    if exec_result["stderr"] and exec_result["returncode"] != 0:
        print(f"[stderr]\n{exec_result['stderr'][:300].rstrip()}")
    print(f"\n{VERDICT[verdict]}  {test['id']}")
    if test.get("answer_note"):
        print(f"[Note] {test['answer_note']}")

    return {"id": test["id"], "title": test["title"], "category": test["category"],
            "difficulty": test["difficulty"], "verdict": verdict, "passed": passed,
            "returncode": exec_result["returncode"],
            "stdout": exec_result["stdout"][:2000],
            "stderr": exec_result["stderr"][:500],
            "response": response[:4000]}


def print_summary(results: list):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    bar = "=" * 72
    print(f"\n\n{bar}\n  HY3 BENCHMARK RESULTS  |  {datetime.now():%Y-%m-%d %H:%M}\n{bar}")
    print(f"  {'ID':<12}{'Title':<34}{'Verdict':<10}")
    print(f"  {'-'*12}{'-'*34}{'-'*10}")
    for r in results:
        print(f"  {r['id']:<12}{r['title'][:33]:<34}{VERDICT.get(r['verdict'],'[?]'):<10}")
    print(f"\n  PASS: {passed}/{total}  ({100*passed/max(total,1):.0f}%)\n{bar}\n")


def main():
    run_ids = sys.argv[1:] or None
    tests = TEST_SUITE if not run_ids else [t for t in TEST_SUITE if t["id"] in run_ids]

    backend_desc = (f"CUDA/gpu-layers={GPU_LAYERS}" if BACKEND == "cuda" else "Metal")
    print(f"\n{'='*72}\n  HY3 Reasoning/Coding Benchmark  |  {len(tests)} questions\n"
          f"  model: {os.path.basename(MODEL_PATH)}  backend={backend_desc}  "
          f"experts={EXPERTS} temp={TEMP} think={THINK}\n{'='*72}")

    results = run_hy3_batch(tests)

    print_summary(results)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"hy3_eval_results_{BACKEND}_e{EXPERTS}_think{THINK}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
