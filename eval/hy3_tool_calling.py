#!/usr/bin/env python3
"""
hy3 tool-calling (function-calling) test harness — think OFF.

The hy3 chat template has no native tool tokens, so tool calling is driven by
prompt engineering: every prompt lists the available tools as JSON schemas and
asks the model to emit exactly one tool call as
    <tool_call>{"name": "...", "arguments": {...}}</tool_call>
or <tool_call>none</tool_call> when no tool fits.

Like hy3_eval.py, this drives hy3-cli ONCE in --batch mode (model loads once for
all cases), streams output live, parses each tool call the moment its
<<<HY3_END>>> marker arrives, validates it against the expected function +
arguments, and — for the tools we can execute — runs the tool and prints the
resulting observation (a single-turn round trip).

Backend defaults to CUDA (--gpu-layers 80). think is OFF (no --think flag).

Env overrides:
  HY3_TOOL_BACKEND   cuda|metal      (default cuda)
  HY3_TOOL_GPU_LAYERS <n>            (default 80)
  HY3_TOOL_MODEL     <path>          (default: autodetected gguf)
  HY3_TOOL_TEMP      <float>         (default 0.0 = greedy, best for tool calls)
  HY3_TOOL_MAX_TOKENS <n>            (default 512)
"""

import os
import re
import sys
import json
import time
import subprocess
import tempfile

# ─── Configuration ────────────────────────────────────────────────────────────

HY3_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HY3_CLI    = os.path.join(HY3_DIR, "hy3-cli")

def _find_model():
    if os.environ.get("HY3_TOOL_MODEL"):
        return os.environ["HY3_TOOL_MODEL"]
    for c in (os.path.join(HY3_DIR, "hy3-gguf", "hy3_q4k_mixed.gguf"),
              os.path.join(os.path.dirname(HY3_DIR), "hy3-gguf", "hy3_q4k_mixed.gguf")):
        if os.path.exists(c):
            return c
    return os.path.join(HY3_DIR, "hy3-gguf", "hy3_q4k_mixed.gguf")

MODEL_PATH = _find_model()
BACKEND    = os.environ.get("HY3_TOOL_BACKEND", "cuda").lower()
GPU_LAYERS = int(os.environ.get("HY3_TOOL_GPU_LAYERS", "80"))
TEMP       = float(os.environ.get("HY3_TOOL_TEMP", "0.0"))
MAX_TOKENS = int(os.environ.get("HY3_TOOL_MAX_TOKENS", "512"))
EXPERTS    = int(os.environ.get("HY3_TOOL_EXPERTS", "8"))  # MoE experts per token

# ─── The tools the model is told it can call ────────────────────────────────────

TOOLS = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'Paris'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
    {
        "name": "multiply",
        "description": "Multiply two numbers and return the product.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "search_flights",
        "description": "Search for flights between two cities on a date.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["origin", "destination", "date"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "body"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount of money from one currency to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string", "description": "ISO code, e.g. USD"},
                "to_currency": {"type": "string", "description": "ISO code, e.g. EUR"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
]

# Executable implementations of the tools (mock, for the round-trip observation).
def _exec_tool(name, args):
    try:
        if name == "get_current_weather":
            unit = args.get("unit", "celsius")
            return {"location": args.get("location"), "temperature": 18,
                    "unit": unit, "conditions": "partly cloudy"}
        if name == "multiply":
            return {"product": float(args["a"]) * float(args["b"])}
        if name == "search_flights":
            return {"flights": [{"flight": "AB123", "price_usd": 780,
                                 "origin": args.get("origin"),
                                 "destination": args.get("destination"),
                                 "date": args.get("date")}]}
        if name == "send_email":
            return {"status": "sent", "to": args.get("to")}
        if name == "convert_currency":
            rate = 0.92  # mock USD->EUR
            return {"converted": round(float(args["amount"]) * rate, 2),
                    "to_currency": args.get("to_currency")}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    return {"error": "unknown tool"}

# ─── Grading helpers ────────────────────────────────────────────────────────────

def _vals(args):
    """All argument values as lowercased strings, for tolerant matching."""
    out = []
    for v in (args or {}).values():
        out.append(str(v).lower())
    return out

def _has(args, needle):
    return any(needle.lower() in v for v in _vals(args))

def _num_present(args, target):
    for v in (args or {}).values():
        try:
            if abs(float(v) - target) < 1e-6:
                return True
        except (TypeError, ValueError):
            pass
    return False

# ─── Test suite ─────────────────────────────────────────────────────────────────
# Each case: id, user query, expected tool name (or None for "no tool"), and a
# validator(args)->bool for the arguments.

TEST_SUITE = [
    {
        "id": "weather_paris",
        "query": "What's the weather like in Paris right now? Use celsius.",
        "expect": "get_current_weather",
        "check": lambda a: _has(a, "paris") and _has(a, "celsius"),
    },
    {
        "id": "multiply",
        "query": "What is 47 multiplied by 89?",
        "expect": "multiply",
        "check": lambda a: _num_present(a, 47) and _num_present(a, 89),
    },
    {
        "id": "flights",
        "query": "Find me flights from New York to Tokyo on 2025-12-01.",
        "expect": "search_flights",
        "check": lambda a: _has(a, "new york") and _has(a, "tokyo") and _has(a, "2025-12-01"),
    },
    {
        "id": "email",
        "query": "Send an email to john@example.com telling him the meeting is at 3pm.",
        "expect": "send_email",
        "check": lambda a: _has(a, "john@example.com") and _has(a, "3pm"),
    },
    {
        "id": "currency",
        "query": "Convert 100 US dollars to euros.",
        "expect": "convert_currency",
        "check": lambda a: _num_present(a, 100) and _has(a, "usd") and _has(a, "eur"),
    },
    {
        "id": "no_tool",
        "query": "Hi! In one sentence, who wrote the play Romeo and Juliet?",
        "expect": None,   # none of the tools apply -> should NOT call a tool
        "check": lambda a: True,
    },
]

# ─── Prompt construction ────────────────────────────────────────────────────────

def build_prompt(query):
    tools_json = json.dumps(TOOLS, ensure_ascii=False, indent=2)
    return (
        "You are a function-calling assistant. You can call the following tools "
        "(described as JSON schemas):\n\n"
        f"{tools_json}\n\n"
        "Instructions:\n"
        "- If the user's request should be handled by one of the tools, respond "
        "with EXACTLY one tool call and nothing else, in this format:\n"
        '  <tool_call>{"name": "<tool_name>", "arguments": {<args>}}</tool_call>\n'
        "- The arguments must be valid JSON and match the tool's schema.\n"
        "- If none of the tools are appropriate, respond with:\n"
        "  <tool_call>none</tool_call>\n"
        "  followed by a brief direct answer.\n\n"
        f"User request: {query}"
    )

# ─── Tool-call extraction ───────────────────────────────────────────────────────

def parse_tool_call(text):
    """Return (kind, payload):
       kind='call'  -> payload is {'name':..., 'arguments':{...}}
       kind='none'  -> payload is None (model declined to call a tool)
       kind='error' -> payload is a short reason string
    """
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    inner = m.group(1) if m else None
    if inner is not None and inner.strip().lower() == "none":
        return ("none", None)

    # candidate JSON: prefer inside the tag, else a ```json block, else first {...}
    candidates = []
    if inner:
        candidates.append(inner)
    for fb in re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
        candidates.append(fb)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))

    for c in candidates:
        c = c.strip()
        try:
            obj = json.loads(c)
        except Exception:  # noqa: BLE001
            # tolerate trailing prose after the JSON object
            try:
                obj, _ = json.JSONDecoder().raw_decode(c)
            except Exception:  # noqa: BLE001
                continue
        if isinstance(obj, dict) and "name" in obj:
            obj.setdefault("arguments", {})
            if not isinstance(obj["arguments"], dict):
                obj["arguments"] = {}
            return ("call", obj)
    return ("error", "no parseable tool call found")

# ─── Grading ────────────────────────────────────────────────────────────────────

VALID_NAMES = {t["name"] for t in TOOLS}

def grade_one(test, response):
    sep = "=" * 72
    print(f"\n{sep}\n  tool-call: {test['id']}\n{sep}")
    print(f"[query] {test['query']}")
    print(f"[raw]   {response.strip()[:400]}")

    kind, payload = parse_tool_call(response)
    verdict, passed, detail = "FAIL", False, ""

    if test["expect"] is None:
        # expected NO tool call
        if kind == "none":
            verdict, passed, detail = "PASS", True, "correctly declined to call a tool"
        elif kind == "call":
            detail = f"unexpected tool call: {payload.get('name')}"
        else:
            # no <tool_call>none> tag but also no call -> acceptable as a decline
            verdict, passed, detail = "PASS", True, "no tool call emitted (implicit decline)"
    else:
        if kind == "call":
            name = payload.get("name")
            args = payload.get("arguments", {})
            if name not in VALID_NAMES:
                detail = f"hallucinated tool name '{name}'"
            elif name != test["expect"]:
                detail = f"wrong tool: got '{name}', expected '{test['expect']}'"
            else:
                try:
                    args_ok = bool(test["check"](args))
                except Exception as e:  # noqa: BLE001
                    args_ok = False
                    detail = f"arg check raised {type(e).__name__}: {e}"
                if args_ok:
                    obs = _exec_tool(name, args)
                    verdict, passed = "PASS", True
                    detail = f"args OK -> executed: {json.dumps(obs, ensure_ascii=False)}"
                elif not detail:
                    detail = f"argument mismatch: {json.dumps(args, ensure_ascii=False)}"
        elif kind == "none":
            detail = "declined to call a tool, but a call was expected"
        else:
            detail = payload  # parse error reason

    tag = {"PASS": "[PASS]", "FAIL": "[FAIL]"}[verdict]
    if kind == "call" and passed:
        print(f"[call]  {payload['name']}({json.dumps(payload['arguments'], ensure_ascii=False)})")
    print(f"{tag}  {test['id']}  — {detail}")
    return {"id": test["id"], "expect": test["expect"], "verdict": verdict,
            "passed": passed, "kind": kind, "detail": detail,
            "response": response.strip()[:2000]}

# ─── Batch driver (mirrors hy3_eval.py) ─────────────────────────────────────────

BEGIN_RE = re.compile(r"^<<<HY3_BEGIN (\d+)>>>$")
END_MARK = "<<<HY3_END>>>"

def run_batch(tests):
    if not os.path.exists(HY3_CLI):
        sys.exit(f"hy3-cli not found: {HY3_CLI} (run `make hy3-cli`)")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"model not found: {MODEL_PATH}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as bf:
        for t in tests:
            p = build_prompt(t["query"])
            bf.write(p.replace("\\", "\\\\").replace("\n", "\\n") + "\n")
        batch_path = bf.name

    cmd = [HY3_CLI, "-m", MODEL_PATH, "-n", str(MAX_TOKENS),
           "-temp", str(TEMP), "-experts", str(EXPERTS), "--batch", batch_path]
    if BACKEND == "cuda":
        cmd += ["--gpu-layers", str(GPU_LAYERS)]
    elif BACKEND == "metal":
        cmd.append("--metal")
    else:
        sys.exit(f"unknown HY3_TOOL_BACKEND={BACKEND!r}")
    # think is intentionally OFF: no --think / --think-low flag.

    print(f"[hy3] backend: {BACKEND}  experts={EXPERTS}  think=off  temp={TEMP}  tools={len(TOOLS)}")
    print(f"[hy3] launching: {' '.join(cmd)}")
    print("[hy3] hy3-cli stderr streams below.\n")

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, encoding="utf-8", errors="replace")

    results = [None] * len(tests)
    cur, body = None, []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        stripped = line.rstrip("\n")
        m = BEGIN_RE.match(stripped)
        if m:
            cur, body = int(m.group(1)), []
            continue
        if cur is not None and stripped == END_MARK:
            if 0 <= cur < len(tests):
                results[cur] = grade_one(tests[cur], "".join(body).strip())
            cur, body = None, []
            continue
        if cur is not None:
            body.append(line)
    proc.wait()
    os.unlink(batch_path)
    if proc.returncode != 0:
        sys.exit(f"\nhy3-cli failed (rc={proc.returncode})")

    for i, t in enumerate(tests):
        if results[i] is None:
            results[i] = {"id": t["id"], "expect": t["expect"], "verdict": "FAIL",
                          "passed": False, "kind": "error",
                          "detail": "no output produced", "response": ""}
    print(f"\n[hy3] batch finished in {time.time() - t0:.0f}s\n")
    return results

# ─── Runner ─────────────────────────────────────────────────────────────────────

def main():
    run_ids = sys.argv[1:] or None
    tests = TEST_SUITE if not run_ids else [t for t in TEST_SUITE if t["id"] in run_ids]

    print(f"\n{'='*72}\n  HY3 Tool-Calling Test  |  {len(tests)} cases  |  think=OFF\n"
          f"  model: {os.path.basename(MODEL_PATH)}  backend={BACKEND}\n{'='*72}")

    results = run_batch(tests)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    bar = "=" * 72
    print(f"\n{bar}\n  TOOL-CALLING RESULTS\n{bar}")
    print(f"  {'ID':<16}{'Expected tool':<22}{'Verdict':<8}")
    print(f"  {'-'*16}{'-'*22}{'-'*8}")
    for r in results:
        exp = r["expect"] or "(no tool)"
        print(f"  {r['id']:<16}{exp:<22}{'[PASS]' if r['passed'] else '[FAIL]':<8}")
    print(f"\n  PASS: {passed}/{total}  ({100*passed//max(total,1)}%)\n{bar}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"hy3_tool_calling_results_{BACKEND}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out}")

if __name__ == "__main__":
    main()
