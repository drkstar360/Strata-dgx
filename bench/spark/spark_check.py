"""bench/spark/spark_check.py - checks a running Strata server (docs/DGX_SPARK.md "Results so far"): the API (OpenAI,
streaming, Anthropic), greedy correctness (a fact, arithmetic, a code function that is run against tests, the same
answer twice, a reasoning answer with thinking on), and speed (a short prompt, prose and code; a 32K prompt with a
code word to recall, and a 32K summary).  Standard library only; it only sends requests.

    python3 bench/spark/spark_check.py                                  # http://127.0.0.1:8080
    python3 bench/spark/spark_check.py --url http://SPARK:8080 --api-key KEY --out results.json

The 32K text comes from this checkout (tools/needle_bench.py's haystack).  Exit code 1 when a check fails."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

R: dict = {}
ROOT = Path(__file__).resolve().parents[2]
KEY = ""


def call(url, path, body=None, headers=None, timeout=1800, stream=False):
    h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + KEY} if KEY else {}), **(headers or {})}
    req = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None, headers=h)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8")
    return (raw if stream else json.loads(raw)), time.time() - t0


def chat(url, content, max_tokens=256, thinking=False, **extra):
    body = {"model": "strata", "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "messages": [{"role": "user", "content": content}], **extra}
    out, wall = call(url, "/v1/chat/completions", body)
    msg = out["choices"][0]["message"]
    return (msg.get("content") or ""), out.get("usage", {}), out.get("timings") or {}, wall, out


def check(name, ok, detail=""):
    R.setdefault("checks", []).append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)


def speed(name, usage, t, wall):
    row = {"name": name, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
           "prompt_per_second": t.get("prompt_per_second"), "predicted_per_second": t.get("predicted_per_second"),
           "prompt_ms": t.get("prompt_ms"), "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
           "wall_s": round(wall, 2)}
    R.setdefault("speed", []).append(row)
    acc = (f", drafts {row['draft_n_accepted']}/{row['draft_n']}" if row["draft_n"] else "")
    print(f"  [speed] {name}: prompt {row['prompt_tokens']} tok at {row['prompt_per_second']} tok/s, "
          f"output {row['completion_tokens']} tok at {row['predicted_per_second']} tok/s{acc}, wall {row['wall_s']} s",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    global KEY
    KEY = a.api_key
    url = a.url.rstrip("/")

    print("== API")
    h, _ = call(url, "/health")
    R["health"] = h
    check("/health loaded", h.get("loaded") is True, f"model {h.get('model')}, context {h.get('max_context')}")
    m, _ = call(url, "/v1/models")
    ids = [x.get("id") for x in m.get("data", [])]
    check("/v1/models", bool(ids), ", ".join(ids))
    text, usage, t, wall, _ = chat(url, "What is 17 * 23? Answer with the number only.", 16)
    check("OpenAI chat (17*23)", "391" in text, repr(text.strip()))
    raw, _ = call(url, "/v1/chat/completions", {"model": "strata", "max_tokens": 16, "temperature": 0, "stream": True,
                                                "chat_template_kwargs": {"enable_thinking": False},
                                                "messages": [{"role": "user", "content": "Say hello in one word."}]},
                  stream=True)
    chunks = [l for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
    check("OpenAI streaming", len(chunks) > 1 and "data: [DONE]" in raw, f"{len(chunks)} chunks")
    out, _ = call(url, "/v1/messages", {"model": "strata", "max_tokens": 16, "temperature": 0,
                                       "thinking": {"type": "disabled"},
                                       "messages": [{"role": "user", "content": "What is 6 * 7? Answer with the number only."}]},
                  headers={"anthropic-version": "2023-06-01"})
    atext = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
    check("Anthropic messages (6*7)", "42" in atext, repr(atext.strip()))

    print("== correctness (temperature 0, thinking off)")
    text, *_ = chat(url, "What is the capital of Australia? Answer with the city name only.", 16)
    check("fact (Canberra)", "canberra" in text.lower(), repr(text.strip()))
    text, *_ = chat(url, "What is 1234 + 5678? Answer with the number only.", 16)
    check("arithmetic (6912)", "6912" in text.replace(",", ""), repr(text.strip()))
    prompt = ("Write a Python function is_prime(n) that returns True if n is a prime number and False otherwise. "
              "Reply with only the code in one ```python block.")
    code, usage, t, wall, _ = chat(url, prompt, 400)
    body = code.split("```python", 1)[-1].split("```", 1)[0] if "```" in code else code
    test = body + "\nassert [n for n in range(-3, 60) if is_prime(n)] == [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59]\n" \
                  "assert is_prime(7919) and not is_prime(7917)\nprint('pass')\n"
    try:
        r = subprocess.run([sys.executable, "-I", "-c", test], capture_output=True, text=True, timeout=20)
        ok, why = r.stdout.strip() == "pass", (r.stderr.strip().splitlines() or ["pass"])[-1]
    except subprocess.TimeoutExpired:
        ok, why = False, "timed out"
    check("code (is_prime runs and passes)", ok, why)
    again, *_ = chat(url, prompt, 400)
    check("greedy output repeats exactly", again == code, f"{len(code)} chars")
    text, usage, t, wall, _ = chat(url, "A farmer has 17 sheep. All but 9 run away. How many sheep are left? "
                                        "Think it through, then give the number.", 2048, thinking=True)
    check("reasoning, thinking on (9)", "9" in text.split()[-5:] or text.strip().endswith("9") or " 9" in text[-40:],
          repr(text.strip()[-80:]))
    speed("reasoning answer, thinking on", usage, t, wall)

    print("== speed")
    text, usage, t, wall, _ = chat(url, "Write a 400-word story about a lighthouse keeper who finds a message in a bottle.",
                                   600)
    speed("short prompt, prose", usage, t, wall)
    code, usage, t, wall, _ = chat(url, "Write a Python module implementing a binary search tree class with insert, "
                                        "delete, search, in-order traversal and height, with docstrings. Code only.", 600)
    speed("short prompt, code", usage, t, wall)
    if (ROOT / "tools" / "needle_bench.py").exists():
        sys.path.insert(0, str(ROOT / "tools"))
        import needle_bench as nb
        n_chars = int(32768 * 0.98 * nb.CHARS_PER_TOKEN)
        hay = nb.haystack(n_chars)
        word = "saffron-quartz-1729"
        cut = len(hay) // 2
        doc = hay[:cut] + f"\n\nThe secret code word is {word}.\n\n" + hay[cut:]
        q = doc + "\n\nWhat is the secret code word mentioned in the text above? Answer with the code word only."
        text, usage, t, wall, _ = chat(url, q, 32)
        check("32K recall (word at 50%)", word in text, repr(text.strip()))
        speed("32K prompt (recall)", usage, t, wall)
        text, usage, t, wall, _ = chat(url, doc[: len(doc) - 2000] + "\n\nSummarize the text above in about 300 words.",
                                       500)
        speed("32K prompt, 500-token summary", usage, t, wall)
    fails = [c["name"] for c in R["checks"] if not c["ok"]]
    print(f"== {len(R['checks']) - len(fails)}/{len(R['checks'])} checks passed" + (f"; failed: {fails}" if fails else ""))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(R, f, indent=1)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
