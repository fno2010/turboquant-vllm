"""Gemma 4 26B: WHT vs Planar KV cache compression benchmark."""
import subprocess, requests, time, json, os, sys

MODEL = "google/gemma-4-26B-A4B-it"
PROMPTS = [
    "What is the capital of Finland?",
    "Calculate 17 * 23 and show your work",
    "Write a product description for a SaaS analytics dashboard",
    "Explain the difference between TCP and UDP in two sentences",
    "Write a Python function that checks if a number is prime",
]

CONFIGS = [
    ("BF16-baseline", {}),
    ("WHT-K3V3",     {"TQ_KV_K_BITS": "3", "TQ_KV_V_BITS": "3", "TQ_KV_ROTATION": "wht"}),
    ("Planar-K3V3",  {"TQ_KV_K_BITS": "3", "TQ_KV_V_BITS": "3", "TQ_KV_ROTATION": "planar"}),
    ("WHT-K4V3",     {"TQ_KV_K_BITS": "4", "TQ_KV_V_BITS": "3", "TQ_KV_ROTATION": "wht"}),
    ("Planar-K4V3",  {"TQ_KV_K_BITS": "4", "TQ_KV_V_BITS": "3", "TQ_KV_ROTATION": "planar"}),
]


def start_vllm(env_vars):
    env = os.environ.copy()
    for k in list(env.keys()):
        if k.startswith("TQ_KV_"):
            del env[k]
    env.update(env_vars)
    proc = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", MODEL,
         "--max-model-len", "4096",
         "--gpu-memory-utilization", "0.92",
         "--enforce-eager",
         "--port", "8000"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc


def wait_healthy(proc, timeout=1200):
    for i in range(timeout // 10):
        if proc.poll() is not None:
            out = proc.stdout.read().decode()[-500:]
            return False, out
        try:
            r = requests.get("http://localhost:8000/health", timeout=3)
            if r.status_code == 200:
                return True, "ready after %ds" % ((i + 1) * 10)
        except Exception:
            pass
        time.sleep(10)
    return False, "timeout"


def run_prompts():
    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        t0 = time.time()
        resp = requests.post("http://localhost:8000/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200, "temperature": 0,
        })
        elapsed = time.time() - t0
        data = resp.json()
        if "choices" in data:
            answer = data["choices"][0]["message"]["content"]
            toks = data.get("usage", {}).get("completion_tokens", 0)
            ptoks = data.get("usage", {}).get("prompt_tokens", 0)
            results.append({"prompt": prompt, "answer": answer[:200],
                            "tokens": toks, "prompt_tokens": ptoks, "time": elapsed})
            print("  [%d] (%.1fs, %d tok) %s" % (i, elapsed, toks, prompt[:50]))
            print("      -> %s" % answer[:120])
        else:
            print("  [%d] ERROR: %s" % (i, json.dumps(data)[:200]))
            results.append({"prompt": prompt, "error": str(data), "time": elapsed})
    return results


def get_gpu_mem():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode().strip()
    return int(out)


print()
print("=" * 70)
print("  Gemma 4 26B: WHT vs Planar KV Cache Compression")
print("=" * 70)

all_results = {}
for name, env_vars in CONFIGS:
    print("\n--- %s ---" % name)

    os.system("pkill -9 -f vllm 2>/dev/null; sleep 3")

    proc = start_vllm(env_vars)
    ok, msg = wait_healthy(proc)

    if not ok:
        print("  FAILED: %s" % msg[-200:])
        all_results[name] = {"status": "failed", "error": msg[-200:]}
        proc.kill()
        continue

    print("  %s" % msg)
    gpu_mem = get_gpu_mem()
    print("  GPU memory: %d MiB" % gpu_mem)

    results = run_prompts()
    total_toks = sum(r.get("tokens", 0) for r in results)
    total_time = sum(r.get("time", 0) for r in results)
    tok_s = total_toks / total_time if total_time > 0 else 0

    all_results[name] = {
        "status": "ok", "gpu_mib": gpu_mem,
        "total_tokens": total_toks, "total_time": total_time,
        "tok_s": tok_s, "results": results,
    }
    print("  TOTAL: %d tokens / %.1fs = %.1f tok/s" % (total_toks, total_time, tok_s))

    proc.kill()
    proc.wait()

# Summary
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
header = "  %-20s %8s %7s %7s %7s" % ("Config", "GPU MiB", "Tokens", "Time", "tok/s")
print(header)
print("-" * 55)
for name, r in all_results.items():
    if r["status"] == "ok":
        print("  %-20s %8d %7d %6.1fs %6.1f" % (
            name, r["gpu_mib"], r["total_tokens"], r["total_time"], r["tok_s"]))
    else:
        print("  %-20s  FAILED" % name)

with open("/root/benchmark_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nResults saved to /root/benchmark_results.json")
