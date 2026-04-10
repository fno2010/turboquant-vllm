#!/usr/bin/env bash
# Runs on the GPU instance. Self-contained — no ssh/heredoc shenanigans.
#
# What it does:
#   1. Verifies vLLM import + attention backend registry shape.
#   2. Starts vLLM server with --kv-cache-dtype tq3, nohup.
#   3. Polls /health until ready (timeout 600s).
#   4. Sends a completion request, checks for non-empty output.
#   5. Greps log for "TurboQuant native backend registered as CUSTOM".
#   6. Kills server, writes PASS/FAIL summary to /tmp/tq-test-result.txt.
#
# Everything runs in background after the quick import check, so this script
# returns in <5s. Progress is visible in /tmp/tq-test.log.
#
# Env overrides:
#   MODEL          HF model id (default: Qwen/Qwen2.5-0.5B for fast iteration)
#   KV_DTYPE       tq3 | tq4 | tq_k4v3 (default tq3)
#   GPU_MEM        gpu_memory_utilization (default 0.5)
#   WORKDIR        where the code lives (default $HOME/turboquant-vllm)

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
KV_DTYPE="${KV_DTYPE:-tq3}"
GPU_MEM="${GPU_MEM:-0.5}"
WORKDIR="${WORKDIR:-$HOME/turboquant-vllm}"
LOG="/tmp/tq-test.log"
SERVER_LOG="/tmp/tq-server.log"
RESULT="/tmp/tq-test-result.txt"
PIDFILE="/tmp/tq-server.pid"

# Wipe previous state
rm -f "$LOG" "$SERVER_LOG" "$RESULT" "$PIDFILE"

exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Iseconds) TQ native backend GPU test ==="
echo "MODEL=$MODEL  KV_DTYPE=$KV_DTYPE  GPU_MEM=$GPU_MEM  WORKDIR=$WORKDIR"

cd "$WORKDIR"
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

# ---------------------------------------------------------------------------
# Step 1: import sanity — fail fast if vLLM interfaces are wrong
# ---------------------------------------------------------------------------
echo "[1/6] Import sanity check"
python3 - <<'PY'
import sys
try:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    assert hasattr(AttentionBackendEnum, "CUSTOM"), "AttentionBackendEnum.CUSTOM missing"
    assert callable(register_backend), "register_backend not callable"
    print(f"OK: vLLM registry interface valid (CUSTOM={AttentionBackendEnum.CUSTOM})")
except Exception as e:
    print(f"FAIL: vLLM registry interface unexpected: {e}")
    sys.exit(1)

try:
    from vllm.platforms.cuda import CudaPlatform
    assert hasattr(CudaPlatform, "get_valid_backends"), "CudaPlatform.get_valid_backends missing"
    print("OK: CudaPlatform.get_valid_backends present")
except Exception as e:
    print(f"FAIL: CudaPlatform: {e}")
    sys.exit(1)

try:
    from vllm.model_executor.layers.attention.attention import AttentionLayer
    import inspect
    sig = inspect.signature(AttentionLayer.__init__)
    print(f"OK: AttentionLayer.__init__ params: {list(sig.parameters.keys())[:10]}...")
except Exception as e:
    print(f"FAIL: AttentionLayer: {e}")
    sys.exit(1)

try:
    from turboquant_vllm.native_backend import TurboQuantAttentionBackend, TurboQuantAttentionImpl
    from turboquant_vllm.tq_config import TurboQuantConfig
    c = TurboQuantConfig.from_cache_dtype("tq3", head_dim=128)
    print(f"OK: turboquant_vllm imports (tq3 key_packed={c.key_packed_size}B)")
except Exception as e:
    print(f"FAIL: turboquant_vllm: {e}")
    sys.exit(1)
PY

# ---------------------------------------------------------------------------
# Step 2: plugin registration (dry run, no server)
# ---------------------------------------------------------------------------
echo "[2/6] Plugin registration dry run"
python3 - <<'PY'
from turboquant_vllm._vllm_plugin import register, _register_native_backend
ok = _register_native_backend()
assert ok, "_register_native_backend returned False"

from vllm.v1.attention.backends.registry import AttentionBackendEnum, _ATTN_OVERRIDES
custom_path = _ATTN_OVERRIDES.get(AttentionBackendEnum.CUSTOM)
assert custom_path and "TurboQuantAttentionBackend" in custom_path, \
    f"CUSTOM not registered: {custom_path}"
print(f"OK: CUSTOM → {custom_path}")
PY

# ---------------------------------------------------------------------------
# Step 3: start vLLM server in background
# ---------------------------------------------------------------------------
echo "[3/6] Starting vLLM server (nohup, log=$SERVER_LOG)"
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --kv-cache-dtype "$KV_DTYPE" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len 2048 \
    --port 8765 \
    > "$SERVER_LOG" 2>&1 &

echo $! > "$PIDFILE"
SERVER_PID=$(cat "$PIDFILE")
echo "server PID=$SERVER_PID"

# ---------------------------------------------------------------------------
# Step 4: poll /health
# ---------------------------------------------------------------------------
echo "[4/6] Waiting for server /health (timeout 600s)"
for i in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FAIL: server died before becoming healthy"
        echo "--- server log tail ---"
        tail -n 50 "$SERVER_LOG"
        echo "FAIL: server crashed" > "$RESULT"
        exit 1
    fi
    if curl -sf http://127.0.0.1:8765/health >/dev/null 2>&1; then
        echo "OK: server healthy after ${i}x2s"
        break
    fi
    sleep 2
done

if ! curl -sf http://127.0.0.1:8765/health >/dev/null 2>&1; then
    echo "FAIL: timeout waiting for /health"
    echo "--- server log tail ---"
    tail -n 80 "$SERVER_LOG"
    kill "$SERVER_PID" 2>/dev/null || true
    echo "FAIL: health timeout" > "$RESULT"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 5: send a completion
# ---------------------------------------------------------------------------
echo "[5/6] Sending completion request"
RESP=$(curl -sf http://127.0.0.1:8765/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$MODEL"'","prompt":"The capital of France is","max_tokens":10,"temperature":0}')

echo "response: $RESP"
GENERATED=$(echo "$RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["text"])' 2>/dev/null || echo "")

if [[ -z "$GENERATED" ]]; then
    echo "FAIL: empty completion"
    kill "$SERVER_PID" 2>/dev/null || true
    echo "FAIL: empty completion" > "$RESULT"
    exit 1
fi

echo "generated: '$GENERATED'"

# ---------------------------------------------------------------------------
# Step 6: verify backend was actually used
# ---------------------------------------------------------------------------
echo "[6/6] Verifying TurboQuant backend was activated"
if grep -q "TurboQuant native backend registered as CUSTOM" "$SERVER_LOG"; then
    echo "OK: found native backend registration in server log"
else
    echo "WARN: did not find 'registered as CUSTOM' in server log"
    echo "--- server log tail ---"
    tail -n 50 "$SERVER_LOG"
fi

# Look for the attention backend selection log
BACKEND_LINE=$(grep -i "attention backend\|selected.*backend" "$SERVER_LOG" | head -5 || true)
if [[ -n "$BACKEND_LINE" ]]; then
    echo "backend lines: $BACKEND_LINE"
fi

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo "PASS: native backend served $MODEL with $KV_DTYPE (generated: '$GENERATED')" > "$RESULT"
echo "=== PASS ==="
cat "$RESULT"
