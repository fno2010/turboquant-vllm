#!/usr/bin/env bash
# Step 2: Serve GLM-5.1 TQ3 checkpoint on 4x A100 and validate
#
# Loads the native TQ3 checkpoint (from HuggingFace or local),
# serves with vLLM TP=4, validates coherent output, measures throughput.
#
# Resources: 4x A100 80GB (~52 GB/GPU used, 25 GB KV budget)
# Time: ~15-30 min (model loading + graph capture + validation)
#
# Usage: run on a 4x A100 instance after setup.
#   bash /root/turboquant-vllm/scripts/glm51-serve.sh

set -euo pipefail
export PATH="/root/.local/bin:$PATH"

# Use HuggingFace checkpoint (uploaded) or local path
MODEL="${MODEL:-varjosoft/GLM-5.1-TQ3-native}"
TP="${TP:-4}"
GPU_MEM="${GPU_MEM:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PORT="${PORT:-8000}"
LOG="/tmp/glm51-serve.log"
SERVER_LOG="/tmp/glm51-serve-server.log"
RESULT="/tmp/glm51-serve-result.txt"

rm -f "$LOG" "$SERVER_LOG" "$RESULT"
exec > >(tee -a "$LOG") 2>&1

echo "=== GLM-5.1 TQ3 Serving Test ==="
echo "Model: $MODEL"
echo "TP: $TP"
echo "Date: $(date -u)"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Kill stale
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

# Pre-flight
echo "Phase 1: Import check..."
python3 -c "
import vllm
print(f'vLLM {vllm.__version__}')
import turboquant_vllm
print('turboquant_vllm OK')
# Verify GLM-5.1 architecture is registered
from vllm.model_executor.models.registry import ModelRegistry
print('GlmMoeDsaForCausalLM registered:', 'GlmMoeDsaForCausalLM' in str(ModelRegistry.models))
" || { echo "FAIL: import check" > "$RESULT"; exit 1; }

echo ""
echo "Phase 2: Starting vLLM server (TP=$TP)..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --port $PORT \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Health poll (longer timeout for 754B model)
echo "Phase 3: Waiting for health..."
TIMEOUT=1800
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server healthy after ${ELAPSED}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "FAIL: server died during startup" > "$RESULT"
        tail -80 "$SERVER_LOG" >> "$RESULT"
        echo "Server died. Log tail:"
        tail -80 "$SERVER_LOG"
        exit 1
    fi
    sleep 15
    ELAPSED=$((ELAPSED + 15))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "FAIL: server did not start in ${TIMEOUT}s" > "$RESULT"
    tail -60 "$SERVER_LOG"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Check compression logs
echo ""
echo "Checking logs..."
grep -i "TurboQuant\|FusedMoE\|CUDA graph\|piecewise\|Capturing\|compressed" "$SERVER_LOG" | head -20 || true

# Smoke test
echo ""
echo "Phase 4: Smoke test..."
RESPONSE=$(curl -sf "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "prompt": "The capital of France is",
        "max_tokens": 50,
        "temperature": 0
    }' 2>&1) || { echo "FAIL: completion request" > "$RESULT"; kill $SERVER_PID; exit 1; }

TEXT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)
echo "Generated: $TEXT"

# Coherence
python3 -c "
text = '''$TEXT'''
text = text.strip()
if len(text) < 5: exit(1)
chars = set(text.replace(' ',''))
if len(chars) <= 2: exit(1)
print(f'OK: coherent ({len(chars)} unique chars)')
" || {
    echo "FAIL: gibberish output" > "$RESULT"
    echo "Output: $TEXT" >> "$RESULT"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
}

# Second test — reasoning
echo ""
echo "Phase 5: Reasoning test..."
RESPONSE2=$(curl -sf "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "prompt": "Write a Python function to check if a number is prime:\n\ndef is_prime(n):",
        "max_tokens": 150,
        "temperature": 0
    }' 2>&1)
TEXT2=$(echo "$RESPONSE2" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)
echo "Generated: $TEXT2"

# Throughput
echo ""
echo "Phase 6: Throughput c=1..."
python3 -c "
import json, time, urllib.request

def run_one(pid):
    start = time.time()
    data = json.dumps({
        'model': '$MODEL',
        'prompt': f'Explain the theory of relativity in simple terms. Version {pid}:',
        'max_tokens': 128,
        'temperature': 0.7,
    }).encode()
    req = urllib.request.Request('http://localhost:$PORT/v1/completions',
                                data=data,
                                headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    tokens = result['usage']['completion_tokens']
    elapsed = time.time() - start
    return tokens, elapsed

total_t, total_s = 0, 0
for i in range(4):
    t, s = run_one(i)
    total_t += t; total_s += s
    print(f'  req {i}: {t} tok in {s:.1f}s = {t/s:.1f} tok/s')
print(f'c=1: {total_t} tok in {total_s:.1f}s = {total_t/total_s:.1f} tok/s')
"

# Memory
echo ""
echo "Phase 7: GPU memory..."
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

echo ""
echo "=== DONE ==="
echo "PASS: GLM-5.1 TQ3 TP=$TP serving validated" > "$RESULT"
echo "Smoke: $TEXT" >> "$RESULT"

grep -i "cuda graph\|piecewise\|captured" "$SERVER_LOG" | tail -5 || true

kill $SERVER_PID 2>/dev/null || true
cat "$RESULT"
