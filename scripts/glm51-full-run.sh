#!/usr/bin/env bash
# GLM-5.1 full pipeline: compress → upload → serve → validate
#
# Designed for a 4x A100 80GB instance (has ~1.5 TB RAM, enough disk).
# Runs compression on CPU, then serves on GPU.
#
# Instance requirements:
#   - 4x A100 80GB (Verda: 4A100.88V in FIN-03, or 8A100.176V in FIN-02)
#   - 500 GB disk (output ~153 GB + temp downloads)
#   - ~64 GB RAM for compression (streaming, low peak)
#
# Usage:
#   ssh root@<ip> "nohup bash /root/turboquant-vllm/scripts/glm51-full-run.sh > /tmp/glm51-full.log 2>&1 &"
#
# Monitor:
#   ssh root@<ip> "tail -20 /tmp/glm51-full.log"
#   ssh root@<ip> "cat /tmp/glm51-full-result.txt"

set -euo pipefail
export PATH="/root/.local/bin:$PATH"

HF_REPO="${HF_REPO:-varjosoft/GLM-5.1-TQ3-native}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/glm51-tq3-native}"
RESULT="/tmp/glm51-full-result.txt"

rm -f "$RESULT"

echo "================================================"
echo "  GLM-5.1 TQ3 Full Pipeline"
echo "  $(date -u)"
echo "================================================"
echo ""

# ─── Step 1: Compress ─────────────────────────────────
echo ">>> Step 1/3: Compress checkpoint (CPU, ~2-4 hours)"
bash /root/turboquant-vllm/scripts/glm51-compress.sh || {
    echo "FAIL: compression step" > "$RESULT"
    exit 1
}
echo ""

# ─── Step 2: Upload ───────────────────────────────────
echo ">>> Step 2/3: Upload to HuggingFace"
python3 /root/turboquant-vllm/scripts/publish_model.py upload \
    "$OUTPUT_DIR" "$HF_REPO" || {
    echo "FAIL: upload step" > "$RESULT"
    exit 1
}
echo "Uploaded: https://huggingface.co/$HF_REPO"
echo ""

# ─── Step 3: Serve & Validate ─────────────────────────
echo ">>> Step 3/3: Serve and validate (GPU, ~15-30 min)"
export MODEL="$OUTPUT_DIR"  # serve from local to skip re-download
export TP=4
bash /root/turboquant-vllm/scripts/glm51-serve.sh || {
    echo "FAIL: serving step" > "$RESULT"
    exit 1
}

echo ""
echo "================================================"
echo "  COMPLETE: GLM-5.1 TQ3 pipeline done"
echo "  Checkpoint: https://huggingface.co/$HF_REPO"
echo "  $(date -u)"
echo "================================================"
echo "PASS: full pipeline" > "$RESULT"
